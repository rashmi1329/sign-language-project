# E2-RTM: transfer grid on the cleaned RTMPose set. Written 2026-10-19; frozen by docs/analysis_plan_rtm.md.
# ARM = 'scratch'  : 4-layer pose transformer (random init) -> T5-small decoder.            cells: 10 per-language + 90 transfer
# ARM = 'uni'      : Uni-Sign 4-stream ST-GCN (ASL pose-only checkpoint) -> Linear(1024,512)+LayerNorm -> T5-small.   cells: 10 + nearest/farthest source per target (20)
# ARM = 'uni_ctrl' : the same network as 'uni' with random init (same seed).                   cells: as 'uni'
# One STREAM = one process on one GPU; it owns a list of languages (RTM_OWN): it trains a per-language model for each owned language (that model is both the
# baseline of that language and the source of its transfers) and then every transfer cell whose SOURCE is an owned language. No stream needs another's checkpoint.
# Training settings are those of e2_cell.py v4 (60 epochs, AdamW lr 3e-4 wd 0.01, OneCycle, batch 16, last epoch, beam 4, max 32 new tokens, full test split),
# except: input 48x69x4 (x, y, conf, observed), and augmentation shifts only body points (hands/face are wrist/nose-relative; see augment_rtm).
import os, sys, json, math, random, time, gc
import numpy as np, torch, torch.nn as nn
from sacrebleu.metrics import BLEU, CHRF
ROOT = os.environ['RTM_ROOT']; sys.path.insert(0, ROOT)
from rtm_data import LANGS, SL, obs_mask
ARM = os.environ.get('RTM_ARM', 'scratch'); SEED = int(os.environ.get('RTM_SEED', '0')); GATE = os.environ.get('RTM_GATE', '0') == '1'
OWN = [x for x in os.environ.get('RTM_OWN', ','.join(LANGS)).split(',') if x]; TAGS = os.environ.get('RTM_STREAM', 'all')
EPOCHS = int(os.environ['RTM_EPOCHS']) if os.environ.get('RTM_EPOCHS') else (2 if GATE else 60);   # RTM_EPOCHS is for CPU tests only; the grid uses 60 (fixed in the analysis plan)
GUARD_S = float(os.environ.get('RTM_GUARD_H', '3')) * 3600
WORK = os.environ.get('RTM_WORK', '/kaggle/working'); CKPT = os.environ.get('RTM_UNI_CKPT', '')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; assert DEV == 'cuda' or os.environ.get('RTM_ALLOW_CPU') == '1', 'no GPU'
if os.environ.get('RTM_EPOCHS'): print('*** RTM_EPOCHS override =', EPOCHS, '- NOT FOR REPORTING ***')
assert ARM in ('scratch', 'uni', 'uni_ctrl'); os.makedirs(WORK, exist_ok=True)
TAG = ('gate_' if GATE else '') + f'{ARM}_s{SEED}_{TAGS}'; OUTJ = f'{WORK}/rtm_results_{TAG}.json'
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
bl = lambda h, r: BLEU().corpus_score(h, [r]).score
T0 = time.time()

# ---------------- DATA (+ checks against the manifest, before any GPU time is spent) ----------------
man = json.load(open(f'{ROOT}/manifest.json')); refs = json.load(open(f'{ROOT}/english_refs.json'))
def load_split(lang):
    z = np.load(f'{ROOT}/rtm_std_{lang}.npz'); out = {'train': ([], [], []), 'dev': ([], [], []), 'test': ([], [], [])}
    for i in sorted(man['languages'][lang]['ids']):
        sp = i.split('/')[0]; x = z['x|' + i].astype(np.float32); o = obs_mask(x, z['m|' + i]).astype(np.float32)[..., None]
        X, Y, K = out[sp]; X.append(np.concatenate([x * o, o], -1)); Y.append(refs[i]); K.append(i)
    return {sp: (np.stack(v[0]).astype(np.float32), v[1], v[2]) for sp, v in out.items()}
data = {l: load_split(l) for l in LANGS}
for l in LANGS:
    for sp in ('train', 'dev', 'test'):
        X, Y, K = data[l][sp]; n = sum(1 for i in man['languages'][l]['ids'] if i.startswith(sp + '/'))
        assert len(X) == len(Y) == len(K) == n and len(set(K)) == n, (l, sp, 'size/key problem')
        assert all(isinstance(y, str) and y.strip() for y in Y) and np.isfinite(X).all() and X.shape[1:] == (48, 69, 4), (l, sp, 'bad reference or pose values')
print('DATA CHECKS PASSED', {l: len(data[l]['train'][0]) for l in LANGS}, flush=True)
if os.environ.get('RTM_FAST'):                                       # CPU smoke test only; never set on Kaggle
    _n = int(os.environ['RTM_FAST']); data = {l: {sp: (v[0][:_n], v[1][:_n], v[2][:_n]) for sp, v in d.items()} for l, d in data.items()}; print('FAST MODE: data truncated to', _n, 'clips per split - NOT FOR REPORTING')

# ---------------- MODELS ----------------
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers.modeling_outputs import BaseModelOutput
_UNI_SD = None
def build():
    tok = T5TokenizerFast.from_pretrained('t5-small'); t5 = T5ForConditionalGeneration.from_pretrained('t5-small'); dm = t5.config.d_model
    if ARM == 'scratch':
        class PoseT5(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(69 * 4, dm); self.pos = nn.Parameter(torch.zeros(1, 48, dm))
                self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, 8, 4 * dm, 0.2, batch_first=True), 4); self.norm = nn.LayerNorm(dm); self.t5 = t5
            def encode(self, x): return self.norm(self.enc(self.inp(x.flatten(2)) + self.pos))
            def forward(self, x, labels): return self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=self.encode(x)), labels=labels).loss
            def generate(self, x, **kw): return self.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=self.encode(x)), max_new_tokens=32, num_beams=4, **kw)
        return PoseT5(), tok
    from stgcn_uni import UniPoseEncoder, load_uni_checkpoint
    global _UNI_SD
    class PoseUniT5(nn.Module):
        def __init__(self):
            super().__init__()
            self.uni = UniPoseEncoder(); self.out = nn.Linear(1024, dm); self.norm = nn.LayerNorm(dm); self.t5 = t5
        def encode(self, x): return self.norm(self.out(self.uni(x)))
        def forward(self, x, labels): return self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=self.encode(x)), labels=labels).loss
        def generate(self, x, **kw): return self.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=self.encode(x)), max_new_tokens=32, num_beams=4, **kw)
    m = PoseUniT5()
    if ARM == 'uni':
        if _UNI_SD is None:
            assert CKPT and os.path.exists(CKPT), 'Uni-Sign checkpoint missing'
            rep = load_uni_checkpoint(m.uni, CKPT); print('UNI-SIGN CHECKPOINT LOAD', {k: v for k, v in rep.items()}, flush=True)
            assert rep['frac'] >= 0.95 and not rep['bad_shape'], ('checkpoint does not fit the encoder', rep)
            _UNI_SD = {k: v.clone() for k, v in m.uni.state_dict().items()}; json.dump(rep, open(f'{WORK}/uni_load_report_{TAG}.json', 'w'))
        else: m.uni.load_state_dict(_UNI_SD)
    return m, tok

def augment_rtm(x, rng):
    """Same ranges as pilot_pipeline.augment (scale 0.85-1.15, shift +-0.1, rotation +-0.2 rad, 50% temporal crop) with ONE change: the shift is applied to body points
    only (points 0-8). Hand points are relative to the wrist and face points to the nose, so shifting them would move the wrist/nose off the origin that defines them.
    Unobserved points are re-zeroed (channels x, y, conf); the observed flag (channel 3) is untouched."""
    B = x.shape[0]
    s = torch.empty(B, 1, 1, 1).uniform_(0.85, 1.15); sh = torch.empty(B, 1, 1, 2).uniform_(-0.1, 0.1); th = torch.empty(B).uniform_(-0.2, 0.2)
    c, sn = torch.cos(th), torch.sin(th); xy = x[..., :2] * s
    xr = torch.stack([c[:, None, None] * xy[..., 0] - sn[:, None, None] * xy[..., 1], sn[:, None, None] * xy[..., 0] + c[:, None, None] * xy[..., 1]], -1)
    shift = torch.zeros_like(xr); shift[:, :, 0:9, :] = sh; xr = xr + shift
    x = torch.cat([xr, x[..., 2:]], -1); x = torch.cat([x[..., :3] * x[..., 3:4], x[..., 3:4]], -1)
    if rng.random() < 0.5:
        T = x.shape[1]; a = rng.randint(0, T // 8); b = T - rng.randint(0, T // 8); x = x[:, torch.linspace(a, b - 1, T).round().long()]
    return x

def decode(model, tok, X, batch=32):
    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch): outs += tok.batch_decode(model.generate(torch.tensor(X[i:i + batch]).to(DEV)), skip_special_tokens=True)
    return outs

def run(target, seed, init_state=None, lr=3e-4, batch=16, want_s4=False):
    torch.manual_seed(seed); rng = random.Random(seed)
    model, tok = build()
    if init_state is not None: model.load_state_dict(init_state)
    model.to(DEV)
    Xtr, Ytr, _ = data[target]['train']; Xte, Yte, Kte = data[target]['test']
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=EPOCHS * math.ceil(len(Xtr) / batch))
    Xt = torch.tensor(Xtr); lab = tok(Ytr, padding=True, truncation=True, max_length=32, return_tensors='pt').input_ids; lab[lab == tok.pad_token_id] = -100
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]; xb = augment_rtm(Xt[idx], rng).to(DEV)
            loss = model(xb, lab[idx].to(DEV)); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item() * len(idx)
    assert math.isfinite(tot), 'non-finite training loss'
    hy = decode(model, tok, Xte); assert len(hy) == len(Yte) == len(Kte), 'decode length mismatch'
    r = dict(target=target, seed=seed, arm=ARM, epochs=EPOCHS, train_loss=round(tot / len(Xt), 4), test_bleu=round(bl(hy, Yte), 3), test_chrf=round(CHRF().corpus_score(hy, [Yte]).score, 3),
             distinct=len(set(hy)), n_test=len(Yte), secs=int(time.time() - t0), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0, keys=Kte, hyps=hy)
    if want_s4:                                                      # mean encoder output over the train split (S4 comparator), eval mode
        model.eval(); acc = 0; n = 0
        with torch.no_grad():
            for i in range(0, len(Xt), 64):
                h = model.encode(Xt[i:i + 64].to(DEV)); acc = acc + h.sum((0, 1)).double().cpu(); n += h.shape[0] * h.shape[1]
        r['s4_mean'] = [round(float(v), 5) for v in (acc / n)]
    return model, r

# ---------------- CELL LIST (fixed in advance) ----------------
sim = json.load(open(f'{ROOT}/similarity_rtm.json'))
dist = lambda s, t: sim[f'{s}|{t}']['S1_dtw'] if f'{s}|{t}' in sim else sim[f'{t}|{s}']['S1_dtw']
if ARM == 'scratch': PAIRS = [(s, t) for s in LANGS for t in LANGS if s != t]
else:
    PAIRS = []
    for t in LANGS:
        srcs = sorted([s for s in LANGS if s != t], key=lambda s: dist(s, t)); PAIRS += [(srcs[0], t), (srcs[-1], t)]
if GATE: OWN, PAIRS = ['en'], [('en', 'lt')]
mine = [(s, t) for s, t in PAIRS if s in OWN]
print(f'{TAG}: owns {OWN}: {len(OWN)} per-language models + {len(mine)} transfers = {len(OWN) + len(mine)} cells', flush=True)
json.dump(PAIRS, open(f'{WORK}/rtm_pairs_{ARM}.json', 'w'))
def save(): json.dump(res, open(OUTJ, 'w'))
def out_of_time(): return time.time() - T0 > GUARD_S

# ---------------- SMOKE TEST ----------------
if not res and not GATE:
    E = EPOCHS; EPOCHS = 1; _m, r = run(OWN[0], SEED); EPOCHS = E; del _m; gc.collect()
    print('SMOKE OK', {k: v for k, v in r.items() if k not in ('keys', 'hyps')}, flush=True)

# ---------------- MAIN LOOP ----------------
stopped = False
for s in OWN:
    todo = [(a, t) for a, t in mine if a == s and f'transfer_{s}_{t}' not in res]
    ck = f'{WORK}/ck_{TAG}_{s}.pt'
    if f'scratch_{s}' in res and not todo: continue
    if out_of_time(): print('TIME GUARD before', s, flush=True); stopped = True; break
    model, r = run(s, SEED, want_s4=True); torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ck)
    if f'scratch_{s}' not in res: res[f'scratch_{s}'] = r; save()
    print(f'scratch_{s}', {k: v for k, v in r.items() if k not in ('keys', 'hyps', 's4_mean')}, flush=True); del model; gc.collect(); torch.cuda.empty_cache()
    st = torch.load(ck)
    for _, t in todo:
        if out_of_time(): print('TIME GUARD before', s, t, flush=True); stopped = True; break
        model, r = run(t, SEED, init_state=st); r['source'] = s; res[f'transfer_{s}_{t}'] = r; save()
        print(f'transfer_{s}_{t}', {k: v for k, v in r.items() if k not in ('keys', 'hyps')}, flush=True); del model; gc.collect(); torch.cuda.empty_cache()
    del st
    if stopped: break
    os.remove(ck)
expected = len(OWN) + len(mine); print(f'COMPLETENESS: {len(res)} of {expected} cells', '' if len(res) == expected else '(INCOMPLETE - rerun to resume)', flush=True)
for k, v in res.items(): assert v['keys'] == data[v['target']]['test'][2], ('test key mismatch', k)
g = [res[f'transfer_{s}_{t}']['test_chrf'] - res[f'scratch_{t}']['test_chrf'] for s, t in mine if f'transfer_{s}_{t}' in res and f'scratch_{t}' in res]
print('mean chrF gain over cells whose target baseline is in this stream: %.3f over %d' % (np.mean(g), len(g)) if g else 'no gains computable in this stream', flush=True)
