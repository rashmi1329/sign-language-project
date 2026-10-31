# EXPERIMENT E2: full source->target transfer grid at a fixed 60 epochs, scratch arm.
#
# The E1 model (4-layer pose transformer, random init, 512-d) -> T5-small decoder. Full grid: 10 per-language models + 90 transfer cells.
# Identical to E1: lr 3e-4 AdamW wd 0.01, OneCycle, batch 16, same augmentation, LAST epoch used
# (no dev-based checkpoint choice), beam-4 decode, max 32 new tokens, test = full split, target text = the same English sentence for every sign language.
# Epochs are fixed at 60 BEFORE running (midpoint of the 30-100 range; E1 showed 30 under-fits and 100 memorises; 60 itself is untested).
# One seed per notebook run: set SEED below. Resumable: results are appended to e2_results_scratch_s{SEED}.json after every cell.
import torch, torch.nn as nn, json, math, random, time, os, gc, numpy as np
from sacrebleu.metrics import BLEU, CHRF
import pilot_pipeline as pp
assert torch.cuda.is_available() or os.environ.get('E2_ALLOW_CPU') == '1', 'no GPU'

SEED = int(os.environ.get('E2_SEED', '0'))
GATE = os.environ.get('E2_GATE', '0') == '1'              # GATE=1: tiny speed/sanity test (en scratch + en->lt transfer, 2 epochs). Results are NOT for reporting.
EPOCHS = 2 if GATE else 60                                # 60 is fixed before running; do not change after seeing any result
if GATE: print('*** GATE MODE: 2 epochs, 1 scratch cell + 1 transfer cell, not for reporting ***')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; LANG = list(d.LANGS); data = d._load_all()
WORK = os.environ.get('E2_WORK', '/kaggle/working'); TAG = ('gate_' if GATE else '') + 'scratch' + f'_s{SEED}'
OUTJ = f'{WORK}/e2_results_{TAG}.json'; CKDIR = f'{WORK}/ck_{TAG}'; os.makedirs(CKDIR, exist_ok=True)
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
bl = lambda h, r: BLEU().corpus_score(h, [r]).score

# ---------------- DATA CHECKS (same as E1; fail before any GPU time is spent) ----------------
EXPECT_TEST = {'zh': 211, 'uk': 209, 'ru': 211, 'bg': 192, 'is': 192, 'de': 185, 'it': 201, 'sv': 212, 'lt': 194, 'en': 214}
for l in LANG:
    for sp, n in (('train', 830), ('dev', 142), ('test', EXPECT_TEST[l])):
        X, Y, K = data[(l, sp)]
        assert len(X) == len(Y) == len(K) == n and len(set(K)) == n, (l, sp, 'size/key problem')
        assert all(isinstance(y, str) and y.strip() for y in Y) and np.isfinite(X).all(), (l, sp, 'bad reference or pose values')
print('DATA CHECKS PASSED')

# ---------------- MODELS ----------------
build = d.build_slt_model

# ---------------- ONE TRAINING RUN (identical loop to E1's run(); optional warm start from a source model for transfer cells) ----------------
def run(target, seed, init_state=None, lr=3e-4, batch=16):
    torch.manual_seed(seed); rng = random.Random(seed)
    model, tok = build()
    if init_state is not None: model.load_state_dict(init_state)
    model.to(DEV)
    Xtr, Ytr, _ = data[(target, 'train')]; Xte, Yte, Kte = data[(target, 'test')]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=EPOCHS * math.ceil(len(Xtr) / batch))
    Xt = torch.tensor(Xtr); lab = tok(Ytr, padding=True, truncation=True, max_length=32, return_tensors='pt').input_ids; lab[lab == tok.pad_token_id] = -100
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]; xb = pp.augment(Xt[idx], rng).to(DEV)
            loss = model(xb, lab[idx].to(DEV)); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item() * len(idx)
    assert math.isfinite(tot), 'non-finite training loss'
    hy = d.decode(model, tok, Xte, DEV); assert len(hy) == len(Yte) == len(Kte), 'decode length mismatch'
    return model, dict(target=target, seed=seed, epochs=EPOCHS, train_loss=round(tot / len(Xt), 4), test_bleu=round(bl(hy, Yte), 3),
                       test_chrf=round(CHRF().corpus_score(hy, [Yte]).score, 3), distinct=len(set(hy)), n_test=len(Yte), secs=int(time.time() - t0), epochs_run=EPOCHS,
                       peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0, keys=Kte, hyps=hy)

# ---------------- CELL LIST (fixed in advance) ----------------
# all 90 ordered pairs
PAIRS = [(s, t) for s in LANG for t in LANG if s != t]
LANG_RUN = ['en'] + [x for x in LANG if x != 'en']
if GATE: LANG_RUN, PAIRS = ['en'], [('en', 'lt')]
print('scratch', 'cells:', len(LANG_RUN), 'scratch-equivalent +', len(PAIRS), 'transfer'); json.dump(PAIRS, open(f'{WORK}/e2_pairs_{TAG}.json', 'w'))

# ---------------- SMOKE TEST (1 epoch, English, discarded) ----------------
if not res:
    EPOCHS_SAVE = EPOCHS; EPOCHS = 1; _m, r = run('en', SEED); EPOCHS = EPOCHS_SAVE; del _m; gc.collect()
    print('SMOKE OK', {k: v for k, v in r.items() if k not in ('keys', 'hyps')})

# ---------------- MAIN LOOP ----------------
spent = sum(v['secs'] for v in res.values())
def save(): json.dump(res, open(OUTJ, 'w'))
for l in LANG_RUN:                                                       # 1) scratch-equivalent model per language; also the source model for its transfers
    key = f'scratch_{l}'; ck = f'{CKDIR}/{l}.pt'
    if key in res and os.path.exists(ck): continue
    model, r = run(l, SEED); torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ck); res[key] = r; spent += r['secs']; save()
    print(key, {k: v for k, v in r.items() if k not in ('keys', 'hyps')}); del model; gc.collect(); torch.cuda.empty_cache()
for s, t in PAIRS:                                                       # 2) transfer: start from the source model, train on the target, same 60 epochs
    key = f'transfer_{s}_{t}'
    if key in res: continue
    if not os.path.exists(f'{CKDIR}/{s}.pt'): print('missing source model', s, 'skipping', key); continue
    st = torch.load(f'{CKDIR}/{s}.pt'); model, r = run(t, SEED, init_state=st); r['source'] = s; res[key] = r; spent += r['secs']; save()
    print(key, {k: v for k, v in r.items() if k not in ('keys', 'hyps')}); del model, st; gc.collect(); torch.cuda.empty_cache()

# ---------------- COMPLETENESS + ALIGNMENT CHECKS ----------------
expected = len(LANG_RUN) + len(PAIRS); got = len(res)
print(f'COMPLETENESS: {got} of {expected} cells', '' if got == expected else '(INCOMPLETE - report as such)')
for k, v in res.items(): assert v['keys'] == data[(v['target'], 'test')][2], ('test key mismatch', k)
for ep_name in ('scratch',):
    v = [res[k]['test_bleu'] for k in res if k.startswith('scratch_')]; print('mean scratch-equivalent test BLEU %.3f over %d models' % (np.mean(v), len(v)))
g = [res[f'transfer_{s}_{t}']['test_bleu'] - res[f'scratch_{t}']['test_bleu'] for s, t in PAIRS if f'transfer_{s}_{t}' in res and f'scratch_{t}' in res]
print('mean transfer gain (BLEU, transfer minus target-scratch): %.3f over %d cells' % (np.mean(g), len(g)) if g else 'no transfer cells')

# ---------------- CLEAN-UP (after every result is saved above): drop the per-language model checkpoints so the Kaggle output holds only the small result files ----------------
# (the queue script downloads the whole output.)  Set E2_KEEP_CKPT=1 to keep them.
import shutil
if os.environ.get('E2_KEEP_CKPT') != '1': shutil.rmtree(CKDIR, ignore_errors=True); print('checkpoints removed; results kept in', OUTJ)
