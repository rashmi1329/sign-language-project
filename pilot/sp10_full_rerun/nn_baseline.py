"""Retrieval baselines on the cleaned RTMPose clips (exploratory; limitations entry 75). No training. Same test clips and corpus chrF/BLEU (sacrebleu) as the models.
Fixed before looking at any score:
 nn_flat   : for each test clip, the English reference of the nearest TRAIN clip of the SAME language; features = the 48x69 (x, y) values with unobserved points zeroed, centred on the train mean, L2-normalised, cosine.
 nn_pooled : same, features = per-point mean and std over observed frames of x and y (276 values), standardised with train statistics, cosine.
 random    : a random train sentence of that language (mean of 30 draws).
 const     : one train sentence for everything, chosen to maximise chrF on the dev references (never on test).
 oracle    : for each test clip the train sentence with the highest sentence chrF to ITS TEST REFERENCE (upper bound of any pure retrieval of train sentences).
Usage: python3 nn_baseline.py -> rtm_analysis/rtm_nn_baseline.json (scores and hypotheses; hypotheses are scored with BLEURT in the cloud)"""
import json, os, glob, numpy as np
from sacrebleu.metrics import BLEU, CHRF
H = os.path.dirname(os.path.abspath(__file__)); LANGS = ['zh','uk','ru','bg','is','de','it','sv','lt','en']
exec(open(f'{H}/rtm_data.py').read().split('def load_lang')[0]) if False else None
import importlib.util
spec = importlib.util.spec_from_file_location('rtm_data', f'{H}/rtm_data.py'); rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
refs = json.load(open(f'{H}/kaggle_dataset_rtm_c/english_refs.json')); chrf = lambda h, r: CHRF().corpus_score(h, [r]).score; bleu = lambda h, r: BLEU().corpus_score(h, [r]).score
v2 = {}
for f in sorted(glob.glob(f'{H}/e2_outputs/rtm2_s[12]_c/rtm_results_scratch_s0_g*_e100_*.json')): v2.update(json.load(open(f)))
test_ids = {l: v2[f'scratch_{l}']['keys'] for l in LANGS}
def feats(x, m):
    xy = x[..., :2] * m[..., None]; flat = xy.reshape(-1)
    w = m[..., None].astype(float); n = np.maximum(w.sum(0), 1); mu = (xy * w).sum(0) / n; sd = np.sqrt(((xy - mu) ** 2 * w).sum(0) / n)
    return flat, np.concatenate([mu.reshape(-1), sd.reshape(-1)])
def nn(Ftr, Fte):
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-6
    return mu, sd
def cos_nn(Ftr, Fte, standardise):
    mu = Ftr.mean(0); sd = (Ftr.std(0) + 1e-6) if standardise else 1.0
    A = (Ftr - mu) / sd; B = (Fte - mu) / sd
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9; B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    return (B @ A.T).argmax(1)
train_pool = sorted({k for k in refs if k.startswith('train/')}); dev_pool = sorted(k for k in refs if k.startswith('dev/'))
print('choosing constant sentence on dev ...', flush=True)
dev_refs = [refs[k] for k in dev_pool]; best = max(train_pool, key=lambda k: chrf([refs[k]] * len(dev_refs), dev_refs)); const = refs[best]; print('  const:', repr(const), flush=True)
test_union = sorted({k for l in LANGS for k in test_ids[l]}); print('oracle over', len(test_union), 'test sentences x', len(train_pool), flush=True)
cands = [refs[k] for k in train_pool]; ch = CHRF(); orc = {}
for k in test_union:
    sc = [ch.sentence_score(c, [refs[k]]).score for c in cands]; orc[k] = cands[int(np.argmax(sc))]
OUT = {'const_sentence': const, 'langs': {}}; rng = np.random.RandomState(0)
for l in LANGS:
    z = rd.load_lang(f'{H}/rtm_clean', l); tr = sorted(k for k in z if k.startswith('train/')); te = test_ids[l]; assert all(k in z for k in te)
    Ftr = [feats(*z[k]) for k in tr]; Fte = [feats(*z[k]) for k in te]; R = [refs[k] for k in te]; trs = [refs[k] for k in tr]; o = {}
    hy = {}
    hy['nn_flat'] = [trs[i] for i in cos_nn(np.stack([a for a, _ in Ftr]), np.stack([a for a, _ in Fte]), False)]
    hy['nn_pooled'] = [trs[i] for i in cos_nn(np.stack([b for _, b in Ftr]), np.stack([b for _, b in Fte]), True)]
    hy['const'] = [const] * len(te); hy['oracle'] = [orc[k] for k in te]
    for n, h in hy.items(): o[n] = dict(chrf=round(chrf(h, R), 3), bleu=round(bleu(h, R), 3), hyps=h)
    rc = [], []
    for _ in range(30): h = [trs[i] for i in rng.randint(0, len(trs), len(te))]; rc[0].append(chrf(h, R)); rc[1].append(bleu(h, R))
    o['random'] = dict(chrf=round(float(np.mean(rc[0])), 3), bleu=round(float(np.mean(rc[1])), 3), chrf_sd=round(float(np.std(rc[0])), 3))
    o['n_test'] = len(te); o['n_train'] = len(tr); OUT['langs'][l] = o
    print(l, {n: (o[n]['chrf'], o[n]['bleu']) for n in ('nn_flat', 'nn_pooled', 'random', 'const', 'oracle')}, flush=True)
json.dump(OUT, open(f'{H}/rtm_analysis/rtm_nn_baseline.json', 'w'))
print('mean over languages (chrF / BLEU):', {n: (round(np.mean([OUT['langs'][l][n]['chrf'] for l in LANGS]), 2), round(np.mean([OUT['langs'][l][n]['bleu'] for l in LANGS]), 2)) for n in ('nn_flat', 'nn_pooled', 'random', 'const', 'oracle')})
