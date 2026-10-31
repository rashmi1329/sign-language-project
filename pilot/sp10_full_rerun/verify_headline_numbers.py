#!/usr/bin/env python3
"""Independent recomputation of the headline numbers of the paper from the RAW per-cell result files.

It does NOT import or reuse any of the project's analysis scripts. It reads only
  - the per-cell result JSONs written by the training notebooks (e2_outputs/*/rtm_results_*.json, e2_results_scratch_s0.json)
  - the distance matrices (similarity_rtm.json for Rounds B/C, s1_new_from_log.json for Round A)
and recomputes Table 3, selected rows of Tables 4 and 5, the R^2 of target and source, and the copy rates.

Run from pilot/sp10_full_rerun :   python3 verify_headline_numbers.py
Output: printed table 'paper value | recomputed value'.
Written for the GitHub release (2026-10-31); QAP p-values use its own random stream, so they agree with the paper
to about +-0.01, not exactly. Slopes, means and counts must agree exactly (to the digits printed in the paper).
"""
import json, glob, os, re, numpy as np
H = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
E = os.path.join(H, 'e2_outputs')
L = ['zh', 'uk', 'ru', 'bg', 'is', 'de', 'it', 'sv', 'lt', 'en']; ix = {l: i for i, l in enumerate(L)}
CODE = dict(zip(['CSL', 'UKL', 'RSL', 'BQN', 'ICL', 'GSG', 'ISE', 'SWL', 'LLS', 'BFI'], L))

def dist(src):
    D = np.zeros((10, 10))
    if src == 'A':
        s = json.load(open(os.path.join(H, 's1_new_from_log.json')))
        for k, v in s.items():
            a, b = k.split('-'); D[ix[CODE[a]], ix[CODE[b]]] = D[ix[CODE[b]], ix[CODE[a]]] = v[0]
    else:
        s = json.load(open(os.path.join(H, 'similarity_rtm.json')))
        for k, v in s.items():
            a, b = k.split('|'); D[ix[a], ix[b]] = D[ix[b], ix[a]] = v['S1_dtw']
    return D

def load(patterns):
    res = {}
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(E, pat))):
            if 'sweep' in f or 'gate' in f: continue
            res.update(json.load(open(f)))
    return res

def mats(res, metric):
    base = np.full(10, np.nan); T = np.full((10, 10), np.nan)
    for k, v in res.items():
        x = float(v['test_' + metric])
        if k.startswith('scratch_'): base[ix[k[8:]]] = x
        elif k.startswith('transfer_'):
            s, t = k[9:].split('_'); T[ix[s], ix[t]] = x
    return base, T

def gain(base, T):
    G = T - base[None, :]; np.fill_diagonal(G, np.nan); return G

def zs(x): return (x - x.mean()) / x.std()
def pairs(keep): return [(i, j) for i in keep for j in keep if i != j]

def slope_pooled(D, G, keep):
    ii = pairs(keep); y = np.array([G[i, j] for i, j in ii]); x = np.array([D[i, j] for i, j in ii]); return np.polyfit(zs(x), y, 1)[0]

def qap(D, G, keep, adj, R=10000, seed=0):
    keep = list(keep); ii = pairs(keep); y = np.array([G[i, j] for i, j in ii]); tg = np.array([j for _, j in ii])
    def dm(v):
        v = v.copy()
        for t in set(tg): v[tg == t] -= v[tg == t].mean()
        return v
    yy = dm(y) if adj else y
    x0= np.array([D[i, j] for i, j in ii])
    if adj:  # slope per SD of the raw (undemeaned) S1 over all pairs, as in the project's convention
        sd = x0.std()
        stat = lambda x: np.polyfit(dm(x), yy, 1)[0] * sd
    else:
        stat = lambda x: np.polyfit(zs(x), yy, 1)[0]
    obs = stat(x0); rng = np.random.RandomState(seed); c = 0
    for _ in range(R):
        p = rng.permutation(len(keep)); m = {keep[a]: keep[p[a]] for a in range(len(keep))}
        c += stat(np.array([D[m[i], m[j]] for i, j in ii])) <= obs
    return obs, (c + 1) / (R + 1)

def r2_fe(T, by):
    ii = pairs(range(10)); y = np.array([T[i, j] for i, j in ii]); g = np.array([(j if by == 't' else i) for i, j in ii])
    fit = np.zeros_like(y)
    for k in set(g): fit[g == k] = y[g == k].mean()
    return 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()

GRIDS = {
    'A':   (['scratch_s0/e2_results_scratch_s0.json'], 'A'),
    'B':   (['rtm_scratch_*/rtm_results_scratch_s0_g*.json'], 'B'),
    'C0':  (['rtm2_s[12]_c/rtm_results_scratch_s0_g*_e100_zscore.json'], 'B'),
    'C1':  (['rtm2_s[abc]_*/rtm_results_scratch_s1_g*_e100_zscore.json'], 'B'),
}
out = []
def rep(name, paper, got, tol=None):
    ok = '' if tol is None else ('OK' if abs(paper - got) <= tol else '<<< DIFFERENT')
    print(f'{name:58s} paper={paper!s:>9}  recomputed={got:9.3f}  {ok}'); out.append((name, paper, got, ok))

PAPER3 = {'A': (13.30, 3.27, -0.25, 38, -0.35, 32), 'B': (12.53, 2.33, 0.61, 62, 0.26, 48),
          'C0': (15.60, 4.69, -0.70, 28, -0.59, 25), 'C1': (15.45, 4.64, -0.62, 13, -0.63, 19)}
print('cells per grid:'); R = {}
for g, (pat, dsrc) in GRIDS.items():
    res = load(pat); R[g] = res
    nb = sum(k.startswith('scratch_') for k in res); nt = sum(k.startswith('transfer_') for k in res)
    print(' ', g, 'baselines', nb, 'transfer cells (incl. self)', nt)
for g, (pat, dsrc) in GRIDS.items():
    res = R[g]; D = dist(dsrc); bc, Tc = mats(res, 'chrf'); bb, Tb = mats(res, 'bleu')
    Gc = gain(bc, Tc); Gb = gain(bb, Tb); p = PAPER3[g]
    print(f'--- Round {g}')
    rep('Table3 mean baseline chrF', p[0], bc.mean(), .006); rep('Table3 mean baseline BLEU', p[1], bb.mean(), .006)
    rep('Table3 mean gain chrF', p[2], np.nanmean(Gc), .006); rep('Table3 positive gain chrF cells (of 90)', p[3], float((Gc > 0).sum()), 0)
    rep('Table3 mean gain BLEU', p[4], np.nanmean(Gb), .006); rep('Table3 positive gain BLEU cells (of 90)', p[5], float((Gb > 0).sum()), 0)
    keep = range(10)
    rep('pooled chrF slope per SD', {'A': -0.104, 'B': -0.159, 'C0': -0.038, 'C1': 0.037}[g], slope_pooled(D, Gc, keep), .0006)
    rep('pooled chrF slope without bg', {'A': -0.057, 'B': -0.079, 'C0': 0.010, 'C1': 0.067}[g], slope_pooled(D, Gc, [i for i in range(10) if L[i] != 'bg']), .0006)
    if g in ('C0', 'C1'):
        diag = np.array([float(res[f'transfer_{l}_{l}']['test_chrf']) for l in L]) - bc
        rep('self-transfer mean chrF change', {'C0': -0.19, 'C1': -0.25}[g], diag.mean(), .006)
        rep('self-transfer positive of 10', {'C0': 3, 'C1': 4}[g], float((diag > 0).sum()), 0)
        rep('R2 target FE on transfer chrF', {'C0': .89, 'C1': .89}[g], r2_fe(Tc, 't'), .006)
        rep('R2 source FE on transfer chrF', {'C0': .02, 'C1': .02}[g], r2_fe(Tc, 's'), .006)
    else:
        rep('R2 target FE on transfer chrF', {'A': .84, 'B': .64}[g], r2_fe(Tc, 't'), .006)
        rep('R2 source FE on transfer chrF', {'A': .02, 'B': .05}[g], r2_fe(Tc, 's'), .006)
    # QAP p for the tests quoted in the abstract/results
    if g in ('C0', 'C1', 'B'):
        o, pv = qap(D, Gc, range(10), adj=True)
        rep('target-adjusted chrF slope per SD', {'B': -0.196, 'C0': -0.224, 'C1': 0.039}[g], o, .002)
        rep('target-adjusted chrF one-sided p (own RNG)', {'B': .033, 'C0': .013, 'C1': .690}[g], pv, .012)
    o, pv = qap(D, Gc, range(10), adj=False)
    rep('pooled chrF one-sided p (own RNG)', {'A': .171, 'B': .246, 'C0': .437, 'C1': .620}[g], pv, .012)
print('\nmismatches:', [o[0] for o in out if o[3] == '<<< DIFFERENT'])
