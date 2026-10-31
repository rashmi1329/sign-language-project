"""Contribution 1 (limitations entry 79): power of three tests of 'S1 predicts gain' in a transfer matrix as a function of how much the TARGET language dominates the outcome.
 A pooled QAP (the pre-specified test), B target-adjusted QAP, C two-way (target + source) adjusted QAP. Same node-relabelling null for all (2000 relabellings of the observed S1_dtw matrix), one-sided negative, alpha 0.05.
Model: gain[s,t] = a_t + b * z(S1[s,t]) + e; a_t ~ N(0, tau^2), e ~ N(0, 0.567^2) (the observed within-target residual SD of the v2 chrF grid); tau set from the target share of the outcome variance.
Usage: python3 power_general.py -> rtm_analysis/rtm_power_general.json"""
import json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); L = ['zh','uk','ru','bg','is','de','it','sv','lt','en']; I = {l: i for i, l in enumerate(L)}
sim = json.load(open(f'{H}/similarity_rtm.json')); D = np.zeros((10, 10))
for k, v in sim.items(): a, b = k.split('|'); D[I[a], I[b]] = D[I[b], I[a]] = v['S1_dtw']
ii = [(i, j) for i in range(10) for j in range(10) if i != j]; tg = np.array([j for _, j in ii]); src = np.array([i for i, _ in ii]); n = 90
Z = np.zeros((n, 20)); Z[np.arange(n), src] = 1; Z[np.arange(n), 10 + tg] = 1; M2 = np.eye(n) - Z @ np.linalg.pinv(Z)
Zt = np.zeros((n, 10)); Zt[np.arange(n), tg] = 1; M1 = np.eye(n) - Zt @ np.linalg.pinv(Zt)
Mc = np.eye(n) - np.ones((n, n)) / n
std = lambda v: (v - v.mean()) / v.std()
x = std(np.array([D[i, j] for i, j in ii])); P = 2000; rng = np.random.RandomState(1)
Xp = np.stack([std(np.array([D[p[i], p[j]] for i, j in ii])) for p in (rng.permutation(10) for _ in range(P))])
def prep(M): Xm = Xp @ M.T; xm = M @ x; return Xm, (Xm ** 2).sum(1), xm, xm @ xm   # M symmetric idempotent: slope = (M x).y / (M x).(M x)
PRE = {'A': prep(Mc), 'B': prep(M1), 'C': prep(M2)}
def power(share, b, sd=0.567, nsim=2000, seed=3):
    r = np.random.RandomState(seed); tau = np.sqrt(share / (1 - share)) * sd if share > 0 else 0.0
    Y = (tau * r.randn(nsim, 10))[:, tg] + b * x[None] + sd * r.randn(nsim, n); out = {}
    for k, (Xm, d, xm, dx) in PRE.items():
        obs = (Y @ xm) / dx; perm = (Xm @ Y.T) / d[:, None]; p = (1 + (perm <= obs[None]).sum(0)) / (P + 1); out[k] = float((p <= .05).mean())
    return out
OUT = {'by_share': {}, 'by_effect': {}, 'mde': {}}
print('power at true slope -0.22 chrF per SD (the seed-0 estimate), residual SD 0.567, vs the share of outcome variance that is the target:')
for share in (0.0, 0.25, 0.5, 0.77, 0.9):
    pw = power(share, -0.22); OUT['by_share'][str(share)] = pw; print(f'  target share {share:.2f}: pooled {pw["A"]:.2f} | target-adjusted {pw["B"]:.2f} | two-way {pw["C"]:.2f}')
print('size (true slope 0):', {s: power(s, 0.0, seed=9) for s in (0.0, 0.77)})
print('\npower at target share 0.77 vs the true slope (chrF per SD of S1):')
for b in (0, -0.05, -0.1, -0.15, -0.22, -0.3, -0.45, -0.6):
    pw = power(0.77, b); OUT['by_effect'][str(b)] = pw; print(f'  {b:+.2f}: pooled {pw["A"]:.2f} | target-adjusted {pw["B"]:.2f} | two-way {pw["C"]:.2f}')
def mde(share, k):
    lo, hi = 0.0, 1.5
    for _ in range(14):
        mid = (lo + hi) / 2
        if power(share, -mid, nsim=800, seed=7)[k] >= .8: hi = mid
        else: lo = mid
    return hi
print('\nminimum true slope (chrF per SD of S1) for 80% power:')
for share in (0.0, 0.5, 0.77, 0.9):
    m = {k: mde(share, k) for k in 'ABC'}; OUT['mde'][str(share)] = m; print(f'  target share {share:.2f}: pooled {m["A"]:.2f} | target-adjusted {m["B"]:.2f} | two-way {m["C"]:.2f}   (S1 range = 4.2 SD)')
json.dump(OUT, open(f'{H}/rtm_analysis/rtm_power_general.json', 'w'), indent=1)
