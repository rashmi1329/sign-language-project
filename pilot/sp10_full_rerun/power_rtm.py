"""Power / minimum detectable effect of the pre-specified S1 test on the scratch v2 grid (exploratory; docs/limitations entry 75).
Model: gain[s,t] = a_t + b * z(S1[s,t]) + e, a_t = observed target means of the gain, e ~ N(0, sd^2), sd = within-target residual SD of the observed grid
(after removing target means and the within-target S1 slope, df = 90-10-1). z = S1 standardised over the 90 pairs, so b is gain per SD of S1.
Test A (pre-specified): pooled QAP, slope of gain on z, node relabelling, one-sided (negative), alpha 0.05, 2000 relabellings (the null depends only on S1, so it is shared by all simulated data sets).
Test B (exploratory, target-adjusted): same QAP but gain and S1 are demeaned within target first (the target effect, 89% of the variance, drops out).
Usage: python3 power_rtm.py -> prints, writes rtm_analysis/rtm_power.json"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f'{HERE}/analyze_rtm2.py').read().split('def run_scratch')[0]; exec(src)
bsrc = open(f'{HERE}/analyze_bleurt.py').read(); exec('B = json.load' + bsrc.split('B = json.load', 1)[1].split('# scratch grids')[0])
res, files = load_files('rtm2_s[12]_*/rtm_results_scratch_s0_g*_e100_*.json')
def gain_grid(vals_base, vals_tr):
    G = np.full((10, 10), np.nan)
    for (s, t), v in vals_tr.items():
        if s != t: G[IDX[s], IDX[t]] = v - vals_base[t]
    return G
grids = {}
for m in ('chrf', 'bleu'):
    base = {k[8:]: v['test_' + m] for k, v in res.items() if k.startswith('scratch_')}
    tr = {tuple(k[9:].split('_')): v['test_' + m] for k, v in res.items() if k.startswith('transfer_')}
    grids[m] = gain_grid(base, tr)
c = cells(['rtm2_s1_c', 'rtm2_s2_c'], 'rtm_results_scratch_s0'); base, selfc, G, N = grid(c); grids['bleurt'] = G
D = dmat('S1_dtw'); ii = [(i, j) for i in range(10) for j in range(10) if i != j]
tgt = np.array([j for _, j in ii]); x = np.array([D[i, j] for i, j in ii]); x = (x - x.mean()) / x.std()
P = 2000; rng = np.random.RandomState(1)
Xp = np.zeros((P, 90))
for k in range(P):
    p = rng.permutation(10); v = np.array([D[p[i], p[j]] for i, j in ii]); Xp[k] = (v - v.mean()) / v.std()
def demean(M):   # within-target demeaning, rows = vectors over the 90 pairs
    M = M.copy()
    for t in range(10): M[..., tgt == t] -= M[..., tgt == t].mean(-1, keepdims=True)
    return M
Xpw = demean(Xp); xw = demean(x[None])[0]
def power(tau, sd, b, nsim, seed):
    # target effects are redrawn in every simulated data set (N(0, tau^2) per target): the node-relabelling null treats them as exchangeable, so conditioning on the observed alignment of target effects with S1 would bias the pooled test
    r = np.random.RandomState(seed); A = (tau * r.randn(nsim, 10))[:, tgt]; Y = A + b * x[None] + sd * r.randn(nsim, 90)
    out = {}
    Yc = Y - Y.mean(1, keepdims=True); so = Yc @ x / 90; sp = (Xp @ Yc.T / 90)
    pA = (1 + (sp <= so[None]).sum(0)) / (P + 1); out['A'] = float((pA <= .05).mean())
    Yw = demean(Y); sow = Yw @ xw / 90; spw = Xpw @ Yw.T / 90
    pB = (1 + (spw <= sow[None]).sum(0)) / (P + 1); out['B'] = float((pB <= .05).mean()); return out
def mde(a, sd, test, nsim=1000):   # a = tau here
    lo, hi = 0.0, 4 * sd
    for _ in range(14):
        mid = (lo + hi) / 2
        if power(a, sd, -mid, nsim, 7)[test] >= .8: hi = mid
        else: lo = mid
    return hi
OUTP = {}; SEED_SD = 0.6   # single-cell run-to-run SD at fixed seed in chrF (limitations entry 73)
for m, Gm in grids.items():
    y = np.array([Gm[i, j] for i, j in ii]); a_t = float(np.std([np.nanmean(Gm[:, t]) for t in range(10)], ddof=1))
    yw = demean(y[None])[0]; bw = (xw @ yw) / (xw @ xw); sse = ((yw - bw * xw) ** 2).sum(); sd = float(np.sqrt(sse / (90 - 10 - 1)))
    tvar = float(a_t ** 2 / np.var(y)); print(f'\n[{m}] residual SD {sd:.3f}, outcome SD {y.std():.3f}, target effects share of variance {tvar:.2f}, observed within-target slope {bw:+.3f}/SD')
    o = dict(resid_sd=sd, outcome_sd=float(y.std()), target_share=tvar, within_slope=float(bw), power={}); print('  true slope (gain per SD of S1) -> power A (pooled, pre-specified) | power B (target-adjusted)')
    for k in (0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        b = -k * sd; pw = power(a_t, sd, b, 2000, 3); o['power'][f'{k}sd'] = pw
        print(f'   {b:+.3f} ({k:.2f} residual SD) -> A {pw["A"]:.2f} | B {pw["B"]:.2f}')
    o['mde80_A'] = float(mde(a_t, sd, 'A')); o['mde80_B'] = float(mde(a_t, sd, 'B'))
    print(f'  minimum effect with 80% power: A {o["mde80_A"]:.3f} per SD of S1 ({o["mde80_A"] * 4.2:.2f} over the 4.2 SD range), B {o["mde80_B"]:.3f} ({o["mde80_B"] * 4.2:.2f} over range)')
    if m == 'chrf':
        o['bounds'] = {}
        print('  more seeds: the residual SD cannot be split into seed noise and pair-specific variance for the scratch arm (one seed). Bounds: best case (all residual is seed noise) divides it by sqrt(k); worst case (none is) leaves it unchanged.')
        for kk in (2, 4, 8):
            sdk = sd / np.sqrt(kk); ma, mb = mde(a_t, sdk, 'A', 600), mde(a_t, sdk, 'B', 600); o['bounds'][kk] = dict(sd=float(sdk), mde80_A=float(ma), mde80_B=float(mb))
            print(f'   best case k={kk}: residual SD {sdk:.3f}; MDE80 A {ma:.3f}, B {mb:.3f}')
    # observed target-adjusted QAP (exploratory), 10000 relabellings
    r2 = np.random.RandomState(11); sob = (xw @ yw) / 90; cnt = 0; R = 10000
    for _ in range(R):
        p = r2.permutation(10); v = np.array([D[p[i], p[j]] for i, j in ii]); v = (v - v.mean()) / v.std(); vw = demean(v[None])[0]; cnt += (vw @ yw) / 90 <= sob
    o['obs_B_p_one_sided'] = float((cnt + 1) / (R + 1)); o['obs_B_slope'] = float(bw); print(f'  OBSERVED target-adjusted QAP (exploratory): slope {bw:+.3f} per SD, one-sided p = {o["obs_B_p_one_sided"]:.4f}')
    OUTP[m] = o
size = power(1.0, 1.0, 0.0, 4000, 5); print('\nsize check (random target effects, true slope 0): A', size['A'], 'B', size['B'], '(should be about 0.05)'); OUTP['size_check'] = size
json.dump(OUTP, open(f'{HERE}/rtm_analysis/rtm_power.json', 'w'), indent=1)
