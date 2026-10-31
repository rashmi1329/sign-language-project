"""Target-adjusted QAP test of S1 against gain (EXPLORATORY: not in docs/analysis_plan_rtm.md; motivated by power_rtm.py after the pre-specified pooled test showed 77% of the gain variance is the target language).
Statistic: slope of gain on S1 after both are demeaned within target; null: node relabelling of S1 (10,000 draws, seed 0), one-sided (negative = closer source, more gain). Every grid x metric x S1 variant x language set is reported.
Also: per-target sign count, leave-one-language-out slope (language dropped as source and target). Usage: python3 target_adjusted.py -> rtm_analysis/rtm_target_adjusted.json"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(f'{HERE}/analyze_rtm2.py').read().split('def run_scratch')[0])
bsrc = open(f'{HERE}/analyze_bleurt.py').read(); exec('B = json.load' + bsrc.split('B = json.load', 1)[1].split('# scratch grids')[0])
import glob
def tadj(G, D, keep, R=10000, seed=0):
    keep = list(keep); ii = [(i, j) for i in keep for j in keep if i != j and not np.isnan(G[i, j])]
    if len(ii) < 20: return None
    y = np.array([G[i, j] for i, j in ii]); tg = np.array([j for _, j in ii]); ts = sorted(set(tg))
    def dm(v):
        v = v.astype(float).copy()
        for t in ts: v[tg == t] -= v[tg == t].mean()
        return v
    yw = dm(y); sd = lambda v: np.sqrt((dm(v) ** 2).sum())
    def stat(v): vw = dm(v); return (vw @ yw) / (vw @ vw)    # within-target OLS slope per unit of S1
    xs = np.array([D[i, j] for i, j in ii]); obs = stat(xs); sdx = (dm(xs)).std()
    rng = np.random.RandomState(seed); cnt = 0
    for _ in range(R):
        p = rng.permutation(len(keep)); m = {keep[a]: keep[p[a]] for a in range(len(keep))}; cnt += stat(np.array([D[m[i], m[j]] for i, j in ii])) <= obs
    sl = [np.polyfit(xs[tg == t], y[tg == t], 1)[0] for t in ts if (tg == t).sum() >= 4]
    return dict(n=len(y), slope_per_unit=float(obs), p_one_sided=float((cnt + 1) / (R + 1)), targets_negative=int(sum(s < 0 for s in sl)), targets=len(sl))
def run(G, name, out, dkeys=('S1_dtw', 'S1_body', 'S1_hands')):
    allk = list(range(10)); nobg = [i for i in allk if LANGS[i] != 'bg']
    for dk in dkeys:
        D = dmat(dk); sdD = np.std([D[i, j] for i in allk for j in allk if i != j])
        for sn, keep in (('all', allk), ('no_bg', nobg)):
            r = tadj(G, D, keep)
            if r is None: continue
            r['slope_per_SD_of_S1'] = r['slope_per_unit'] * sdD; out[f'{name}|{dk}|{sn}'] = r
            print(f'  {name} {dk} {sn}: n={r["n"]}, within-target slope {r["slope_per_SD_of_S1"]:+.3f} per SD of S1, one-sided p {r["p_one_sided"]:.4f}, negative in {r["targets_negative"]}/{r["targets"]} targets')
        if dk == 'S1_dtw':   # leave one language out (all-language set), fewer draws
            lo = []
            for l in allk:
                r = tadj(G, D, [i for i in allk if i != l], R=2000)
                if r: lo.append((LANGS[l], r['slope_per_unit'] * sdD, r['p_one_sided']))
            out[f'{name}|{dk}|leave_one_out'] = lo; print('    leave-one-language-out (slope/SD, p):', ', '.join(f'{a} {b:+.2f}/{c:.3f}' for a, b, c in lo))
OUT2 = {}
def grid_metric(res, m):
    base = {k[8:]: v['test_' + m] for k, v in res.items() if k.startswith('scratch_')}; G = np.full((10, 10), np.nan); N = np.full((10, 10), np.nan)
    for k, v in res.items():
        if k.startswith('transfer_'):
            s, t = k[9:].split('_')
            if s != t and t in base: G[IDX[s], IDX[t]] = v['test_' + m] - base[t]
            if s != t and f'transfer_{t}_{t}' in res: N[IDX[s], IDX[t]] = v['test_' + m] - res[f'transfer_{t}_{t}']['test_' + m]
    return G, N
v2, _ = load_files('rtm2_s[12]_*/rtm_results_scratch_s0_g*_e100_*.json')
fr = {}
for f in sorted(glob.glob(f'{HERE}/e2_outputs/rtm_scratch_*/rtm_results_scratch_s0_g*.json')): fr.update(json.load(open(f)))
for m in ('chrf', 'bleu'):
    G, N = grid_metric(v2, m); print(f'\n== v2 grid {m} =='); run(G, f'v2|{m}|gain', OUT2); run(N, f'v2|{m}|net', OUT2, ('S1_dtw',))
    G, _ = grid_metric(fr, m); print(f'\n== frozen 60-epoch grid {m} =='); run(G, f'frozen|{m}|gain', OUT2)
for tag, pre in (('v2', ['rtm2_s1_c', 'rtm2_s2_c']), ('frozen', ['rtm_scratch_a', 'rtm_scratch_b_k'])):
    c = cells(pre, 'rtm_results_scratch_s0'); base, selfc, G, N = grid(c); print(f'\n== {tag} grid bleurt =='); run(G, f'{tag}|bleurt|gain', OUT2)
    if tag == 'v2': run(N, f'{tag}|bleurt|net', OUT2, ('S1_dtw',))
json.dump(OUT2, open(f'{HERE}/rtm_analysis/rtm_target_adjusted.json', 'w'), indent=1)
