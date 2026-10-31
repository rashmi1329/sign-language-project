"""CPU-only extension analyses on the RTMPose scratch v2 grid (exploratory, run after seeing the main results).
1. How large an S1 effect is excluded: slope CIs (target-cluster bootstrap) in chrF per SD of S1 and as the change from the closest to the farthest pair.
2. What explains the transfer score: variance decomposition with target / source fixed effects and S1 (adjusted R2, so extra parameters are penalised)."""
import json, glob, numpy as np
L = ['zh', 'uk', 'ru', 'bg', 'is', 'de', 'it', 'sv', 'lt', 'en']; rng = np.random.default_rng(0)
S = json.load(open('similarity_rtm.json')); D = {}
for k, v in S.items():
    a, b = k.split('|'); D[a, b] = D[b, a] = v
R = {}
for d in ['rtm2_s1_c', 'rtm2_s2_c']:
    for f in glob.glob(f'e2_outputs/{d}/rtm_results_scratch_s0_g*_e100_zscore.json'): R.update(json.load(open(f)))
out = {}
for metric in ['test_chrf', 'test_bleu']:
    base = {l: R[f'scratch_{l}'][metric] for l in L}; selfc = {l: R[f'transfer_{l}_{l}'][metric] for l in L}
    rows = [(s, t, R[f'transfer_{s}_{t}'][metric]) for s in L for t in L if s != t]
    for key in ['S1_dtw', 'S1_body', 'S1_hands']:
        for nm, drop in [('all', set()), ('no_bg', {'bg'})]:
            rr = [(s, t, v) for s, t, v in rows if s not in drop and t not in drop]
            x = np.array([D[s, t][key] for s, t, v in rr]); sd = x.std(); xs = (x - x.mean()) / sd
            for dv, yf in [('gain', lambda s, t, v: v - base[t]), ('net', lambda s, t, v: v - selfc[t])]:
                y = np.array([yf(s, t, v) for s, t, v in rr]); b = np.polyfit(xs, y, 1)[0]
                tg = sorted({t for s, t, v in rr}); bs = []
                for _ in range(4000):
                    pick = rng.choice(len(tg), len(tg)); idx = np.concatenate([[i for i, (s, t, v) in enumerate(rr) if t == tg[p]] for p in pick])
                    if xs[idx].std() > 0: bs.append(np.polyfit(xs[idx], y[idx], 1)[0])
                lo90, hi90 = np.percentile(bs, [5, 95]); lo95, hi95 = np.percentile(bs, [2.5, 97.5])
                span = (x.max() - x.min()) / sd   # SDs between the closest and farthest pair
                out[f'{metric}|{key}|{nm}|{dv}'] = dict(n=len(rr), slope=float(b), ci90=[float(lo90), float(hi90)], ci95=[float(lo95), float(hi95)], span_sd=float(span),
                                                       max_effect_90=float(max(abs(lo90), abs(hi90)) * span), y_sd=float(y.std()))
# variance decomposition of the transfer score (all 90 pairs)
dec = {}
for metric in ['test_chrf', 'test_bleu']:
    rows = [(s, t, R[f'transfer_{s}_{t}'][metric]) for s in L for t in L if s != t]; y = np.array([v for s, t, v in rows]); n = len(y)
    def design(parts):
        cols = [np.ones(n)]
        if 'target' in parts: cols += [np.array([1.0 * (t == l) for s, t, v in rows]) for l in L[1:]]
        if 'source' in parts: cols += [np.array([1.0 * (s == l) for s, t, v in rows]) for l in L[1:]]
        if 'S1' in parts: x = np.array([D[s, t]['S1_dtw'] for s, t, v in rows]); cols.append((x - x.mean()) / x.std())
        return np.column_stack(cols)
    def r2(parts):
        X = design(parts); beta = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ beta; ss = (res ** 2).sum(); tot = ((y - y.mean()) ** 2).sum(); k = X.shape[1] - 1
        return dict(r2=float(1 - ss / tot), adj_r2=float(1 - (ss / (n - k - 1)) / (tot / (n - 1))), k=k)
    dec[metric] = {nm: r2(p) for nm, p in [('S1 only', ['S1']), ('target only', ['target']), ('source only', ['source']), ('target+source', ['target', 'source']), ('target+source+S1', ['target', 'source', 'S1'])]}
json.dump(dict(equivalence=out, decomposition=dec), open('rtm_analysis/rtm_extend_cpu.json', 'w'), indent=1)
for k in ['test_chrf|S1_dtw|all|gain', 'test_chrf|S1_dtw|all|net', 'test_chrf|S1_dtw|no_bg|net', 'test_chrf|S1_body|all|net', 'test_chrf|S1_hands|all|net', 'test_bleu|S1_dtw|all|net']:
    o = out[k]; print(k, 'slope %+.3f/SD  90%% CI [%+.2f, %+.2f]  95%% CI [%+.2f, %+.2f]  S1 spans %.1f SD -> largest effect inside the 90%% CI over the full S1 range: %.2f (SD of the outcome %.2f)' % (o['slope'], *o['ci90'], *o['ci95'], o['span_sd'], o['max_effect_90'], o['y_sd']))
for m, d in dec.items():
    print('\n' + m + ' transfer score, 90 pairs'); [print('  %-18s R2 %.3f  adj R2 %+.3f  (%d predictors)' % (k, v['r2'], v['adj_r2'], v['k'])) for k, v in d.items()]
