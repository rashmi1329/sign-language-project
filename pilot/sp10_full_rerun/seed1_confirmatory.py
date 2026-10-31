"""Seed-1 confirmatory analysis exactly as written in docs/analysis_plan_rtm.md ('Confirmatory seed', 2026-10-21).
PRIMARY (decides replication): target-adjusted QAP, S1_dtw, all 10 languages, chrF gain over the target's own seed-1 baseline, within-target demeaned slope, node-relabelling null (10,000 draws, seed 0), one-sided negative, alpha 0.05, seed-1 grid alone.
SECONDARY (same sign check, Holm-adjusted as one family, not used for the decision): BLEU gain, without bg, two-way adjusted, S1_body, S1_hands, two-seed mean. (BLEURT gain NOT run here: needs BLEURT scoring of the seed-1 hypotheses.)
EXPLORATORY (not in the plan; labelled as such): seed-to-seed noise, target-ranking replication, pair-specific agreement across seeds.
Usage: python3 seed1_confirmatory.py dry|primary|secondary|noise  -> rtm_analysis/rtm_seed1_confirmatory.json (merged per mode). 'dry' reruns the seed-0 v2 test to check this script reproduces the earlier numbers."""
import json, os, sys, glob, numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
t = open(f'{HERE}/target_adjusted.py').read(); exec(t.split('OUT2 = {}')[0]); exec('def grid_metric' + t.split('def grid_metric', 1)[1].split('v2, _ = load_files')[0])
tw = open(f'{HERE}/two_way_hub.py').read(); exec('def twoway' + tw.split('def twoway', 1)[1].split('OUT = {')[0])
OUTF = f'{HERE}/rtm_analysis/rtm_seed1_confirmatory.json'; OUTD = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
allk = list(range(10)); nobg = [i for i in allk if LANGS[i] != 'bg']
def sdS1(D): return np.std([D[i, j] for i in allk for j in allk if i != j])
def prim(G, D, R=10000):
    r = tadj(G, D, allk, R=R); r['slope_per_SD_of_S1'] = r['slope_per_unit'] * sdS1(D); return r
mode = sys.argv[1]
s0, _ = load_files('rtm2_s[12]_*/rtm_results_scratch_s0_g*_e100_*.json')
s1, files1 = load_files('rtm2_s[abc]_*/rtm_results_scratch_s1_g*_e100_zscore.json')
assert len(s1) == 110, f'seed-1 grid incomplete: {len(s1)} cells (plan requires all 110)'
if mode == 'dry':
    r = prim(grid_metric(s0, 'chrf')[0], dmat('S1_dtw')); print('SEED-0 reproduction (earlier reported: slope -0.22 per SD, one-sided p 0.013):', r)
elif mode == 'primary':
    G1 = grid_metric(s1, 'chrf')[0]; r = prim(G1, dmat('S1_dtw')); OUTD['primary_seed1_chrf_gain_S1_dtw_all'] = r
    print('PRIMARY seed 1:', r); print('REPLICATED' if (r['p_one_sided'] < 0.05 and r['slope_per_SD_of_S1'] < 0) else 'NOT REPLICATED', '(p<0.05 and negative slope required)')
elif mode == 'secondary':
    res = {}
    for m in ('chrf', 'bleu'):
        G1 = grid_metric(s1, m)[0]
        for dk in ('S1_dtw', 'S1_body', 'S1_hands'):
            D = dmat(dk)
            for sn, keep in (('all', allk), ('no_bg', nobg)):
                if (m, dk, sn) == ('chrf', 'S1_dtw', 'all'): continue   # that is the primary
                r = tadj(G1, D, keep); r['slope_per_SD_of_S1'] = r['slope_per_unit'] * sdS1(D); res[f'seed1|{m}|tadj|{dk}|{sn}'] = r
            r = twoway(G1, D, allk); res[f'seed1|{m}|twoway|{dk}|all'] = r
    G0 = grid_metric(s0, 'chrf')[0]; G1 = grid_metric(s1, 'chrf')[0]; GM = (G0 + G1) / 2   # two-seed mean of every cell's gain
    for dk in ('S1_dtw',):
        D = dmat(dk); r = tadj(GM, D, allk); r['slope_per_SD_of_S1'] = r['slope_per_unit'] * sdS1(D); res[f'twoseed|chrf|tadj|{dk}|all'] = r
    keys = list(res); ps = np.array([res[k]['p_one_sided'] for k in keys]); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run_max = 0
    for rank, idx in enumerate(order): run_max = max(run_max, (m - rank) * ps[idx]); adj[idx] = min(1, run_max)
    for k, a in zip(keys, adj): res[k]['p_holm'] = float(a); sl = res[k].get('slope_per_SD_of_S1', res[k].get('slope_per_SD')); print(f'{k}: slope {sl:+.3f}/SD, p {res[k]["p_one_sided"]:.4f}, Holm {a:.3f}')
    OUTD['secondary'] = res
elif mode == 'noise':
    def mats(res):
        Y = np.full((10, 10), np.nan); Bs = np.full(10, np.nan)
        for k, v in res.items():
            if k.startswith('scratch_'): Bs[IDX[k[8:]]] = v['test_chrf']
            elif k.startswith('transfer_'):
                s, tt = k[9:].split('_')
                if s != tt: Y[IDX[s], IDX[tt]] = v['test_chrf']
        return Y, Bs
    Y0, B0 = mats(s0); Y1, B1 = mats(s1); m = ~np.isnan(Y0) & ~np.isnan(Y1); y0, y1 = Y0[m], Y1[m]; out = {}
    out['n_cells'] = int(m.sum()); out['cell_corr_raw'] = float(np.corrcoef(y0, y1)[0, 1]); d = y1 - y0
    out['seed_diff_SD_per_cell'] = float(d.std(ddof=1)); out['per_cell_seed_variance'] = float((d ** 2).mean() / 2); out['total_var_cells_seed0'] = float(y0.var(ddof=1)); out['seed_noise_share_of_total'] = out['per_cell_seed_variance'] / float(np.concatenate([y0, y1]).var(ddof=1))
    tmean0 = np.nanmean(Y0, axis=0); tmean1 = np.nanmean(Y1, axis=0); smean0 = np.nanmean(Y0, axis=1); smean1 = np.nanmean(Y1, axis=1)
    out['target_mean_spearman'] = [float(x) for x in stats.spearmanr(tmean0, tmean1)]; out['source_mean_spearman'] = [float(x) for x in stats.spearmanr(smean0, smean1)]; out['baseline_spearman'] = [float(x) for x in stats.spearmanr(B0, B1)]
    def resid(Y):   # remove target and source means (two-way additive fit) -> pair-specific part
        Z = np.where(np.isnan(Y), np.nan, Y); g = np.nanmean(Z); r_ = np.nanmean(Z, axis=1, keepdims=True); c_ = np.nanmean(Z, axis=0, keepdims=True); return Z - r_ - c_ + g
    R0, R1 = resid(Y0), resid(Y1); mm = ~np.isnan(R0) & ~np.isnan(R1); out['pair_specific_corr'] = float(np.corrcoef(R0[mm], R1[mm])[0, 1]); out['pair_specific_SD_seed0'] = float(R0[mm].std()); out['pair_specific_SD_seed1'] = float(R1[mm].std())
    out['target_mean_seed0'] = {LANGS[i]: float(tmean0[i]) for i in range(10)}; out['target_mean_seed1'] = {LANGS[i]: float(tmean1[i]) for i in range(10)}
    print(json.dumps(out, indent=1)); OUTD['exploratory_noise'] = out
json.dump(OUTD, open(OUTF, 'w'), indent=1)
