"""Analysis of the v2 rerun (docs/analysis_plan_rtm.md, v2 addendum). Reads e2_outputs/*/ result files; works on partial data and says so.
 scratch v2 (100 epochs, fixed scaling, seed 0): rtm2_s1_*, rtm2_s2_* ; Uni-Sign arms: seed 0 from rtm_uni_* (frozen code, 60 epochs), seeds 1-3 from rtm2_ua_*, rtm2_ub_*.
Usage: python3 analyze_rtm2.py   -> prints, writes rtm_analysis/rtm_analysis_v2.json. BLEURT is a separate step."""
import json, glob, os, re, itertools
import numpy as np
from scipy import stats
LANGS = ['zh','uk','ru','bg','is','de','it','sv','lt','en']; IDX = {l: i for i, l in enumerate(LANGS)}
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = {}
sim = json.load(open(f'{HERE}/similarity_rtm.json'))
def dmat(key):
    D = np.zeros((10, 10))
    for k, v in sim.items():
        a, b = k.split('|'); D[IDX[a], IDX[b]] = D[IDX[b], IDX[a]] = v[key]
    return D
def slope(x, y): x = (x - x.mean()) / x.std(); return np.polyfit(x, y, 1)[0]
def pairs(G, D, keep):
    ii = [(i, j) for i in keep for j in keep if i != j and not np.isnan(G[i, j])]
    return np.array([D[i, j] for i, j in ii]), np.array([G[i, j] for i, j in ii]), ii
def qap(G, D, keep, draws=10000, seed=0):
    rng = np.random.RandomState(seed); x, y, ii = pairs(G, D, keep)
    if len(y) < 8: return None
    obs = slope(x, y); c1 = c2 = 0; keep = list(keep)
    for _ in range(draws):
        p = rng.permutation(len(keep)); m = {keep[a]: keep[p[a]] for a in range(len(keep))}
        s = slope(np.array([D[m[i], m[j]] for i, j in ii]), y); c1 += s <= obs; c2 += abs(s) >= abs(obs)
    return dict(n=len(y), slope=float(obs), p_one_sided_negative=(c1 + 1) / (draws + 1), p_two_sided=(c2 + 1) / (draws + 1))
def clustered(G, D, keep, B=2000, seed=0):
    rng = np.random.RandomState(seed); x, y, ii = pairs(G, D, keep); tg = np.array([j for _, j in ii]); ts = sorted(set(tg))
    if len(ts) < 3: return None
    bs = []
    for _ in range(B):
        idx = np.concatenate([np.where(tg == t)[0] for t in rng.choice(ts, len(ts))])
        if x[idx].std() > 0: bs.append(slope(x[idx], y[idx]))
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None
def within(G, D, keep):
    rs = []
    for t in keep:
        ss = [s for s in keep if s != t and not np.isnan(G[s, t])]
        if len(ss) >= 4: rs.append(stats.spearmanr([D[s, t] for s in ss], [G[s, t] for s in ss])[0])
    neg = sum(r < 0 for r in rs); n = len(rs)
    return dict(targets=n, mean_rho=float(np.mean(rs)) if rs else None, negative=int(neg), sign_test_p_one_sided=float(stats.binomtest(neg, n, .5, 'greater').pvalue) if n else None)
def load_files(pattern, skip=('gate', 'sweep')):
    res, files = {}, []
    for f in sorted(glob.glob(f'{HERE}/e2_outputs/{pattern}')):
        if any(s in os.path.basename(f) for s in skip): continue
        if 'rtm2_uc_a' in f: continue   # account A's duplicate run of seed 2 (same seed and code as rtm2_uc_c): kept as a replicate, not mixed into the seed-2 cells
        d = json.load(open(f)); files.append(os.path.relpath(f, HERE)); res.update(d)
    return res, files
def report_tests(tag, G, m):
    for name, keep in (('all languages', list(range(10))), ('without bg', [i for i in range(10) if LANGS[i] != 'bg'])):
        for dk in ('S1_dtw', 'S1_body', 'S1_hands'):
            D = dmat(dk); q = qap(G, D, keep)
            if q is None: print(f'  [{tag} {m}] {name} {dk}: too few cells'); continue
            line = f'  [{tag} {m}] {name} {dk}: n={q["n"]} slope {q["slope"]:+.3f}/SD, QAP one-sided(neg) p={q["p_one_sided_negative"]:.3f}, two-sided p={q["p_two_sided"]:.3f}'
            if dk == 'S1_dtw':
                ci = clustered(G, D, keep); w = within(G, D, keep); line += f' | target-cluster CI {ci} | within-target {w}'; OUT[f'{tag}|{m}|{name}'] = dict(qap=q, ci=ci, within=w)
            print(line)
def run_scratch():
    res, files = load_files('rtm2_s[12]_*/rtm_results_scratch_s0_g*_e100_*.json')
    nb = sum(k.startswith('scratch_') for k in res); nt = sum(k.startswith('transfer_') and k.split('_')[1] != k.split('_')[2] for k in res); ns = sum(k.startswith('transfer_') and k.split('_')[1] == k.split('_')[2] for k in res)
    print(f'SCRATCH v2 files {files}\n cells: {nb} baselines, {nt} transfers, {ns} self-transfers (of 10, 90, 10)', '' if (nb, nt, ns) == (10, 90, 10) else '  *** PARTIAL: not the planned test ***')
    OUT['scratch_v2_cells'] = [nb, nt, ns]
    scales = sorted({v.get('scale') for v in res.values()}); eps = sorted({v.get('ep_official') for v in res.values()}); print(' scaling used:', scales, ' epochs:', eps, '(more than one scaling = the two halves disagree: do NOT merge)' if len(scales) > 1 else '')
    sc = sorted(glob.glob(f'{HERE}/e2_outputs/rtm2_s1_*/scale_choice.json'))
    if sc: print(' dev sweep (mean dev chrF over is,en,de,zh):', json.load(open(sc[0])).get('mean_dev_chrf'))
    if not res: return
    for m in ('chrf', 'bleu'):
        base = {k[8:]: v['test_' + m] for k, v in res.items() if k.startswith('scratch_')}; dev = {k[8:]: v['dev_chrf'] for k, v in res.items() if k.startswith('scratch_') and 'dev_chrf' in v}
        G = np.full((10, 10), np.nan); N = np.full((10, 10), np.nan); selfg = {}
        for k, v in res.items():
            if not k.startswith('transfer_'): continue
            s, t = k[9:].split('_')
            if s == t:
                if t in base: selfg[t] = v['test_' + m] - base[t]
            elif t in base: G[IDX[s], IDX[t]] = v['test_' + m] - base[t]
        for k, v in res.items():
            if k.startswith('transfer_'):
                s, t = k[9:].split('_')
                if s != t and f'transfer_{t}_{t}' in res: N[IDX[s], IDX[t]] = v['test_' + m] - res[f'transfer_{t}_{t}']['test_' + m]
        print(f'\n[{m}] baselines mean {np.mean(list(base.values())):.3f}:', {l: round(v, 2) for l, v in base.items()})
        if m == 'chrf' and dev: print('   dev chrF of baselines:', {l: round(v, 2) for l, v in dev.items()})
        g = G[~np.isnan(G)]; n = N[~np.isnan(N)]
        if len(g): print(f'[{m}] gain vs baseline: {len(g)} cells, mean {g.mean():+.3f}, positive {int((g > 0).sum())}')
        if selfg: print(f'[{m}] self-transfer gain (same language, 100 more epochs): mean {np.mean(list(selfg.values())):+.3f} over {len(selfg)} languages:', {l: round(v, 2) for l, v in selfg.items()})
        if len(n): print(f'[{m}] NET gain (transfer minus self-transfer of the target): {len(n)} cells, mean {n.mean():+.3f}, positive {int((n > 0).sum())}')
        OUT[f'scratch_v2_{m}'] = dict(baseline=base, self_gain=selfg, mean_gain=float(g.mean()) if len(g) else None, mean_net_gain=float(n.mean()) if len(n) else None)
        report_tests('gain', G, m)
        if len(n): report_tests('NET', N, m)
def uni_cells(arm, seed):
    pat = f'*/rtm_results_{arm}_s{seed}_g*.json'
    res, files = load_files(pat); return res
def run_uni():
    seeds = [0, 1, 2, 3]; R = {(a, s): uni_cells(a, s) for a in ('uni', 'uni_ctrl') for s in seeds}
    print('\nUNI cells found (of 30 per arm and seed):', {f'{a}_s{s}': len(r) for (a, s), r in R.items() if r}); OUT['uni_cells'] = {f'{a}_s{s}': len(r) for (a, s), r in R.items()}
    def gains(res, m):
        base = {k[8:]: v['test_' + m] for k, v in res.items() if k.startswith('scratch_')}; g = {}
        for k, v in res.items():
            if k.startswith('transfer_'):
                s, t = k[9:].split('_')
                if t in base: g[(s, t)] = v['test_' + m] - base[t]
        return base, g
    for m in ('chrf', 'bleu'):
        per = {}
        for (a, s), r in R.items():
            if not r: continue
            base, g = gains(r, m); per[(a, s)] = (r, base, g)
            print(f'  [{m}] {a} seed {s}: {len(base)} baselines (mean {np.mean(list(base.values())) if base else float("nan"):.3f}), {len(g)} gains, mean gain {np.mean(list(g.values())) if g else float("nan"):+.3f}')
        # pretrained minus control, per cell, averaged over the seeds that have both arms
        common = {}
        for s in seeds:
            if ('uni', s) in per and ('uni_ctrl', s) in per:
                a, c = R[('uni', s)], R[('uni_ctrl', s)]
                for k in set(a) & set(c): common.setdefault(k, []).append(a[k]['test_' + m] - c[k]['test_' + m])
        if common:
            d = np.array([np.mean(v) for v in common.values()]); rng = np.random.RandomState(0); bs = [rng.choice(d, len(d)).mean() for _ in range(10000)]
            nseeds = sorted({s for s in seeds if ('uni', s) in per and ('uni_ctrl', s) in per})
            print(f'  [{m}] pretrained minus control: {len(d)} cells (seed-averaged over seeds {nseeds}), mean {d.mean():+.3f}, 95% bootstrap CI over cells [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]')
            OUT[f'uni_vs_ctrl_{m}'] = dict(cells=len(d), seeds=nseeds, mean=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        # descriptive S1 slope of the seed-averaged gain, 20 transfer cells per arm
        D = dmat('S1_dtw')
        for a in ('uni', 'uni_ctrl'):
            acc = {}
            for s in seeds:
                if (a, s) in per:
                    for (src, t), v in per[(a, s)][2].items(): acc.setdefault((src, t), []).append(v)
            if len(acc) >= 8:
                x = np.array([D[IDX[s_], IDX[t_]] for s_, t_ in acc]); y = np.array([np.mean(v) for v in acc.values()])
                print(f'  [{m}] {a}: seed-averaged gain on S1_dtw over {len(acc)} transfer cells: slope {slope(x, y):+.3f}/SD (descriptive, no p-value)')
run_scratch(); run_uni()
os.makedirs(f'{HERE}/rtm_analysis', exist_ok=True); json.dump(OUT, open(f'{HERE}/rtm_analysis/rtm_analysis_v2.json', 'w'), indent=1, default=str)
