"""BLEURT version of the v2 analysis (exploratory plan v2 addendum). Reads rtm_analysis/rtm_bleurt_cells.json (score_bleurt.py); reuses the tests of analyze_rtm2.py unchanged (QAP 10,000 draws seed 0, target-cluster bootstrap, within-target Spearman).
Usage: python3 analyze_bleurt.py  -> prints and writes rtm_analysis/rtm_analysis_bleurt.json"""
import json, os, numpy as np
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyze_rtm2.py')).read().split('def load_files')[0]; exec(src)
B = json.load(open(f'{HERE}/rtm_analysis/rtm_bleurt_cells.json')); RES = {}
def cells(prefixes, fname):
    out = {}
    for k, v in B.items():
        path, cell = k.split('::')
        if any(path.startswith(p) for p in prefixes) and os.path.basename(path).startswith(fname): out[cell] = v['bleurt']
    return out
def grid(c):
    base = np.array([c.get(f'scratch_{l}', np.nan) for l in LANGS]); G = np.full((10, 10), np.nan); N = np.full((10, 10), np.nan); selfc = np.array([c.get(f'transfer_{l}_{l}', np.nan) for l in LANGS])
    for s in LANGS:
        for t in LANGS:
            v = c.get(f'transfer_{s}_{t}')
            if v is None or s == t: continue
            G[IDX[s], IDX[t]] = v - base[IDX[t]]; N[IDX[s], IDX[t]] = v - selfc[IDX[t]]
    return base, selfc, G, N
def tests(tag, G):
    keep_all = list(range(10)); keep_nobg = [i for i in keep_all if LANGS[i] != 'bg']; out = {}
    for name, keep in [('all', keep_all), ('no_bg', keep_nobg)]:
        for dk in ['S1_dtw', 'S1_body', 'S1_hands']:
            q = qap(G, dmat(dk), keep); 
            if q is None: continue
            q['cluster_ci95'] = clustered(G, dmat(dk), keep) if dk == 'S1_dtw' else None; q['within'] = within(G, dmat(dk), keep) if dk == 'S1_dtw' else None
            out[f'{name}|{dk}'] = q
            print(f'  [{tag}] {name} {dk}: n={q["n"]} slope {q["slope"]:+.3f}/SD, QAP one-sided(neg) p={q["p_one_sided_negative"]:.3f}, two-sided p={q["p_two_sided"]:.3f}' + (f' | cluster CI {q["cluster_ci95"]}' if q['cluster_ci95'] else ''))
    return out
# scratch grids (RTM v2, RTM frozen 60 ep)
for name, prefixes, fname in [('RTM scratch v2 (100 ep, z-score)', ['rtm2_s1_c', 'rtm2_s2_c'], 'rtm_results_scratch_s0'), ('RTM scratch frozen (60 ep)', ['rtm_scratch_a', 'rtm_scratch_b_k'], 'rtm_results_scratch_s0')]:
    c = cells(prefixes, fname); base, selfc, G, N = grid(c); print(f'\n{name}: {len(c)} cells, baseline BLEURT mean {np.nanmean(base):.2f}, mean gain {np.nanmean(G):+.3f} (positive {int((G > 0).sum())} of {int((~np.isnan(G)).sum())})' + (f', net gain {np.nanmean(N):+.3f}' if not np.isnan(selfc).all() else ''))
    RES[name] = dict(baseline_mean=float(np.nanmean(base)), baseline=dict(zip(LANGS, map(float, base))), gain_mean=float(np.nanmean(G)), gain_tests=tests('gain', G))
    if not np.isnan(selfc).all(): RES[name]['net_gain_mean'] = float(np.nanmean(N)); RES[name]['net_tests'] = tests('net', N)
# MediaPipe full-data scratch grid (descriptive: its own S1 is not in similarity_rtm.json)
c = {k.split('::')[1]: v['bleurt'] for k, v in B.items() if k.startswith('scratch_s0/')}
if c:
    b = [v for k, v in c.items() if k.startswith('scratch_')]; t = [v for k, v in c.items() if k.startswith('transfer_')]
    print(f'\nMediaPipe scratch (60 ep, full data): baseline BLEURT mean {np.mean(b):.2f}, transfer mean {np.mean(t):.2f}'); RES['MediaPipe scratch'] = dict(baseline_mean=float(np.mean(b)), transfer_mean=float(np.mean(t)))
# Uni-Sign arms
def uni(arm, seed):
    pf = {'uni': 'rtm_results_uni_s%d_', 'uni_ctrl': 'rtm_results_uni_ctrl_s%d_'}[arm] % seed; c = {}
    for k, v in B.items():
        path, cell = k.split('::')
        if 'rtm2_uc_a' in path: continue
        if os.path.basename(path).startswith(pf): c[cell] = v['bleurt']
    return c
print('\nUni-Sign arms (BLEURT x100)'); U = {}
for arm in ('uni', 'uni_ctrl'):
    for s in range(4):
        c = uni(arm, s)
        if not c: continue
        base, selfc, G, N = grid(c); U[(arm, s)] = (base, G); print(f'  {arm} seed {s}: {len(c)} cells, baseline mean {np.nanmean(base):.2f}, mean gain {np.nanmean(G):+.3f}')
        RES[f'{arm}_s{s}'] = dict(baseline_mean=float(np.nanmean(base)), gain_mean=float(np.nanmean(G)))
rng = np.random.RandomState(0); common = [s for s in range(4) if ('uni', s) in U and ('uni_ctrl', s) in U]
if common:
    cellsd = []
    for s in common:
        for arm, sg in (('uni', 1), ('uni_ctrl', -1)): pass
        bu, gu = U[('uni', s)]; bc, gc = U[('uni_ctrl', s)]; cellsd.append(np.concatenate([bu - bc, (gu - gc)[~np.isnan(gu)]]))
    d = np.mean(np.stack(cellsd), axis=0); bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(4000)]
    print(f'  pretrained minus control: seeds {common}, {len(d)} cells, mean {d.mean():+.2f} BLEURT, 95% bootstrap CI over cells [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]'); RES['pretrained_minus_control'] = dict(seeds=common, mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
c = {k.split('::')[1]: v['bleurt'] for k, v in B.items() if 'rtm2_uc_a' in k}
if c: b = [v for k, v in c.items() if k.startswith('scratch_')]; print(f'  seed-2 replicate on account A: baseline mean {np.mean(b):.2f}')
json.dump(RES, open(f'{HERE}/rtm_analysis/rtm_analysis_bleurt.json', 'w'), indent=1)
