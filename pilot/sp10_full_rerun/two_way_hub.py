"""(1) Two-way adjusted QAP: gain and S1 are both residualised on target AND source dummies (least squares), so only pair-specific closeness is left; statistic = slope of residual gain on residual S1,
null = node relabelling of S1 (10,000 draws, seed 0), one-sided negative. (2) Hubness diagnostic: does a language's mean S1 distance to the others track how much of its pose is observed?
EXPLORATORY (limitations entry 77). Usage: python3 two_way_hub.py -> rtm_analysis/rtm_two_way_hub.json"""
import json, os, glob, numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
t = open(f'{HERE}/target_adjusted.py').read(); exec(t.split('OUT2 = {}')[0]); exec('def grid_metric' + t.split('def grid_metric', 1)[1].split('v2, _ = load_files')[0])
v2, _ = load_files('rtm2_s[12]_*/rtm_results_scratch_s0_g*_e100_*.json'); fr = {}
for f in sorted(glob.glob(f'{HERE}/e2_outputs/rtm_scratch_*/rtm_results_scratch_s0_g*.json')): fr.update(json.load(open(f)))
def twoway(G, D, keep, R=10000, seed=0):
    keep = list(keep); ii = [(i, j) for i in keep for j in keep if i != j and not np.isnan(G[i, j])]; n = len(ii)
    Z = np.zeros((n, 2 * len(keep)));
    for r, (s, tt) in enumerate(ii): Z[r, keep.index(s)] = 1; Z[r, len(keep) + keep.index(tt)] = 1
    M = np.eye(n) - Z @ np.linalg.pinv(Z)   # residual-maker (rank-deficient dummies handled by pinv)
    y = M @ np.array([G[i, j] for i, j in ii]); xs = np.array([D[i, j] for i, j in ii]); sd = np.std([D[i, j] for i in keep for j in keep if i != j])
    st = lambda v: (lambda xr: (xr @ y) / (xr @ xr))(M @ v); obs = st(xs); rng = np.random.RandomState(seed); cnt = 0
    for _ in range(R):
        p = rng.permutation(len(keep)); m = {keep[a]: keep[p[a]] for a in range(len(keep))}; cnt += st(np.array([D[m[i], m[j]] for i, j in ii])) <= obs
    return dict(n=n, slope_per_SD=float(obs * sd), p_one_sided=float((cnt + 1) / (R + 1)))
OUT = {'two_way': {}}; allk = list(range(10)); nobg = [i for i in allk if LANGS[i] != 'bg']
def go(G, name):
    for dk in ('S1_dtw', 'S1_body', 'S1_hands'):
        for sn, keep in (('all', allk), ('no_bg', nobg)):
            r = twoway(G, dmat(dk), keep); OUT['two_way'][f'{name}|{dk}|{sn}'] = r; print(f'  {name} {dk} {sn}: pair-specific slope {r["slope_per_SD"]:+.3f} per SD, one-sided p {r["p_one_sided"]:.4f}')
for m in ('chrf', 'bleu'):
    print(f'== v2 {m}'); go(grid_metric(v2, m)[0], f'v2|{m}'); print(f'== frozen {m}'); go(grid_metric(fr, m)[0], f'frozen|{m}')
for tag, pre in (('v2', ['rtm2_s1_c', 'rtm2_s2_c']), ('frozen', ['rtm_scratch_a', 'rtm_scratch_b_k'])):
    base, selfc, G, N = grid(cells(pre, 'rtm_results_scratch_s0')); print(f'== {tag} bleurt'); go(G, f'{tag}|bleurt')
# hubness diagnostic
import importlib.util
spec = importlib.util.spec_from_file_location('rtm_data', f'{HERE}/rtm_data.py'); rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
obs = {}
for l in LANGS:
    z = rd.load_lang(f'{HERE}/rtm_clean', l, split='train'); ks = sorted(z)[:300]; M = np.stack([z[k][1] for k in ks])   # (n,T,69)
    obs[l] = dict(body=float(M[:, :, 0:9].mean()), hands=float(M[:, :, 9:51].mean()), face=float(M[:, :, 51:69].mean()), all=float(M.mean()))
print('\nshare of points observed (train clips, first 300):'); [print(f'  {l}: ' + ' '.join(f'{k} {v:.2f}' for k, v in obs[l].items())) for l in LANGS]
OUT['observed'] = obs; OUT['hub'] = {}
for dk in ('S1_dtw', 'S1_body', 'S1_hands'):
    D = dmat(dk); md = [np.mean([D[i, j] for j in range(10) if j != i]) for i in range(10)]
    for part in ('all', 'body', 'hands'):
        r, p = stats.spearmanr(md, [obs[l][part] for l in LANGS]); OUT['hub'][f'{dk}~observed_{part}'] = (float(r), float(p)); print(f'  mean {dk} distance of a language to the others vs its observed share of {part} points: rho {r:+.2f}, p {p:.3f}')
json.dump(OUT, open(f'{HERE}/rtm_analysis/rtm_two_way_hub.json', 'w'), indent=1)
