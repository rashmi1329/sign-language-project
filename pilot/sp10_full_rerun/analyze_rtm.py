"""Analysis of the RTMPose rerun, as fixed in docs/analysis_plan_rtm.md. Seed 0. Reads e2_outputs/rtm_{scratch,uni}_{a,b_k}/ shard files.
Works on partial data (missing cells are skipped and the counts are printed; a partial result is labelled PARTIAL). Not in this script: BLEURT (separate step on saved hyps).
Usage: python3 analyze_rtm.py            -> prints, and writes rtm_analysis/rtm_analysis.json"""
import json, glob, os, itertools
import numpy as np
from scipy import stats
LANGS = ['zh','uk','ru','bg','is','de','it','sv','lt','en']; IDX = {l: i for i, l in enumerate(LANGS)}
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = {}
def load(arm):
    res = {}
    fam = 'scratch' if arm == 'scratch' else 'uni'   # the uni and uni_ctrl result files sit in the same rtm_uni_* output folders
    for f in sorted(glob.glob(f'{HERE}/e2_outputs/rtm_{fam}_*/rtm_results_{arm}_s0_g*.json')): res.update(json.load(open(f)))
    return res
sim = json.load(open(f'{HERE}/similarity_rtm.json'))
def dmat(key):
    D = np.zeros((10, 10))
    for k, v in sim.items():
        a, b = k.split('|'); D[IDX[a], IDX[b]] = D[IDX[b], IDX[a]] = v[key]
    return D
def gains(res, metric):
    G = np.full((10, 10), np.nan); base = {}
    for k, v in res.items():
        if k.startswith('scratch_'): base[k[8:]] = v['test_' + metric]
    for k, v in res.items():
        if k.startswith('transfer_'):
            s, t = k[9:].split('_'); 
            if t in base: G[IDX[s], IDX[t]] = v['test_' + metric] - base[t]
    return G, base
def slope(x, y):
    x = (x - x.mean()) / x.std(); return np.polyfit(x, y, 1)[0]
def pairs(G, D, keep):
    ii = [(i, j) for i in keep for j in keep if i != j and not np.isnan(G[i, j])]
    return np.array([D[i, j] for i, j in ii]), np.array([G[i, j] for i, j in ii]), ii
def qap(G, D, keep, draws=10000, seed=0):
    rng = np.random.RandomState(seed); x, y, ii = pairs(G, D, keep)
    if len(y) < 8: return None
    obs = slope(x, y); cnt1 = cnt2 = 0; keep = list(keep)
    for _ in range(draws):
        p = rng.permutation(len(keep)); m = {keep[a]: keep[p[a]] for a in range(len(keep))}   # relabel nodes of the distance matrix
        xp = np.array([D[m[i], m[j]] for i, j in ii]); s = slope(xp, y)
        cnt1 += s <= obs; cnt2 += abs(s) >= abs(obs)
    return dict(n=len(y), slope=float(obs), p_one_sided_negative=(cnt1 + 1) / (draws + 1), p_two_sided=(cnt2 + 1) / (draws + 1))
def clustered(G, D, keep, B=2000, seed=0):
    rng = np.random.RandomState(seed); x, y, ii = pairs(G, D, keep); tg = np.array([j for _, j in ii]); ts = sorted(set(tg))
    if len(ts) < 3: return None
    bs = []
    for _ in range(B):
        pick = rng.choice(ts, len(ts)); idx = np.concatenate([np.where(tg == t)[0] for t in pick])
        if x[idx].std() > 0: bs.append(slope(x[idx], y[idx]))
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def within(G, D, keep):
    rs = []
    for t in keep:
        ss = [s for s in keep if s != t and not np.isnan(G[s, t])]
        if len(ss) >= 4: rs.append(stats.spearmanr([D[s, t] for s in ss], [G[s, t] for s in ss])[0])
    neg = sum(r < 0 for r in rs); n = len(rs)
    return dict(targets=n, mean_rho=float(np.mean(rs)) if rs else None, negative=int(neg), sign_test_p_one_sided=float(stats.binomtest(neg, n, .5, 'greater').pvalue) if n else None)
def run_scratch():
    res = load('scratch'); n_base = sum(k.startswith('scratch_') for k in res); n_tr = sum(k.startswith('transfer_') for k in res)
    print(f'SCRATCH cells found: {n_base} baselines, {n_tr} transfers of 10 + 90', '' if (n_base, n_tr) == (10, 90) else '  *** PARTIAL: results below are not the planned test ***')
    OUT['scratch_cells'] = [n_base, n_tr]
    if not res: return
    for m in ('chrf', 'bleu'):
        G, base = gains(res, m); OUT[f'baseline_{m}'] = base
        tr = [res[k]['test_' + m] for k in res if k.startswith('transfer_')]
        print(f'\n[{m}] baselines mean {np.mean(list(base.values())):.3f} :', {l: round(v, 2) for l, v in base.items()})
        g = G[~np.isnan(G)]
        if len(g): print(f'[{m}] gains computable {len(g)}; mean gain {g.mean():+.3f}; positive {int((g > 0).sum())}; transfer-cell mean score {np.mean(tr):.3f}')
        for tag, keep in (('all languages', list(range(10))), ('without bg', [i for i in range(10) if LANGS[i] != 'bg'])):
            for dk in ('S1_dtw', 'S1_body', 'S1_hands'):
                D = dmat(dk); q = qap(G, D, keep)
                if q is None: print(f'  [{m}] {tag} {dk}: too few cells'); continue
                line = f'  [{m}] {tag} {dk}: n={q["n"]} slope {q["slope"]:+.3f}/SD, QAP one-sided(neg) p={q["p_one_sided_negative"]:.3f}, two-sided p={q["p_two_sided"]:.3f}'
                if dk == 'S1_dtw':
                    ci = clustered(G, D, keep); w = within(G, D, keep); line += f' | target-cluster CI {ci} | within-target {w}'; OUT[f'{m}|{tag}'] = dict(qap=q, ci=ci, within=w)
                print(line)
    s4 = {k[8:]: np.array(v['s4_mean']) for k, v in res.items() if k.startswith('scratch_') and 's4_mean' in v}
    if len(s4) == 10:
        S4 = np.zeros((10, 10))
        for a, b in itertools.combinations(LANGS, 2): S4[IDX[a], IDX[b]] = S4[IDX[b], IDX[a]] = 1 - np.dot(s4[a], s4[b]) / np.linalg.norm(s4[a]) / np.linalg.norm(s4[b])
        G, _ = gains(res, 'chrf'); q = qap(G, S4, list(range(10))); print('\n[chrf] S4 comparator (cosine of mean encoder outputs), all languages:', q); OUT['S4'] = q
def run_uni():
    ru, rc = load('uni'), load('uni_ctrl')
    print(f'\nUNI cells: pretrained {len(ru)}, control {len(rc)} (of 30 each)')
    OUT['uni_cells'] = [len(ru), len(rc)]
    common = sorted(set(ru) & set(rc))
    if not common: return
    for m in ('chrf', 'bleu'):
        d = np.array([ru[k]['test_' + m] - rc[k]['test_' + m] for k in common]); rng = np.random.RandomState(0)
        bs = [rng.choice(d, len(d)).mean() for _ in range(10000)]
        print(f'[{m}] pretrained minus control over {len(d)} paired cells: mean {d.mean():+.3f}, 95% bootstrap CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]')
        OUT[f'uni_vs_ctrl_{m}'] = dict(n=len(d), mean=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        for name, r in (('pretrained', ru), ('control', rc)):
            G, base = gains(r, m); g = G[~np.isnan(G)]
            print(f'   {name}: baselines {len(base)}, mean baseline {np.mean(list(base.values())) if base else float("nan"):.3f}, mean transfer gain {g.mean() if len(g) else float("nan"):+.3f} over {len(g)} cells')
run_scratch(); run_uni()
os.makedirs(f'{HERE}/rtm_analysis', exist_ok=True); json.dump(OUT, open(f'{HERE}/rtm_analysis/rtm_analysis.json', 'w'), indent=1)
