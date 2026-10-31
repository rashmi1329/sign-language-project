#!/usr/bin/env python3
"""Second independent check (see verify_headline_numbers.py for the first): numbers of the Results section that are not in Tables 3-5.
Everything is recomputed from the raw per-cell files; no analysis script of the project is imported.

 1. Same-seed rerun of the pretrained arm (seed 2 on two accounts): identical cells, share of differing translations, SD of the chrF difference.
 2. Run-to-run noise between the two seeds of Round C (SD of the cell difference, cell correlation, noise share), target-ranking agreement.
 3. Pair-specific part of gain: the paper's shortcut formula (row/column means) and the exact two-way least-squares fit, side by side.
 4. Pretrained arm vs random-init control: per-seed baselines and gains, mean paired chrF difference, bootstrap over cells.
 5. Share of outputs that are verbatim training sentences (needs english_refs.json, see README) -- skipped if the file is absent.
 6. Language identification from clip-level framing features (ten features, five-fold random forest) from rtm_qc/clip_table.csv.
 7. S1 of Round A: the 45 values in s1_new_from_log.json against the printed values in the Kaggle log of the MediaPipe extraction.
Run from pilot/sp10_full_rerun :  python3 verify_secondary_numbers.py
"""
import json, glob, os, re, numpy as np
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
E = os.path.join(H, 'e2_outputs')
L = ['zh', 'uk', 'ru', 'bg', 'is', 'de', 'it', 'sv', 'lt', 'en']; ix = {l: i for i, l in enumerate(L)}
def load(pats):
    r = {}
    for p in pats:
        for f in sorted(glob.glob(os.path.join(E, p))):
            if 'sweep' in f or 'gate' in f: continue
            r.update(json.load(open(f)))
    return r
def P(label, paper, got, fmt='%.3f'): print(f'  {label:60s} paper: {paper!s:>14}   recomputed: {fmt % got if isinstance(got, float) else got}')

print('1. same-seed rerun (pretrained arm, seed 2: rtm2_uc_a vs rtm2_uc_c)')
A = load(['rtm2_uc_a/rtm_results_uni_s2_g*.json']); C = load(['rtm2_uc_c/rtm_results_uni_s2_g*.json']); ks = sorted(set(A) & set(C))
d = np.array([float(A[k]['test_chrf']) - float(C[k]['test_chrf']) for k in ks])
have_hyps = all('hyps' in A[k] and 'hyps' in C[k] for k in ks)   # the public copy has no saved translations
if have_hyps: nd = sum(a != c for k in ks for a, c in zip(A[k]['hyps'], C[k]['hyps'])); nt = sum(len(A[k]['hyps']) for k in ks)
P('cells compared', 30, float(len(ks)), '%.0f'); P('cells with identical chrF', 0, float(sum(d == 0)), '%.0f')
if have_hyps: P('translations compared', 5970, float(nt), '%.0f'); P('share of translations that differ (%)', 96.1, 100 * nd / nt, '%.1f')
else: print('  translations compared / share of translations that differ: skipped (saved translations are not part of this copy)')
P('SD of per-cell chrF difference', 0.87, float(d.std(ddof=1))); P('max |difference|', 1.84, float(abs(d).max()), '%.2f')

print('2. noise between the two Round C seeds')
def mat(r):
    T = np.full((10, 10), np.nan); b = np.zeros(10)
    for k, v in r.items():
        if k.startswith('scratch_'): b[ix[k[8:]]] = float(v['test_chrf'])
        elif k.startswith('transfer_'):
            s, t = k[9:].split('_')
            if s != t: T[ix[s], ix[t]] = float(v['test_chrf'])
    return T, b
T0, b0 = mat(load(['rtm2_s[12]_c/rtm_results_scratch_s0_g*_e100_zscore.json'])); T1, b1 = mat(load(['rtm2_s[abc]_*/rtm_results_scratch_s1_g*_e100_zscore.json']))
idx = [(i, j) for i in range(10) for j in range(10) if i != j]; y0 = np.array([T0[i, j] for i, j in idx]); y1 = np.array([T1[i, j] for i, j in idx]); dd = y0 - y1
P('SD of the seed difference per cell', 0.79, float(dd.std(ddof=1))); P('per-run SD', 0.56, float(dd.std(ddof=1) / np.sqrt(2))); P('cell correlation across seeds', 0.89, float(np.corrcoef(y0, y1)[0, 1]))
P('noise share of total cell variance', 0.11, float((dd.var(ddof=1) / 2) / np.concatenate([y0, y1]).var(ddof=1)))
tm0, tm1 = np.nanmean(T0, 0), np.nanmean(T1, 0)
P('target means range, seed 0 (min)', 10.8, float(tm0.min()), '%.1f'); P('target means range, seed 0 (max)', 16.5, float(tm0.max()), '%.1f')
P('Spearman of target means across seeds', 1.00, float(stats.spearmanr(tm0, tm1)[0]), '%.2f'); P('Spearman of baselines across seeds', 0.84, float(stats.spearmanr(b0, b1)[0]), '%.2f')
G0, G1 = T0 - b0[None, :], T1 - b1[None, :]
P('Spearman of mean gain by target', 0.44, float(stats.spearmanr(np.nanmean(G0, 0), np.nanmean(G1, 0))[0]), '%.2f')
P('Spearman of source means (transfer chrF)', 0.15, float(stats.spearmanr(np.nanmean(T0, 1), np.nanmean(T1, 1))[0]), '%.2f')

print('3. pair-specific part of gain: shortcut formula vs exact two-way least squares')
def add(Y): return Y - np.nanmean(Y, 1, keepdims=True) - np.nanmean(Y, 0, keepdims=True) + np.nanmean(Y)
m = ~np.isnan(T0); a0, a1 = add(T0), add(T1)
P('shortcut (row/column means): correlation of the two seeds', 0.16, float(np.corrcoef(a0[m], a1[m])[0, 1]))
P('shortcut: SD seed 0 / seed 1', '0.52 / 0.56', float(a0[m].std()), '%.3f (seed 0)')
X = np.zeros((90, 19))
for r, (i, j) in enumerate(idx):
    X[r, 0] = 1
    if i > 0: X[r, i] = 1
    if j > 0: X[r, 9 + j] = 1
res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
o0, o1 = res(y0), res(y1)
print(f'  exact two-way fit: correlation {np.corrcoef(o0, o1)[0, 1]:.3f}, SD {o0.std():.3f} / {o1.std():.3f}   <-- differs from the paper (0.16; 0.52/0.56); see KNOWN_GAPS')

print('4. pretrained arm vs random-init control')
runs = {'pre s0': 'rtm_uni_*/rtm_results_uni_s0_g*.json', 'pre s1': 'rtm2_ua_c/rtm_results_uni_s1_g*.json', 'pre s2': 'rtm2_uc_c/rtm_results_uni_s2_g*.json', 'pre s3': 'rtm2_ub_c/rtm_results_uni_s3_g*.json',
        'ctrl s0': 'rtm_uni_*/rtm_results_uni_ctrl_s0_g*.json', 'ctrl s1': 'rtm2_ud_c/rtm_results_uni_ctrl_s1_g*.json'}
R = {k: load([v]) for k, v in runs.items()}
def gains(r):
    b = {k[8:]: float(v['test_chrf']) for k, v in r.items() if k.startswith('scratch_')}
    return np.mean(list(b.values())), np.mean([float(v['test_chrf']) - b[k.split('_')[-1]] for k, v in r.items() if k.startswith('transfer_')])
for k, paper in zip(R, ['+0.08', '-0.11', '-0.02', '+0.18', '+0.27', '+0.43']):
    bm, g = gains(R[k]); print(f'  {k}: mean baseline chrF {bm:.2f} (paper range 15.0-15.6 for pretrained, 13.84 / 13.58 control), mean transfer gain {g:+.2f} (paper {paper})')
cm = lambda r: {k: float(v['test_chrf']) for k, v in r.items()}
dif = []
for p_, c_ in (('pre s0', 'ctrl s0'), ('pre s1', 'ctrl s1')):
    Pp, Cc = cm(R[p_]), cm(R[c_]); dif.append(np.array([Pp[k] - Cc[k] for k in sorted(Pp)]))
D = np.stack(dif)                       # 2 seeds x 30 cells
rng = np.random.RandomState(0)
cl = [D[:, rng.randint(0, 30, 30)].mean() for _ in range(10000)]          # resample the 30 cells, keep both seeds together
ind = [rng.choice(D.ravel(), 60).mean() for _ in range(10000)]
P('mean paired chrF difference pretrained - control', 1.57, float(D.mean()), '%.2f')
print(f'  95% bootstrap interval, cells resampled with both seeds together: [{np.percentile(cl, 2.5):.2f}, {np.percentile(cl, 97.5):.2f}]  (paper [1.19, 1.95]);  60 draws treated as independent: [{np.percentile(ind, 2.5):.2f}, {np.percentile(ind, 97.5):.2f}]')

print('5. share of outputs that are verbatim training sentences')
rp = os.path.join(H, 'kaggle_dataset_rtm_c', 'english_refs.json')
if os.path.exists(rp):
    refs = json.load(open(rp)); norm = lambda s: re.sub(r'\s+', ' ', s.strip().lower()); train = {norm(v) for k, v in refs.items() if k.startswith('train/')}
    def share(files, kind):          # kind 'base' = the ten per-language models only; 'transfer' = transfer cells only
        n = c = 0
        for f in files:
            for k, v in json.load(open(f)).items():
                if not (isinstance(v, dict) and 'hyps' in v): continue
                if (kind == 'base') != k.startswith('scratch_'): continue
                n += len(v['hyps']); c += sum(norm(h) in train for h in v['hyps'])
        return 100 * c / n
    fs = lambda p: [f for f in sorted(glob.glob(os.path.join(E, p))) if 'sweep' not in f and 'gate' not in f]
    arms = [('Round A', 58.4, ['scratch_s0/e2_results_scratch_s0.json']), ('Round B', 83.8, ['rtm_scratch_*/rtm_results_scratch_s0_g*.json']),
            ('Round C seed 0', 38.3, ['rtm2_s[12]_c/rtm_results_scratch_s0_g*_e100_zscore.json']), ('control seed 0', 47.6, ['rtm_uni_*/rtm_results_uni_ctrl_s0_g*.json']),
            ('pretrained, seeds 0-3 (seed 2 = rtm2_uc_c)', 8.5, ['rtm_uni_*/rtm_results_uni_s0_g*.json', 'rtm2_ua_c/rtm_results_uni_s1_g*.json', 'rtm2_uc_c/rtm_results_uni_s2_g*.json', 'rtm2_ub_c/rtm_results_uni_s3_g*.json'])]
    print('  (the paper figure is for the ten per-language baseline models only; transfer cells are shown for comparison and are NOT what the paper reports)')
    for name, paper, pats in arms:
        files = [f for p in pats for f in fs(p)]
        P(name + ' - baseline models', paper, share(files, 'base'), '%.1f'); print(f'  {"":60s} transfer cells of the same grid: {share(files, "transfer"):.1f}%')
else: print('  skipped: kaggle_dataset_rtm_c/english_refs.json is not part of the release (SP-10 reference sentences); place it there to run this part')

print('6. language identification from framing features')
try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    df = pd.read_csv(os.path.join(H, 'rtm_qc', 'clip_table.csv'), keep_default_na=False); yv = df.lang.values
    ten = ['sh_px', 'ok_body', 'ok_face', 'ok_lh', 'ok_rh', 'vis_lh', 'vis_rh', 'wrist_out_l', 'wrist_out_r', 'wrist_y_mean']
    acc = lambda f: float(cross_val_score(RandomForestClassifier(200, random_state=0, n_jobs=-1), df[f].values, yv, cv=5).mean())
    P('ten framing features', 0.727, acc(ten)); P('shoulder width alone', 0.237, acc(['sh_px']))
    print('  note: rtm_quality_check.py itself reports 0.796 because it uses 14 features (adds frames, lr_ok, mot_all, wrist_mot); the paper quotes the ten-feature variant, whose original run was not saved')
except Exception as e: print('  skipped:', e)

print('7. Round A S1 against the printed values of the MediaPipe extraction log')
lg = os.path.join(H, 'full_poses', 'notebooka9f823024d.log')
if os.path.exists(lg):
    txt = ''.join(ev.get('data', '') for ev in json.load(open(lg)))
    log = {m.group(1): float(m.group(2)) for m in re.finditer(r"([A-Z]{3}-[A-Z]{3}): \{'S1_dtw': ([0-9.eE+-]+)", txt)}
    s = json.load(open(os.path.join(H, 's1_new_from_log.json')))
    bad = [k for k in s if k not in log or abs(log[k] - s[k][0]) > 1e-9]
    P('pairs in s1_new_from_log.json found in the log with equal S1_dtw', 45, float(len(s) - len(bad)), '%.0f')
else: print('  skipped: log not present')
