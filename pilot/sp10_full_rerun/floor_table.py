"""Contribution 2 (limitations entry 79): what do chrF / BLEU score WITHOUT reading the signing? Point estimates and a sentence-bootstrap (resampling test sentences within each language, same sentences for every system)
for: random train sentence, one constant sentence, pose nearest neighbour (flat / pooled), oracle retrieval (ceiling), and the trained systems (frozen scratch, v2 scratch, random-init control seed 0, pretrained seed 0).
Metric = mean over the 10 languages of corpus chrF / BLEU. Also saves 3 random-sentence draws per language for BLEURT scoring. Usage: python3 floor_table.py [B] -> rtm_analysis/rtm_floor_table.json, rtm_analysis/rtm_random_hyps.json"""
import json, glob, os, sys, numpy as np
from sacrebleu.metrics import BLEU, CHRF
H = os.path.dirname(os.path.abspath(__file__)); L = ['zh','uk','ru','bg','is','de','it','sv','lt','en']; B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
refs = json.load(open(f'{H}/kaggle_dataset_rtm_c/english_refs.json')); nn = json.load(open(f'{H}/rtm_analysis/rtm_nn_baseline.json'))['langs']
def load(pat):
    r = {}
    for f in sorted(glob.glob(f'{H}/e2_outputs/{pat}')): r.update(json.load(open(f)))
    return r
v2 = load('rtm2_s[12]_c/rtm_results_scratch_s0_g*_e100_*.json'); fr = load('rtm_scratch_*/rtm_results_scratch_s0_g*.json'); uni = load('rtm_uni_*/rtm_results_uni_s0_g*.json'); ctl = load('rtm_uni_*/rtm_results_uni_ctrl_s0_g*.json')
keys = {l: v2[f'scratch_{l}']['keys'] for l in L}
train = {l: None for l in L}
import importlib.util
spec = importlib.util.spec_from_file_location('rtm_data', f'{H}/rtm_data.py'); rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
trs = {}
for l in L: z = np.load(f'{H}/rtm_clean/rtm_std_{l}.npz'); trs[l] = [refs[k[2:]] for k in z.files if k.startswith('x|train/')]
sysh = {}   # system -> language -> {key: hyp}
def from_res(res): return {l: dict(zip(res[f'scratch_{l}']['keys'], res[f'scratch_{l}']['hyps'])) for l in L}
sysh['frozen scratch (60 ep)'] = from_res(fr); sysh['v2 scratch (100 ep)'] = from_res(v2); sysh['random-init control (seed 0)'] = from_res(ctl); sysh['pretrained Uni-Sign (seed 0)'] = from_res(uni)
for name in ('const', 'nn_flat', 'nn_pooled', 'oracle'): sysh[{'const': 'one constant sentence', 'nn_flat': 'pose nearest neighbour (flat)', 'nn_pooled': 'pose nearest neighbour (pooled)', 'oracle': 'oracle retrieval (ceiling)'}[name]] = {l: dict(zip(keys[l], nn[l][name]['hyps'])) for l in L}
rr = np.random.RandomState(5)
def rand_h(l): return {k: trs[l][rr.randint(len(trs[l]))] for k in keys[l]}
names = ['random train sentence'] + list(sysh)
CH, BL = CHRF(), BLEU(); CACHE = f'{H}/rtm_analysis/_floor_stats.npz'; stats = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
def prep(sname, hyp_by_lang):
    for l in L:
        key = f'{sname}|{l}'
        if key + '|c' in stats: continue
        r = [refs[k] for k in keys[l]]; h = [hyp_by_lang[l][k] for k in keys[l]]
        stats[key + '|c'] = np.array(CH._extract_corpus_statistics(h, [r])); stats[key + '|b'] = np.array(BL._extract_corpus_statistics(h, [r]))
for s_, hb in sysh.items(): prep(s_, hb)
NR = 10
for d in range(NR): prep(f'random#{d}', {l: {k: trs[l][rr.randint(len(trs[l]))] for k in keys[l]} for l in L})
np.savez(CACHE, **stats)
def score(sname, idx_by_lang):
    c, b = [], []
    for l in L:
        i = idx_by_lang[l]; c.append(CH._compute_f_score(stats[f'{sname}|{l}|c'][i].sum(0))); b.append(BL._compute_score_from_stats(stats[f'{sname}|{l}|b'][i].sum(0)).score)
    return np.mean(c), np.mean(b)
full = {l: np.arange(len(keys[l])) for l in L}; point = {'random train sentence': tuple(np.mean([score(f'random#{d}', full) for d in range(NR)], axis=0))}
for s_ in sysh: point[s_] = score(s_, full)
print('point estimates (mean over 10 languages)  chrF   BLEU'); [print(f'  {s_:34s} {point[s_][0]:6.2f} {point[s_][1]:6.2f}') for s_ in names]
rb = np.random.RandomState(0); boot = {s_: [] for s_ in names}
for it in range(B):
    idx = {l: rb.randint(0, len(keys[l]), len(keys[l])) for l in L}
    boot['random train sentence'].append(score(f'random#{rb.randint(NR)}', idx))
    for s_ in sysh: boot[s_].append(score(s_, idx))
boot = {s_: np.array(v) for s_, v in boot.items()}; OUT = {'point': {s_: list(point[s_]) for s_ in names}, 'ci95': {}, 'diff': {}, 'B': B}
for s_ in names: OUT['ci95'][s_] = {m: [float(np.percentile(boot[s_][:, i], 2.5)), float(np.percentile(boot[s_][:, i], 97.5))] for i, m in enumerate(('chrf', 'bleu'))}
print(f'\n95% sentence-bootstrap CI (B={B})'); [print(f'  {s_:34s} chrF [{OUT["ci95"][s_]["chrf"][0]:5.2f}, {OUT["ci95"][s_]["chrf"][1]:5.2f}]  BLEU [{OUT["ci95"][s_]["bleu"][0]:5.2f}, {OUT["ci95"][s_]["bleu"][1]:5.2f}]') for s_ in names]
print('\npaired differences (bootstrap mean and 95% CI; same resampled sentences)')
for a, bq in (('v2 scratch (100 ep)', 'one constant sentence'), ('pretrained Uni-Sign (seed 0)', 'one constant sentence'), ('random-init control (seed 0)', 'one constant sentence'), ('frozen scratch (60 ep)', 'one constant sentence'),
              ('v2 scratch (100 ep)', 'pose nearest neighbour (pooled)'), ('pretrained Uni-Sign (seed 0)', 'pose nearest neighbour (pooled)'), ('random-init control (seed 0)', 'pose nearest neighbour (pooled)'), ('pretrained Uni-Sign (seed 0)', 'random-init control (seed 0)')):
    d = boot[a] - boot[bq]; OUT['diff'][f'{a} - {bq}'] = {m: [float(d[:, i].mean()), float(np.percentile(d[:, i], 2.5)), float(np.percentile(d[:, i], 97.5))] for i, m in enumerate(('chrf', 'bleu'))}
    print(f'  {a} - {bq}: chrF {d[:,0].mean():+.2f} [{np.percentile(d[:,0],2.5):+.2f}, {np.percentile(d[:,0],97.5):+.2f}]   BLEU {d[:,1].mean():+.2f} [{np.percentile(d[:,1],2.5):+.2f}, {np.percentile(d[:,1],97.5):+.2f}]')
json.dump(OUT, open(f'{H}/rtm_analysis/rtm_floor_table.json', 'w'), indent=1)
rh = {l: [[trs[l][rr.randint(len(trs[l]))] for _ in keys[l]] for _ in range(3)] for l in L}; json.dump(rh, open(f'{H}/rtm_analysis/rtm_random_hyps.json', 'w'))
