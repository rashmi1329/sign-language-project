"""BLEURT-20-D12 (x100) of the retrieval baselines of nn_baseline.py (nn_flat, nn_pooled, const, oracle; 'random' was not saved as text). Same scorer settings as score_bleurt.py.
Needs rtm_analysis/rtm_nn_baseline.json and the v2 scratch result files (test ids). Writes rtm_analysis/rtm_nn_bleurt.json."""
import json, glob, numpy as np, torch
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
torch.set_num_threads(2)
LANGS = ['zh','uk','ru','bg','is','de','it','sv','lt','en']; refs = json.load(open('kaggle_dataset_rtm_c/english_refs.json')); nn = json.load(open('rtm_analysis/rtm_nn_baseline.json'))
v2 = {}
for f in sorted(glob.glob('e2_outputs/rtm2_s[12]_c/rtm_results_scratch_s0_g*_e100_*.json')): v2.update(json.load(open(f)))
n = 'lucadiliello/BLEURT-20-D12'; m = BleurtForSequenceClassification.from_pretrained(n).eval(); t = BleurtTokenizer.from_pretrained(n); cache = {}
def score(pairs):
    todo = sorted({p for p in pairs if p not in cache}, key=lambda p: len(p[0]) + len(p[1]))
    with torch.no_grad():
        for i in range(0, len(todo), 32):
            b = todo[i:i + 32]; inp = t([p[0] for p in b], [p[1] for p in b], padding='longest', truncation=True, max_length=128, return_tensors='pt')
            for p, s in zip(b, m(**inp).logits.flatten().tolist()): cache[p] = s
    return [cache[p] for p in pairs]
out = {}
for l in LANGS:
    keys = v2[f'scratch_{l}']['keys']; out[l] = {}
    for v in ('nn_flat', 'nn_pooled', 'const', 'oracle'):
        h = nn['langs'][l][v]['hyps']; assert len(h) == len(keys)
        out[l][v] = 100 * float(np.mean(score([(refs[k], x) for k, x in zip(keys, h)])))
    print(l, {k: round(x, 2) for k, x in out[l].items()}, flush=True)
out['mean'] = {v: float(np.mean([out[l][v] for l in LANGS])) for v in ('nn_flat', 'nn_pooled', 'const', 'oracle')}; print('mean', {k: round(x, 2) for k, x in out['mean'].items()})
json.dump(out, open('rtm_analysis/rtm_nn_bleurt.json', 'w'), indent=1)
