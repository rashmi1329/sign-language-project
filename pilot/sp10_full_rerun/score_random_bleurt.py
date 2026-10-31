"""BLEURT-20-D12 (x100) of the random-train-sentence baseline (3 draws per language, floor_table.py), same scorer settings as score_bleurt.py. Output rtm_analysis/rtm_random_bleurt.json."""
import json, glob, numpy as np, torch
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
torch.set_num_threads(2)
L = ['zh','uk','ru','bg','is','de','it','sv','lt','en']; refs = json.load(open('kaggle_dataset_rtm_c/english_refs.json')); rh = json.load(open('rtm_analysis/rtm_random_hyps.json'))
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
for l in L:
    keys = v2[f'scratch_{l}']['keys']; out[l] = [100 * float(np.mean(score([(refs[k], x) for k, x in zip(keys, h)]))) for h in rh[l]]; print(l, [round(x, 2) for x in out[l]], flush=True)
out['mean'] = float(np.mean([np.mean(out[l]) for l in L])); print('mean', round(out['mean'], 2)); json.dump(out, open('rtm_analysis/rtm_random_bleurt.json', 'w'), indent=1)
