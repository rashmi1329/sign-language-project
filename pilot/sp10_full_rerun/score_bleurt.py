"""BLEURT-20-D12 on every saved translation (CPU or GPU). Same scorer as the MediaPipe pilot (lucadiliello/BLEURT-20-D12 via bleurt_pytorch, padding=longest, max_length 128, batch 32, x100); it reproduces the stored pilot values exactly (checked 2026-10-21: scratch bg-bg 21.675, transfer de-bg 25.385, scratch en-en 25.785).
Needs transformers==4.48.3, tokenizers==0.21.0, huggingface_hub<1.0 (the bleurt_pytorch package breaks on newer transformers).
Writes rtm_analysis/rtm_bleurt_cells.json: {"<folder>/<file>::<cell>": {"bleurt": mean x100, "n": sentences}}. Identical (reference, translation) pairs are scored once. Resumable: finished files are skipped.
Run from pilot/sp10_full_rerun."""
import json, glob, os, sys, time, numpy as np, torch
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
torch.set_num_threads(int(os.environ.get('THREADS', '2')))
OUT = 'rtm_analysis/rtm_bleurt_cells.json'; refs = json.load(open('kaggle_dataset_rtm_c/english_refs.json'))
ORDER = ['e2_outputs/rtm2_s1_c/rtm_results_scratch_*', 'e2_outputs/rtm2_s2_c/rtm_results_scratch_*', 'e2_outputs/rtm_uni_*/rtm_results_uni_s*', 'e2_outputs/rtm_uni_*/rtm_results_uni_ctrl_s*',
         'e2_outputs/rtm2_u*_c/rtm_results_uni_s*', 'e2_outputs/rtm2_u*_c/rtm_results_uni_ctrl_s*', 'e2_outputs/rtm_scratch_*/rtm_results_scratch_*', 'e2_outputs/scratch_s0/e2_results_scratch_s0.json', 'e2_outputs/rtm2_uc_a/rtm_results_uni_s*']
files = []
for pat in ORDER:
    for f in sorted(glob.glob(pat)):
        if any(x in os.path.basename(f) for x in ('gate', 'sweep')) or f in files: continue
        files.append(f)
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
n = 'lucadiliello/BLEURT-20-D12'; m = BleurtForSequenceClassification.from_pretrained(n).eval(); t = BleurtTokenizer.from_pretrained(n)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; m.to(dev); cache = {}
def score(pairs):
    todo = [p for p in dict.fromkeys(pairs) if p not in cache]; todo.sort(key=lambda p: len(p[0]) + len(p[1]))
    with torch.no_grad():
        for i in range(0, len(todo), 32):
            b = todo[i:i + 32]; inp = t([p[0] for p in b], [p[1] for p in b], padding='longest', truncation=True, max_length=128, return_tensors='pt').to(dev)
            for p, s in zip(b, m(**inp).logits.flatten().tolist()): cache[p] = s
    return [cache[p] for p in pairs]
t0 = time.time(); total = 0
for fi, f in enumerate(files):
    if all(f'{os.path.relpath(f, "e2_outputs")}::{c}' in done for c in json.load(open(f))): continue
    d = json.load(open(f))
    for c, v in d.items():
        if 'hyps' not in v: continue
        sc = score([(refs[k], h) for k, h in zip(v['keys'], v['hyps'])]); done[f'{os.path.relpath(f, "e2_outputs")}::{c}'] = dict(bleurt=100 * float(np.mean(sc)), n=len(sc)); total += len(sc)
    json.dump(done, open(OUT, 'w')); print(f'[{fi + 1}/{len(files)}] {f}  cells so far {len(done)}  unique pairs scored {len(cache)}  {time.time() - t0:.0f}s', flush=True)
print('DONE', len(done), 'cells', flush=True)
