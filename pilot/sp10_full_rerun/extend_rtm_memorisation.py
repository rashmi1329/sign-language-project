"""Memorisation share of the translations (exploratory, CPU only). A hypothesis counts as verbatim-train when, lower-cased and whitespace-normalised, it equals one of the 830 SP-10 train reference sentences (the pooled train set; each language trains on a subset, so this slightly overstates). Baseline (per-language) cells only, so the MediaPipe row is comparable with limitations entry 49 (58%).
Run from pilot/sp10_full_rerun."""
import json, glob, collections, re, statistics as st
norm = lambda s: re.sub(r'\s+', ' ', s.strip().lower())
refs = json.load(open('kaggle_dataset_rtm_c/english_refs.json')); train = {norm(v) for k, v in refs.items() if k.startswith('train/')}
test_in_train = {k: norm(v) in train for k, v in refs.items() if k.startswith('test/')}
print('train sentences %d unique texts %d; test refs whose text also occurs in train: %d of %d (%.1f%%)' % (sum(k.startswith('train/') for k in refs), len(train), sum(test_in_train.values()), len(test_in_train), 100*sum(test_in_train.values())/len(test_in_train)))
ONLY=True
arms = collections.defaultdict(lambda: [0, 0, 0, 0])   # hyps, verbatim-in-train, hyps whose ref is in train, verbatim AND correct (== ref)
def add(arm, d):
    for k, v in d.items():
        if 'hyps' not in v: continue
        if ONLY and not (k.startswith('scratch_') or k.startswith('uni_') or k.startswith('uni_ctrl_')): continue
        for key, h in zip(v['keys'], v['hyps']):
            a = arms[arm]; a[0] += 1; a[1] += norm(h) in train; a[2] += test_in_train.get(key, False); a[3] += (norm(h) == norm(refs[key]))
for f in glob.glob('e2_outputs/rtm2_s*_c/rtm_results_scratch_s0_g*_e100_zscore.json'): add('RTM scratch v2 (100 ep, scaling)', json.load(open(f)))
for f in glob.glob('e2_outputs/rtm_scratch_*/rtm_results_scratch_s0_g*.json'): add('RTM scratch frozen (60 ep)', json.load(open(f)))
for f in glob.glob('e2_outputs/scratch_s0/e2_results_scratch_s0.json'): add('MediaPipe scratch (60 ep)', json.load(open(f)))
for s in range(4):
    for f in glob.glob(f'e2_outputs/*/rtm_results_uni_s{s}_g*.json'): add('RTM pretrained (seeds 0-3)', json.load(open(f)))
for f in glob.glob('e2_outputs/*/rtm_results_uni_ctrl_s0_g*.json'): add('RTM random-init control (seed 0)', json.load(open(f)))
print('%-36s %8s %14s %16s' % ('arm', 'hyps', 'verbatim train', 'exactly right'))
for k, (n, v, tt, ok) in arms.items(): print('%-36s %8d %13.1f%% %15.1f%%' % (k, n, 100*v/n, 100*ok/n))
