"""What goes with the 'target effect'? (exploratory, limitations entry 76). Per target language (v2 scratch grid, seed 0): mean transfer chrF over the 9 sources, mean gain, baseline chrF, vs
 test-set properties (size; chrF/BLEURT of ONE fixed sentence on that language's test set = how generic-friendly the test set is; oracle-retrieval chrF), data properties (share of test frames with a hand observed, RTMPose),
 and model properties (baseline train loss, share of baseline outputs that are verbatim train sentences).
Spearman across the 10 languages (n=10: low power, no correction, descriptive). Usage: python3 target_difficulty.py -> rtm_analysis/rtm_target_difficulty.json"""
import json, glob, os, numpy as np, importlib.util
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); L = ['zh','uk','ru','bg','is','de','it','sv','lt','en']
spec = importlib.util.spec_from_file_location('rtm_data', f'{H}/rtm_data.py'); rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
v2 = {}
for f in sorted(glob.glob(f'{H}/e2_outputs/rtm2_s[12]_c/rtm_results_scratch_s0_g*_e100_*.json')): v2.update(json.load(open(f)))
nn = json.load(open(f'{H}/rtm_analysis/rtm_nn_baseline.json'))['langs']; nb = json.load(open(f'{H}/rtm_analysis/rtm_nn_bleurt.json')); refs = json.load(open(f'{H}/kaggle_dataset_rtm_c/english_refs.json'))
trainset = {v.strip().lower() for k, v in refs.items() if k.startswith('train/')}
T = {}
for t in L:
    tr = [v2[f'transfer_{s}_{t}']['test_chrf'] for s in L if s not in (t,)]; base = v2[f'scratch_{t}']
    z = rd.load_lang(f'{H}/rtm_clean', t, split='test'); hand = np.mean([np.mean(m[:, 9:51].reshape(m.shape[0], 2, 21).any(2).any(1)) for x, m in z.values()])
    hy = base['hyps']
    T[t] = dict(transfer_chrf=float(np.mean(tr)), gain=float(np.mean(tr) - base['test_chrf']), baseline_chrf=base['test_chrf'], n_test=base['n_test'], const_chrf=nn[t]['const']['chrf'], const_bleurt=nb[t]['const'], oracle_chrf=nn[t]['oracle']['chrf'],
                nn_pooled_chrf=nn[t]['nn_pooled']['chrf'], hand_frames=float(hand), train_loss=base['train_loss'], verbatim_train=float(np.mean([h.strip().lower() in trainset for h in hy])), distinct=base.get('distinct'))
print(f'{"lang":4s} {"trans":>6s} {"gain":>6s} {"base":>6s} {"n":>4s} {"const":>6s} {"orac":>6s} {"hand":>5s} {"loss":>6s} {"verb":>5s}')
for t in L: r = T[t]; print(f'{t:4s} {r["transfer_chrf"]:6.2f} {r["gain"]:+6.2f} {r["baseline_chrf"]:6.2f} {r["n_test"]:4d} {r["const_chrf"]:6.2f} {r["oracle_chrf"]:6.2f} {r["hand_frames"]:5.2f} {r["train_loss"]:6.3f} {r["verbatim_train"]:5.2f}')
OUT = {'per_language': T, 'spearman': {}}
for y in ('transfer_chrf', 'gain'):
    print(f'\nSpearman of target mean {y} with:')
    for x in ('baseline_chrf', 'n_test', 'const_chrf', 'const_bleurt', 'oracle_chrf', 'nn_pooled_chrf', 'hand_frames', 'train_loss', 'verbatim_train'):
        a = [T[t][y] for t in L]; b = [T[t][x] for t in L]; r, p = stats.spearmanr(a, b); OUT['spearman'][f'{y}~{x}'] = (float(r), float(p)); print(f'  {x:15s} rho {r:+.2f}  p {p:.3f}')
a = np.array([T[t]['transfer_chrf'] for t in L]); b = np.array([T[t]['const_chrf'] for t in L]); print(f'\nR2 of target mean transfer chrF on const-sentence chrF of the test set alone (linear): {np.corrcoef(a, b)[0, 1] ** 2:.2f}')
json.dump(OUT, open(f'{H}/rtm_analysis/rtm_target_difficulty.json', 'w'), indent=1)
