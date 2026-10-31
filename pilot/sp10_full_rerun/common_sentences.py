"""Is the target effect test-sentence composition? (exploratory, limitations entry 78). Rescore every v2 and frozen scratch cell on the sentences that ALL 10 languages have in their test set, and
split the sentence-level chrF into sentence / language / residual variance. Usage: python3 common_sentences.py -> rtm_analysis/rtm_common_sentences.json"""
import json, glob, os, numpy as np
from scipy import stats
from sacrebleu.metrics import CHRF
H = os.path.dirname(os.path.abspath(__file__)); L = ['zh','uk','ru','bg','is','de','it','sv','lt','en']
refs = json.load(open(f'{H}/kaggle_dataset_rtm_c/english_refs.json')); ch = CHRF()
def load(pat):
    r = {}
    for f in sorted(glob.glob(f'{H}/e2_outputs/{pat}')): r.update(json.load(open(f)))
    return r
grids = {'v2 (100 ep)': load('rtm2_s[12]_c/rtm_results_scratch_s0_g*_e100_*.json'), 'frozen (60 ep)': load('rtm_scratch_*/rtm_results_scratch_s0_g*.json')}
OUT = {}
for name, res in grids.items():
    keys = {l: set(res[f'scratch_{l}']['keys']) for l in L}; C = sorted(set.intersection(*keys.values())); print(f'\n== {name}: {len(C)} sentences are in all 10 test sets (per-language test sizes {[len(keys[l]) for l in L]})')
    Cset = set(C); sc = {}   # sentence-level chrF per cell
    def sent(cell): v = res[cell]; return {k: ch.sentence_score(h, [refs[k]]).score for k, h in zip(v['keys'], v['hyps']) if k in Cset}
    def corpus(cell): v = res[cell]; kk = [(k, h) for k, h in zip(v['keys'], v['hyps']) if k in Cset]; return CHRF().corpus_score([h for _, h in kk], [[refs[k] for k, _ in kk]]).score
    full = {t: np.mean([res[f'transfer_{s}_{t}']['test_chrf'] for s in L if s != t and f'transfer_{s}_{t}' in res]) for t in L}
    com = {t: np.mean([corpus(f'transfer_{s}_{t}') for s in L if s != t and f'transfer_{s}_{t}' in res]) for t in L}
    rho, p = stats.spearmanr([full[t] for t in L], [com[t] for t in L]); print(f' target mean transfer chrF, own test set -> common sentences (Spearman of the two rankings {rho:+.2f}, p {p:.4f}):')
    for t in L: print(f'   {t}: {full[t]:6.2f} -> {com[t]:6.2f}')
    print(f' spread of target means (max-min): own test set {max(full.values()) - min(full.values()):.2f}, common sentences {max(com.values()) - min(com.values()):.2f}; SD {np.std(list(full.values())):.2f} -> {np.std(list(com.values())):.2f}')
    S = np.zeros((10, len(C)))
    for i, t in enumerate(L):
        m = [sent(f'transfer_{s}_{t}') for s in L if s != t and f'transfer_{s}_{t}' in res]; S[i] = [np.mean([d[k] for d in m]) for k in C]
    g = S.mean(); ss_tot = ((S - g) ** 2).sum(); ss_lang = len(C) * ((S.mean(1) - g) ** 2).sum(); ss_sent = 10 * ((S.mean(0) - g) ** 2).sum(); ss_res = ss_tot - ss_lang - ss_sent
    print(f' sentence-level chrF (mean over the 9 sources), 10 languages x {len(C)} sentences: variance share  language {ss_lang / ss_tot:.3f}, sentence {ss_sent / ss_tot:.3f}, language x sentence residual {ss_res / ss_tot:.3f}')
    OUT[name] = dict(n_common=len(C), full=full, common=com, spearman=[rho, p], share=dict(language=ss_lang / ss_tot, sentence=ss_sent / ss_tot, residual=ss_res / ss_tot))
json.dump(OUT, open(f'{H}/rtm_analysis/rtm_common_sentences.json', 'w'), indent=1)
