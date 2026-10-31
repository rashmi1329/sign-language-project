"""Test families and Holm-Bonferroni adjusted p-values for the S1-vs-gain tests (EXPLORATORY bookkeeping; limitations entry 75).
Family P (pre-specified in docs/analysis_plan_rtm.md item 5 and the v2 addendum): pooled QAP, S1_dtw, all 10 languages, chrF, one-sided negative: frozen gain, v2 gain, v2 net gain.
Family T (post hoc, target-adjusted QAP, target_adjusted.py), S1_dtw, all languages, gain: 3 metrics x 2 grids (net gain gives identical values: the target constant drops out).
Family ALL: every one-sided p-value computed in this analysis (pooled and target-adjusted; chrF, BLEU, BLEURT; frozen, v2; gain and net; S1_dtw, S1_body, S1_hands; all and no-bg) -- the 'whole garden' view.
Usage: python3 holm_family.py -> rtm_analysis/rtm_holm.json"""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); A = f'{H}/rtm_analysis/'
fz = json.load(open(A + 'rtm_analysis.json')); v2 = json.load(open(A + 'rtm_analysis_v2.json')); bl = json.load(open(A + 'rtm_analysis_bleurt.json')); ta = json.load(open(A + 'rtm_target_adjusted.json'))
def holm(items):
    order = sorted(items, key=lambda kv: kv[1]); m = len(order); out = {}; run = 0
    for r, (k, p) in enumerate(order):
        run = max(run, min(1.0, (m - r) * p)); out[k] = run
    return out
P = [('frozen chrF gain (pooled)', fz['chrf|all languages']['qap']['p_one_sided_negative']), ('v2 chrF gain (pooled)', v2['gain|chrf|all languages']['qap']['p_one_sided_negative']), ('v2 chrF net gain (pooled)', v2['NET|chrf|all languages']['qap']['p_one_sided_negative'])]
T = [(f'{g}|{m}|gain target-adjusted', ta[f'{g}|{m}|gain|S1_dtw|all']['p_one_sided']) for g in ('frozen', 'v2') for m in ('chrf', 'bleu', 'bleurt')]
ALL = []
for k, v in ta.items():
    if k.endswith('|leave_one_out') or '|net|' in k: continue
    ALL.append((k + ' (target-adjusted)', v['p_one_sided']))
for tag, d in (('v2 gain', v2), ):
    for m in ('chrf', 'bleu'):
        for kind, pre in (('gain', 'gain'), ('net', 'NET')):
            for sn in ('all languages', 'without bg'):
                ALL.append((f'v2|{m}|{kind}|S1_dtw|{sn} (pooled)', d[f'{pre}|{m}|{sn}']['qap']['p_one_sided_negative']))
for m in ('chrf', 'bleu'):
    for sn in ('all languages', 'without bg'): ALL.append((f'frozen|{m}|gain|S1_dtw|{sn} (pooled)', fz[f'{m}|{sn}']['qap']['p_one_sided_negative']))
for gname, tag in (('RTM scratch v2 (100 ep, z-score)', 'v2'), ('RTM scratch frozen (60 ep)', 'frozen')):
    for tests, kind in ((bl[gname]['gain_tests'], 'gain'), (bl[gname].get('net_tests', {}), 'net')):
        for kk, q in tests.items(): ALL.append((f'{tag}|bleurt|{kind}|{kk} (pooled)', q['p_one_sided_negative']))
res = {}
for name, fam in (('P', P), ('T', T), ('ALL', ALL)):
    adj = holm(fam); res[name] = {k: dict(raw=p, holm=adj[k]) for k, p in fam}
    print(f'\nFamily {name}: m={len(fam)} tests, smallest raw p {min(p for _, p in fam):.4f}, smallest Holm p {min(adj.values()):.4f}, Holm p<0.05: {sum(a < .05 for a in adj.values())}')
    if name != 'ALL':
        for k, p in sorted(fam, key=lambda kv: kv[1]): print(f'   {k}: raw {p:.4f}, Holm {adj[k]:.4f}')
    else:
        for k, p in sorted(fam, key=lambda kv: kv[1])[:8]: print(f'   {k}: raw {p:.4f}, Holm {adj[k]:.3f}')
json.dump(res, open(A + 'rtm_holm.json', 'w'), indent=1)
