import numpy as np, json, ctypes, subprocess, hashlib, itertools, sys, time
from rtm_data import LANGS, SL, load_lang
subprocess.check_call(['gcc','-O3','-shared','-fPIC','-o','/tmp/rtmdtwm.so','rtm_dtw_masked.c','-lm'])
lib = ctypes.CDLL('/tmp/rtmdtwm.so'); lib.dtw_masked.restype = ctypes.c_double
lib.dtw_masked.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
def dtw(a, ma, b, mb):
    return lib.dtw_masked(a.ctypes.data, ma.ctypes.data, b.ctypes.data, mb.ctypes.data, a.shape[0], b.shape[0], a.shape[1])
SEL = {'S1_dtw': np.arange(69), 'S1_body': np.arange(0, 9), 'S1_hands': np.arange(9, 51)}
def prep(d, sel):
    return {k: (np.ascontiguousarray(x[:, sel, :2]), np.ascontiguousarray(m[:, sel].astype(np.uint8))) for k, (x, m) in d.items()}
if __name__ == '__main__':
    root = 'rtm_clean'; t0 = time.time()
    D = {l: load_lang(root, l, 'train') for l in LANGS}
    P = {name: {l: prep(D[l], sel) for l in LANGS} for name, sel in SEL.items()}
    old = json.load(open('full_poses/sp10_out/similarity.json')); out = {}
    for a, b in itertools.combinations(LANGS, 2):
        ids = sorted(set(D[a]) & set(D[b])); r = {'S1_n_sentences': len(ids)}
        for name in SEL:
            r[name] = float(np.mean([dtw(*P[name][a][i], *P[name][b][i]) for i in ids]))
        o = old[f'{a}|{b}']; r['S2_same_family_hist'] = o['S2_same_family_hist']; r['S2_same_family_zhanghu'] = o['S2_same_family_zhanghu']
        out[f'{a}|{b}'] = r; print(a, b, len(ids), round(r['S1_dtw'], 4), round(time.time() - t0), flush=True)
    json.dump(out, open('similarity_rtm.json', 'w'), indent=1)
    print('sha256', hashlib.sha256(open('similarity_rtm.json','rb').read()).hexdigest())
