import json,glob,itertools,numpy as np
LANGS=['zh','uk','ru','bg','is','de','it','sv','lt','en']; IDX={l:i for i,l in enumerate(LANGS)}
sim=json.load(open('similarity_rtm.json'))
def dm(key):
    D=np.zeros((10,10))
    for k,v in sim.items():
        a,b=k.split('|'); D[IDX[a],IDX[b]]=D[IDX[b],IDX[a]]=v[key]
    return D
def load(dirs,pat):
    r={}
    for d in dirs:
        for f in sorted(glob.glob(f'e2_outputs/{d}/{pat}')):
            if 'sweep' in f: continue
            r.update(json.load(open(f)))
    return r
GR={'B':load(['rtm_scratch_a','rtm_scratch_b_k'],'rtm_results_scratch_s0_g*.json'),'C0':load(['rtm2_s1_c','rtm2_s2_c'],'rtm_results_scratch_s0_g*_zscore.json'),'C1':load(['rtm2_sa_a','rtm2_sb_b_k','rtm2_sc_c'],'rtm_results_scratch_s1_g*_zscore.json')}
def gain(res,m='chrf'):
    base={k[8:]:v['test_'+m] for k,v in res.items() if k.startswith('scratch_')}; G=np.full((10,10),np.nan)
    for k,v in res.items():
        if k.startswith('transfer_'):
            s,t=k[9:].split('_')
            if s!=t: G[IDX[s],IDX[t]]=v['test_'+m]-base[t]
    return G
def slope(x,y): x=(x-x.mean())/x.std(); return np.polyfit(x,y,1)[0]
def qap(G,D,R=10000,seed=0):
    keep=list(range(10)); ii=[(i,j) for i in keep for j in keep if i!=j]; y=np.array([G[i,j] for i,j in ii]); x=np.array([D[i,j] for i,j in ii]); obs=slope(x,y); rng=np.random.RandomState(seed); c1=c2=0
    for _ in range(R):
        p=rng.permutation(10); s=slope(np.array([D[p[i],p[j]] for i,j in ii]),y); c1+=s<=obs; c2+=abs(s)>=abs(obs)
    return round(float(obs),3),round((c1+1)/(R+1),3),round((c2+1)/(R+1),3)
def s4(res):
    v={k[8:]:np.array(x['s4_mean']) for k,x in res.items() if k.startswith('scratch_') and 's4_mean' in x}; S=np.zeros((10,10))
    for a,b in itertools.combinations(LANGS,2): S[IDX[a],IDX[b]]=S[IDX[b],IDX[a]]=1-v[a]@v[b]/np.linalg.norm(v[a])/np.linalg.norm(v[b])
    return S
for g in ['B','C0','C1']:
    G=gain(GR[g])
    for dk in ['S1_body','S1_hands']: print(g,dk,'slope,p1,p2',qap(G,dm(dk)))
    try: print(g,'comparator',qap(G,s4(GR[g])))
    except Exception as e: print(g,'comparator n/a',e)
