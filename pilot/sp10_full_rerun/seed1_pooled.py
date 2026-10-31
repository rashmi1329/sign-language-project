import json,glob,numpy as np
LANGS=['zh','uk','ru','bg','is','de','it','sv','lt','en']; IDX={l:i for i,l in enumerate(LANGS)}
sim=json.load(open('similarity_rtm.json')); D=np.zeros((10,10))
for k,v in sim.items():
    a,b=k.split('|'); D[IDX[a],IDX[b]]=D[IDX[b],IDX[a]]=v['S1_dtw']
res={}
for d in ['rtm2_sa_a','rtm2_sb_b_k','rtm2_sc_c']:
    for f in sorted(glob.glob(f'e2_outputs/{d}/rtm_results_scratch_s1_g*_zscore.json')): res.update(json.load(open(f)))
print(len(res))
def grid(m):
    base={k[8:]:v['test_'+m] for k,v in res.items() if k.startswith('scratch_')}; G=np.full((10,10),np.nan); N=G.copy()
    for k,v in res.items():
        if k.startswith('transfer_'):
            s,t=k[9:].split('_')
            if s!=t:
                G[IDX[s],IDX[t]]=v['test_'+m]-base[t]; N[IDX[s],IDX[t]]=v['test_'+m]-res[f'transfer_{t}_{t}']['test_'+m]
    return G,N
def slope(x,y): x=(x-x.mean())/x.std(); return np.polyfit(x,y,1)[0]
def qap(G,keep,R=10000,seed=0):
    keep=list(keep); ii=[(i,j) for i in keep for j in keep if i!=j]; y=np.array([G[i,j] for i,j in ii]); x=np.array([D[i,j] for i,j in ii]); obs=slope(x,y); rng=np.random.RandomState(seed); c=0
    for _ in range(R):
        p=rng.permutation(len(keep)); m={keep[a]:keep[p[a]] for a in range(len(keep))}; c+=slope(np.array([D[m[i],m[j]] for i,j in ii]),y)<=obs
    return round(float(obs),3),round((c+1)/(R+1),3)
allk=list(range(10)); nobg=[i for i in allk if LANGS[i]!='bg']
for m in ['chrf','bleu']:
    G,N=grid(m); print(m,'gain all',qap(G,allk),'nobg',qap(G,nobg),'| net all',qap(N,allk),'nobg',qap(N,nobg))
