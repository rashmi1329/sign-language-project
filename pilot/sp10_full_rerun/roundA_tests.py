import json, numpy as np, os
from scipy import stats
H=os.getcwd()
LANGS=['zh','uk','ru','bg','is','de','it','sv','lt','en']; IDX={l:i for i,l in enumerate(LANGS)}
CODE=dict(zip(['CSL','UKL','RSL','BQN','ICL','GSG','ISE','SWL','LLS','BFI'],LANGS))
s1=json.load(open('s1_new_from_log.json')); D=np.zeros((10,10))
for k,v in s1.items():
    a,b=k.split('-'); D[IDX[CODE[a]],IDX[CODE[b]]]=D[IDX[CODE[b]],IDX[CODE[a]]]=v[0]
res=json.load(open('e2_outputs/scratch_s0/e2_results_scratch_s0.json'))
bl=json.load(open('rtm_analysis/rtm_bleurt_cells.json'))
bl={k.split('::')[1]:v['bleurt'] for k,v in bl.items() if k.startswith('scratch_s0/')}
def grid(m):
    val=lambda k: bl[k] if m=='bleurt' else res[k]['test_'+m]
    base={k[8:]:val(k) for k in res if k.startswith('scratch_')}; G=np.full((10,10),np.nan)
    for k in res:
        if k.startswith('transfer_'):
            s,t=k[9:].split('_')
            if s!=t: G[IDX[s],IDX[t]]=val(k)-base[t]
    return G,base
def slope(x,y): x=(x-x.mean())/x.std(); return np.polyfit(x,y,1)[0]
def qap(G,keep,R=10000,seed=0):
    keep=list(keep); ii=[(i,j) for i in keep for j in keep if i!=j]; y=np.array([G[i,j] for i,j in ii]); x=np.array([D[i,j] for i,j in ii])
    obs=slope(x,y); rng=np.random.RandomState(seed); c=0
    for _ in range(R):
        p=rng.permutation(len(keep)); m={keep[a]:keep[p[a]] for a in range(len(keep))}
        c+=slope(np.array([D[m[i],m[j]] for i,j in ii]),y)<=obs
    return len(y),obs,(c+1)/(R+1)
def tadj(G,keep,R=10000,seed=0):
    keep=list(keep); ii=[(i,j) for i in keep for j in keep if i!=j]; y=np.array([G[i,j] for i,j in ii]); tg=np.array([j for _,j in ii]); ts=sorted(set(tg))
    def dm(v):
        v=v.astype(float).copy()
        for t in ts: v[tg==t]-=v[tg==t].mean()
        return v
    yw=dm(y); stat=lambda v:(lambda vw:(vw@yw)/(vw@vw))(dm(v))
    xs=np.array([D[i,j] for i,j in ii]); obs=stat(xs); rng=np.random.RandomState(seed); c=0
    for _ in range(R):
        p=rng.permutation(len(keep)); m={keep[a]:keep[p[a]] for a in range(len(keep))}; c+=stat(np.array([D[m[i],m[j]] for i,j in ii]))<=obs
    sdD=np.std([D[i,j] for i in range(10) for j in range(10) if i!=j])
    sl=[np.polyfit(xs[tg==t],y[tg==t],1)[0] for t in ts]
    return obs*sdD,(c+1)/(R+1),int(sum(s<0 for s in sl)),len(sl)
out={}
allk=list(range(10)); nobg=[i for i in allk if LANGS[i]!='bg']
for m in ['chrf','bleu','bleurt']:
    G,base=grid(m); g=G[~np.isnan(G)]
    n,sl,p=qap(G,allk); n2,sl2,p2=qap(G,nobg); a=tadj(G,allk)
    out[m]=dict(mean_base=float(np.mean(list(base.values()))),mean_gain=float(g.mean()),pos=int((g>0).sum()),pooled=(sl,p),pooled_nobg=(sl2,p2),tadj=a)
    print(m,json.dumps(out[m]))
json.dump(out,open('rtm_analysis/rtm_roundA_tests.json','w'),indent=1)
