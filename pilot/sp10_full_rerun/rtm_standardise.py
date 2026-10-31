import numpy as np, json, glob, hashlib, os, collections, csv, sys
from rtm_std_lib import unisign_parts, stack_parts
LANGS=['zh','uk','ru','bg','is','de','it','sv','lt','en']; OUT='rtm_clean'; os.makedirs(OUT,exist_ok=True)
MAXFRAMES=600
meta={}
for d in ['rtm_extract_a','rtm_extract_b_k']: meta.update(json.load(open(f'e2_outputs/{d}/rtm_out/rtm_meta.json')))
src={}
for f in glob.glob('e2_outputs/*/rtm_out/rtm_*.npz'):
    l=f.split('rtm_')[-1][:-4]
    if l in LANGS: src[l]=f
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
# 1) exact-duplicate groups per language (hash of keypoints AND scores)
dropped=[]; kept_n=collections.Counter(); manifest={'rules':{'drop_exact_duplicate_groups':'all members (labels differ, footage identical)','drop_frames_gt':MAXFRAMES,'thr':0.3},'src_sha256':{}, 'languages':{}}
RUN=[a for a in sys.argv[1:] if a in LANGS] or LANGS
for l in RUN:
    z=np.load(src[l]); ks=sorted(k[2:] for k in z.files if k.startswith('k|'))
    H=collections.defaultdict(list)
    for k in ks: H[hashlib.md5(z['k|'+k].tobytes()+z['s|'+k].tobytes()).hexdigest()].append(k)
    dupset={k for v in H.values() if len(v)>1 for k in v}
    X={};M={};bad_scale=0; keep=[]
    for k in ks:
        reason=None
        if k in dupset: reason='duplicate_footage'
        elif meta[f'{l}|{k}']['frames']>MAXFRAMES: reason='over_%d_frames'%MAXFRAMES
        if reason: dropped.append((l,k,reason)); continue
        W,Hh=meta[f'{l}|{k}']['W'],meta[f'{l}|{k}']['H']
        kp=z['k|'+k].astype(np.float32); sc=z['s|'+k].astype(np.float32)
        parts,scale=unisign_parts(kp*np.array([W,Hh],np.float32),sc)
        if scale==0: bad_scale+=1
        X[k]=stack_parts(parts).astype(np.float16)
        sel=[0]+list(range(3,11))+list(range(91,112))+list(range(112,133))+[i for i in list(range(23,40))[::2]]+list(range(83,91))+[53]
        M[k]=((kp[:,sel,0]>=0)&(kp[:,sel,0]<=1)&(kp[:,sel,1]>=0)&(kp[:,sel,1]<=1))
        keep.append(k)
    np.savez_compressed(f'{OUT}/rtm_std_{l}.npz',**{'x|'+k:X[k] for k in keep},**{'m|'+k:M[k] for k in keep})
    c=collections.Counter(k.split('/')[0] for k in keep); kept_n[l]=len(keep)
    json.dump({'lang':l,'src_sha256':sha(src[l]),'kept':len(keep),'splits':dict(c),'zero_scale_clips':bad_scale,'ids':keep,'dropped':[(a,b,r) for a,b,r in dropped if a==l]},open(f'{OUT}/part_{l}.json','w'))
    print(l,len(ks),'->',len(keep),dict(c),'zero_scale',bad_scale,flush=True)
