import numpy as np, json, glob, os, hashlib, csv, sys, collections
BASE='e2_outputs'; LANGS=['zh','uk','ru','bg','is','de','it','sv','lt','en']
OUT='rtm_qc'; os.makedirs(OUT,exist_ok=True)
BODY=[0,3,4,5,6,7,8,9,10]; LH=list(range(91,112)); RH=list(range(112,133)); FACE=list(range(23,40,2))+list(range(83,91))+[53]
meta={}
for d in ['rtm_extract_a','rtm_extract_b_k']:
    meta.update(json.load(open(f'{BASE}/{d}/rtm_out/rtm_meta.json')))
def path(l):
    for d in ['rtm_extract_a','rtm_extract_b_k']:
        p=f'{BASE}/{d}/rtm_out/rtm_{l}.npz'
        if os.path.exists(p): return p
rows=[]; hashes=collections.defaultdict(list); issues=collections.Counter(); structural={}
for l in LANGS:
    z=np.load(path(l)); ks=sorted(k[2:] for k in z.files if k.startswith('k|')); ss=sorted(k[2:] for k in z.files if k.startswith('s|'))
    structural[l]={'n_k':len(ks),'n_s':len(ss),'keys_match':ks==ss,'meta_match':set(f'{l}|{k}' for k in ks)==set(k for k in meta if k.startswith(l+'|'))}
    for k in ks:
        kp=z['k|'+k].astype(np.float32); sc=z['s|'+k].astype(np.float32); m=meta[f'{l}|{k}']
        W,H=m['W'],m['H']; r={'lang':l,'id':k,'split':k.split('/')[0],'frames':m['frames'],'persons':m['persons'],'no_person':m['no_person']}
        r['finite']=bool(np.isfinite(kp).all() and np.isfinite(sc).all()); r['kp_min']=float(kp.min()); r['kp_max']=float(kp.max()); r['sc_min']=float(sc.min()); r['sc_max']=float(sc.max())
        hashes[hashlib.md5(kp.tobytes()).hexdigest()].append(f'{l}|{k}')
        inp=(kp[...,0]>=0)&(kp[...,0]<=1)&(kp[...,1]>=0)&(kp[...,1]<=1)
        ok=(sc>0.3)&inp
        # shoulders (5 = left, 6 = right)
        sh=np.linalg.norm((kp[:,5]-kp[:,6])*[W,H],axis=1); r['sh_px']=float(np.median(sh))
        r['lr_ok']=float(np.mean(kp[:,5,0]>kp[:,6,0]))  # person's left shoulder appears on image right
        for nm,idx in [('body',BODY),('face',FACE),('lh',LH),('rh',RH)]:
            r['ok_'+nm]=float(ok[:,idx].mean())
        for nm,idx in [('lh',LH),('rh',RH)]:
            r['vis_'+nm]=float((ok[:,idx].sum(1)>=10).mean())   # hand usable by proposed rule
        r['wrist_out_l']=float((~inp[:,9]).mean()); r['wrist_out_r']=float((~inp[:,10]).mean())
        # motion: wrist displacement per frame / shoulder width (frames resampled to 48 so per-clip normalised)
        scale=max(r['sh_px'],1.0)
        w=(kp[:,[9,10]]*[W,H])
        d=np.linalg.norm(np.diff(w,axis=0),axis=2)/scale
        r['wrist_mot']=float(d.mean()); r['wrist_jump_max']=float(d.max())
        allp=kp[:,BODY+FACE+LH+RH]*[W,H]; r['mot_all']=float(np.linalg.norm(np.diff(allp,axis=0),axis=2).mean()/scale)
        r['static']=bool(r['mot_all']<0.002)
        r['wrist_y_mean']=float(kp[:,[9,10],1].mean())
        rows.append(r)
    print(l,len(ks),flush=True)
# flags
for r in rows:
    f=[]
    if not r['finite']: f.append('nonfinite')
    if r['kp_max']>3 or r['kp_min']<-3: f.append('kp_wild')
    if r['persons']>1.5: f.append('multi_person')
    if r['no_person']>0: f.append('no_person')
    if r['sh_px']<25: f.append('tiny_signer')
    if r['sh_px']>200: f.append('huge_signer')
    if r['lr_ok']<0.5: f.append('lr_swapped')
    if r['static']: f.append('static')
    if r['wrist_jump_max']>1.5: f.append('wrist_jump')
    if max(r['vis_lh'],r['vis_rh'])<0.2: f.append('no_hand_visible')
    if r['frames']<10: f.append('very_short')
    if r['ok_body']<0.7: f.append('body_conf_low')
    r['flags']='|'.join(f)
    for x in f: issues[x]+=1
dups=[v for v in hashes.values() if len(v)>1]
import pandas as pd
df=pd.DataFrame(rows); df.to_csv(f'{OUT}/clip_table.csv',index=False)
feat=['frames','sh_px','ok_body','ok_face','ok_lh','ok_rh','vis_lh','vis_rh','wrist_out_l','wrist_out_r','wrist_mot','mot_all','wrist_y_mean','lr_ok']
pl=df.groupby('lang')[feat].mean().loc[LANGS]
fl=df.assign(flagged=df['flags']!='').groupby('lang').flagged.mean().loc[LANGS]
pl['flagged']=fl
pl.round(3).to_csv(f'{OUT}/per_language.csv')
# eta^2 of each feature by language (framing/signer confound proxy)
eta={}
for c in feat:
    g=df.groupby('lang')[c]; gm=df[c].mean(); ssb=sum(len(x)*(x.mean()-gm)**2 for _,x in g); sst=((df[c]-gm)**2).sum(); eta[c]=float(ssb/sst)
# language-ID accuracy from the framing features (5-fold, LR + forest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
X=df[feat].values; y=df.lang.values
acc=float(cross_val_score(RandomForestClassifier(200,random_state=0,n_jobs=-1),X,y,cv=5).mean())
rep={'structural':structural,'n_clips':len(df),'issue_counts':dict(issues),'n_flagged':int((df['flags']!='').sum()),
 'dup_groups':len(dups),'dup_examples':dups[:10],'dup_cross_lang':sum(1 for g in dups if len({x.split('|')[0] for x in g})>1),
 'split_counts':df.groupby(['lang','split']).size().unstack().loc[LANGS].to_dict('index'),
 'eta2_by_language':eta,'lang_id_acc_from_framing_features':acc,'chance':0.1}
# ids across splits disjoint
bad=0
for l in LANGS:
    s=df[df.lang==l]; ids=s.id.str.split('/').str[1]; bad+=int(ids.duplicated().sum())
rep['id_collisions_across_splits']=bad
json.dump(rep,open(f'{OUT}/qc_report.json','w'),indent=1)
print(json.dumps({k:rep[k] for k in rep if k not in('structural','split_counts','dup_examples')},indent=1))
print(pl.round(3).to_string())
print(df.groupby('lang')['flags'].apply(lambda s:(s!='').sum()).loc[LANGS].to_dict())
