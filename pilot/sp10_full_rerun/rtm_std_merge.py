import json,csv,collections,hashlib,glob
LANGS=['zh','uk','ru','bg','is','de','it','sv','lt','en']; OUT='rtm_clean'
P=[json.load(open(f'{OUT}/part_{l}.json')) for l in LANGS]
man={'rules':{'drop_exact_duplicate_groups':'all members (footage identical, labels differ)','drop_frames_gt':600,'thr':0.3,'normalisation':'Uni-Sign load_part_kp/crop_scale port (rtm_std_lib.py)'},'src_sha256':{p['lang']:p['src_sha256'] for p in P},'languages':{p['lang']:{k:p[k] for k in ('kept','splits','zero_scale_clips','ids')} for p in P}}
json.dump(man,open(f'{OUT}/manifest.json','w'))
D=[r for p in P for r in p['dropped']]
with open(f'{OUT}/dropped.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['lang','id','reason']); w.writerows(D)
print('dropped',len(D),dict(collections.Counter(r[2] for r in D)),'kept',sum(p['kept'] for p in P))
for p in P: print(p['lang'],p['kept'],p['splits'],'zero_scale',p['zero_scale_clips'])
