# RTMPose full extraction of SP-10 (all clips of the chosen languages) in the Uni-Sign input format.  Written 2026-10-18.
# What it stores per clip: the 48 uniformly spaced frames the experiments use (same nearest-index rule as pilot_pipeline.resample:
# idx = round(linspace(0, t-1, 48))), 133 COCO-WholeBody keypoints per frame as pixel/[W,H] (x,y in 0..1 of the picture, values outside 0..1 mean
# "predicted outside the picture", NOT clipped) and the 133 confidence scores.  No joint is masked or cleaned here; masking/normalisation is
# done by the training code (Uni-Sign rules: confidence <= 0.3 -> zero, body-box crop).  This keeps every later choice reversible.
# Parallelism: RX_WORKERS processes (default 2 per GPU), each pinned to one GPU with CUDA_VISIBLE_DEVICES and taking every W-th clip.
# Resumable: finished clips are read back from RX_OUT/parts and from RX_PRIOR (an earlier output attached as input).  Output: rtm_<lang>.npz
# (keys 'k|<split>/<id>' float16 (48,133,2) and 's|<split>/<id>' float16 (48,133)), rtm_meta.json, rtm_extract_report.json.
import os, sys, json, time, subprocess, urllib.request, collections, glob, traceback, hashlib
OUT = os.environ.get('RX_OUT', '/kaggle/working/rtm_out'); TMP = os.environ.get('RX_TMP', '/tmp/rtm_clips'); PARTS = f'{OUT}/parts'
LANGS_ALL = ['zh', 'uk', 'ru', 'bg', 'is', 'de', 'it', 'sv', 'lt', 'en']
LANGS = os.environ.get('RX_LANGS', ','.join(LANGS_ALL)).split(',')
MODE = os.environ.get('RX_MODE', 'auto')              # 'performance' | 'lightweight' | 'balanced' | 'auto' (calibrated, rule below)
DEVICE = os.environ.get('RX_DEVICE', 'cuda'); CAP_H = float(os.environ.get('RX_CAP_H', '9')); SMOKE = int(os.environ.get('RX_SMOKE', '0'))
AUTO_BUDGET_H = float(os.environ.get('RX_AUTO_H', '3.0'))   # auto rule: 'performance' if the projected whole-dataset time on the GPUs of this machine is <= this, else 'lightweight'
INSTALL = os.environ.get('RX_INSTALL', '1') == '1'; PRIOR = os.environ.get('RX_PRIOR', '')
T_FIXED = 48
os.makedirs(OUT, exist_ok=True); os.makedirs(PARTS, exist_ok=True); os.makedirs(TMP, exist_ok=True)
T0 = time.time()
def log(m, flush=True):
    line = f'[{time.time()-T0:8.1f}s] {m}'; print(line, flush=flush)
    try: open(f'{OUT}/run.log', 'a').write(line + '\n')
    except Exception: pass

def pip(*a): return subprocess.run([sys.executable, '-m', 'pip'] + list(a), capture_output=True, text=True)
def install():
    CAND = os.environ.get('RX_ORT', '1.22.0,1.21.1,1.20.1').split(',')
    NV = ['nvidia-cudnn-cu12', 'nvidia-cublas-cu12', 'nvidia-cuda-runtime-cu12', 'nvidia-cufft-cu12', 'nvidia-curand-cu12', 'nvidia-cuda-nvrtc-cu12']
    TEST = ("import onnxruntime as o\n(o.preload_dlls() if hasattr(o, 'preload_dlls') else None)\nfrom rtmlib import Wholebody\n"
            "wb = Wholebody(mode='lightweight', backend='onnxruntime', device='cuda')\nprint('PROV', wb.pose_model.session.get_providers())\n")
    pip('uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu')
    for pk in (['--no-deps', 'rtmlib'], ['opencv-python-headless', 'tqdm']):          # rtmlib lists the CPU wheel as a dependency -> clashes with onnxruntime-gpu
        r = pip('install', '-q', *pk); log(f'pip install {pk} rc={r.returncode} {r.stderr[-300:] if r.returncode else ""}')
    if DEVICE != 'cuda':
        r = pip('install', '-q', 'onnxruntime'); log(f'pip install onnxruntime rc={r.returncode}'); return
    r = pip('install', '-q', *NV); log(f'pip install nvidia cu12 libs rc={r.returncode} {r.stderr[-300:] if r.returncode else ""}')
    for v in CAND:
        pip('uninstall', '-y', 'onnxruntime-gpu'); r = pip('install', '-q', f'onnxruntime-gpu=={v}'); log(f'onnxruntime-gpu=={v}: pip rc={r.returncode}')
        if r.returncode: continue
        t = subprocess.run([sys.executable, '-c', TEST], capture_output=True, text=True)
        pl = [l for l in t.stdout.split(chr(10)) if l.startswith('PROV')]            # stdout only: stderr warnings also contain the word CUDAExecutionProvider
        hits = [l.strip()[:240] for l in (t.stdout + t.stderr).split(chr(10)) if ('PROV' in l or 'Failed to' in l or 'Require' in l) and '%|' not in l][:5]
        log(f'onnxruntime-gpu=={v}: test rc={t.returncode} :: ' + ' || '.join(hits))
        if pl and 'CUDAExecutionProvider' in pl[-1]: log(f'USING onnxruntime-gpu=={v}'); return
    msg = f'ABORT: no onnxruntime-gpu in {CAND} created a CUDAExecutionProvider'; log(msg)
    json.dump(dict(error=msg), open(f'{OUT}/rtm_extract_report.json', 'w')); raise SystemExit(msg)

def ngpu():
    try: return max(1, len([l for l in subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.split(chr(10)) if l.startswith('GPU')]))
    except Exception: return 1

# ---------------------------------------------------------------- clip index (same source and rule as sp10_driver.STAGE_A)
def build_rows():
    meta = {}
    for part in ('train', 'dev', 'test'):
        p = f'{TMP}/{part}.json'
        if not os.path.exists(p):
            try:
                data = urllib.request.urlopen(urllib.request.Request(f'https://raw.githubusercontent.com/MLSLT/SP-10/main/dataset/{part}.json', headers={'User-Agent': 'sp10-grid'}), timeout=60).read(); json.loads(data)
            except Exception:
                import base64
                r = json.load(urllib.request.urlopen(urllib.request.Request(f'https://api.github.com/repos/MLSLT/SP-10/contents/dataset/{part}.json', headers={'User-Agent': 'sp10-grid'}), timeout=60)); data = base64.b64decode(r['content'])
            open(p, 'wb').write(data)
        meta[part] = json.load(open(p))
    rows = []
    for part, items in meta.items():
        for obj in items:
            for s in obj['sign_list']:
                if s.get('video_url') and s['lang'] in LANGS: rows.append(dict(split=part, id=str(obj['id']), lang=s['lang'], url=s['video_url'], text=s.get('text', '')))
    rows.sort(key=lambda r: (LANGS.index(r['lang']), r['split'], r['id']))
    return rows
def key_of(r): return f"{r['split']}/{r['id']}"

def fetch(r, tries=6):
    dst = f"{TMP}/clips/{r['lang']}/{r['split']}_{r['id']}.mp4"
    if os.path.exists(dst) and os.path.getsize(dst) > 1000: return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True); err = None
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(r['url'], headers={'User-Agent': 'Mozilla/5.0'}), timeout=60) as resp:
                data = resp.read(); want = resp.headers.get('Content-Length')
            if want and len(data) != int(want): raise IOError(f'short read {len(data)} of {want}')
            if len(data) < 1000 or (b'ftyp' not in data[:4096] and b'moov' not in data[:4096]): raise IOError(f'not an mp4 ({len(data)} bytes)')
            open(dst + '.part', 'wb').write(data); os.replace(dst + '.part', dst); return dst
        except Exception as e: err = e; time.sleep(min(60, 3 * 2 ** a))
    raise IOError(f'download failed: {err}')

def frames_at(path):
    """All frames are decoded (seeking is unreliable on these mp4s); only the 48 resample indices go to the network.  Returns (frames dict idx->BGR, n_frames, W, H, fps)."""
    import cv2, numpy as np
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    t = len(fr)
    if t == 0: raise IOError('no frames decoded')
    idx = np.clip(np.round(np.linspace(0, t - 1, T_FIXED)).astype(int), 0, t - 1)       # identical rule to pilot_pipeline.resample
    H, W = fr[0].shape[:2]
    return idx, {int(i): fr[int(i)] for i in set(idx.tolist())}, t, W, H, fps

def run_clip(wb, path):
    import numpy as np
    idx, fr, t, W, H, fps = frames_at(path); cache = {}; npers = []; tt = time.time()
    for i in sorted(fr):
        kp, sc = wb(fr[i].astype(np.uint8)); npers.append(len(kp))
        if len(kp) == 0: cache[i] = (np.zeros((133, 2), np.float32), np.zeros(133, np.float32))
        else:
            j = int(np.argmax(sc.mean(axis=1))); cache[i] = ((kp[j] / np.array([W, H], np.float32)).astype(np.float32), sc[j].astype(np.float32))   # most confident person; coordinates / [W,H] as in Uni-Sign's demo
    dt = time.time() - tt
    K = np.stack([cache[int(i)][0] for i in idx]).astype(np.float16); S = np.stack([cache[int(i)][1] for i in idx]).astype(np.float16)
    return K, S, dict(W=W, H=H, frames=t, fps=round(float(fps), 2), nrun=len(fr), secs=round(dt, 3), persons=round(float(np.mean(npers)), 2), no_person=int(sum(1 for n in npers if n == 0)))

def load_wb(mode):
    from rtmlib import Wholebody
    import onnxruntime as ort
    if DEVICE == 'cuda' and hasattr(ort, 'preload_dlls'): ort.preload_dlls()
    wb = Wholebody(mode=mode, to_openpose=False, backend='onnxruntime', device=DEVICE)
    if DEVICE == 'cuda':
        prov = wb.pose_model.session.get_providers(); dprov = wb.det_model.session.get_providers()
        if 'CUDAExecutionProvider' not in prov or 'CUDAExecutionProvider' not in dprov:
            raise SystemExit(f'ABORT: DEVICE=cuda but sessions run on pose={prov} det={dprov}')
    import numpy as np
    wb(np.zeros((240, 320, 3), np.uint8)); return wb                                    # warm-up

def worker(w, W, gpu, rows, mode, deadline, q):
    """One process.  Pinned to one GPU before any CUDA library is loaded."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    wb = load_wb(mode); mine = rows[w::W]; K, S, META, FAIL = {}, {}, {}, {}; part_i = 0; n_done = 0; last_logged = 0; tw = time.time(); LOOK = 24
    def flush():
        nonlocal K, S, META, part_i
        if not K: return
        p = f'{PARTS}/w{w}_{part_i:04d}.npz'; np.savez(p + '.tmp.npz', **{f'k|{k}': v for k, v in K.items()}, **{f's|{k}': v for k, v in S.items()}); os.replace(p + '.tmp.npz', p)
        json.dump(META, open(f'{PARTS}/w{w}_{part_i:04d}.json', 'w')); part_i += 1; K, S, META = {}, {}, {}
    ex = ThreadPoolExecutor(2)
    futs = {i: ex.submit(fetch, mine[i]) for i in range(min(LOOK, len(mine)))}
    for n, r in enumerate(mine):
        if time.time() > deadline: log(f'w{w}: time cap reached after {n_done} clips of {len(mine)}'); break
        nxt = n + LOOK
        if nxt < len(mine): futs[nxt] = ex.submit(fetch, mine[nxt])
        try:
            path = futs.pop(n).result(); k = f"{r['lang']}|{key_of(r)}"
            Kc, Sc, m = run_clip(wb, path); K[k] = Kc; S[k] = Sc; META[k] = m; n_done += 1
            try: os.remove(path)
            except Exception: pass
        except SystemExit: raise
        except Exception as e:
            FAIL[f"{r['lang']}|{key_of(r)}"] = str(e)[:200]; json.dump(FAIL, open(f'{PARTS}/w{w}_fail.json', 'w'))
        if len(K) >= 150: flush()
        if n_done and n_done % 100 == 0 and n_done != last_logged:
            last_logged = n_done; el = time.time() - tw; log(f'w{w} gpu{gpu}: {n_done}/{len(mine)} clips, {el/n_done:.2f} s/clip, eta {(len(mine)-n_done)*el/n_done/3600:.2f} h, fails {len(FAIL)}')
    if FAIL and time.time() < deadline:                      # one retry pass after a pause (the server may have been throttling)
        log(f'w{w}: retrying {len(FAIL)} failed clips after 60 s'); time.sleep(60); byk = {f"{r['lang']}|{key_of(r)}": r for r in mine}
        for k in list(FAIL):
            if time.time() > deadline: break
            try:
                Kc, Sc, m = run_clip(wb, fetch(byk[k])); K[k] = Kc; S[k] = Sc; META[k] = m; n_done += 1; del FAIL[k]
            except SystemExit: raise
            except Exception as e: FAIL[k] = str(e)[:200]
        json.dump(FAIL, open(f'{PARTS}/w{w}_fail.json', 'w'))
    flush(); q.put((w, n_done, len(mine), len(FAIL)))

def npz_items(p, load=True):
    """Yield (kind, 'lang|split/id', array).  Parts carry the language in the key; final rtm_<lang>.npz files carry it in the file name."""
    import numpy as np, re
    z = np.load(p); m = re.match(r'rtm_([a-z]+)\.npz$', os.path.basename(p))
    for f in z.files:
        kind, rest = f.split('|', 1)
        if m: rest = f'{m.group(1)}|{rest}'
        yield kind, rest, (z[f] if load else None)

def done_keys():
    keys = set()
    for d in [PARTS] + ([PRIOR] if PRIOR else []):
        for p in glob.glob(f'{d}/**/*.npz', recursive=True):
            if p.endswith('.tmp.npz'): continue
            try:
                for kind, k, _ in npz_items(p, load=False):
                    if kind == 'k': keys.add(k)
            except Exception as e: log(f'unreadable {p}: {e}')
    return keys

def main():
    if INSTALL: install()
    import numpy as np
    G = ngpu() if DEVICE == 'cuda' else 1
    log(f'langs {LANGS}; mode {MODE}; device {DEVICE}; GPUs {G}; cap {CAP_H} h')
    try: log('nvidia-smi: ' + subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip().replace(chr(10), ' | '))
    except Exception: pass
    rows = build_rows()
    if SMOKE: rows = [r for L in LANGS for r in [x for x in rows if x['lang'] == L][:SMOKE]]
    log(f'index: {len(rows)} clips; per language {dict(collections.Counter(r["lang"] for r in rows))}')
    have = done_keys(); todo = [r for r in rows if f"{r['lang']}|{key_of(r)}" not in have]; log(f'already extracted {len(have)}; to do {len(todo)}')
    mode = MODE
    if mode == 'auto':                                    # calibration: 12 fixed clips (first dev clip of each of the 10 languages + 2 more), identical in every shard
        cal = []; bd = collections.defaultdict(list)
        for r in build_rows_all() if set(LANGS) != set(LANGS_ALL) else rows:
            if r['split'] == 'dev': bd[r['lang']].append(r)
        for L in LANGS_ALL: cal += bd[L][:1]
        cal = cal[:10]; wb = load_wb('performance'); secs = frames = 0
        for r in cal:
            _, _, m = run_clip(wb, fetch(r)); secs += m['secs']; frames += m['nrun']
        spf = secs / max(1, frames); proj = 11741 * 45 * spf / G / 3600; del wb
        mode = 'performance' if proj <= AUTO_BUDGET_H else 'lightweight'
        log(f'AUTO CALIBRATION: performance {spf*1000:.1f} ms/frame on {frames} frames -> projected whole dataset {proj:.2f} h on {G} GPU(s) (budget {AUTO_BUDGET_H} h) -> mode = {mode}')
        json.dump(dict(spf_performance=spf, projected_h=proj, chosen=mode), open(f'{OUT}/rtm_auto_mode.json', 'w'))
    json.dump(dict(mode=mode, langs=LANGS, device=DEVICE), open(f'{OUT}/rtm_mode.json', 'w'))
    if not todo: log('nothing to extract'); done(finalize(rows, mode)); return
    W = int(os.environ.get('RX_WORKERS', str(2 * G if DEVICE == 'cuda' else 2)))
    import multiprocessing as mp
    ctx = mp.get_context('spawn'); q = ctx.Queue(); deadline = T0 + CAP_H * 3600
    procs = [ctx.Process(target=worker, args=(w, W, w % G, todo, mode, deadline, q)) for w in range(W)]
    for p in procs: p.start()
    res = []
    while len(res) < W:
        try: res.append(q.get(timeout=120))
        except Exception:
            if not any(p.is_alive() for p in procs): break
    for p in procs: p.join(timeout=60)
    log(f'workers done: {res}')
    done(finalize(rows, mode))

def done(miss):
    """Exit non-zero when clips are missing, so Kaggle marks the run ERROR (outputs are still saved) instead of COMPLETE with partial data."""
    if miss and not SMOKE: raise SystemExit(f'INCOMPLETE: {miss} clips missing (time cap, failures or crashed workers); output is partial, resume with RX_PRIOR')

def build_rows_all():
    global LANGS
    keep = LANGS; LANGS = LANGS_ALL
    try: return build_rows()
    finally: LANGS = keep

def finalize(rows, mode):
    import numpy as np
    K, S = {}, {}; META = {}; FAIL = {}
    import re
    for d in [PARTS] + ([PRIOR] if PRIOR else []):
        for p in sorted(glob.glob(f'{d}/**/*.npz', recursive=True)):
            if p.endswith('.tmp.npz'): continue                       # half-written file from a killed run
            try:
                for kind, k, a in npz_items(p): (K if kind == 'k' else S)[k] = a
            except Exception as e: log(f'unreadable {p}: {e}')
        for p in glob.glob(f'{d}/**/*.json', recursive=True):
            nm = os.path.basename(p)
            try:
                if re.match(r'w\d+_\d+\.json$', nm) or nm == 'rtm_meta.json': META.update(json.load(open(p)))
                elif re.match(r'w\d+_fail\.json$', nm): FAIL.update(json.load(open(p)))
            except Exception as e: log(f'unreadable {p}: {e}')
    for k in list(FAIL):
        if k in K: del FAIL[k]                                         # failed once, extracted in a later pass or run
    rep = dict(mode=mode, langs=LANGS, clips_indexed=len(rows), clips_extracted=len([k for k in K]), failures=len(FAIL), per_language={})
    for L in LANGS:
        ks = [k for k in K if k.startswith(L + '|')]
        np.savez_compressed(f'{OUT}/rtm_{L}.npz', **{f'k|{k.split("|",1)[1]}': K[k] for k in ks}, **{f's|{k.split("|",1)[1]}': S[k] for k in ks})
        n_idx = sum(1 for r in rows if r['lang'] == L); ms = [META[k] for k in ks if k in META]
        rep['per_language'][L] = dict(indexed=n_idx, extracted=len(ks), mean_secs=round(float(np.mean([m['secs'] for m in ms])), 2) if ms else None,
                                      frames_no_person=int(sum(m['no_person'] for m in ms)), WxH=sorted({f"{m['W']}x{m['H']}" for m in ms}))
    json.dump(META, open(f'{OUT}/rtm_meta.json', 'w')); json.dump(FAIL, open(f'{OUT}/rtm_failures.json', 'w'), indent=1)
    rep['wall_h'] = round((time.time() - T0) / 3600, 2); json.dump(rep, open(f'{OUT}/rtm_extract_report.json', 'w'), indent=1); log('REPORT ' + json.dumps(rep))
    miss = rep['clips_indexed'] - rep['clips_extracted']
    if not SMOKE and miss: log(f'INCOMPLETE: {miss} clips missing (time cap or failures). Re-run with RX_PRIOR pointing at this output to resume.')
    else:
        import shutil; shutil.rmtree(PARTS, ignore_errors=True); log('complete; parts removed')
    return miss

if __name__ == '__main__':
    main()
