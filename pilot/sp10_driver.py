"""
sp10_driver.py — SP-10 multi-parallel transfer grid: data, poses, similarity (Run 2, CPU) and the grid (Runs 3-5, GPU).
Reuses pose canonicalisation / extraction / distance functions from pilot_pipeline.py.

SP-10 (Yin et al. 2022): 10 sign languages signing the same sentences from Spreadthesign; train 830 / dev 142 / test 214
sentence ids; per sentence a video per language and the spoken-language text.  English text (lang "en", British SL) is the
common target for every source language, as in the many-to-English setting of MLSLT and in Section VIII of the paper.

Stages: A fetch metadata + clips -> B poses -> C similarity (before any training) -> D grid (one seed per run) -> E analysis.
"""
import os, sys, json, time, random, collections, subprocess, math
import numpy as np

OUT = os.environ.get("SP10_OUT", "/kaggle/working/sp10_out")
DATA = os.environ.get("SP10_DATA", "/tmp/sp10_data")
IN_POSES = os.environ.get("SP10_POSES", "")      # Runs 3-5: folder holding poses.npz + similarity.json from Run 2's output
SMOKE = int(os.environ.get("SP10_SMOKE", "0"))   # >0: only this many sentence ids per split (pipeline test)
SEED = int(os.environ.get("SP10_SEED", "0"))
EPOCHS = int(os.environ.get("SP10_EPOCHS", "30"))
LANGS = ["zh", "uk", "ru", "bg", "is", "de", "it", "sv", "lt", "en"]   # SP-10 language codes (spoken-language tags of the sign languages)
SIGN = {"zh": "CSL", "uk": "UKL", "ru": "RSL", "bg": "BQN", "is": "ICL", "de": "GSG", "it": "ISE", "sv": "SWL", "lt": "LLS", "en": "BFI"}
FAMILY = {"zh": "chinese", "uk": "french", "ru": "french", "bg": "french", "is": "french", "de": "german", "it": "french", "sv": "swedish", "lt": "french", "en": "banzsl"}
FAMILY_ZHANGHU = {"uk": "A", "ru": "A", "bg": "A", "is": "A", "it": "A", "lt": "A", "en": "B", "zh": "C", "de": "D", "sv": "E"}  # Zhang, Hu, Yu & Chen 2025 grouping
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True)
    open(os.path.join(OUT, "run.log"), "a").write(line + "\n")

# ============================================================================ STAGE A: metadata and clips
def STAGE_A():
    """Fetch train/dev/test.json from the MLSLT/SP-10 GitHub repo (via the GitHub contents API) and download every clip
    from media.spreadthesign.com with 16 threads.  Writes OUT/sp10_index.json (one row per clip: split, id, lang, text, path)."""
    import base64, urllib.request
    from concurrent.futures import ThreadPoolExecutor
    meta = {}
    for part in ("train", "dev", "test"):
        p = os.path.join(DATA, f"{part}.json")
        if not os.path.exists(p):
            try:      # raw file first; the contents API (base64) as fallback
                data = urllib.request.urlopen(urllib.request.Request(f"https://raw.githubusercontent.com/MLSLT/SP-10/main/dataset/{part}.json", headers={"User-Agent": "sp10-grid"}), timeout=60).read()
                json.loads(data)
            except Exception:
                r = json.load(urllib.request.urlopen(urllib.request.Request(
                    f"https://api.github.com/repos/MLSLT/SP-10/contents/dataset/{part}.json", headers={"User-Agent": "sp10-grid"}), timeout=60))
                data = base64.b64decode(r["content"])
            open(p, "wb").write(data)
        meta[part] = json.load(open(p))
        if SMOKE:
            meta[part] = meta[part][:SMOKE]
    rows = []
    for part, items in meta.items():
        for obj in items:
            for s in obj["sign_list"]:
                if s.get("video_url"):
                    rows.append(dict(split=part, id=obj["id"], lang=s["lang"], text=s["text"], url=s["video_url"],
                                     path=os.path.join(DATA, "videos", part, str(obj["id"]), f"{s['lang']}.mp4")))
    log(f"index: {len(rows)} clips over {sum(len(v) for v in meta.values())} sentence ids; per language {dict(collections.Counter(r['lang'] for r in rows))}")
    def fetch(r):
        if os.path.exists(r["path"]) and os.path.getsize(r["path"]) > 0:
            return True
        os.makedirs(os.path.dirname(r["path"]), exist_ok=True)
        for attempt in range(3):
            try:
                req = urllib.request.Request(r["url"], headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=60).read()
                open(r["path"], "wb").write(data); return True
            except Exception as e:
                time.sleep(2 * (attempt + 1))
        return False
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(fetch, rows))
    for r, o in zip(rows, ok):
        r["downloaded"] = bool(o)
    log(f"downloaded {sum(ok)} of {len(rows)} clips")
    json.dump(rows, open(os.path.join(OUT, "sp10_index.json"), "w"), ensure_ascii=False)
    # English reference text per sentence id (the common target)
    ref = {f"{r['split']}/{r['id']}": r["text"] for r in rows if r["lang"] == "en"}
    json.dump(ref, open(os.path.join(OUT, "english_refs.json"), "w"), ensure_ascii=False, indent=0)
    log(f"english references for {len(ref)} sentence ids; stage A done")

# ============================================================================ STAGE B: poses
def _worker(args):
    import pilot_pipeline as pp
    items, mdir, out_path = args
    pose_lm, hand_lm = pp.make_landmarkers(mdir); res = {}
    for key, path in items:
        try:
            r = pp.extract_video(path, pose_lm, hand_lm, max_frames=200)
            if r is not None:
                seq, _ = pp.canonical_pose(*r); res[key] = seq
        except Exception as e:
            print("extract fail", key, e)
    np.savez_compressed(out_path, **res); return out_path

def STAGE_B(workers=4):
    """MediaPipe Tasks pose+hand landmarkers on every downloaded clip, canonical (T,55,3) form, one npz per language."""
    from multiprocessing import Pool
    mdir = os.path.join(DATA, "models"); os.makedirs(mdir, exist_ok=True)
    for u in ["https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"]:
        dst = os.path.join(mdir, u.rsplit("/", 1)[-1])
        if not os.path.exists(dst):
            subprocess.run(["curl", "-sL", "-o", dst, u], check=True)
    rows = [r for r in json.load(open(os.path.join(OUT, "sp10_index.json"))) if r["downloaded"]]
    for lang in LANGS:
        outp = os.path.join(OUT, f"poses_{lang}.npz")
        if os.path.exists(outp):
            continue
        items = [(f"{r['split']}/{r['id']}", r["path"]) for r in rows if r["lang"] == lang]
        chunks = [items[i::workers] for i in range(workers)]
        parts = [(c, mdir, outp.replace(".npz", f".part{i}.npz")) for i, c in enumerate(chunks) if c]
        t = time.time()
        with Pool(len(parts)) as pool:
            done = pool.map(_worker, parts)
        res = {}
        for p in done:
            z = np.load(p); res.update({k: z[k] for k in z.files}); os.remove(p)
        np.savez_compressed(outp, **res)
        log(f"{lang} ({SIGN[lang]}): poses for {len(res)} of {len(items)} clips in {round(time.time()-t)} s")
    audit = {}
    for lang in LANGS:
        z = np.load(os.path.join(OUT, f"poses_{lang}.npz")); n = fr = hp = 0
        for k in z.files:
            s = z[k]; n += 1; fr += len(s); hp += np.isfinite(s[:, 13:, 0]).any(1).sum()
        audit[lang] = {"clips": n, "mean_frames": round(fr / max(n, 1), 1), "frames_with_a_hand": round(hp / max(fr, 1), 3)}
    json.dump(audit, open(os.path.join(OUT, "pose_audit.json"), "w"), indent=1); log(f"audit {audit}"); log("stage B done")

# ============================================================================ STAGE C: similarity before training
def STAGE_C():
    """S1 kinematic DTW over the shared TRAIN sentences (the paper's S1), S1b handshape MMD, S1c location MMD, S2 family
    labels (two classifications).  45 unordered pairs.  Written to OUT/similarity.json before any grid run exists."""
    import pilot_pipeline as pp
    poses = {l: np.load(os.path.join(OUT, f"poses_{l}.npz")) for l in LANGS}
    train_keys = {l: {k for k in poses[l].files if k.startswith("train/")} for l in LANGS}
    hand, loc, seqs = {}, {}, {}
    for l in LANGS:
        H, Lc, S = [], [], {}
        for k in sorted(train_keys[l]):
            s = poses[l][k]; H.append(pp.hand_shape_vectors(s)); Lc.append(pp.location_vectors(s))
            s48, _ = pp.resample(s, np.ones((len(s), 3), bool)); S[k] = s48      # NaNs kept; DTW masks missing joints (audit 2026-09-22)
        hand[l] = np.concatenate(H); loc[l] = np.concatenate(Lc); loc[l] = loc[l][np.isfinite(loc[l]).all(1)]; seqs[l] = S
        log(f"{l}: {len(S)} train clips, {len(hand[l])} hand frames")
    sim = {}
    for i, a in enumerate(LANGS):
        for b in LANGS[i + 1:]:
            shared = sorted(set(seqs[a]) & set(seqs[b]))
            d = [pp.dtw_distance(seqs[a][k], seqs[b][k], np.isfinite(seqs[a][k][..., 0]), np.isfinite(seqs[b][k][..., 0])) for k in shared]
            sim[f"{a}|{b}"] = {"S1_dtw": float(np.mean(d)), "S1_n_sentences": len(d), "S1b_hand_mmd": float(pp.mmd_rbf(hand[a], hand[b])),
                               "S1c_loc_mmd": float(pp.mmd_rbf(loc[a], loc[b])), "S2_same_family_hist": int(FAMILY[a] == FAMILY[b]),
                               "S2_same_family_zhanghu": int(FAMILY_ZHANGHU[a] == FAMILY_ZHANGHU[b])}
            log(f"{SIGN[a]}-{SIGN[b]}: {sim[f'{a}|{b}']}")
    json.dump(sim, open(os.path.join(OUT, "similarity.json"), "w"), indent=1)
    log("stage C done — similarity fixed before any training")

# ============================================================================ STAGE D: the grid (one seed per run)
def _load_all():
    import pilot_pipeline as pp
    src = IN_POSES or OUT
    poses = {l: np.load(os.path.join(src, f"poses_{l}.npz")) for l in LANGS}
    refs = json.load(open(os.path.join(src, "english_refs.json")))
    data = {}
    for l in LANGS:
        for split in ("train", "dev", "test"):
            X, Y, K = [], [], []
            for k in sorted(poses[l].files):
                if not k.startswith(split + "/") or k not in refs:
                    continue
                s48, _ = pp.resample(poses[l][k], np.ones((len(poses[l][k]), 3), bool))
                X.append(pp.fill_nan_zero(s48)); Y.append(refs[k]); K.append(k)
            data[(l, split)] = (np.stack(X).astype(np.float32), Y, K)
    return data

def build_slt_model(d=256):
    """Pose encoder (4-layer transformer over per-frame joint vectors) feeding a T5-small decoder via its encoder-hidden
    interface; the decoder is text-initialised (the paper's design), the encoder is trained from scratch per cell."""
    import torch, torch.nn as nn
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    tok = T5TokenizerFast.from_pretrained("t5-small")
    t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
    dm = t5.config.d_model
    class PoseT5(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(55 * 4, dm); self.pos = nn.Parameter(torch.zeros(1, 48, dm))
            enc = nn.TransformerEncoderLayer(dm, 8, 4 * dm, 0.2, batch_first=True); self.enc = nn.TransformerEncoder(enc, 4)
            self.norm = nn.LayerNorm(dm); self.t5 = t5
        def encode(self, x):
            return self.norm(self.enc(self.inp(x.flatten(2)) + self.pos))
        def forward(self, x, labels):
            from transformers.modeling_outputs import BaseModelOutput
            return self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=self.encode(x)), labels=labels).loss
        def generate(self, x, **kw):
            from transformers.modeling_outputs import BaseModelOutput
            return self.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=self.encode(x)), max_new_tokens=32, num_beams=4, **kw)
    return PoseT5(), tok

def train_slt(model, tok, Xtr, Ytr, Xdv, Ydv, epochs, seed, device, lr=3e-4, batch=16):
    import torch, pilot_pipeline as pp
    torch.manual_seed(seed); rng = random.Random(seed); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(Xtr) / batch); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Xt = torch.tensor(Xtr); lab = tok(Ytr, padding=True, truncation=True, max_length=32, return_tensors="pt").input_ids
    lab[lab == tok.pad_token_id] = -100
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]; xb = pp.augment(Xt[idx], rng).to(device)
            loss = model(xb, lab[idx].to(device)); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            score = chrf_score(decode(model, tok, Xdv, device), Ydv)
            if score > best:
                best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); return best

def decode(model, tok, X, device, batch=32):
    import torch
    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            ids = model.generate(torch.tensor(X[i:i + batch]).to(device)); outs += tok.batch_decode(ids, skip_special_tokens=True)
    return outs

def chrf_score(hyps, refs):
    from sacrebleu.metrics import CHRF
    return CHRF().corpus_score(hyps, [refs]).score

def score_all(hyps, refs):
    """BLEU and chrF in-run (sacreBLEU).  BLEURT is scored afterwards from the saved hypotheses (STAGE_F), because the
    BLEURT package pins an older transformers than the training stage uses."""
    from sacrebleu.metrics import BLEU, CHRF
    return {"bleu": BLEU().corpus_score(hyps, [refs]).score, "chrf": CHRF().corpus_score(hyps, [refs]).score, "bleurt": float("nan")}

def save_hyps(cond, s, t, hyps, keys):
    hd = os.path.join(OUT, f"hyps_seed{SEED}"); os.makedirs(hd, exist_ok=True)
    json.dump({"condition": cond, "source": s, "target": t, "seed": SEED, "keys": keys, "hyps": hyps},
              open(os.path.join(hd, f"{cond}_{s}_{t}.json"), "w"), ensure_ascii=False)

def STAGE_D():
    """One seed: 10 from-scratch models (also the pretrained sources), then 90 fine-tunes.  Every cell appends to
    OUT/grid_seed{SEED}.csv with BLEU, chrF, BLEURT on the target's test split; resumable."""
    import torch, csv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = _load_all()
    gpath = os.path.join(OUT, f"grid_seed{SEED}.csv")
    done = set()
    if os.path.exists(gpath):
        done = {(r["condition"], r["source"], r["target"]) for r in csv.DictReader(open(gpath))}
    else:
        open(gpath, "w").write("condition,source,target,seed,dev_chrf,test_bleu,test_chrf,test_bleurt,epochs\n")
    def rec(cond, s, t, dev, sc):
        open(gpath, "a").write(f"{cond},{s},{t},{SEED},{dev:.3f},{sc['bleu']:.3f},{sc['chrf']:.3f},{sc['bleurt']:.3f},{EPOCHS}\n")
        log(f"{cond} {SIGN[s]}->{SIGN[t]}: dev chrF {dev:.2f} test BLEU {sc['bleu']:.2f} chrF {sc['chrf']:.2f} BLEURT {sc['bleurt']:.2f}")
    for l in LANGS:                                            # scratch = source model for l
        ck = os.path.join(OUT, f"src_{l}_seed{SEED}.pt")
        if ("scratch", l, l) in done and os.path.exists(ck):
            continue
        model, tok = build_slt_model(); torch.manual_seed(SEED)
        dev = train_slt(model, tok, *data[(l, "train")][:2], *data[(l, "dev")][:2], EPOCHS, SEED, device)
        torch.save(model.state_dict(), ck)
        hyps = decode(model, tok, data[(l, "test")][0], device); save_hyps("scratch", l, l, hyps, data[(l, "test")][2])
        rec("scratch", l, l, dev, score_all(hyps, data[(l, "test")][1]))
    for s in LANGS:
        for t in LANGS:
            if s == t or ("transfer", s, t) in done:
                continue
            model, tok = build_slt_model(); model.load_state_dict(torch.load(os.path.join(OUT, f"src_{s}_seed{SEED}.pt")))
            dev = train_slt(model, tok, *data[(t, "train")][:2], *data[(t, "dev")][:2], EPOCHS, SEED, device)
            hyps = decode(model, tok, data[(t, "test")][0], device); save_hyps("transfer", s, t, hyps, data[(t, "test")][2])
            rec("transfer", s, t, dev, score_all(hyps, data[(t, "test")][1]))
    log(f"stage D seed {SEED} done")

# ============================================================================ STAGE E: analysis (after all seeds)
def STAGE_E(grid_files, similarity_path):
    """Join gains (transfer minus scratch, same target and seed) with similarities; mixed model per metric."""
    import pandas as pd, statsmodels.formula.api as smf, warnings; warnings.filterwarnings("ignore")
    from scipy import stats
    g = pd.concat([pd.read_csv(f) for f in grid_files]); sim = json.load(open(similarity_path))
    bl = os.path.join(OUT, "bleurt.csv")
    if os.path.exists(bl):
        b = pd.read_csv(bl); g = g.drop(columns=["test_bleurt"]).merge(b, on=["condition", "source", "target", "seed"], how="left")
    rows = []
    for _, r in g[g.condition == "transfer"].iterrows():
        sc = g[(g.condition == "scratch") & (g.target == r.target) & (g.seed == r.seed)].iloc[0]
        key = f"{r.source}|{r.target}" if f"{r.source}|{r.target}" in sim else f"{r.target}|{r.source}"; s = sim[key]
        rows.append(dict(source=r.source, target=r.target, seed=int(r.seed), gain_bleurt=r.test_bleurt - sc.test_bleurt, gain_chrf=r.test_chrf - sc.test_chrf,
                         gain_bleu=r.test_bleu - sc.test_bleu, S1=s["S1_dtw"], S1b=s["S1b_hand_mmd"], S1c=s["S1c_loc_mmd"],
                         fam_hist=s["S2_same_family_hist"], fam_zh=s["S2_same_family_zhanghu"]))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    res = {}
    for metric in ("gain_bleurt", "gain_chrf", "gain_bleu"):
        if df[metric].isna().all():
            continue
        d = df.dropna(subset=[metric]).copy()
        for col in ("S1", "S1b", "S1c"):
            if not (d[col].std() > 0):
                continue
            d[col + "_z"] = (d[col] - d[col].mean()) / d[col].std()
            m = smf.mixedlm(f"{metric} ~ {col}_z", d, groups=d["target"], re_formula="1").fit(reml=False)
            res[f"{metric}~{col}"] = {"coef": float(m.params[col + "_z"]), "ci": [float(x) for x in m.conf_int().loc[col + "_z"]], "p": float(m.pvalues[col + "_z"])}
        for col in ("fam_hist", "fam_zh"):
            if d[col].nunique() < 2:
                continue
            m = smf.mixedlm(f"{metric} ~ {col}", d, groups=d["target"], re_formula="1").fit(reml=False)
            res[f"{metric}~{col}"] = {"coef": float(m.params[col]), "ci": [float(x) for x in m.conf_int().loc[col]], "p": float(m.pvalues[col])}
        wt = {}
        for t, sub in d.groupby(["source", "target"]).mean(numeric_only=True).reset_index().groupby("target"):
            wt[t] = float(stats.spearmanr(sub["S1"], sub[metric]).statistic)
        res[f"{metric}~S1_within_target_spearman"] = wt
    json.dump(res, open(os.path.join(OUT, "readings.json"), "w"), indent=1); log(json.dumps(res, indent=1)); log("stage E done")

# ============================================================================ STAGE F: BLEURT from saved hypotheses
def STAGE_F(hyps_dirs, refs_path, out_csv):
    """Run in an environment with transformers<5 and bleurt-pytorch (pip install transformers==4.44.2
    git+https://github.com/lucadiliello/bleurt-pytorch.git).  Scores every saved hypothesis file with BLEURT-20-D12."""
    import torch, glob, csv
    from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
    refs = json.load(open(refs_path))
    m = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20-D12").eval(); t = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20-D12")
    w = csv.writer(open(out_csv, "w")); w.writerow(["condition", "source", "target", "seed", "test_bleurt"])
    for hd in hyps_dirs:
        for f in sorted(glob.glob(os.path.join(hd, "*.json"))):
            h = json.load(open(f)); R = [refs[k] for k in h["keys"]]; sc = []
            with torch.no_grad():
                for i in range(0, len(R), 32):
                    inp = t(R[i:i + 32], h["hyps"][i:i + 32], padding="longest", truncation=True, max_length=128, return_tensors="pt")
                    sc += m(**inp).logits.flatten().tolist()
            w.writerow([h["condition"], h["source"], h["target"], h["seed"], round(100 * float(np.mean(sc)), 3)])
    log(f"stage F done -> {out_csv}")

if __name__ == "__main__":
    for s in sys.argv[1:]:
        globals()[f"STAGE_{s}"]()
