"""
sp10_fetch_rest.py -- recover the SP-10 clips that Run 2 never downloaded, extract their poses, merge them with the
poses Run 2 already has, and recompute the similarity matrix on the complete data.

Why this exists.  The Run 2 log (pilot/sp10_out/run.log) says 'downloaded 6858 of 11741 clips'.  4,883 clips (42%) were
never fetched: 478 of 1,186 sentence ids have no clip in any language.  Every later stage (similarity, the 90-pair grid,
the paper's numbers) used only the clips that arrived.  A HEAD check on 12 of the missing URLs returned HTTP 200, so the
failure looks transient.  This script re-fetches only the missing clips.

Three stages, run as two Kaggle notebooks so that each ends with a saved, checkable output:
  D1_download   (notebook 1, CPU, Internet ON)   re-fetch the missing clips, with verification, politeness and a report.
                Writes clips/, sp10_index_v2.json, fetch_report.json.
  D2_extract    (notebook 2, CPU, Internet ON)   MediaPipe poses for the NEW clips only, merged with Run 2's poses.
  D3_finish     (notebook 2, CPU)                pose audit, similarity on all train clips, counts table, sha256 manifest.

Nothing from Run 2 is modified: its dataset is read-only input and every file written here goes to a new output folder.
Everything resumes: re-running a stage skips clips and languages that already exist.

Inputs it looks for under /kaggle/input (override the root with SP10_INPUT for a test):
  notebook 1: Run 2's output   (found by its sp10_index.json)
  notebook 2: Run 2's output   (found by poses_zh.npz)  AND  notebook 1's output (found by sp10_index_v2.json)
"""
import os, sys, json, glob, time, hashlib, collections, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, ".")
# D1 needs nothing from the driver, so notebook 1 ships this file alone.  D2 and D3 import sp10_driver themselves
# (LANGS, SIGN, _worker, STAGE_C), and both read the same SP10_OUT / SP10_DATA environment variables.
OUT = os.environ.get("SP10_OUT", "/kaggle/working/sp10_out")       # all new files go here (the saved Kaggle output)
DATA = os.environ.get("SP10_DATA", "/tmp/sp10_data")
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
CLIPS = os.path.join(OUT, "clips")                # downloaded videos: clips/<split>/<id>/<lang>.mp4
INPUT = os.environ.get("SP10_INPUT", "/kaggle/input")
SMOKE = int(os.environ.get("SP10_FETCH_SMOKE", "0"))   # >0: only this many missing clips (pipeline test)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True)
    open(os.path.join(OUT, "run.log"), "a").write(line + "\n")


def find_one(name):
    """The single file called name under the input root; refuses to guess if there are zero or several."""
    hits = sorted(glob.glob(os.path.join(INPUT, "**", name), recursive=True))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {name} under {INPUT}, found {len(hits)}: {hits[:5]} -- attach the right notebook outputs")
    return hits[0]


# ============================================================================ D1: download the missing clips
def _fetch_one(row, tries=6):
    """One clip.  A clip counts as fetched only if the HTTP read is complete (matches Content-Length when the server
    sends it), is large enough, and looks like an MP4 ('ftyp' or 'moov' box in the first 4 KB).  Written to .part then renamed, so a
    killed run never leaves a half file that looks finished.  Backoff doubles per try (3, 6, 12, ... capped at 60 s).
    Returns (ok, error text).  Run 2 swallowed every exception; this keeps the reason."""
    dst = os.path.join(CLIPS, row["split"], str(row["id"]), f"{row['lang']}.mp4")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return True, None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    err = None
    for a in range(tries):
        try:
            req = urllib.request.Request(row["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                want = r.headers.get("Content-Length")
            if want and len(data) != int(want):
                raise IOError(f"short read {len(data)} of {want}")
            if len(data) < 1000 or (b"ftyp" not in data[:4096] and b"moov" not in data[:4096]):
                raise IOError(f"not an mp4 ({len(data)} bytes)")
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return True, None
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(min(60, 3 * 2 ** a))
    return False, err


def _run_pass(items, workers, tag, batch=300):
    """Fetch the items in batches.  After each batch the failure rate is logged; if more than half of a batch failed the
    server is probably throttling us, so pause 3 minutes and halve the workers (min 1) instead of hammering it."""
    errs, ok_n = {}, 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(_fetch_one, chunk))
        bad = [(r, e) for r, (o, e) in zip(chunk, res) if not o]
        ok_n += len(chunk) - len(bad)
        for r, e in bad:
            errs[r["url"]] = e
        log(f"[{tag}] {i + len(chunk)}/{len(items)} done; batch failures {len(bad)}/{len(chunk)}; workers {workers}")
        if len(chunk) >= 20 and len(bad) > len(chunk) / 2 and i + batch < len(items):
            log(f"[{tag}] more than half the batch failed -> pause 180 s, workers {workers} -> {max(1, workers // 2)}")
            time.sleep(180); workers = max(1, workers // 2)
    return ok_n, errs


def D1_download():
    old_idx = find_one("sp10_index.json")
    rows = json.load(open(old_idx))
    for r in rows:
        r["source"] = "run2" if r["downloaded"] else "pending"
    todo = [r for r in rows if not r["downloaded"]]
    if SMOKE:
        todo = todo[:SMOKE]
    log(f"index {len(rows)} clips; Run 2 had {sum(r['downloaded'] for r in rows)}; to fetch {len(todo)}")

    # three passes: 6 workers, then the failures again after 2 minutes with 2 workers, then after 5 minutes with 1
    report = {"to_fetch": len(todo), "passes": []}
    pending, final_errs = todo, {}
    for tag, workers, wait in (("pass1", 6, 0), ("pass2", 2, 120), ("pass3", 1, 300)):
        if not pending:
            break
        if wait:
            log(f"waiting {wait} s before {tag} on {len(pending)} failures"); time.sleep(wait)
        ok_n, errs = _run_pass(pending, workers, tag)
        report["passes"].append({"pass": tag, "attempted": len(pending), "fetched": ok_n, "failed": len(errs)})
        pending = [r for r in pending if r["url"] in errs]
        final_errs = errs

    # write index v2: paths are relative to this output folder so notebook 2 can find the clips wherever it is mounted
    for r in todo:
        p = os.path.join("clips", r["split"], str(r["id"]), f"{r['lang']}.mp4")
        full = os.path.join(OUT, p)
        if os.path.exists(full) and os.path.getsize(full) > 1000:
            r["downloaded"], r["source"], r["path"] = True, "new", p
        else:
            r["source"] = "missing"
    json.dump(rows, open(os.path.join(OUT, "sp10_index_v2.json"), "w"), ensure_ascii=False)

    n_new = sum(r["source"] == "new" for r in rows)
    report["fetched_new"] = n_new
    report["still_missing"] = sum(r["source"] == "missing" for r in rows)
    report["total_with_clip"] = sum(r["downloaded"] for r in rows)
    report["failure_reasons"] = dict(collections.Counter(str(e)[:60] for e in final_errs.values()))
    report["failed_urls"] = [r["url"] for r in pending]
    by = collections.Counter((r["split"], r["downloaded"]) for r in rows)
    report["by_split"] = {s: {"have": by[(s, True)], "lack": by[(s, False)]} for s in ("train", "dev", "test")}
    json.dump(report, open(os.path.join(OUT, "fetch_report.json"), "w"), indent=1)
    log(json.dumps({k: v for k, v in report.items() if k != "failed_urls"}))
    log("D1 done")


# ============================================================================ D2: poses for the new clips, merged
def D2_extract(workers=4):
    from multiprocessing import Pool
    import sp10_driver as d
    idx_path = find_one("sp10_index_v2.json")
    base = os.path.dirname(idx_path)
    old_dir = os.path.dirname(find_one("poses_zh.npz"))
    rows = json.load(open(idx_path))
    mdir = os.path.join(DATA, "models"); os.makedirs(mdir, exist_ok=True)
    for u in ["https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"]:
        dst = os.path.join(mdir, u.rsplit("/", 1)[-1])
        if not os.path.exists(dst):
            subprocess.run(["curl", "-sL", "-o", dst, u], check=True)
    failures = {}
    for lang in d.LANGS:
        merged_p = os.path.join(OUT, f"poses_{lang}.npz")
        new_p = os.path.join(OUT, f"poses_new_{lang}.npz")
        if os.path.exists(merged_p):
            continue
        items = [(f"{r['split']}/{r['id']}", os.path.join(base, r["path"])) for r in rows if r["lang"] == lang and r["source"] == "new"]
        t = time.time()
        if os.path.exists(new_p):                        # extraction finished earlier; only the merge was lost
            new = dict(np.load(new_p))
        else:
            chunks = [items[i::workers] for i in range(workers)]
            parts = [(c, mdir, new_p.replace(".npz", f".part{i}.npz")) for i, c in enumerate(chunks) if c]
            new = {}
            if parts:
                with Pool(len(parts)) as pool:
                    done = pool.map(d._worker, parts)
                for p in done:
                    z = np.load(p); new.update({k: z[k] for k in z.files}); os.remove(p)
            np.savez_compressed(new_p, **new)            # new-only copy is kept as its own output file
        old = np.load(os.path.join(old_dir, f"poses_{lang}.npz"))
        overlap = set(new) & set(old.files)
        assert not overlap, f"{lang}: {len(overlap)} keys are in both Run 2 poses and the new poses"
        res = {k: old[k] for k in old.files}; res.update(new)
        tmp = merged_p + ".tmp.npz"
        np.savez_compressed(tmp, **res); os.replace(tmp, merged_p)
        expect = sum(1 for r in rows if r["lang"] == lang and r["downloaded"])
        lost = [k for k, _ in items if k not in new]
        failures[lang] = lost
        log(f"{lang} ({d.SIGN[lang]}): old {len(old.files)} + new {len(new)} = {len(res)} poses; clips in index {expect}; "
            f"new clips without a pose {len(lost)}; {round(time.time() - t)} s")
    if failures:
        json.dump(failures, open(os.path.join(OUT, "extract_failures.json"), "w"), indent=1)
    log("D2 done")


# ============================================================================ D3: audit, similarity, counts, manifest
def D3_finish():
    import sp10_driver as d
    idx_path = find_one("sp10_index_v2.json")
    old_dir = os.path.dirname(find_one("poses_zh.npz"))
    rows = json.load(open(idx_path))
    # English references: Run 2's file, checked against the index so a mismatch cannot pass silently
    refs = json.load(open(os.path.join(old_dir, "english_refs.json")))
    from_idx = {f"{r['split']}/{r['id']}": r["text"] for r in rows if r["lang"] == "en"}
    assert refs == from_idx, "english_refs.json from Run 2 does not match the index"
    json.dump(refs, open(os.path.join(OUT, "english_refs.json"), "w"), ensure_ascii=False, indent=0)

    audit, counts = {}, {}
    for lang in d.LANGS:
        z = np.load(os.path.join(OUT, f"poses_{lang}.npz")); n = fr = hp = 0
        cs = collections.Counter(k.split("/")[0] for k in z.files)
        for k in z.files:
            s = z[k]; n += 1; fr += len(s); hp += np.isfinite(s[:, 13:, 0]).any(1).sum()
        audit[lang] = {"clips": n, "mean_frames": round(fr / max(n, 1), 1), "frames_with_a_hand": round(hp / max(fr, 1), 3)}
        counts[lang] = dict(cs)
    json.dump(audit, open(os.path.join(OUT, "pose_audit.json"), "w"), indent=1)
    json.dump(counts, open(os.path.join(OUT, "clip_counts.json"), "w"), indent=1)
    log(f"clips per language and split: {counts}")
    log(f"audit {audit}")

    d.STAGE_C()                                          # S1 DTW etc. on ALL train clips now present, before any training

    man = {}                                             # sha256 of every file we wrote (clips/ excluded; notebook 1 holds them)
    for p in sorted(glob.glob(os.path.join(OUT, "*"))):
        if os.path.isfile(p):
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            man[os.path.basename(p)] = {"bytes": os.path.getsize(p), "sha256": h}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    log("D3 done; manifest has " + str(len(man)) + " files")
