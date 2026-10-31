"""
pilot_pipeline.py — matched-volume cross-lingual ISLR transfer pilot.

Every function here is called from the notebook in the order of the stages in
docs/pilot_design.md.  Nothing in this file reads a result before it is produced;
similarity (Stage 3) is computed and saved before any model is trained (Stage 4).

Pose convention used everywhere (after `canonical_pose`):
  array of shape (T, 55, 3): 13 upper-body pose joints + 21 joints of hand A + 21 joints of hand B,
  (x, y, z) in a body-centred frame: origin = midpoint of the shoulders, unit = shoulder width.
  Hand A / hand B are SPATIAL slots (A = the hand with the smaller mean image-x over its detected
  frames), NOT anatomical left/right: the extractors' handedness labels are discarded so that every
  dataset (Tasks, legacy Holistic, unknown hand order) is treated identically.  Missing joints are NaN.
  Per-frame mask (T, 3) says whether pose / hand A / hand B were detected.  [audit 2026-09-22: naming
  aligned with the logic; no change to the representation]
"""
import os, json, math, random, zipfile, tarfile, io, time
import numpy as np

# ----------------------------------------------------------------------------- constants
UPPER_POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]   # MediaPipe 33-joint indices
T_FIXED = 48            # frames per clip after resampling
N_JOINTS = 13 + 21 + 21
LANGS = ["asl", "tsl", "rsl", "lsfb"]     # order fixed
# Language-family labels (D4).  Two classifications are carried, as the paper does for its own readings R1/R2,
# because the field's sources disagree on Russian SL: the historical classification (Wittmann 1991 as summarised by
# Reagan 2021) places RSL in the French-origin family, while Power, Grimm & List (2020) recover a separate Russian
# group.  ASL and LSFB (a daughter of LSF) are French-lineage under both; Turkish SL is unaffiliated under both.
# With four languages and six unordered pairs, D4 is DESCRIPTIVE only (see fit_readings).  [audit 2026-09-22]
FAMILY = {"asl": "french", "tsl": "turkish", "rsl": "russian", "lsfb": "french"}            # Power et al. 2020 grouping (RSL separate)
FAMILY_HIST = {"asl": "french", "tsl": "turkish", "rsl": "french", "lsfb": "french"}        # historical classification (RSL French-origin)

# ----------------------------------------------------------------------------- Stage 2: pose canonicalisation
def canonical_pose(pose33, lh21, rh21):
    """pose33: (T,33,3) MediaPipe normalized xyz (NaN where absent); lh21/rh21: the two hand arrays (T,21,3) in
    whatever order the extractor gave them.  Returns (T,55,3) body-centred coordinates with the hands re-ordered
    into spatial slots A (columns 13:34) and B (columns 34:55), and a (T,3) detection mask [pose, hand A, hand B]."""
    T = pose33.shape[0]
    ls, rs = pose33[:, 11, :2], pose33[:, 12, :2]                  # shoulders
    centre = np.nanmean(np.stack([ls, rs], 0), 0)                   # (T,2)
    width = np.linalg.norm(ls - rs, axis=1)                         # (T,)
    width = np.where(np.isfinite(width) & (width > 1e-3), width, np.nan)
    # fill centre/width over frames where the pose was missing
    for arr in (centre.T, width[None, :]):
        for row in arr:
            idx = np.where(np.isfinite(row))[0]
            if len(idx) == 0:
                row[:] = 0.0 if arr is centre.T else 1.0
            else:
                row[:] = np.interp(np.arange(T), idx, row[idx])
    # Hand slots are ordered by position, not by the extractor's handedness label, so that every dataset
    # (Tasks, legacy Holistic, unknown hand order) is treated identically: slot A = hand with the smaller
    # mean image-x over the frames in which it is detected; slot B = the other.
    mx = [np.nanmean(h[:, :, 0]) if np.isfinite(h[:, 0, 0]).any() else np.inf for h in (lh21, rh21)]
    hand_a, hand_b = (rh21, lh21) if mx[1] < mx[0] else (lh21, rh21)
    out = np.concatenate([pose33[:, UPPER_POSE_IDX], hand_a, hand_b], axis=1).astype(np.float32)  # (T,55,3)
    out[:, :, 0] = (out[:, :, 0] - centre[:, 0:1]) / width[:, None]
    out[:, :, 1] = (out[:, :, 1] - centre[:, 1:2]) / width[:, None]
    out[:, :, 2] = out[:, :, 2] / width[:, None]
    mask = np.stack([np.isfinite(pose33[:, 0, 0]), np.isfinite(hand_a[:, 0, 0]), np.isfinite(hand_b[:, 0, 0])], 1)
    return out, mask

def resample(seq, mask, T=T_FIXED):
    """Uniformly resample a (t,55,3) clip to T frames (nearest index); keeps NaNs."""
    t = seq.shape[0]
    idx = np.clip(np.round(np.linspace(0, t - 1, T)).astype(int), 0, t - 1)
    return seq[idx], mask[idx]

def fill_nan_zero(seq):
    """Model input: NaN -> 0 and an extra channel flag per joint saying 'observed'."""
    obs = np.isfinite(seq[..., 0]).astype(np.float32)[..., None]     # (T,55,1): per-joint observation flag, same joint order
    return np.concatenate([np.nan_to_num(seq, nan=0.0), obs], -1)    # (T,55,4) -> flattened to 220 inputs per frame in the model

# ----------------------------------------------------------------------------- Stage 2: extraction with MediaPipe Tasks
def make_landmarkers(model_dir):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=os.path.join(model_dir, "pose_landmarker_full.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=os.path.join(model_dir, "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    return pose, hand

_TS = {"base": 0}      # VIDEO-mode landmarkers need strictly increasing timestamps across clips in one process

def extract_video(path, pose_lm, hand_lm, max_frames=160):
    """Run pose + hand landmarkers on every frame. Returns pose33 (T,33,3), lh (T,21,3), rh (T,21,3) with NaNs."""
    import cv2, mediapipe as mp
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not (1.0 <= fps <= 120.0):
        fps = 25.0
    P, L, R = [], [], []
    i = 0
    base = _TS["base"]
    while True:
        ok, fr = cap.read()
        if not ok or i >= max_frames:
            break
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        ts = base + int((i + 1) * 1000 / fps)
        pr = pose_lm.detect_for_video(img, ts)
        hr = hand_lm.detect_for_video(img, ts)
        p = np.full((33, 3), np.nan, np.float32)
        if pr.pose_landmarks:
            p = np.array([[l.x, l.y, l.z] for l in pr.pose_landmarks[0]], np.float32)
        lh = np.full((21, 3), np.nan, np.float32); rh = np.full((21, 3), np.nan, np.float32)
        for lm, hd in zip(hr.hand_landmarks, hr.handedness):
            arr = np.array([[l.x, l.y, l.z] for l in lm], np.float32)
            if hd[0].category_name == "Left":
                lh = arr
            else:
                rh = arr
        P.append(p); L.append(lh); R.append(rh); i += 1
    cap.release()
    _TS["base"] = base + int((i + 2) * 1000 / fps) + 1000
    if not P:
        return None
    return np.stack(P), np.stack(L), np.stack(R)

# ----------------------------------------------------------------------------- Stage 1: matched subsampling
def matched_subset(records, n_signs, k_per_sign, seed=0):
    """records: list of dict(clip_id, gloss, signer, ...).  Keep the n_signs glosses with the most clips,
    cap each at k_per_sign clips (random, fixed seed).  Returns (subset, k_used)."""
    rng = random.Random(seed)
    by = {}
    for r in records:
        by.setdefault(r["gloss"], []).append(r)
    top = sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:n_signs]
    k_used = min(k_per_sign, min(len(v) for _, v in top))
    out = []
    for g, v in top:
        v = sorted(v, key=lambda r: r["clip_id"]); rng.shuffle(v)
        out.extend(v[:k_used])
    return out, k_used

def matched_subset_shared(records_by_lang, concept_of, n_signs, k_per_sign, seed=0):
    """OPTIONAL alternative to matched_subset (audit 2026-09-22, issue 1).  Selects the SAME concepts in every
    language: concept_of[lang](gloss) -> English concept string; keeps the n_signs concepts that have at least
    k_per_sign clips in EVERY language (ranked by their minimum count across languages), then caps every language at
    exactly K clips per concept with a fixed seed, so matched volume is preserved (K = min over languages).  Returns
    ({lang: subset}, K, concepts).  Not the default: the pre-registered design (pilot_design.md, Section 3) selects
    the 100 most frequent signs per language independently, because the four vocabularies overlap little (AUTSL has
    226 signs) and the classifier head is replaced at transfer anyway; enable with PILOT_SHARED_CONCEPTS=1 in the
    driver to run the shared-vocabulary variant instead, and record which variant was run."""
    rng = random.Random(seed)
    by = {}
    for lang, recs in records_by_lang.items():
        by[lang] = {}
        for r in recs:
            c = concept_of[lang](r["gloss"])
            if c:
                by[lang].setdefault(c, []).append(r)
    shared = set.intersection(*[set(v) for v in by.values()])
    eligible = sorted([c for c in shared if min(len(by[l][c]) for l in by) >= k_per_sign],
                      key=lambda c: (-min(len(by[l][c]) for l in by), c))[:n_signs]
    if not eligible:
        raise ValueError("no concept is shared by all languages with enough clips; use matched_subset instead")
    K = min(len(by[l][c]) for l in by for c in eligible)
    K = min(K, k_per_sign)
    out = {}
    for l in by:
        out[l] = []
        for c in eligible:
            v = sorted(by[l][c], key=lambda r: r["clip_id"]); rng.shuffle(v)
            for r in v[:K]:
                r = dict(r); r["concept"] = c; out[l].append(r)
    return out, K, eligible

def split_by_signer(subset, frac=(0.7, 0.1, 0.2), seed=0):
    """Assign whole signers to train/dev/test so no signer appears in two splits, targeting the fractions.
    Signs with no test clip after the split are still trained on; evaluation is per clip."""
    rng = random.Random(seed)
    signers = sorted({r["signer"] for r in subset})
    rng.shuffle(signers)
    counts = {s: 0 for s in signers}
    for r in subset:
        counts[r["signer"]] += 1
    total = len(subset); assign = {}
    order = sorted(signers, key=lambda s: -counts[s])
    # greedy: fill test, then dev, rest train (largest signers go to train so every sign keeps training data)
    test_target, dev_target = frac[2] * total, frac[1] * total
    test, dev = 0, 0
    for s in reversed(order):                     # smallest signers first into test/dev
        if test < test_target:
            assign[s] = "test"; test += counts[s]
        elif dev < dev_target:
            assign[s] = "dev"; dev += counts[s]
        else:
            assign[s] = "train"
    for r in subset:
        r["split"] = assign[r["signer"]]
    return subset

# ----------------------------------------------------------------------------- Stage 3: similarity, computed BEFORE training
def hand_shape_vectors(seq):
    """Per frame, per detected hand (slots A and B pooled; slot identity is irrelevant here): 21 joints relative
    to the wrist, scale-normalised -> 63-d vector.  Only detected hands contribute; missing hands are skipped,
    not zero-filled, so detection rate does not enter the hand-shape distribution."""
    vs = []
    for hs in (seq[:, 13:34], seq[:, 34:55]):
        ok = np.isfinite(hs[:, 0, 0])
        if ok.sum() == 0:
            continue
        h = hs[ok] - hs[ok][:, :1]
        scale = np.linalg.norm(h[:, 9], axis=1, keepdims=True) + 1e-6     # wrist -> middle MCP
        vs.append((h / scale[:, :, None]).reshape(len(h), -1))
    return np.concatenate(vs, 0) if vs else np.zeros((0, 63))

def location_vectors(seq):
    """Per frame: wrist positions of hand A and hand B in the body frame -> 4-d vector; NaN where absent.
    The callers keep only frames in which BOTH wrists are observed (`isfinite(...).all(1)`), so D2 describes the
    joint hand-location distribution of two-handed frames; one-handed frames do not enter D2.  Known limitation,
    recorded 2026-09-22; D1 is the primary measure."""
    return np.concatenate([seq[:, 13 + 0, :2], seq[:, 34 + 0, :2]], 1)

def mmd_rbf(X, Y, n_max=3000, seed=0):
    """Estimator (documented after the 2026-09-22 audit): the BIASED V-statistic of the SQUARED maximum mean
    discrepancy, MMD^2_b = mean K(x,x') + mean K(y,y') - 2 mean K(x,y), with diagonal terms included, using a
    Gaussian kernel K(u,v) = exp(-||u-v||^2 / h) whose bandwidth h is the median of the strictly positive squared
    pairwise distances over the pooled sample (median heuristic).  Each cloud is subsampled to at most n_max points
    with a fixed seed, so the value is deterministic.  The bias is O(1/n + 1/m), identical for every language pair
    at equal n, m, so it does not affect rankings.  Returns 0.0 when every pairwise distance is zero (identical
    degenerate clouds) instead of NaN.  Memory: two (n+m)^2 float64 matrices (d2 and K) plus the Gram product, ~0.9 GB peak at n_max=3000."""
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), min(n_max, len(X)), replace=False)]
    Y = Y[rng.choice(len(Y), min(n_max, len(Y)), replace=False)]
    Z = np.concatenate([X, Y]).astype(np.float64)
    sq = (Z ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)          # (n+m)^2 matrix, no (n+m)^2 x D tensor
    pos = d2[d2 > 0]
    if pos.size == 0:                                                      # all points coincide: distributions identical
        return 0.0
    bw = float(np.median(pos)); K = np.exp(-d2 / bw)
    n, m = len(X), len(Y)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())

def dtw_distance(a, b, ma=None, mb=None):
    """Dynamic time warping between two sequences a (T1, J, 3) and b (T2, J, 3) with the classic recurrence
    C[i,j] = cost(i,j) + min(C[i-1,j], C[i,j-1], C[i-1,j-1]), C[0,0] = 0, borders = inf, and the accumulated cost
    normalised by T1 + T2 (an upper bound on the warping-path length; with both clips resampled to T_FIXED this is a
    constant 2*T_FIXED, so it only fixes the scale).  Design choice, kept.
    Frame cost (changed in the 2026-09-22 audit): joints missing in EITHER frame are excluded and the squared
    distance over the observed joints is rescaled by J / n_observed before the square root, so that a dataset with
    a lower hand-detection rate is not pushed away from every other language merely by its zeros.  Before the audit
    the callers passed zero-filled (T, J*3) arrays, which made missingness part of the distance.  ma, mb: (T, J)
    boolean observation masks; when omitted, all joints count.  Frames with no jointly observed joint cost 0."""
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    if a.ndim == 2:                                                        # legacy flat input (T, J*3)
        a = a.reshape(len(a), -1, 3); b = b.reshape(len(b), -1, 3)
    T1, T2, J = len(a), len(b), a.shape[1]
    ma = np.ones((T1, J), bool) if ma is None else ma
    mb = np.ones((T2, J), bool) if mb is None else mb
    both = ma[:, None, :] & mb[None, :, :]                                  # (T1, T2, J)
    diff2 = ((np.nan_to_num(a)[:, None] - np.nan_to_num(b)[None]) ** 2).sum(-1) * both
    nobs = both.sum(-1)
    D = np.sqrt(diff2.sum(-1) * (J / np.maximum(nobs, 1)))
    D[nobs == 0] = 0.0
    C = np.full((T1 + 1, T2 + 1), np.inf); C[0, 0] = 0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    return float(C[T1, T2] / (T1 + T2))

def concept_matched_dtw(clips_a, clips_b, concepts, max_per_concept=5):
    """clips_x: dict concept -> list of (T,55,3) clips already resampled to T_FIXED (NaN where missing).  For each
    shared concept: mean masked DTW over the cross product of the first max_per_concept clips of each side (the
    lists were built in the fixed-seed subset order, so the cap is deterministic); returns the mean over concepts
    and the number of concepts used.  Masks come from the NaN pattern of each clip (audit 2026-09-22)."""
    vals = []
    for c in concepts:
        A = clips_a.get(c, [])[:max_per_concept]; B = clips_b.get(c, [])[:max_per_concept]
        if A and B:
            vals.append(np.mean([dtw_distance(a, b, np.isfinite(a[..., 0]), np.isfinite(b[..., 0])) for a in A for b in B]))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)

# ----------------------------------------------------------------------------- Stage 4: model and training
def build_model(n_classes, d=128, layers=4, heads=4, dropout=0.2):
    import torch, torch.nn as nn
    class PoseTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(N_JOINTS * 4, d)
            self.pos = nn.Parameter(torch.zeros(1, T_FIXED, d))
            enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, n_classes)
        def features(self, x):                     # x: (B,T,55,4)
            h = self.inp(x.flatten(2)) + self.pos
            return self.norm(self.enc(h).mean(1))  # (B,d)
        def forward(self, x):
            return self.head(self.features(x))
    return PoseTransformer()

def augment(x, rng):
    """Train-time only: small random scale/shift/rotation in the image plane, and time jitter."""
    import torch
    B = x.shape[0]
    s = torch.empty(B, 1, 1, 1).uniform_(0.85, 1.15)
    sh = torch.empty(B, 1, 1, 2).uniform_(-0.1, 0.1)
    th = torch.empty(B).uniform_(-0.2, 0.2)
    c, sn = torch.cos(th), torch.sin(th)
    xy = x[..., :2] * s
    xr = torch.stack([c[:, None, None] * xy[..., 0] - sn[:, None, None] * xy[..., 1],
                      sn[:, None, None] * xy[..., 0] + c[:, None, None] * xy[..., 1]], -1) + sh
    x = torch.cat([xr, x[..., 2:]], -1)
    # Audit 2026-09-22: the line here used to be dead code (`... if False else x`).  Its intent was correct: the
    # shift/rotation above moves zero-filled MISSING joints away from zero, which would give unobserved joints a
    # spurious coordinate signal.  Re-zero the xyz channels of unobserved joints; the observed flag (channel 3) is
    # left untouched.  Missing joints are therefore zero + flag, never imputed.
    x = torch.cat([x[..., :3] * x[..., 3:4], x[..., 3:4]], -1)
    if rng.random() < 0.5:                                           # random temporal crop + resample
        T = x.shape[1]; a = rng.randint(0, T // 8); b = T - rng.randint(0, T // 8)
        idx = torch.linspace(a, b - 1, T).round().long()
        x = x[:, idx]
    return x

def train_model(model, Xtr, ytr, Xdv, ydv, epochs, lr, seed, device, batch=32, log=None):
    """Fixed-budget training used identically for the from-scratch baseline and for transfer fine-tuning: AdamW
    (weight decay 0.05), one-cycle schedule peaking at lr, `epochs` epochs, batch 32, label smoothing 0.1, gradient
    clipping 1.0, train-time augmentation, and model selection on DEV accuracy only (the test split is never seen
    here).  Returns the best dev accuracy and the per-epoch dev curve; the model is left at the best-dev checkpoint."""
    import torch, torch.nn.functional as F
    torch.manual_seed(seed); rng = random.Random(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(len(Xtr) / batch))
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    best, best_state, curve = -1.0, None, []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb = augment(Xtr_t[idx], rng).to(device); yb = ytr_t[idx].to(device)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        acc = evaluate(model, Xdv, ydv, device)
        curve.append(acc)
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log: log(f"ep {ep+1}/{epochs} dev {acc:.3f}")
    model.load_state_dict(best_state)
    return best, curve

def evaluate(model, X, y, device, batch=128):
    import torch
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch]).to(device)
            correct += (model(xb).argmax(1).cpu().numpy() == y[i:i + batch]).sum()
    return correct / max(1, len(X))

def transfer_head(model, n_classes, seed=0):
    """Transfer initialisation: every encoder parameter (inp, pos, enc, norm) keeps the SOURCE model's weights; only
    the classifier `head` is replaced by a fresh nn.Linear for the target's classes.  Nothing is frozen: the caller
    then fine-tunes ALL parameters on the target with the same train_model budget as the from-scratch baseline.
    Audit 2026-09-22: the new head is now initialised under torch.manual_seed(seed) so a cell is reproducible
    regardless of what ran before it in the same process."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    model.head = nn.Linear(model.head.in_features, n_classes)
    return model

# ----------------------------------------------------------------------------- Stage 5: analysis
def fit_readings(df, n_perm=10000, seed=0):
    """Statistical readings (rewritten in the 2026-09-22 audit to match the sample structure).

    Sample structure: the unit of analysis is the ordered source->target PAIR (12 pairs for 4 languages); the seeds are
    replicates of the same pair, not independent observations, and the distances are pair-level constants.  Every
    inferential statistic is therefore computed on SEED-AVERAGED gains (one row per pair).  Seeds enter only through
    the per-pair standard deviation, which is reported.

    PRIMARY (Reading A/B in pilot_design.md): OLS of seed-averaged gain on the standardised distance (12 rows,
    df = 10) with its t-based 95% interval, plus an exact-style PERMUTATION test of the slope (the pair->distance
    assignment is shuffled n_perm times with a fixed seed; two-sided p = share of |slope_perm| >= |slope_obs|).
    The permutation p is the one the readings use; it makes no distributional assumption and is valid at n = 12.

    EXPLORATORY (pre-specified in the design, retained, but NOT used to decide a reading): the mixed model with a
    random intercept per target language (4 groups: a variance component estimated from 4 groups is unreliable and
    the fit frequently sits on the boundary) and OLS with target-clustered standard errors (4 clusters: far below
    what cluster-robust inference needs).  Both are fitted on the seed-averaged rows and labelled exploratory.

    DESCRIPTIVE: same-family mean difference under BOTH labelings (D4: with four languages, 1 vs 5 or 3 vs 3 pairs,
    no test is claimed); within-target Spearman rank correlations on n = 3 sources per target (only the values
    +/-1, +/-0.5 are possible; reported as a sign pattern, not as evidence on their own).

    df columns: source, target, seed, gain, dist_mmd_hand, dist_mmd_loc, dist_dtw, dist_model, same_family
    (and optionally same_family_hist)."""
    import pandas as pd, statsmodels.formula.api as smf, warnings
    from scipy import stats
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    out = {"design_note": "unit = ordered pair; seeds averaged before inference; n_pairs below", "n_rows_total": int(len(df)),
           "n_seeds": int(df["seed"].nunique())}
    # ---- seed-averaged pair table
    agg = df.groupby(["source", "target"]).agg(gain=("gain", "mean"), gain_sd=("gain", "std"), n_seeds=("gain", "size")).reset_index()
    consts = df.groupby(["source", "target"]).first().reset_index()
    for col in ["dist_mmd_hand", "dist_mmd_loc", "dist_dtw", "dist_model", "same_family", "same_family_hist"]:
        if col in consts:
            agg[col] = consts[col].values
    out["n_pairs"] = int(len(agg)); out["n_targets"] = int(agg["target"].nunique())
    out["pairs"] = agg.to_dict(orient="records")

    def perm_slope(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        obs = np.polyfit(x, y, 1)[0]
        cnt = 0
        for _ in range(n_perm):
            cnt += abs(np.polyfit(rng.permutation(x), y, 1)[0]) >= abs(obs)
        return float(obs), float((cnt + 1) / (n_perm + 1))

    out["primary"] = {}; out["exploratory"] = {}
    for col in ["dist_mmd_hand", "dist_mmd_loc", "dist_dtw", "dist_model"]:
        if col not in agg or agg[col].isna().any() or not (agg[col].std() > 0):
            continue
        z = (agg[col] - agg[col].mean()) / agg[col].std()
        d = agg.assign(z=z)
        o = smf.ols("gain ~ z", d).fit()
        slope, p_perm = perm_slope(z, d["gain"])
        out["primary"][col] = {"n_pairs": int(len(d)), "slope_per_sd": float(o.params["z"]), "ci95_t": [float(v) for v in o.conf_int().loc["z"]],
                               "p_t_df10": float(o.pvalues["z"]), "p_permutation": p_perm, "r2": float(o.rsquared)}
        # exploratory, pre-specified in the design: mixed model (4 target groups) and target-clustered OLS (4 clusters)
        ex = {}
        try:
            m = smf.mixedlm("gain ~ z", d, groups=d["target"], re_formula="1").fit(reml=False)
            ex["mixedlm_target_re"] = {"coef": float(m.params["z"]), "se": float(m.bse["z"]), "p": float(m.pvalues["z"]),
                                       "ci": [float(v) for v in m.conf_int().loc["z"]], "n_groups": int(d["target"].nunique()),
                                       "converged": bool(getattr(m, "converged", True)), "note": "4 groups: variance component unreliable"}
        except Exception as e:
            ex["mixedlm_target_re"] = {"error": str(e)[:120]}
        try:
            c = smf.ols("gain ~ z", d).fit(cov_type="cluster", cov_kwds={"groups": d["target"]})
            ex["ols_cluster_target"] = {"coef": float(c.params["z"]), "p": float(c.pvalues["z"]), "n_clusters": int(d["target"].nunique()),
                                        "note": "4 clusters: cluster-robust p not reliable"}
        except Exception as e:
            ex["ols_cluster_target"] = {"error": str(e)[:120]}
        out["exploratory"][col] = ex
    # ---- D4 family labels: descriptive under both classifications
    out["descriptive"] = {}
    for col in ["same_family", "same_family_hist"]:
        if col in agg and agg[col].nunique() == 2:
            g1 = agg[agg[col] == 1]["gain"]; g0 = agg[agg[col] == 0]["gain"]
            out["descriptive"][col] = {"n_same": int(len(g1)), "n_diff": int(len(g0)), "mean_gain_same": float(g1.mean()),
                                       "mean_gain_diff": float(g0.mean()), "difference": float(g1.mean() - g0.mean()),
                                       "note": "descriptive; 4 languages cannot support a family-level test"}
    # ---- within-target rank pattern (n = 3 sources per target)
    wt = {}
    for t, sub in agg.groupby("target"):
        if len(sub) >= 3 and sub["dist_mmd_hand"].notna().all():
            wt[t] = {"spearman_D1_vs_gain": float(stats.spearmanr(sub["dist_mmd_hand"], sub["gain"]).statistic), "n_sources": int(len(sub)),
                     # a tie for the largest gain is NOT counted as support (audit 2026-09-22)
                     "closest_source_has_largest_gain": bool(sub.loc[sub["dist_mmd_hand"].idxmin(), "gain"] > sub.drop(sub["dist_mmd_hand"].idxmin())["gain"].max())}
    out["descriptive"]["within_target"] = wt
    out["descriptive"]["n_targets_closest_is_best"] = int(sum(v["closest_source_has_largest_gain"] for v in wt.values()))
    # ---- reading decision, exactly as pre-registered (Reading A needs both conditions)
    if "dist_mmd_hand" in out["primary"]:
        pr = out["primary"]["dist_mmd_hand"]
        cond_slope = pr["slope_per_sd"] < 0 and pr["ci95_t"][1] < 0 and pr["p_permutation"] < 0.05
        cond_rank = out["descriptive"]["n_targets_closest_is_best"] >= 3
        out["reading"] = "A" if (cond_slope and cond_rank) else "B_or_inconclusive"
        out["reading_conditions"] = {"slope_negative_ci_excludes_0_and_perm_p<0.05": bool(cond_slope), "closest_is_best_in_>=3_of_4_targets": bool(cond_rank)}
    return out

# ----------------------------------------------------------------------------- Stage 1: dataset records and clip readers
# Each loader returns a list of records {clip_id, gloss, signer, src} for ONE language, and a reader
# that turns a record into (pose33, lh21, rh21) arrays with NaN for missing joints.  WLASL and AUTSL are
# extracted from video with MediaPipe Tasks 0.10.32 (pose_landmarker_full + hand_landmarker); Slovo comes
# with landmarks from the same extractor family and version; LSFB ships legacy-Holistic landmarks.

def records_wlasl(json_path, available):
    """available: dict video_id -> path of the mp4 inside the mirror repo (the mirror spreads clips over data_0..data_N)."""
    d = json.load(open(json_path))
    recs = []
    for g in d:
        for inst in g["instances"]:
            vid = inst["video_id"]
            if vid in available:
                recs.append(dict(clip_id=vid, gloss=g["gloss"], signer=str(inst["signer_id"]), src="wlasl", repo_path=available[vid],
                                 frame_start=inst.get("frame_start", 1), frame_end=inst.get("frame_end", -1)))
    return recs

def records_autsl(labels_csv, signlist_csv, available_samples):
    """labels_csv rows: 'signerX_sampleY,classId'; available_samples: set of 'signerX_sampleY' present as _color.mp4."""
    import csv
    id2en = {int(r["ClassId"]): r["EN"] for r in csv.DictReader(open(signlist_csv))}
    recs = []
    for line in open(labels_csv):
        parts = line.strip().split(",")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        name, cid = parts[0], int(parts[1])
        if name in available_samples:
            recs.append(dict(clip_id=name, gloss=id2en[cid], signer=name.split("_")[0], src="autsl"))
    return recs

def records_slovo(repo="karinversan/slovo-full-landmarks"):
    """Read only the metadata columns of every landmark block over HTTP (a few seconds per block)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(); recs = []
    for path in sorted(fs.glob(f"datasets/{repo}/data/*/*.parquet")):
        with fs.open(path, "rb") as f:
            tab = pq.ParquetFile(f).read(columns=["video_id", "gloss", "signer_id"]).to_pylist()
        for r in tab:
            recs.append(dict(clip_id=r["video_id"], gloss=r["gloss"], signer=str(r["signer_id"]), src="slovo", block=path))
    return recs

LSFB_EXCLUDE = {"ls", "gsign", "fs", "ns", "pt"}      # corpus annotation codes, not lexical signs; fixed before running

def records_lsfb(tar_paths, lexical_only=True):
    recs = []
    for tp in tar_paths:
        tf = tarfile.open(tp)
        meta = {}
        for m in tf.getmembers():
            key, _, field = m.name.partition(".")
            if field in ("label.txt", "signer.txt", "linguistic_metadata.json"):
                meta.setdefault(key, {})[field] = tf.extractfile(m).read().decode()
        for key, f in meta.items():
            if "label.txt" not in f:
                continue
            if lexical_only and "linguistic_metadata.json" in f and json.loads(f["linguistic_metadata.json"]).get("sign_type") != "lexical":
                continue
            gl = f["label.txt"].strip()
            if gl.lower() in LSFB_EXCLUDE or len(gl) < 2:
                continue
            recs.append(dict(clip_id=key, gloss=gl, signer=f.get("signer.txt", "?").strip(), src="lsfb", tar=tp))
    return recs

def read_lsfb_clip(rec):
    tf = tarfile.open(rec["tar"])
    def arr(field):
        a = np.load(io.BytesIO(tf.extractfile(rec["clip_id"] + "." + field).read())).astype(np.float32)
        a[np.all(np.abs(a) < 1e-6, axis=2)] = np.nan          # all-zero joint = not detected
        return a
    up = arr("pose.upper_pose.npy")                               # (T,23,3): MediaPipe pose joints 0..22
    pose33 = np.full((up.shape[0], 33, 3), np.nan, np.float32); pose33[:, :23] = up
    return pose33, arr("pose.left_hand.npy"), arr("pose.right_hand.npy")

def read_slovo_clip(row):
    """row: one parquet record (dict) from the Slovo landmarks blocks."""
    def arr(key, mask_key, J):
        a = np.array(row[key], np.float32).reshape(-1, J, 3)
        m = np.array(row[mask_key], bool)
        a[~m] = np.nan
        return a
    return arr("pose", "pose_mask", 33), arr("hand_0", "hand_0_mask", 21), arr("hand_1", "hand_1_mask", 21)
