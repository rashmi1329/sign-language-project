"""
pilot_pipeline.py (SLIM COPY for the Kaggle re-run notebooks).  Every function below is a verbatim copy of the same-named
function in pilot/pilot_pipeline.py (sha256 104edfb56e78...a74982e, 552 lines); the constants are verbatim too.  Only the
functions that sp10_driver.py and sp10_fetch_rest.py call are included: canonical_pose, resample, fill_nan_zero,
make_landmarkers, extract_video, hand_shape_vectors, location_vectors, mmd_rbf, dtw_distance, augment.  The ISLR
pilot's model, training and splitting code is not needed here and is left out.  Equality with the original is checked by
build_sp10_rerun_notebooks.py (per-function source comparison) before the notebooks are written.
"""
import os, json, math, random, zipfile, tarfile, io, time
import numpy as np

UPPER_POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
T_FIXED = 48
N_JOINTS = 13 + 21 + 21

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
    return np.concatenate([np.nan_to_num(seq, nan=0.0), obs], -1)

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

_TS = {"base": 0}

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
