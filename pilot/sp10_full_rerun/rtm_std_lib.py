"""Uni-Sign pose normalisation, numpy port of load_part_kp / crop_scale in Uni-Sign datasets.py
(github.com/ZechengLi19/Uni-Sign). Logic copied line for line; only torch was removed.
Input: kp_px (T,133,2) in PIXELS, sc (T,133). Output: dict part -> (T,n,3) [x,y,conf], parts body9, left21, right21, face18."""
import numpy as np, copy
THR = 0.3
FACE_IDX = [i for i in list(range(23, 23 + 17))[::2]] + [i for i in range(83, 83 + 8)] + [53]

def crop_scale(motion, thr):
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:, 0]); xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1]); ymax = max(valid_coords[:, 1])
    ratio = 1
    scale = max(xmax - xmin, ymax - ymin) * ratio
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result, scale, [xs, ys]

def unisign_parts(kp_px, sc):
    kps_with_scores = {}; scale = None
    for part in ['body', 'left', 'right', 'face_all']:
        if part == 'body':
            kps = kp_px[:, [0] + [i for i in range(3, 11)], :]; conf = sc[:, [0] + [i for i in range(3, 11)]]
        elif part == 'left':
            kps = kp_px[:, 91:112, :]; kps = kps - kps[:, 0:1, :]; conf = sc[:, 91:112]
        elif part == 'right':
            kps = kp_px[:, 112:133, :]; kps = kps - kps[:, 0:1, :]; conf = sc[:, 112:133]
        else:
            kps = kp_px[:, FACE_IDX, :]; kps = kps - kps[:, -1:, :]; conf = sc[:, FACE_IDX]
        if part == 'body':
            result, scale, _ = crop_scale(np.concatenate([kps, conf[..., None]], axis=-1), THR)
        else:
            result = np.concatenate([kps, conf[..., None]], axis=-1)
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :2] = result[..., :2] / scale
                result = np.clip(result, -1, 1)
                result[result[..., 2] <= THR] = 0
        kps_with_scores[part] = result
    return kps_with_scores, scale

def stack_parts(d):
    return np.concatenate([d['body'], d['left'], d['right'], d['face_all']], axis=1)  # (T,69,3)
