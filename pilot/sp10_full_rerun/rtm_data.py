"""Shared loader for the cleaned RTMPose set (pilot/sp10_full_rerun/rtm_clean). Used by S1, the training cells and the analysis.
Point layout of a standardised clip (T=48, 69, 3): body 0-8, left hand 9-29, right hand 30-50, face 51-68; channels x, y, conf (Uni-Sign normalised).
obs mask: conf>0 (Uni-Sign already zeroes conf<=0.3) AND point inside the picture AND, for hands, at least HAND_MIN of 21 points observed in that frame."""
import numpy as np, json, os
LANGS = ['zh', 'uk', 'ru', 'bg', 'is', 'de', 'it', 'sv', 'lt', 'en']
SL = {'body': slice(0, 9), 'left': slice(9, 30), 'right': slice(30, 51), 'face': slice(51, 69)}
HAND_MIN = 10
def obs_mask(x, inpic):
    """x (T,69,3) float, inpic (T,69) bool -> (T,69) bool"""
    m = (x[..., 2] > 0) & inpic
    for h in ('left', 'right'):
        s = SL[h]; keep = m[:, s].sum(1) >= HAND_MIN; m[:, s] = m[:, s] & keep[:, None]
    return m
def load_lang(root, lang, split=None, with_mask=True):
    """returns dict id -> (x float32 (T,69,3), obs bool (T,69))  (ids like 'train/410')"""
    z = np.load(os.path.join(root, f'rtm_std_{lang}.npz')); out = {}
    for k in z.files:
        if not k.startswith('x|'): continue
        i = k[2:]
        if split and not i.startswith(split + '/'): continue
        x = z[k].astype(np.float32); m = obs_mask(x, z['m|' + i]) if with_mask else None
        out[i] = (x, m)
    return out
