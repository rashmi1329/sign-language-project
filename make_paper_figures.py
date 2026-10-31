#!/usr/bin/env python3
"""Reproduce the four figures of "Target Effects and Kinematic Similarity in Sign Language
Transfer Matrices" from the project's own result files.

  Fig. 1  model architecture (TikZ via pdflatex if installed, otherwise redrawn with matplotlib)
  Fig. 2  Round C seed 0: (a) chrF gain vs S1 for the 90 ordered pairs, (b) gain matrix
  Fig. 3  power of the three tests vs target share, and minimum detectable slope
  Fig. 4  chrF / BLEU / BLEURT of no-training baselines vs trained systems

Put this file in the project root (the folder that contains `pilot/`) and run

    python make_paper_figures.py              # all four, saved to paper_figures/
    python make_paper_figures.py --show       # also open them in matplotlib windows
    python make_paper_figures.py --only 2 3   # only some figures

Needs: numpy, matplotlib only. If pdflatex (TeX Live / MiKTeX) is installed, Fig. 1 is compiled
from TikZ (plus pdftoppm or PyMuPDF for the PNG); if not, it is drawn with matplotlib.
Every number is read from a file; nothing is typed in by hand.
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
RERUN = ROOT / 'pilot' / 'sp10_full_rerun'
ANA = RERUN / 'rtm_analysis'
OUTDIR = ROOT / 'paper_figures'
LANGS_ORDER = ['zh', 'uk', 'ru', 'bg', 'is', 'de', 'it', 'sv', 'lt', 'en']   # order used by the analysis code
LANGS = sorted(LANGS_ORDER)                                                     # bg de en is it lt ru sv uk zh (as drawn)
IDX = {l: i for i, l in enumerate(LANGS_ORDER)}

# palette used in Figs. 3 and 4
BLUE, ORANGE, AQUA, GRAY, LGRAY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9992', '#d9d8d2'


def need(path):
    if not Path(path).exists():
        sys.exit(f'missing file: {path}\n(run this script from the project root, next to the pilot/ folder)')
    return path


def jload(path):
    with open(need(path), encoding='utf8') as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------- Fig. 1
FIG1_TEX = r"""
\documentclass[border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,calc,backgrounds,shapes.geometric}
\hyphenpenalty=10000 \exhyphenpenalty=10000
\begin{document}
\begin{tikzpicture}[
  font=\footnotesize, >=Stealth,
  blk/.style={draw, rounded corners=1.5pt, align=center, inner sep=3pt, fill=blue!7},
  io/.style={draw, rounded corners=1.5pt, align=center, inner sep=3pt, fill=gray!14},
  ar/.style={-{Stealth[length=1.8mm]}, line width=0.6pt},
  grp/.style={draw, dashed, rounded corners=3pt, inner sep=5pt, line width=0.6pt},
]
\node[io,  text width=2.6cm, minimum height=2.3cm] (n1) at (1.4,0) {Sign video clip\\[1pt](one language)};
\node[blk, text width=3.5cm, minimum height=2.3cm] (n2) at (5.1,0) {Landmark extraction\\[1pt]Round A: MediaPipe pose and hand landmarkers. Rounds B, C: RTMPose, 69-point layout};
\node[blk, text width=5.4cm, minimum height=2.3cm] (n3) at (10.4,0) {Canonical pose\\[1pt]A: 13 upper-body + 2$\times$21 hand joints, $(x,y,z)$, shoulder-centered, scaled by shoulder width. B, C: 69 points, $(x,y)$ and confidence; hands relative to the wrist, face to the nose};
\node[blk, text width=3.9cm, minimum height=2.3cm] (n4) at (16.0,0) {Fixed-length input $X$\\[1pt]resample to 48 frames; missing point $\rightarrow$ 0 plus ``observed'' flag\\$X\in\mathbb{R}^{48\times J\times4}$, $J=55$ (A) or $69$ (B, C)};
\draw[ar] (n1) -- (n2); \draw[ar] (n2) -- (n3); \draw[ar] (n3) -- (n4);
\node[blk, text width=1.9cm, minimum height=1.7cm] (e1) at (1.3,-4.6) {Flatten\\per frame\\$4J=220$ or $276$};
\node[blk, text width=2.5cm, minimum height=1.7cm] (e2) at (4.5,-4.6) {Linear $4J\rightarrow512$ + learned position embedding (48)};
\node[blk, text width=2.9cm, minimum height=1.7cm] (e3) at (7.9,-4.6) {Transformer encoder\\4 layers, 8 heads,\\FFN 2048, dropout 0.2};
\node[blk, text width=1.6cm, minimum height=1.7cm] (e4) at (10.5,-4.6) {Layer-Norm};
\node[io,  text width=1.9cm, minimum height=1.7cm] (h)  at (12.7,-4.6) {Encoder output\\$H\in\mathbb{R}^{48\times512}$};
\draw[ar] (e1) -- (e2); \draw[ar] (e2) -- (e3); \draw[ar] (e3) -- (e4); \draw[ar] (e4) -- (h);
\begin{scope}[on background layer]
\node[grp, fit=(e1)(e2)(e3)(e4), label={[font=\footnotesize\itshape, anchor=south east, inner sep=2pt]north east:Pose encoder}] (gE) {};
\end{scope}
\node[blk, text width=2.5cm, minimum height=1.7cm] (d1) at (15.5,-4.6) {T5-small decoder\\6 layers, 8 heads,\\cross-attention to $H$};
\node[blk, text width=2.2cm, minimum height=1.7cm] (d2) at (15.5,-7.1) {LM head, beam search (4 beams, $\leq$32 tokens)};
\node[io,  text width=2.2cm, minimum height=1.1cm] (o)  at (11.7,-7.1) {English sentence};
\draw[ar] (h) -- (d1); \draw[ar] (d1) -- (d2); \draw[ar] (d2) -- (o);
\begin{scope}[on background layer]
\node[grp, fit=(d1)(d2), label={[font=\footnotesize\itshape, anchor=south west, inner sep=2pt]north west:Text decoder}] (gD) {};
\end{scope}
\draw[ar] (n4.south) -- ++(0,-0.95) -| (e1.north);
\node[font=\footnotesize, text width=7.5cm, align=left, anchor=north west] at (0.0,-6.4) {T5-small's own encoder is bypassed: $H$ is passed to the decoder in its place. Pose encoder: fresh initialization in a from-scratch model, source-model weights in a transfer cell. Decoder: text-pretrained T5-small initialization in a from-scratch model, source-model weights in a transfer cell.};
\node[draw, rounded corners=1.5pt, align=left, font=\footnotesize, text width=17.4cm, inner sep=3pt, anchor=north west] at (-0.4,-8.7) {\textbf{Training (every from-scratch model and every fine-tune):} token-level cross-entropy on English text; AdamW (weight decay 0.01), one-cycle schedule with peak learning rate $3\times10^{-4}$, batch 16, gradient clipping 1.0; train-time augmentation: scale $\pm$15\%, shift $\pm$0.1 (body points only in Rounds B, C), rotation $\pm$0.2~rad, temporal crop ($p=0.5$). \textbf{Schedule by round:} A and B, 60 epochs, last epoch; C, 100 epochs, last epoch, with a fixed per-joint z-score scaling of the input inside the model.};
\end{tikzpicture}
\end{document}
"""


def fig1_matplotlib():
    """Fig. 1 drawn with matplotlib (no LaTeX needed). Same boxes, text and arrows as the TikZ version."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8})
    fig = plt.figure(figsize=(7.38, 4.88))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-0.6, 18.15); ax.set_ylim(-11.1, 1.3); ax.axis('off')
    BLK, IO = '#ececff', '#ebebeb'

    def box(cx, cy, w, h, text, fc):
        w += 0.2
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.07',
                                    fc=fc, ec='black', lw=0.6, zorder=2))
        ax.text(cx, cy, text, ha='center', va='center', zorder=3, linespacing=1.25)
        return dict(l=cx - w / 2, r=cx + w / 2, t=cy + h / 2, b=cy - h / 2, cx=cx, cy=cy)

    def arrow(p, q, style='-|>', rad=0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=0.7, color='black', zorder=1,
                                     shrinkA=0, shrinkB=0))

    n1 = box(1.4, 0, 2.6, 2.3, 'Sign video clip\n(one language)', IO)
    n2 = box(5.1, 0, 3.5, 2.3, 'Landmark extraction\nRound A: MediaPipe pose\nand hand landmarkers.\n'
                               'Rounds B, C: RTMPose,\n69-point layout', BLK)
    n3 = box(10.4, 0, 5.4, 2.3, 'Canonical pose\nA: 13 upper-body + 2\u00d721 hand joints,\n'
                                '$(x,y,z)$, shoulder-centered, scaled by shoulder\n'
                                'width.  B, C: 69 points, $(x,y)$ and confidence;\n'
                                'hands relative to the wrist, face to the nose', BLK)
    n4 = box(16.0, 0, 3.9, 2.3, 'Fixed-length input $X$\nresample to 48 frames; missing\n'
                                'point \u2192 0 plus \u201cobserved\u201d flag\n'
                                '$X\\in\\mathbb{R}^{48\\times J\\times 4}$, $J=55$\n(A) or 69 (B, C)', BLK)
    for a, b in ((n1, n2), (n2, n3), (n3, n4)):
        arrow((a['r'], 0), (b['l'], 0))
    y2 = -4.6
    e1 = box(1.3, y2, 1.9, 1.7, 'Flatten\nper frame\n$4J=220$ or $276$', BLK)
    e2 = box(4.5, y2, 2.5, 1.7, 'Linear $4J\\rightarrow 512$\n+ learned position\nembedding (48)', BLK)
    e3 = box(7.9, y2, 2.9, 1.7, 'Transformer encoder\n4 layers, 8 heads,\nFFN 2048, dropout 0.2', BLK)
    e4 = box(10.5, y2, 1.6, 1.7, 'Layer-Norm', BLK)
    h = box(12.7, y2, 1.9, 1.7, 'Encoder output\n$H\\in\\mathbb{R}^{48\\times 512}$', IO)
    for a, b in ((e1, e2), (e2, e3), (e3, e4), (e4, h)):
        arrow((a['r'], y2), (b['l'], y2))
    d1 = box(15.5, y2, 2.5, 1.7, 'T5-small decoder\n6 layers, 8 heads,\ncross-attention to $H$', BLK)
    d2 = box(15.5, -7.1, 2.2, 1.7, 'LM head, beam\nsearch (4 beams,\n$\\leq$32 tokens)', BLK)
    o = box(11.7, -7.1, 2.2, 1.1, 'English sentence', IO)
    arrow((h['r'], y2), (d1['l'], y2)); arrow((d1['cx'], d1['b']), (d2['cx'], d2['t']))
    arrow((d2['l'], -7.1), (o['r'], -7.1))
    # wrap arrow from the input row into the encoder
    ax.plot([n4['cx'], n4['cx'], e1['cx']], [n4['b'], n4['b'] - 0.95, n4['b'] - 0.95], color='black', lw=0.7, zorder=1)
    arrow((e1['cx'], n4['b'] - 0.95), (e1['cx'], e1['t']))
    # dashed groups
    s = 0.18
    def group(boxes, label, anchor):
        l = min(b['l'] for b in boxes) - s; r = max(b['r'] for b in boxes) + s
        t = max(b['t'] for b in boxes) + s; b_ = min(b['b'] for b in boxes) - s
        ax.add_patch(FancyBboxPatch((l, b_), r - l, t - b_, boxstyle='round,pad=0,rounding_size=0.1', fc='none',
                                    ec='black', lw=0.7, ls=(0, (4, 3)), zorder=0))
        if anchor == 'right':
            ax.text(r - 0.05, t + 0.08, label, ha='right', va='bottom', style='italic')
        else:
            ax.text(l + 0.05, t + 0.08, label, ha='left', va='bottom', style='italic')
    group([e1, e2, e3, e4], 'Pose encoder', 'right')
    group([d1, d2], 'Text decoder', 'left')
    # note on the bypassed T5 encoder
    note = ['T5-small\u2019s own encoder is bypassed: $H$ is passed to the decoder',
            'in its place. Pose encoder: fresh initialization in a from-scratch',
            'model, source-model weights in a transfer cell. Decoder:',
            'text-pretrained T5-small initialization in a from-scratch model,',
            'source-model weights in a transfer cell.']
    for i, ln in enumerate(note):
        ax.text(0.0, -6.55 - i * 0.37, ln, ha='left', va='center')
    # training box with bold lead-ins
    top = -8.7; bottom = -10.45
    ax.add_patch(FancyBboxPatch((-0.4, bottom), 17.4, top - bottom, boxstyle='round,pad=0,rounding_size=0.07',
                                fc='white', ec='black', lw=0.7, zorder=2))
    lines = [[('Training (every from-scratch model and every fine-tune):', True),
              (' token-level cross-entropy on English text; AdamW (weight decay 0.01), one-cycle', False)],
             [('schedule with peak learning rate $3\\times 10^{-4}$, batch 16, gradient clipping 1.0; '
               'train-time augmentation: scale \u00b115%, shift \u00b10.1 (body points only in', False)],
             [('Rounds B, C), rotation \u00b10.2 rad, temporal crop ($p=0.5$). ', False),
              ('Schedule by round:', True),
              (' A and B, 60 epochs, last epoch; C, 100 epochs, last epoch, with a', False)],
             [('fixed per-joint z-score scaling of the input inside the model.', False)]]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    for i, parts in enumerate(lines):
        x = -0.28; y = top - 0.27 - i * 0.37
        for txt, bold in parts:
            t = ax.text(x, y, txt, ha='left', va='center', fontweight='bold' if bold else 'normal', zorder=3)
            bb = t.get_window_extent(rend)
            x += inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
    path = OUTDIR / 'fig1_architecture.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig1():
    if not shutil.which('pdflatex'):
        print('Fig. 1: pdflatex not found, drawing it with matplotlib instead (install MiKTeX or TeX Live for the exact TikZ version).')
        return fig1_matplotlib()
    with tempfile.TemporaryDirectory() as td:
        tex = Path(td) / 'fig1_architecture.tex'
        tex.write_text(FIG1_TEX, encoding='utf8')
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex.name],
                           cwd=td, capture_output=True, text=True)
        pdf = Path(td) / 'fig1_architecture.pdf'
        if not pdf.exists():
            print('Fig. 1: pdflatex failed.\n' + r.stdout[-1500:])
            return None
        shutil.copy(pdf, OUTDIR / 'fig1_architecture.pdf')
        png = OUTDIR / 'fig1_architecture.png'
        if shutil.which('pdftoppm'):
            subprocess.run(['pdftoppm', '-r', '200', '-png', '-singlefile', str(pdf),
                            str(png.with_suffix(''))], check=True)
        else:
            try:
                import fitz                                    # PyMuPDF
                fitz.open(str(pdf))[0].get_pixmap(dpi=200).save(str(png))
            except Exception:
                print('Fig. 1: PDF saved, but no pdftoppm / PyMuPDF to make a PNG.')
                return OUTDIR / 'fig1_architecture.pdf'
        return png


# ----------------------------------------------------------------------------------------- Fig. 2
def round_c_seed0():
    """Round C, seed 0: 10 baselines + 90 transfers (100 epochs, z-score). Same files and rules as analyze_rtm2.py."""
    res = {}
    files = sorted(glob.glob(str(RERUN / 'e2_outputs' / 'rtm2_s[12]_c' / 'rtm_results_scratch_s0_g*_e100_*.json')))
    files = [f for f in files if not any(s in os.path.basename(f) for s in ('gate', 'sweep'))]
    if not files:
        sys.exit('no Round C result files found under pilot/sp10_full_rerun/e2_outputs/rtm2_s1_c and rtm2_s2_c')
    for f in files:
        res.update(jload(f))
    base = {k[8:]: v['test_chrf'] for k, v in res.items() if k.startswith('scratch_')}
    G = np.full((10, 10), np.nan)
    for k, v in res.items():
        if k.startswith('transfer_'):
            s, t = k[9:].split('_')
            if s != t:
                G[IDX[s], IDX[t]] = v['test_chrf'] - base[t]
    sim = jload(RERUN / 'similarity_rtm.json')
    D = np.zeros((10, 10))
    for k, v in sim.items():
        a, b = k.split('|')
        D[IDX[a], IDX[b]] = D[IDX[b], IDX[a]] = v['S1_dtw']
    return G, D, len(base)


def std_slope(x, y):
    x = (x - x.mean()) / x.std()
    return np.polyfit(x, y, 1)[0]


def qap_one_sided(G, D, draws=10000, seed=0):
    """Node-relabelling QAP test (rows and columns of D permuted together); one-sided for a negative slope."""
    rng = np.random.RandomState(seed)
    ii = [(i, j) for i in range(10) for j in range(10) if i != j and not np.isnan(G[i, j])]
    y = np.array([G[i, j] for i, j in ii])
    obs = std_slope(np.array([D[i, j] for i, j in ii]), y)
    c = 0
    for _ in range(draws):
        p = rng.permutation(10)
        s = std_slope(np.array([D[p[i], p[j]] for i, j in ii]), y)
        c += s <= obs
    return obs, (c + 1) / (draws + 1)


def fig2():
    G, D, nb = round_c_seed0()
    ii = [(i, j) for i in range(10) for j in range(10) if i != j]
    x_raw = np.array([D[i, j] for i, j in ii]); y = np.array([G[i, j] for i, j in ii])
    x = (x_raw - x_raw.mean()) / x_raw.std()
    slope, p = qap_one_sided(G, D)
    print(f'Fig. 2: {len(y)} pairs, slope {slope:+.3f} chrF/SD, one-sided QAP p = {p:.3f}')

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.57), gridspec_kw={'width_ratios': [1, 1.08], 'wspace': 0.32})
    # (a) scatter
    a.scatter(x, y, s=26, color='#4c72b0', alpha=0.85, edgecolor='none', zorder=3)
    k, c0 = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    a.plot(xs, k * xs + c0, color='#c0392b', lw=2, zorder=4)
    a.axhline(0, color='gray', lw=1, ls='--', zorder=1)
    a.set_xlabel('S1: standardised masked DTW distance (SD)\n(a)')
    a.set_ylabel('chrF gain over target baseline')
    a.text(0.03, 0.04, f'slope {slope:+.2f} chrF per SD, one-sided QAP p = {p:.2f}\nn = {len(y)} ordered pairs'
           .replace('slope -', 'slope \u2212'), transform=a.transAxes, fontsize=9.5, va='bottom')
    # (b) matrix, rows = source, columns = target, both in alphabetical order
    order = [IDX[l] for l in LANGS]
    M = G[np.ix_(order, order)]
    vmax = np.nanmax(np.abs(M))
    cmap = plt.get_cmap('RdBu').copy(); cmap.set_bad('white')
    im = b.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(10):                                        # hatched diagonal: self-transfer, not among the 90 pairs
        b.add_patch(Rectangle((i - .5, i - .5), 1, 1, facecolor='white', edgecolor='#aaaaaa', hatch='////', lw=0.8))
    b.set_xticks(range(10)); b.set_xticklabels(LANGS, fontsize=10)
    b.set_yticks(range(10)); b.set_yticklabels(LANGS, fontsize=10)
    b.tick_params(length=0)
    for s_ in b.spines.values():
        s_.set_visible(False)
    b.set_xlabel('Target language\n(b)'); b.set_ylabel('Source language')
    cb = fig.colorbar(im, ax=b, fraction=0.045, pad=0.04); cb.set_label('chrF gain', fontsize=10)
    cb.outline.set_visible(False)
    path = OUTDIR / 'fig2_roundC_seed0.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    return path, fig


# ----------------------------------------------------------------------------------------- Fig. 3
def fig3():
    P = jload(ANA / 'rtm_power_general.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
    sh = [0.0, 0.25, 0.5, 0.77, 0.9]
    names = {'A': ('pooled QAP (pre-specified)', BLUE), 'B': ('target-adjusted QAP', ORANGE),
             'C': ('two-way adjusted QAP', AQUA)}
    for k, (lab, c) in names.items():
        yv = [P['by_share'][str(s)][k] for s in sh]
        ax[0].plot(sh, yv, '-o', color=c, lw=2, ms=5, mfc=c, mec='white', mew=1.2, label=lab)
    ax[0].axhline(0.8, color=GRAY, lw=1, ls='--'); ax[0].text(0.01, 0.82, '80% power', color='#52514e', fontsize=8)
    ax[0].axvline(0.77, color=LGRAY, lw=1)
    ax[0].text(0.775, 0.03, 'this study', color='#52514e', fontsize=8, rotation=90, va='bottom')
    ax[0].set_xlabel('share of outcome variance associated with the target')
    ax[0].set_ylabel('power (true slope -0.22 chrF per SD of S1)')
    ax[0].set_ylim(0, 1.02); ax[0].set_xlim(-0.02, 0.95)
    ax[0].set_title('Pooled-test power falls as the target share rises', loc='left', fontsize=9.5)
    M = P['mde']; ms = [0.0, 0.5, 0.77, 0.9]
    for k, (lab, c) in names.items():
        ax[1].plot(ms, [M[str(s)][k] for s in ms], '-o', color=c, lw=2, ms=5, mfc=c, mec='white', mew=1.2)
    ax[1].set_xlabel('share of outcome variance associated with the target')
    ax[1].set_ylabel('smallest true slope detected with 80% power\n(chrF per SD of S1)')
    ax[1].set_ylim(0, 0.95); ax[1].set_xlim(-0.02, 0.95)
    ax[1].set_title('Minimum detectable slope', loc='left', fontsize=9.5)
    ax[1].text(0.9, 0.84, 'pooled', color=BLUE, ha='right', fontsize=8.5)
    ax[1].text(0.9, 0.2, 'target-adjusted', color=ORANGE, ha='right', fontsize=8.5)
    ax[1].text(0.9, 0.42, 'two-way', color=AQUA, ha='right', fontsize=8.5)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = OUTDIR / 'fig3_power.png'
    fig.savefig(path, dpi=200)
    return path, fig


# ----------------------------------------------------------------------------------------- Fig. 4
def fig4():
    F = jload(ANA / 'rtm_floor_table.json')['point']             # name -> [chrF, BLEU]
    NNB = jload(ANA / 'rtm_nn_bleurt.json')['mean']              # nn_flat, nn_pooled, const, oracle (BLEURT x100)
    RND = jload(ANA / 'rtm_random_bleurt.json')['mean']          # random sentence (BLEURT x100)
    AB = jload(ANA / 'rtm_analysis_bleurt.json')                 # trained systems (BLEURT x100)
    RA = jload(ANA / 'rtm_roundA_tests.json')                    # Round A (MediaPipe) mean baselines
    bl = lambda k: AB[k]['baseline_mean']
    # label, chrF, BLEU, BLEURT x100, kind (f = no-training baseline, t = trained, o = oracle)
    rows = [
        ('random sentence',   *F['random train sentence'],                RND, 'f'),
        ('constant sentence', *F['one constant sentence'],                NNB['const'], 'f'),
        ('pose NN (flat)',    *F['pose nearest neighbour (flat)'],        NNB['nn_flat'], 'f'),
        ('pose NN (pooled)',  *F['pose nearest neighbour (pooled)'],      NNB['nn_pooled'], 'f'),
        ('Round A',           RA['chrf']['mean_base'], RA['bleu']['mean_base'], bl('MediaPipe scratch'), 't'),
        ('Round B',           *F['frozen scratch (60 ep)'],               bl('RTM scratch frozen (60 ep)'), 't'),
        ('control',           *F['random-init control (seed 0)'],         bl('uni_ctrl_s0'), 't'),
        ('Round C',           *F['v2 scratch (100 ep)'],                  bl('RTM scratch v2 (100 ep, z-score)'), 't'),
        ('pretrained',        *F['pretrained Uni-Sign (seed 0)'],         bl('uni_s0'), 't'),
        ('oracle retrieval',  *F['oracle retrieval (ceiling)'],           NNB['oracle'], 'o'),
    ]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.9))
    for a, (ttl, idx) in zip(axs, (('chrF', 1), ('BLEU', 2), ('BLEURT x100', 3))):
        v = [r[idx] for r in rows]
        cols = [LGRAY if r[4] == 'o' else (GRAY if r[4] == 'f' else BLUE) for r in rows]
        y = np.arange(len(rows))[::-1]
        a.barh(y, v, color=cols, height=0.62)
        a.set_yticks(y); a.set_yticklabels([r[0] for r in rows] if idx == 1 else [], fontsize=10)
        a.tick_params(axis='y', length=0)
        for yi, vi in zip(y, v):
            a.text(vi + max(v) * 0.01, yi, f'{vi:.1f}', va='center', fontsize=9.5, color='#0b0b0b')
        a.set_title(ttl, loc='left', fontsize=11); a.set_xlim(0, max(v) * 1.22)
    fig.text(0.01, 0.01, 'grey = no-training baselines; blue = trained systems; light grey = oracle ceiling',
             fontsize=9, color='#52514e')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    print('Fig. 4 values (chrF, BLEU, BLEURT):')
    for r in rows:
        print(f'   {r[0]:<18} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.2f}')
    path = OUTDIR / 'fig4_floor.png'
    fig.savefig(path, dpi=200)
    return path, fig


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', type=int, choices=[1, 2, 3, 4], help='figure numbers to make')
    ap.add_argument('--show', action='store_true', help='open the matplotlib figures in windows')
    args = ap.parse_args()
    todo = args.only or [1, 2, 3, 4]
    if not args.show:
        matplotlib.use('Agg')
    OUTDIR.mkdir(exist_ok=True)
    made = []
    if 1 in todo:
        p = fig1()
        if p:
            made.append(p)
    for n, fn in ((2, fig2), (3, fig3), (4, fig4)):
        if n in todo:
            path, fig = fn()
            made.append(path)
            if not args.show:
                plt.close(fig)
    print('\nsaved:')
    for p in made:
        print('  ', p)
    if args.show:
        plt.show()


if __name__ == '__main__':
    try:
        main()
    finally:
        # double-clicked on Windows: keep the console open so the output can be read
        if os.name == 'nt' and len(sys.argv) == 1 and sys.stdin and sys.stdin.isatty():
            input('\nPress Enter to close...')
