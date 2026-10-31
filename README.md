# Target Effects and Kinematic Similarity in Sign Language Transfer Matrices

Code and saved results for the paper by Kulkarni, Pathania and Suhasini S. (Vellore Institute of Technology, Chennai), submitted to Scientific Reports.

## What the project is about

Sign-language translation models are normally trained one language at a time. This project asks whether a sign language that *moves* more like another one transfers better to it.

We use SP-10, a corpus of ten sign languages signing the same English sentences. For each language we train a small pose-to-English model (a pose encoder feeding T5-small). We then start from each language's model and train it on every other language, which gives 90 ordered source-to-target pairs. The gain of a pair is the target's score after transfer minus the score of the target's own model. Each pair also gets a kinematic distance between the two languages' pose trajectories (masked DTW, called S1). The question is whether that distance predicts the gain. It is tested with a pooled permutation test over three training rounds (MediaPipe poses, RTMPose poses, and an RTMPose confirmatory run with a second seed) and with a pretrained Uni-Sign encoder arm. The results are in the paper; this repository holds the code and per-run outputs behind its numbers.

## How to read this repository

1. **`PROVENANCE_MAP.md`**: every table, figure and quoted number of the paper, with the script that produces it, the raw file it reads, and how it was checked.
2. **`STEP_INDEX.md`**: the pipeline in order, from data to figures, and which result folder belongs to which round.
3. **Where things are:**
   * `pilot/sp10_full_rerun/`: the scripts (training-cell code `e2_*` / `e3_*`, pose processing `rtm_*`, statistics and checks).
   * `pilot/sp10_full_rerun/e2_outputs/`: per-cell scores, one folder per run.
   * `pilot/sp10_full_rerun/rtm_analysis/`: outputs of the statistics scripts.
   * `pilot/`: Round A (MediaPipe) code and notebooks.
   * `make_paper_figures.py` and `paper_figures_as_in_manuscript/`: the figures.
   * `MANIFEST_SHA256.txt`: checksum of every file.

The SP-10 videos, poses and sentences are not included; SP-10 is available from its authors (Yin et al., 2022, github.com/MLSLT/SP-10). SP-10, RTMPose, Uni-Sign and T5 have their own terms.

The pretrained-encoder arm (`e2_rtm_cell.py`, `e3_rtm_cell.py`) imports `stgcn_uni.py`, which adapts the ST-GCN pose encoder of Uni-Sign (Li et al., 2025; github.com/ZechengLi19/Uni-Sign) and is not included here. To re-run that arm, take `stgcn_layers/gcn_utils.py` and `stgcn_layers/stgcn_block.py` from the Uni-Sign repository and the released checkpoint (CC-BY-NC-4.0, non-commercial use).

