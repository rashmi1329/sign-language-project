# Provenance map: paper -> script -> raw file -> check

Status codes used in the last column:

* **R** = recomputed independently from the raw per-cell files by `verify_headline_numbers.py` / `verify_secondary_numbers.py` (code written separately from the project's analysis scripts; matches the paper to the digits printed).
* **S** = the project's own analysis script was re-run on 2026-10-31 on the full project files and reproduced the stored output file exactly.
* **O** = checked against the stored output file only (the script was not re-run, or cannot be re-run from this folder).
* **X** = cannot be checked from this folder (needs data or a model that is not shipped), or no script was found.

Paths are relative to `pilot/sp10_full_rerun/` unless they start with `docs/` or `paper_figures`. "cells" = per-cell result JSONs in `e2_outputs/<run>/` (each holds train loss, chrF, BLEU, dev scores, seconds and peak GPU memory; the saved translations and test-clip ids that the original files also held are not part of this copy).

## Section 3, design

| Paper statement | Script | Raw file / output | Check |
|---|---|---|---|
| 11,741 clips extracted; 74 duplicate groups, 156 clips; six Icelandic clips over 600 frames; 162 removed; Bulgarian lost 73; 11,579 kept | `rtm_extract.py`, `rtm_quality_check.py`, `rtm_standardise.py`, `rtm_std_merge.py` | `rtm_qc/qc_report.json`, `rtm_clean/cleaning_summary.json` (counts only) | O (counts read from the files and agree) |
| 778-828 train, 181-214 test clips per language; 92 test sentences in all ten | `rtm_standardise.py`; `common_sentences.py` | `rtm_clean/cleaning_summary.json`; `rtm_analysis/rtm_common_sentences.json` | O |
| Standardisation (Uni-Sign rules), 69 points x 48 frames x 4 channels | `rtm_std_lib.py`, `rtm_data.py` | code | O (code read against the text) |
| Dev sweep of input scaling: 18.02 / 16.58 / 13.66 | `e3_rtm_cell.py` (sweep mode) | `e2_outputs/rtm2_s1_c/scale_choice.json` | O |
| Architecture and training settings (4 layers, 512, 8 heads, FF 2048, dropout 0.2, AdamW 3e-4, wd 0.01, one-cycle, batch 16, clip 1.0, beam 4, 32 tokens, augmentation ranges, body-only shift) | `e3_rtm_cell.py` (lines 94-155) | code; `sha256(e3_rtm_cell.py)` = 5ce2fb16a70a... | O |
| S1 definition (Algorithm 1); S1_body, S1_hands | `rtm_s1.py`, `rtm_dtw_masked.c` | `similarity_rtm.json` (sha256 b4d584ec..., equals the hash written in `docs/analysis_plan_rtm.md`) | O |
| S1 computed before any training (Rounds B, C) | `rtm_s1.py` | file times: `similarity_rtm.json` 2026-10-19 05:50 UTC; first Round B result 2026-10-19 15:20 UTC (local file times) | O |
| Round A S1 (MediaPipe) | `pilot/sp10_fetch_rest.py` (Kaggle step 2) | `full_poses/notebooka9f823024d.log`; `s1_new_from_log.json` | R (45 of 45 values equal the log) |
| Confirmatory plan, SHA-256 d65e325d, mtime 2026-10-21 05:27 UTC, first seed-1 file 07:55 UTC | - | `docs/analysis_plan_rtm.md`; result-file times | O (hash and times reproduce; self-reported) |

## Table 2 and Table 3 (grids; scores and gain)

| Item | Script | Raw files | Check |
|---|---|---|---|
| Table 2 baseline BLEU 3.27 / 2.33 / 4.69 / 4.64; pretrained 4.6; control 3.6 | `verify_headline_numbers.py`; `floor_table.py` | cells of each round; `rtm_analysis/rtm_floor_table.json` | R |
| Table 3: mean baseline chrF and BLEU, mean gain, positive-cell counts (all four grids), self-transfer gain and positive counts | `verify_headline_numbers.py` | cells of Rounds A, B, C0, C1 | R (all rows) |
| Round B baseline loss 0.27 vs 0.03; Round A 0.054 vs 0.012; Round C 0.003 vs 0.0015 | - | `train_loss` field of the cells | O |

## Section 4.2, target and source effects

| Statement | Script | Raw files | Check |
|---|---|---|---|
| Target R^2 0.89 / source 0.02 (C seed 0 and 1); 0.84 / 0.02 (A); 0.64 / 0.05 (B) | `verify_headline_numbers.py` | cells | R |
| Adjusted R^2 (0.87, below zero), target + source 91%, + S1 91.5% | `extend_rtm_cpu.py` | `rtm_analysis/rtm_extend_cpu.json` | O (not recomputed) |
| BLEU 72% / 5% / 79%; gain 70% / 40% | `extend_rtm_cpu.py`; limitations entry 86 | `rtm_analysis/rtm_extend_cpu.json` | O |
| Target means 10.8 (bg) to 16.5 (zh) / 16.6; rank correlation 1.00; baselines 0.84 (p 0.002); gain by target 0.44 (0.20); source means 0.15 (0.68) | `seed1_confirmatory.py noise` | `rtm_analysis/rtm_seed1_confirmatory.json` (exploratory_noise) | R and S |
| Mean gain by target (-1.97 it ... +0.66 lt), by source (-1.10 to -0.27), SD 0.90 / 0.26 | `make_paper_figures.py` reads the cells | cells | R (recomputed) |
| 92 common sentences: rho 0.94, spread 5.7 -> 7.8; B 0.83, 3.0 -> 4.4; sentence 64% / language 2% | `common_sentences.py` | `rtm_analysis/rtm_common_sentences.json` | O (needs reference sentences to re-run) |
| Target mean chrF vs own baseline 0.85, NN 0.76, size 0.54, loss -0.60, fixed sentence -0.11, oracle 0.03, copy share -0.14, hand share -0.04 | `target_difficulty.py` | `rtm_analysis/rtm_target_difficulty.json` | O |
| Framing: random forest on ten framing features 72.7% (chance 10%), shoulder width alone 23.7% | `rtm_quality_check.py` (14-feature version, 79.6%); ten-feature variant in `verify_secondary_numbers.py` | clip-level table (not part of this copy) | X in this copy: the ten-feature number cannot be recomputed without the clip-level table; **the original ten-feature run was not saved** |

## Table 4 (pooled test) and Table 5 (target-adjusted test)

| Rows | Script | Output file | Check |
|---|---|---|---|
| Round A chrF, BLEU, BLEURT, with and without bg | `roundA_tests.py` | `rtm_analysis/rtm_roundA_tests.json` | S; chrF slope and p also R |
| Round B and Round C seed 0 chrF, BLEU, net | `analyze_rtm.py`, `analyze_rtm2.py` | `rtm_analysis/rtm_analysis.json`, `rtm_analysis_v2.json` | S; chrF slopes and p also R |
| BLEURT rows (B, C0) | `score_bleurt.py` -> `analyze_bleurt.py` | `rtm_analysis/rtm_bleurt_cells.json`, `rtm_analysis_bleurt.json` | O: BLEURT scores not recomputed; the statistics on them were re-run |
| Round C seed 1 pooled (chrF +0.037/0.620; BLEU +0.101/0.768; without bg) | `seed1_pooled.py` (prints only), `pooled_variants.py` | printed; limitations entry 85 | R for chrF; the BLEU row was reproduced by re-running `seed1_pooled.py` |
| Target-clustered intervals B [-0.57, +0.26], C0 [-0.32, +0.24] | `analyze_rtm.py`, `analyze_rtm2.py` | json | S |
| Table 5: nine post-hoc rows and seeds 0+1 | `target_adjusted.py`, `roundA_tests.py`, `seed1_confirmatory.py` | `rtm_target_adjusted.json`, `rtm_roundA_tests.json`, `rtm_seed1_confirmatory.json` | S; chrF rows for B, C0, C1 also R |
| Holm adjustments (0.74-0.84; 0.049, 0.051, >= 0.094; all 1.0 on seed 1) | `holm_family.py`, `seed1_confirmatory.py secondary` | `rtm_holm.json`, `rtm_seed1_confirmatory.json` | S |
| Confirmatory result: +0.04 chrF per SD, p 0.69, 4 of 10 targets negative; two-way -0.14 (0.17); seeds 0+1 -0.09 (0.089, 7/10) | `seed1_confirmatory.py primary/secondary` | `rtm_seed1_confirmatory.json` | R and S |
| S1_body, S1_hands, comparator slopes | `pooled_variants.py` | printed; limitations entry 85 | O |
| Hubs (Icelandic nearest for 7, Chinese for 3, ...); hand-share correlation +0.65 (p 0.043) | `two_way_hub.py` | `rtm_analysis/rtm_two_way_hub.json` | O (needs pose arrays to re-run) |

## Section 4.5-4.7 (power, noise, floors, copying, pretrained arm)

| Statement | Script | Output | Check |
|---|---|---|---|
| Power 0.97 / 0.80 / 0.54 / 0.29 / 0.17; 0.88; 0.46; 5-6% under the null; minimum slopes 0.52 / 0.19 / 0.37; S1 spans 4.2 SD | `power_general.py`; `power_rtm.py` | `rtm_analysis/rtm_power_general.json`, `rtm_power.json` | S (identical) |
| Noise: SD 0.79, 0.56 per run, 11%, cell correlation 0.89 | `seed1_confirmatory.py noise` | json | R, S |
| Pair-specific SD 0.52 / 0.56 and seed correlation 0.16 | same | json | S, but **the method is a shortcut; the exact value is 0.07** |
| Same-seed rerun: 0 of 30 identical, 96.1% of 5,970 translations differ, SD 0.87, max 1.84 | `verify_secondary_numbers.py` (no stored script) | `e2_outputs/rtm2_uc_a/`, `rtm2_uc_c/` | R for the chrF figures (0 of 30, SD 0.87, max 1.84); the 96.1% needs the saved translations, which are not part of this copy |
| Floors: constant sentence 14.7, random 9.2, NN 12.6 / 13.8, oracle 38.0 / 15.3 / 44.9; trained systems; bootstrap intervals | `nn_baseline.py`, `floor_table.py`, `score_nn_bleurt.py`, `score_random_bleurt.py` | `rtm_floor_table.json`, `rtm_nn_baseline.json`, `rtm_nn_bleurt.json`, `rtm_random_bleurt.json` | O (point values for trained systems agree with R; the retrieval baselines need the pose arrays) |
| Copy rates 83.8 / 58.4 / 47.6 / 38.3 / 8.5%; none of 214 test references in train | `extend_rtm_memorisation.py` | `rtm_analysis/rtm_memorisation.txt` | R, **for the ten per-language models only** |
| Pretrained arm: per-seed baselines 15.0-15.6, gains +0.08 / -0.11 / -0.02 / +0.18, control +0.27 / +0.43; +1.57 chrF [1.19, 1.95], +0.95 BLEU, +0.63 BLEURT | `analyze_rtm2.py`, `analyze_bleurt.py` | `rtm_analysis_v2.json`, `rtm_analysis_bleurt.json` | R for chrF values and the interval; S for the json |

## Figures

| Figure | Script | Inputs | Check |
|---|---|---|---|
| Fig. 1 architecture | `make_paper_figures.py` (TikZ or matplotlib) | none (diagram) | the code of the model matches the diagram (read); the image is redrawn, not pixel-identical |
| Fig. 2 gain vs S1 and gain matrix | `make_paper_figures.py` | cells of Round C seed 0, `similarity_rtm.json` | plotted points equal the data; **manuscript image file is an earlier rendering** |
| Fig. 3 power | same | `rtm_power_general.json` | same |
| Fig. 4 floors | same | `rtm_floor_table.json`, `rtm_nn_bleurt.json`, `rtm_random_bleurt.json`, `rtm_analysis_bleurt.json`, `rtm_roundA_tests.json` | same; every bar value equals the table values |

The four image files exactly as used in the manuscript are in `paper_figures_as_in_manuscript/`; `make_paper_figures.py` regenerates them into `paper_figures/`.

## Table 1 and the related-work claims

Table 1 and the literature statements come from the cited papers, not from code. They are not covered by this release.
