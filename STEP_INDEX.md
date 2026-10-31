# Pipeline step by step

All paths are relative to this folder. "Kaggle" = notebook run on Kaggle GPUs/CPUs; each run was a notebook built from the cell code in this folder, pushed with the Kaggle command line, and its output downloaded into `e2_outputs/<run>/`. The notebooks as pushed, their builders, the launch scripts and the Kaggle run logs are not included because the paper does not state anything that depends on them; only the saved per-cell results are.

## Step 0. Corpus
SP-10 (Yin et al. 2022; github.com/MLSLT/SP-10): ten sign languages, the same sentences, 830 / 142 / 185-214 train / dev / test. The clip list and English texts are read from the public SP-10 metadata by `rtm_extract.py` and `pilot/sp10_fetch_rest.py`. Not redistributed here.

## Step 1. Round A poses and S1 (MediaPipe)
Kaggle notebooks `pilot/sp10_run2_data.ipynb` (download + poses of the clips that downloaded), `pilot/sp10_rerun_fetch.ipynb` (re-fetch of the clips Run 2 missed), `pilot/sp10_rerun_poses.ipynb` (poses for the new clips, merge, similarity), code `pilot/pilot_pipeline.py`, `pilot/sp10_driver.py`, `pilot/sp10_fetch_rest.py`. Evidence: `pilot/sp10_full_rerun/full_poses/notebooka9f823024d.log` (extraction and S1 values), `s1_new_from_log.json` (45 values, equal to the log).

## Step 2. Round A training (60 epochs, seed 0, 100 cells)
`pilot/sp10_full_rerun/e2_notebooks/e2_scratch_s0.ipynb` runs `e2_cell.py` with the slim libraries `pilot/pilot_pipeline_slim2.py` and `pilot/sp10_driver_seedfix.py` (pinned by hash inside the notebook). Output: `e2_outputs/scratch_s0/e2_results_scratch_s0.json`.

## Step 3. RTMPose extraction (Rounds B, C, pretrained arm)
`rtm_extract.py` on Kaggle (two Kaggle runs, accounts a and b_k; their extraction reports are in `e2_outputs/rtm_extract_*`), 133 whole-body keypoints per frame at 48 frames per clip.

## Step 4. Quality checks and cleaning
`rtm_quality_check.py` -> `rtm_qc/` (structure, duplicate footage, framing features); standardisation and cleaning `rtm_std_lib.py`, `rtm_standardise.py`, `rtm_std_merge.py` -> `rtm_clean/cleaning_summary.json` (counts per language; the per-clip lists are not included); loader `rtm_data.py`.

## Step 5. Similarity S1 (Rounds B, C)
`rtm_s1.py` + `rtm_dtw_masked.c` -> `similarity_rtm.json` (sha256 b4d584ec...). Computed before any RTMPose training; hash recorded in `docs/analysis_plan_rtm.md`.

## Step 6. Analysis plan
`docs/analysis_plan_rtm.md`: frozen plan (2026-10-19), v2 deviation (2026-10-20), confirmatory seed (2026-10-21, hash d65e325d).

## Step 7. Training cells (Kaggle)
Round B and pretrained seed 0: `e2_rtm_cell.py` (+ `stgcn_uni.py`, `rtm_data.py`). Round C and later seeds: `e3_rtm_cell.py`. Every run folder in `e2_outputs/` has the per-cell scores; the first Round C run also has the scaling sweep and `scale_choice.json`. The saved translations and test-clip lists are not included (see README).

## Step 8. Scoring
chrF and BLEU (sacreBLEU) are computed inside the cell. BLEURT-20-D12: `score_bleurt.py` (all cells), `score_nn_bleurt.py`, `score_random_bleurt.py` -> `rtm_analysis/rtm_bleurt_cells.json`, `rtm_nn_bleurt.json`, `rtm_random_bleurt.json`.

## Step 9. Statistics
`analyze_rtm.py` (Round B), `analyze_rtm2.py` (Round C seed 0, pretrained arm), `roundA_tests.py` (Round A), `analyze_bleurt.py`, `target_adjusted.py`, `two_way_hub.py`, `holm_family.py`, `seed1_confirmatory.py`, `seed1_pooled.py`, `pooled_variants.py`, `power_rtm.py`, `power_general.py`, `extend_rtm_cpu.py`, `extend_rtm_memorisation.py`, `common_sentences.py`, `target_difficulty.py`; floors `nn_baseline.py`, `floor_table.py`. Outputs in `rtm_analysis/`.

## Step 10. Figures
`make_paper_figures.py` (reads the cells and the files in `rtm_analysis/`) -> `paper_figures/` (created when run). The four images exactly as used in the manuscript are in `paper_figures_as_in_manuscript/`.

## Step 11. Recomputing Table 3, the R^2 values and the secondary numbers
`verify_headline_numbers.py` and `verify_secondary_numbers.py` recompute them from the per-cell files and print them next to the paper's values.
