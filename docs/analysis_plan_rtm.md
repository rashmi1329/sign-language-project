# Analysis plan: full RTMPose rerun (frozen 2026-10-19 before any training; nothing below has been run)

Data: pilot/sp10_full_rerun/rtm_clean (11,579 clips; manifest sha256 a1bc8bd9dadbc732a1c61c157fd68fd93e0584a80984114c97aada8b7deb2739). Distances: pilot/sp10_full_rerun/similarity_rtm.json (sha256 b4d584ec4054ed8c2c11efe931da318df298c184fa588ecc380519a993a18c1c).
Seed 0 only (seeds 1-2 cut for time; no seed spread is available and this is stated as a limitation).

## Fixed before running
1. S1 (primary) = masked DTW over all 69 Uni-Sign points (x, y), mean over shared train sentence ids, 45 unordered pairs. S1_body (points 0-8) and S1_hands (points 9-50) are secondary. Observed mask = confidence above 0.3, inside the picture, hand frame dropped if fewer than 10 of 21 points observed.
2. Scratch arm: 4-layer pose transformer into T5-small, 60 epochs, lr 3e-4, last epoch, beam 4 (as e2_cell.py v4). Input 48 x 69 x 4 (x, y, conf, observed). 100 cells: 10 per-language models and 90 transfers (each starts from the source language model).
3. Pretrained arm: Uni-Sign ST-GCN (ASL checkpoint) into Linear(1024,512) + T5-small, plus a random-init control with the same code and seed. Cells per arm: 10 per-language models and, for each target, the nearest and the farthest source by S1_dtw (20 transfers). Same training settings as the scratch arm.
4. Gain = transfer score minus the target's own per-language score. Primary metric chrF; secondary BLEU-4, then BLEURT-20 on saved translations.
5. Primary test: QAP node relabelling (10,000 draws, seed 0), slope of gain on standardised S1_dtw over the 90 ordered pairs. Reported with: pooled OLS slope, target-clustered OLS, within-target Spearman (sign test over 10 targets), S1_body and S1_hands separately, and the model-derived comparator S4 (cosine distance between mean scratch-encoder outputs per language).
6. Every test is also run without Bulgarian (bg) as source and target. Both are reported.
7. Pretrained arm comparison: per cell, chrF(pretrained) minus chrF(random-init control); mean difference with a bootstrap interval over the 30 cells. Also whether the S1 slope differs between arms (descriptive, 20 transfer cells).

## Decision rules
- S1 slope negative with QAP p < 0.05 on chrF and the same sign without bg: report as an association at matched content. Otherwise report as not supported. No metric or variant is swapped after results.
- Pretrained minus control interval excluding 0: report as an effect of pretraining on this input. Otherwise: no effect shown.

## Known limits (stated in the paper)
Filming confound (language identifiable from framing at 72.7%); bg video-sentence link weak (limitations entries 55-57); 74 duplicate-footage groups removed; Uni-Sign pretraining data may include sign languages in SP-10 (CSL) and its stage-1 data is not fully known; lightweight RTMPose mode; one seed; no sentence-level ground truth for poses.

## Additions made during implementation (2026-10-19, before any Kaggle run)
- Augmentation (scale 0.85-1.15, rotation +-0.2, shift +-0.1) acts on body points 0-8 only; unobserved points re-zeroed; 50% temporal crop.
- Pretrained checkpoint: how2sign_pose_only_slt.pth. Control: same ST-GCN, random init.
- Stream layouts. Scratch a=[[zh,uk,ru],[bg,is,de]], b_k=[[it,sv],[lt,en]]. Uni a=[[is],[en,uk,ru]], b_k=[[lt,bg],[zh,de,it,sv]]. Each stream owns its languages; the per-language model is both baseline and source of its transfers.
- Primary metric chrF is my choice; veto before the first push.
- Cut rules: no new launch after 17:30 IST; seeds 1-2 dropped.


## v2 rerun (2026-10-20): deviation from this frozen plan, made after seeing the seed-0 results (limitations 65-68), so exploratory
Why: the 60-epoch scratch models are under-fitted (mean train loss 0.27 against 0.03 for the transfer cells), the input scale of hands and face is about 8 times below the body, and there is no control for the extra 60 epochs a transfer cell gets (entries 62, 65, 67).
Changes, scratch arm only (cell e3_rtm_cell.py; with no new environment variable it equals e2_rtm_cell.py):
1. 100 epochs per cell (baseline and transfer), seed 0.
2. Fixed input scaling inside the model, after augmentation, constants pooled over the train splits of all 10 languages (computed in the cell): none, per-part std, or per-joint z-score. Chosen by a dev sweep in the first notebook: languages is, en, de, zh, each scaling, 100 epochs, seed 0; rule = highest mean dev chrF; if the sweep is incomplete, per-joint z-score. The second half of the grid uses the first half's choice (written into its notebook before pushing; z-score if that output is missing); the constants are hashed and printed so the halves can be compared.
3. Ten self-transfer cells (source = target, 100 more epochs from the language's own model), the control for the extra training.
4. Dev chrF and BLEU are recorded for every cell.
Pretrained arm (Uni-Sign and random-init control): recipe unchanged (60 epochs). Seed 0 comes from rtm-uni-a / rtm-uni-b-k (frozen code); added: uni seeds 1, 2, 3 and control seed 1, only as far as compute allows. The 4-seed and 2-seed targets are targets, not guarantees; whatever finishes is reported with its seed count.
Analysis proposed (veto before results): same QAP test on chrF gain as in item 5 above; additionally the same test on net gain = chrF(s to t) minus chrF(t to t) (self-transfer), which removes the extra-training effect. Both are reported; neither is swapped after results. The frozen 60-epoch grid (entry 65) stays in the paper next to the v2 grid.
Split into independent notebooks (a failed or partial notebook costs only its own cells): s1 = sweep + source groups zh, uk, ru, bg (44 grid cells + 12 sweep runs); s2 = groups is, de, it, sv, lt, en (66 cells); ua = uni seeds 1, 2; ub = uni seed 3 + control seed 1. Each starts no new cell after 5.2 h of notebook time (cap 12 h). Expected: s1 about 3.4 h, s2 about 4.1 h (from the measured 258 s per 60-epoch cell, scaled to 100 epochs, 2 GPUs); ua and ub unmeasured (estimate 2-4 h per seed and arm).


## Confirmatory seed (2026-10-21): written BEFORE any seed-1 run starts; exploratory origin of the hypothesis stated
Why: the pre-specified pooled test (item 5) does not reject on the seed-0 grids and has 80% power only for 0.46 chrF per SD of S1 (limitations 75). A target-adjusted version (gain and S1 demeaned within target) was run AFTER seeing that and gives within-target slope -0.22 chrF per SD, one-sided p 0.013 (v2 grid), also nominally significant on the frozen grid and in BLEURT; after Holm over everything run, nothing survives (entries 75, 77). The seed-1 grid below is a new, independent training run to test that lead.
Data: second seed (seed 1) of the scratch v2 grid: 100 epochs, per-joint z-score input scaling FIXED (the choice of the seed-0 sweep; no new sweep), 10 baselines + 90 transfers + 10 self-transfers = 110 cells, cell e3_rtm_cell.py (sha256 prefix 5ce2fb16a70a, identical to seed 0). Three independent Kaggle notebooks: rtm2_sa (zh, uk, ru, bg as sources), rtm2_sb (is, de, it, sv), rtm2_sc (lt, en). Output files rtm_results_scratch_s1_g*_e100_zscore.json.
Confirmatory test (fixed now, one test, no correction needed): target-adjusted QAP, S1_dtw, all 10 languages, chrF gain over the target's seed-1 baseline, within-target demeaned slope, node-relabelling null (10,000 draws, seed 0), one-sided negative, alpha 0.05, on the seed-1 grid ALONE (script target_adjusted.py logic, applied to the seed-1 files). Result is replicated if p < 0.05 and the slope is negative; otherwise it is reported as not replicated, and the seed-0 lead is described as not reproduced by an independent run.
Secondary, reported with the same sign check but not used for the decision, Holm-adjusted as one family: BLEURT gain, BLEU gain, without bg, two-way adjusted, S1_body, S1_hands, net gain (identical to gain under the target-adjusted test), and a two-seed analysis (mean of seeds 0 and 1 for every cell, then the same target-adjusted test).
Expected power: simulated in power_rtm.py: about 0.85 if the true slope is the seed-0 estimate (-0.22 chrF/SD), about 0.6 at -0.15, about 0.4 at -0.11; the seed-0 estimate is probably inflated by selection, so power is likely below 0.85.
Not decided by this run: the cause of the hubs, the signer/filming confound, and the one-signer-per-language limit; a replication of the slope would not remove those.
