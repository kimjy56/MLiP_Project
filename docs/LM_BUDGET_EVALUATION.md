# LM training budget comparison

2026-10-25 KST. IDs: LM-RANK-GPU-03 / EVAL-LM-RANK-02. User requested further improvement; this tested the planned training-budget extension and did not yield an improvement.

| Method | Hangul (239) | Non-Hangul (2133) | Weighted |
|---|---:|---:|---:|
| AR06 | 239/239 | 2130/2133 | 0.99958 |
| 300-step LM | 235/239 | 2126/2133 | 0.98730 |
| 1200-step LM | 232/239 | 2124/2133 | 0.97823 |
| AR06 + 0.20 LM (1200) | 239/239 | 2130/2133 | 0.99958 |
| Original margin<=1.0 gate | 239/239 | 2130/2133 | 0.99958 |

The 1200-step LM fixes 1 and breaks 6 against the 300-step LM (Hangul 1 fix/4 breaks, non-Hangul 0 fixes/2 breaks). Weighted difference -0.00907; vessel-cluster bootstrap95%CI [-0.04098,+0.00845], which includes zero. Point estimate is worse, but statistical worsening is not established. Both fixed fusion modes change zero AR06 predictions. All gold IDs remain in fixed top35; LM selection errors rise from11 to16.

Only the step budget300->1200 and associated cosine LR horizon changed. Original base, seed42, pairs, augmentation, negatives2, rank beta0.1/tau1, rowCE, batch4/accum2, language cycles0.7/0.3 and LoRA rank16/alpha32 were retained. Pair content/order hash matches the previous training. No heldout-based endpoint selection. Evaluation reused all83,020 candidates/scores in identical row order from EVAL-LM-RANK-01; official evaluator subset comparison passed.

Training consumed9600 draws: Hangul6792, non-Hangul2808. Unique pairs3627 (Hangul819/all819, non-Hangul2808/6446); max repetition9. Training optimizer loop293.327s; exhaustive fixed-candidate evaluation474.5s. RTX5080, torch2.7.1+cu128, transformers4.57.6, existing user-authorized .venv Python3.11.15 (server target3.11.17 difference retained). Models saved separately; previous weights unchanged. No ZIP generated.

Decision: reject the1200-step candidate for adoption, retain AR06 and prior300-step research baseline. Increased exposure alone did not solve candidate discrimination on this clean fold. Repetition/overfitting is a possible explanation, not a verified cause. Lexical negatives may differ from actual model confusions; assess this using training-fold-only inner holdouts rather than tuning external fold0. Next controlled work: CE-only ablation at equal draw budget, then fit-inner-OOF hard-negative mining and checkpoint selection. Do not raise fusion weights based on these external heldout outcomes.

Limitations: explored public fold0 only, not untouched final test, not5-fold OOF, no official server performance claim. Training-budget comparison does not isolate the LR horizon. Legacy V1/V2 weights were unavailable, but both current300/1200 checkpoints are retained.

Evidence: `project_log/experiments/LM-RANK-GPU-03.json`, `project_log/experiments/EVAL-LM-RANK-02.json`, `reports/noisy_channel/EVAL-LM-RANK-02/official_lm_vs_previous.txt`, paired predictions and changes CSVs in that folder.
