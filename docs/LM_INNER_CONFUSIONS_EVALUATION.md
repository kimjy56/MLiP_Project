# Inner-confusion LM evaluation

2026-10-25 KST. EVAL-LM-RANK-03 / LM-RANK-GPU-04.

| Method | Hangul239 | Non-Hangul2133 | Weighted |
|---|---:|---:|---:|
| AR06 |239/239|2130/2133|0.99958|
| Lexical-negative300-step LM |235/239|2126/2133|0.98730|
| Inner-confusion300-step LM |235/239|2126/2133|0.98730|
| AR06 + new LM0.20 |239/239|2130/2133|0.99958|
| Original gate margin<=1.0 |239/239|2130/2133|0.99958|

New/previous LM top1 predictions are identical on all2372 rows, fixes0/breaks0, delta0 with paired bootstrapCI[0,0]. Both fusion modes also change zero AR06 predictions. Official eval_kit subset evaluator passed all methods and paired LM comparison. Gold inclusion in fixedtop35 is100% in both languages;11 LM errors are candidate selection errors.

Only training negative source changed at equal300-step budget. All4685 original pairs used cross-fitted AR06 wrong-name candidates;2580 augmented pairs retained lexical fallback. Positive pair hash, seed, objective/weight, optimizer, augmentation and sampling were retained. All83020 candidate IDs/scores/order equal EVAL-LM-RANK-01. GPU exhaustive scoring493.3s; this is diagnostic scoring, not deployed gated runtime.

Decision: no demonstrated incremental top1 accuracy from inner-confusion negatives on this clean fold; keep baseline. This does not establish that scores or robustness are unchanged. Do not claim model-wide ineffectiveness, server improvement or statistical generalization from this fold. Further investigation should inspect fit-inner validation score separation and pre-specified robustness cases before tuning loss weights or mining reverse-LM confusions. Current mining uses matcher confusions, not reverse-LM confusions.

Single explored fold0 only, not fivefoldOOF or untouched final test. Existing .venv Python3.11.15 differs from server3.11.17; torch2.7.1+cu128, transformers4.57.6, RTX5080. No retraining or ZIP in this evaluation.

Evidence: project_log/experiments/EVAL-LM-RANK-03.json; reports/noisy_channel/EVAL-LM-RANK-03/official_lm_vs_previous.txt and paired_predictions.csv.
