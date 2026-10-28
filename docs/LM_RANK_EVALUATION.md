# GPU LM evaluation — EVAL-LM-RANK-01

2026-10-25 KST. Clean official group-fold 0 only: 2,372 rows, Hangul 239 / non-Hangul 2,133. All targets unseen in fit folds. Public train was previously explored; this is not an untouched final test or five-fold OOF.

| Method | Hangul | Non-Hangul | Weighted | Fix / Break vs baseline |
|---|---:|---:|---:|---:|
| AR06 baseline | 239/239 (100%) | 2130/2133 (99.859%) | 0.99958 | — |
| New reverse LM alone | 235/239 (98.326%) | 2126/2133 (99.672%) | 0.98730 | 0 / 8 |
| Baseline + 0.20 LM | 239/239 | 2130/2133 | 0.99958 | 0 / 0 |
| Original AS05 gate, margin <=1.0 | 239/239 | 2130/2133 | 0.99958 | 0 / 0 |

Official `eval_kit/evaluate.py --subset --compare` confirms these values. Standalone delta is -0.01228 (vessel-cluster bootstrap 95% CI [-0.02959, -0.00056]); fusion delta is zero. LM weighted 95% CI [0.96985, 0.99903]. No previous LM weight checkpoint is locally available for matched comparison, so this is not evidence that the new training improves or worsens the old LM.

Both language groups have 100% target inclusion in the fixed 35 candidates. LM errors are 4 Hangul + 7 non-Hangul, all candidate selection errors; retrieval does not explain them. The LM changes 9 baseline predictions (8 newly wrong, 1 wrong-to-different-wrong), with no fixes. Ungated and gated fusion change zero predictions; only 8 rows satisfy the margin gate. Same-name IDs remain indistinguishable by name likelihood. Row-level errors and known facts are saved in `reports/noisy_channel/EVAL-LM-RANK-01/public_heldout_lm_errors.csv`; detailed causal claims are not established.

AR06/EN_V3 was rebuilt only on fit folds, lookup disabled, with no gold IDs passed to retrieval. Fixed top35 contains 83,020 pairs. Candidates differ from prior cache: 81,626 pairs shared, 1,394 unique per set. Prior cached LM comparisons are therefore not matched experiments.

RTX 5080 float32, batch8 within each record, TF32 disabled. First two pairs checked against single-pair scorer: maximum absolute mean-logp difference 2.08e-5. Unsupported records: 0. Full candidate scoring took 476.35 s (encoding, GPU scoring, ranking and progress writes). Baseline index/fit 2.68 s, baseline retrieval/scoring 39.91 s, LM load 2.38 s; measured total 522.09 s. Peak allocated GPU memory 3,014,085,120 bytes. This exhaustive diagnostic runtime is not production gated runtime. Python 3.11.15 differs from server 3.11.17; torch 2.7.1+cu128 and transformers 4.57.6. Tokenizer emitted the existing regex warning; tokenizer behavior was preserved for checkpoint consistency.

Decision: retain the baseline. This clean fold shows no added ID accuracy from fixed fusion; do not increase fusion weight based on this heldout result. Further work should use pre-specified robustness evaluation and matched CE-only ablations before adoption. No model changes, retraining or ZIP generation occurred.

Evidence: `project_log/experiments/EVAL-LM-RANK-01.json`, `reports/noisy_channel/EVAL-LM-RANK-01/official_{baseline,lm,fusion,gated}.txt`, paired predictions and candidate hashes in the same folder.
