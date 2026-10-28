# LFM2.5-350M evaluation

2026-10-25 KST. EVAL-LFM25-03, checkpoint LM-LFM25-03. Public clean fold0 only:2372 rows, Hangul239 / non-Hangul2133, all unseen target IDs relative to training folds. Same83020 candidate IDs/scores/order as previous evaluations.

| Method | Hangul | Non-Hangul | Weighted |
|---|---:|---:|---:|
| AR06 |239/239 (100%)|2130/2133 (99.859%)|0.99958|
| HyperCLOVA inner-confusion LM |235/239 (98.326%)|2126/2133 (99.672%)|0.98730|
| LFM2.5-350M LM |232/239 (97.071%)|2130/2133 (99.859%)|0.97908|
| AR06 +0.20 LFM |239/239|2130/2133|0.99958|
| Original margin<=1.0 gate |239/239|2130/2133|0.99958|

Against previous LM:7 fixes /6 breaks (Hangul3/6, non-Hangul4/0). More raw fixes do not imply better weighted metric: net3 Hangul errors outweigh4 non-Hangul fixes. Weighted delta-0.00822, vessel-cluster bootstrap95%CI[-0.04087,+0.01728] includes0; no statistically established difference. LFM weighted95%CI[0.94758,0.99958]. All targets appear in top35; LFM has7 Hangul and3 non-Hangul selection errors. AR06 fusion/gate make zero prediction changes. Official subset evaluator passed four modes and matched LM comparison.

Numerical validation: initial batch/single check exceeded legacy1e-4 tolerance by ~1.76e-4, with and without cuDNN; failed runs EVAL-LFM25-01/02 preserved. Run03 disabled cuDNN/TF32, calibrated mean-logp tolerance1e-3 and expanded first-record check to all35 names; maximum observed difference0.00023503495114152173. No target labels used for these numerical settings. Near-top single refinement triggered0 rows. Posthoc single-pair scoring of all17 rows with changed LM predictions or new-LM errors reproduced every batched top1 (0 mismatches). This is not an exhaustive single audit, nor a universal error bound. Do not conceal the numeric protocol difference from the HyperCLOVA comparison.

RTX5080 exhaustive candidate scoring732.02s, including encoding/scoring/ranking/progress writes. Not a deployed gated runtime comparison. Python3.11.15 (server target3.11.17), torch2.7.1+cu128, transformers4.57.6; v5 tokenizer metadata handled via local v4-compatible fast-tokenizer loader. Same300 training steps and positive/negative data, but tokenization, adapted params344064 and architecture differ; equal steps do not imply equal tokens/compute.

Decision: retain AR06 for ID matching; no measured fusion benefit from switching LLM. LFM remains the user-selected research checkpoint and is not deleted or reverted. Korean discrimination is the main observed deficit, but causal training explanations remain unconfirmed. Single explored fold, not untouched final test/fullOOF; no official server claim or ZIP.

Evidence: project_log/experiments/EVAL-LFM25-03.json; reports/noisy_channel/EVAL-LFM25-03/official_lm_vs_previous.txt, paired_predictions.csv, single_pair_numerical_audit.json.
