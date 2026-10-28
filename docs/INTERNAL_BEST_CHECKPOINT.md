# Internal Best checkpoint selection

2026-10-25 KST. Implementation IMPL-INTERNAL-BEST-01; review REVIEW-INTERNAL-DEV-01 addressed.

Default trainer mode is now `--checkpoint-selection internal-best`. `--checkpoint-selection last` preserves the last-step approach for comparison. Existing LoRA target expansion, rank/alpha, rank-beta, optimizer and server package pins are retained.

The external heldout fold is never used for selection. Within outer fit rows, vessels are stratified by whether any observation contains Hangul, then source IDs are split deterministically by seed. Defaults: `--dev-fraction 0.05 --dev-max-vessels 128 --dev-every 25`. Cap is vessels, not pairs. `--dev-max-vessels 0` removes cap; all rows of a selected vessel belong to Dev. Actual row/vessel/pair counts and split hash are saved. Current seed42/fold0 dry check:9560 train rows,290 Dev rows/128 vessels,129 unique Dev pairs; external2372 rows untouched. This check did not train a model.

Train-pair construction, learned transliteration mappings, augmentation and collision checks consume only internal-train rows. Both replacement and cycle samplers index only resulting training pairs. Internal Dev uses only original clean name pairs, without augmentation. Every source vessel's descendants remain on the training side where the mapping was fit.

When confusion CSV options are supplied in internal-best mode, the trainer verifies the supplied provenance but does NOT reuse that outer-fit mined CSV. It rebuilds AR06 cross-fitted miners on the internal-train rows only; internal_dev.crossfit_confusions excludes the probe fold for each fit and records all fit/probe sample IDs. This is necessary because filtering a precomputed outer-fit CSV would still leave models fitted on internal Dev observations. Rebuilding adds CPU preprocessing cost. Generated internal_confusions.csv and miner_fit_audit.json hashes are recorded. With last mode the previous CSV path remains available. Dev negatives are fixed catalog lexical confusions, not a fitted Dev miner.

Every25 steps and at final step the trainer evaluates a language-weighted (hangul-mass/1-hangul-mass) SFT+rank criterion. Row/token positive-SFT reduction follows the selected mode. Token sufficient statistics and rank losses are aggregated separately per language, independent of the last batch size. This is an internal surrogate loss, not official vessel-ID accuracy. Nonfinite Dev loss aborts. Before the first evaluation best loss/step are null in JSON. Equal losses keep the earliest best step.

Adapter snapshots clone CPU tensors, making CPU and CUDA snapshots independent of later optimizer updates. Best adapter is also written to best_adapter.safetensors. Before final merge the selected in-memory state is restored with strict key/shape checking. adapter.safetensors and merged model represent that Best step; no automatic final full-fit retraining occurs. Selection step/loss, interval, source split, clean Dev-pair and negative-candidate hashes are saved in record and model/noisy_channel_manifest.json. Existing schema remains1 with added metadata.

Timing separates internal_dev_seconds (validation plus best snapshot copying/saving) from training_seconds; optimization_loop_seconds includes both. Coverage and completed_optimizer_steps describe total optimization, not only exposure up to selected Best. Source weights/older runs are preserved and no submission ZIP is created.

Example options to append to a fresh training command:

```powershell
--checkpoint-selection internal-best --dev-fraction 0.05 --dev-max-vessels 128 --dev-every 25
```

Validation:5 internal tests (including tiny complete CPU training for cycle/replacement and last mode, forced Best restoration, JSON log before first Dev evaluation, and crossfit fit/probe isolation);11 channel tests;2 LFM tests passed. These are correctness tests, not training-performance evidence. Existing user-authorized .venv Python3.11.15 differs from server3.11.17. No full real-data training started by this implementation task. Next training must use a fresh experiment ID. Already-started processes do not acquire source edits automatically.
