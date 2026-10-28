# Inner-confusion LM training

2026-10-25 KST. Experiment LM-RANK-GPU-04, mining INNER-CONFUSIONS-01.

External fold0 was excluded from mining and training. For each inner fold1,2,3,4 AR06 was fit on the other three groups; lookup disabled. Top8 were collected without injecting gold. Training validates source IDs, source fold, observation, catalog positive/negative and mining artifact hashes, excludes all known positive names for an observation, and takes two highest-ranked distinct wrong names. This is matcher-confusion mining, not reverse-LM-confusion mining.

Original4685 pairs receive9370 mined negatives; augmented2580 pairs keep5160 lexical slots. The positive pair hash equals the prior300-step run. Original base, seed42, 300 steps, batch4/accum2, LR2e-4, LoRA16/alpha32, rowCE, beta0.1/tau1, language cycles0.7/0.3 and augmentation retained. Only negative source changes. Existing user-authorized .venv Python3.11.15 / torch2.7.1+cu128 / transformers4.57.6; server Python3.11.17 differs.

Training completed on RTX5080.11 tests passed. Source records and older models preserved. Output: artifacts/LM-RANK-GPU-04/model and adapter.safetensors. Accuracy still pending: use exact EVAL-LM-RANK-01 candidate CSV for comparison. Do not promote or claim improvement before evaluation. No ZIP created.
