# LM improvement implementation

Implemented defaults: row-mean target/EOS CE + 0.1 listwise ranking loss, two catalog lexical confusions per positive, temperature 1, language mass 0.7/0.3 with shuffled per-language queues, fit-only Hangul token correspondence augmentation, CUDA required. Digit-bearing suffixes are preserved in synthetic augmentation. Same normalized names and fit-known alternative positives are excluded as negatives. No heldout observations enter augmentation or mining.

PowerShell, after the exact `.venv-server` runtime is available:

```powershell
.\.venv-server\Scripts\python.exe -m unittest discover -s tests -p "test_*channel*.py" -v
.\.venv-server\Scripts\python.exe experiments/train_noisy_channel.py --experiment-id LM-RANK-GPU-01 --device cuda --steps 300
```

The command uses existing project `data/`, official group folds, heldout fold 0, local pretrained weights. It trains and saves only; it does not evaluate accuracy or create a submission ZIP. Reserve a new ID for each run. For factor comparisons use `--rank-beta 0`, `--sampling replacement`, `--loss-reduction token`, `--no-learned-hangul`; hold other settings and sampled draws constant where possible. Combined defaults are not an attribution experiment.

Coverage logs report unique pairs seen by language, maximum repetition and positive target tokens. Training time includes the optimizer loop; saving and preparation are separate. Candidate mining uses catalog-name lexical similarity, not a fitted baseline's errors. Expanded Hangul pairs require full alphabetic token coverage from unambiguous fit-fold alignments; they are synthetic hypotheses, not new observed labels.

Status on 2026-10-25 KST: syntax checks passed. Numerical regression tests and GPU training remain blocked because documented uv setup could not obtain Windows Python 3.11.17. RTX 5080 inventory alone does not verify CUDA execution. No accuracy improvement is claimed. Record: `project_log/experiments/IMPL-LM-AR06-01.json`.


## 2026-10-25 KST — IMPL-LM-ENV-02 correction

Existing `.venv` confirmed: Python 3.11.15, torch 2.7.1+cu128, transformers 5.20.1. Only `.venv-server` is absent. The earlier general statement that a virtual environment was absent is incorrect. RTX 5080 CUDA forward/backward passed with finite gradients. New training tests 4 passed, retrieval tests 2 passed; existing V1/V2 prompt tests have 3 tokenizer decode TypeErrors. No training completed. Next: resolve tokenizer/version compatibility (target transformers 4.57.6, Python 3.11.17) before actual LM training. Evidence: `project_log/experiments/IMPL-LM-ENV-02.json`.
