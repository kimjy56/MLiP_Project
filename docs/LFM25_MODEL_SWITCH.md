# LFM2.5-350M reverse-LM switch

2026-10-25 KST. User selected https://huggingface.co/LiquidAI/LFM2.5-350M (instruction model, not Base). Default training checkpoint is now artifacts/pretrained/lfm2.5-350m. Old checkpoints and official server requirements remain unchanged.

The official model has16 hybrid layers (10conv,6attention). Our q/v LoRA adapts the6attention blocks only; convolution and other base parameters stay frozen. inject_attention_lora now validates attention count from layer_types for model_type=lfm2. Llama retains its old all-layer validation. Architecture type and adapted layer count are saved in adapter metadata. AutoModelForCausalLM scoring remains architecture-independent.

Official config https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/config.json declares transformers5.0.0rc1. Existing transformers4.57.6 has native LFM2. A tiny hybrid model passed GPU forward/backward, LoRA merge-equivalence and save/reload. Existing11channel tests passed. This does not verify the real350M checkpoint/tokenizer. Actual download was blocked by huggingface.co DNS in the restricted tool environment; no package upgrades or network policy changes were made.

Run in ordinary PowerShell:

```powershell
cd C:\Projects\ML_midterm
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\start_lm_gpu.ps1
```

Launcher defaults to fresh experiment LM-LFM25-01. Downloader resolves the official repository commit before downloading, pins it in download_manifest.json and hashes files; subsequent runs retain that revision. Launcher tests CUDA/transformers/hybrid LoRA, then trains300steps using existing inner-confusion CSV/provenance. No ZIP. If an ID already exists use -ExperimentId with a fresh value. Download manifest and per-run source hashes preserve weight provenance. Training itself remains offline/local-only. Existing server package pins are not changed.

Real350M training and accuracy remain pending. Pair normalization and task-prefix objectives are retained but tokenization naturally differs, so equal steps do not imply equal tokens/compute. Retain AR06 until measured same-candidate ID benefit. Evidence: project_log/experiments/IMPL-LFM25-01.json.


## 2026-10-25 KST — LM-LFM25-03 trained

[x] Actual downloaded LFM2.5-350M GPU300-step training completed and merged model/adapter saved in artifacts/LM-LFM25-03/. Adapted q/v params344064 over6attention blocks. Version4.57.6 compatibility loader retains original tokenizer files but maps v5 TokenizersBackend to PreTrainedTokenizerFast with extra_special_tokens={}. Actual Korean/EOS round-trip plus tiny hybrid GPU tests2 and channel tests11 passed. LM-LFM25-01/02 failed before training and remain separate records. [ ] Accuracy evaluation pending; no improvement claim/ZIP. Next: evaluate with fresh EVAL-LFM25-01 ID, model artifacts/LM-LFM25-03/model and exact reports/noisy_channel/EVAL-LM-RANK-01/candidates.csv. Launcher next default fresh LM-LFM25-04. Commit suggestion: fix(lfm): support v5 tokenizer metadata on pinned transformers4 and train GPU checkpoint.
