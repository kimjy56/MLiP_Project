"""Verify saved weights and write a separate recovery record; never retrain."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.noisy_channel_common import now_kst, sha256, write_json, configure_cpu

def main():
    configure_cpu(2)
    import torch
    from name_decoder.noisy_channel import NoisyChannelScorer, MANIFEST_NAME
    source = ROOT / "artifacts/LM-RANK-GPU-02"
    destination = ROOT / "project_log/experiments/LM-RANK-GPU-02-RECOVERY-01.json"
    if destination.exists():
        raise FileExistsError("Recovery record already exists")
    record = json.loads((ROOT / "project_log/experiments/LM-RANK-GPU-02.json").read_text(encoding="utf-8-sig"))
    history = json.loads((source / "training_history.json").read_text(encoding="utf-8"))
    if [row["step"] for row in history] != list(range(1,301)):
        raise ValueError("Expected full 300-step history")
    if not (source / "adapter.safetensors").is_file():
        raise FileNotFoundError("Adapter missing")
    # load validates every checkpoint hash in the training manifest.
    scorer = NoisyChannelScorer.load(source / "model")
    scorer.model.to("cuda")
    scorer.device = torch.device("cuda")
    score = scorer.score_pair("SEA STAR", "SEA STAR")
    if not torch.isfinite(torch.tensor(score.mean_log_probability)):
        raise ValueError("Nonfinite recovered score")
    manifest = json.loads((source / "model" / MANIFEST_NAME).read_text(encoding="utf-8"))
    record.update(experiment_id="LM-RANK-GPU-02-RECOVERY-01", source_experiment_id="LM-RANK-GPU-02",
        status="trained_unvalidated_recovered", recorded_at_kst=now_kst(),
        purpose="Recover completed training after final result JSON serialization failure",
        recovery_at_kst=now_kst(), completed_optimizer_steps=300,
        failure={"source_run_error":"NumPy int32 JSON serialization after model saving",
                 "recovery":"Original source record and weights preserved; separate verified record"},
        observations=["All 300 optimizer history entries exist; final loss is training loss only.",
                      "Saved model hashes verified by NoisyChannelScorer.load.",
                      "Saved model loaded and completed synthetic CUDA likelihood check.",
                      "Original record remained running because failure reporting also failed serialization."],
        final_training_loss=history[-1]["loss"],
        validation_status="accuracy_not_measured",
        artifacts=[str(source.resolve())],
        artifact_sha256={"adapter.safetensors":sha256(source / "adapter.safetensors"),
            "training_history.json":sha256(source / "training_history.json"),
            "manifest":sha256(source / "model" / MANIFEST_NAME)},
        decision="Weights recovered without retraining; accuracy validation remains pending",
        next_actions=["Evaluate fixed candidates on heldout fold; do not infer accuracy from training loss."])
    record["cost"].update(training_seconds=None, saving_seconds=None,
        model_bytes=sum(p.stat().st_size for p in (source / "model").rglob("*") if p.is_file()),
        adapter_bytes=(source / "adapter.safetensors").stat().st_size,
        timing_scope="Final in-memory timing/coverage values lost at serialization failure; not reconstructed")
    record["checkpoint_sha256"]=manifest["checkpoint_sha256"]
    write_json(destination, record)
    print("Recovered 300-step model; checkpoint hashes and CUDA scoring passed.")

if __name__ == "__main__":
    main()
