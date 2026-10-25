"""AS-05 row-independent reverse-channel reranking for the submission bundle."""
import json
import math
from pathlib import Path


class NoisyChannelReranker:
    def __init__(self, model_dir, catalog_df):
        from channel.catalog import CatalogNames

        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "as05_config.json").read_text(encoding="utf-8"))
        self.weight = float(self.config["channel_weight"])
        self.margin = float(self.config["max_baseline_margin"])
        self.top_k = int(self.config["candidate_limit"])
        if not (math.isfinite(self.weight) and self.weight > 0 and
                math.isfinite(self.margin) and self.margin >= 0 and self.top_k >= 2):
            raise ValueError("Invalid AS-05 reranking settings")
        self.checkpoint = model_dir / "noisy_channel_weights"
        self.catalog = CatalogNames(catalog_df.to_dict("records"))
        self.scorer = None
        self.eligible_rows = 0
        self.scored_pairs = 0
        self.changed_rows = 0
        self.unsupported_rows = 0

    def _load(self):
        if self.scorer is not None:
            return
        import torch
        from channel.noisy_channel import NoisyChannelScorer

        torch.set_num_threads(4)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        scorer = NoisyChannelScorer.load(self.checkpoint)
        if torch.cuda.is_available():
            scorer.model.to("cuda")
            scorer.device = torch.device("cuda")
        self.scorer = scorer

    def select(self, observed_name, scored_candidates, catalog_index):
        """Return one candidate index. No labels, cross-row cache or time budget.

        Only the original top-35 are eligible. Within the gated row, pruning is
        exact: log probabilities <= 0, so a candidate's original score is an
        upper bound on its fused score. Stop only below the best fused score.
        """
        initial = scored_candidates[0][1]
        if len(scored_candidates) < 2 or not isinstance(observed_name, str) or not observed_name.strip():
            return initial
        shortlist = sorted(scored_candidates, key=lambda pair: (-pair[0], catalog_index.vessel_ids[pair[1]]))[:self.top_k]
        if shortlist[0][0] - shortlist[1][0] > self.margin:
            return initial
        if len({self.catalog.name_by_id[catalog_index.vessel_ids[idx]] for _, idx in shortlist}) == 1:
            return initial
        self.eligible_rows += 1
        self._load()
        best_score = -math.inf
        best_index = initial
        best_id = catalog_index.vessel_ids[initial]
        name_scores = {}  # Same observed row only; retains ALL distinct vessel IDs.
        for baseline_score, index in shortlist:
            if baseline_score < best_score - 1e-9:
                break
            vessel_id = catalog_index.vessel_ids[index]
            catalog_name = self.catalog.name_by_id[vessel_id]
            if catalog_name not in name_scores:
                try:
                    logp = self.scorer.score_pair(catalog_name, observed_name).mean_log_probability
                except (ValueError, TypeError):
                    # The learned tokenizer contract rejects empty/overlength/
                    # special-token names. The rule engine still handles these.
                    self.unsupported_rows += 1
                    return initial
                if not math.isfinite(logp) or logp > 1e-6:
                    raise FloatingPointError("Invalid reverse-channel log probability")
                name_scores[catalog_name] = min(logp, 0.0)
                self.scored_pairs += 1
            combined = float(baseline_score) + self.weight * name_scores[catalog_name]
            if combined > best_score or (combined == best_score and vessel_id < best_id):
                best_score, best_index, best_id = combined, index, vessel_id
        self.changed_rows += int(best_index != initial)
        return best_index
