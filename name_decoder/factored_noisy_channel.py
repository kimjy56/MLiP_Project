"""Factored Noisy Channel Matcher: Full Specification-Based Bayesian Matching.

Formulation:
    P(vessel v | observation x) \propto P(x | v) * P(v)
    where:
        log P(x | v) = w_name * log P_LLM(name_obs | name_v)
                     + w_ton * log P_stat(ton_obs | ton_v)
                     + w_flag * log P_stat(flag_obs | flag_v)
                     + w_type * log P_stat(type_obs | type_v)
                     + log P_num(number_match)
        log P(v)     = log P_prior(vessel_popularity)

This removes arbitrary heuristic name similarity scores and evaluates
pure generative likelihood from both language (LLM) and physical specifications.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from name_decoder.noisy_channel import NoisyChannelScorer
from name_decoder.catalog import canonical_name


class FactoredNoisyChannelMatcher:
    """Combines LLM reverse name likelihood with statistical physical spec likelihoods."""

    def __init__(
        self,
        scorer: NoisyChannelScorer,
        catalog_df: pd.DataFrame,
        model_dict: Optional[Dict[str, Any]] = None,
        ton_sigma: float = 0.035,
        w_name: float = 1.0,
        w_ton: float = 1.0,
        w_flag: float = 0.5,
        w_type: float = 0.3,
        num_mismatch_penalty: float = -8.0,
    ) -> None:
        self.scorer = scorer
        self.catalog_df = catalog_df.copy()
        self.ton_sigma = ton_sigma
        self.w_name = w_name
        self.w_ton = w_ton
        self.w_flag = w_flag
        self.w_type = w_type
        self.num_mismatch_penalty = num_mismatch_penalty

        # Catalog lookup tables
        self.vessel_info: Dict[str, Dict[str, Any]] = {}
        for _, row in self.catalog_df.iterrows():
            vid = str(row["vessel_id"])
            gt = -1.0
            try:
                if pd.notna(row["gross_tonnage"]):
                    gt = float(row["gross_tonnage"])
            except (ValueError, TypeError):
                gt = -1.0

            self.vessel_info[vid] = {
                "vessel_id": vid,
                "vessel_name": str(row["vessel_name"]) if pd.notna(row["vessel_name"]) else "",
                "canonical_name": canonical_name(str(row["vessel_name"])) if pd.notna(row["vessel_name"]) else "",
                "gross_tonnage": gt,
                "flag_name": str(row["flag_name"]).strip() if pd.notna(row["flag_name"]) else "UNK",
                "vessel_type": str(row["vessel_type"]).strip() if pd.notna(row["vessel_type"]) else "UNK",
            }

        # Spec lifts / transition tables from fitted model
        self.flag_log_lift: Dict[str, Dict[str, float]] = {}
        self.type_log_lift: Dict[str, Dict[str, float]] = {}
        if model_dict is not None:
            self.flag_log_lift = model_dict.get("flag_log_lift", {})
            self.type_log_lift = model_dict.get("type_log_lift", {})

    def score_tonnage(self, obs_ton: float, alt_ton: float, cat_ton: float) -> float:
        """Compute Gaussian log likelihood of observed tonnage given catalog tonnage."""
        if obs_ton <= 0 or cat_ton <= 0:
            return 0.0  # Missing tonnage is neutral

        diff = abs(obs_ton - cat_ton)
        if alt_ton > 0:
            diff = min(diff, abs(alt_ton - cat_ton))

        rel_err = diff / max(1.0, cat_ton)

        # Exact match bonus
        if diff <= 1.0:
            return 3.5

        # Normal Gaussian error density: -0.5 * (rel / sigma)^2
        z = rel_err / self.ton_sigma
        log_prob = -0.5 * (z ** 2)

        # Clip severe penalty so single measurement error doesn't completely veto
        return float(max(-6.0, log_prob))

    def score_flag(self, obs_flag: str, cat_flag: str) -> float:
        """Compute log likelihood / lift for flag match."""
        if not obs_flag or obs_flag in ("UNK", "nan", "NONE"):
            return 0.0

        key = f"{obs_flag}|ALL"
        if key in self.flag_log_lift:
            return self.flag_log_lift[key].get(cat_flag, 0.0)

        # Fallback exact string match
        return 2.5 if obs_flag.upper() == cat_flag.upper() else -1.0

    def score_type(self, obs_type: str, cat_type: str) -> float:
        """Compute log likelihood / lift for vessel type match."""
        if not obs_type or obs_type in ("UNK", "nan", "NONE"):
            return 0.0

        if obs_type in self.type_log_lift:
            return self.type_log_lift[obs_type].get(cat_type, 0.0)

        return 2.0 if obs_type.upper() == cat_type.upper() else -1.0

    def score_number_match(self, obs_name: str, cat_name: str) -> float:
        """Check for number mismatch (e.g., SHIP 1 vs SHIP 2)."""
        import re
        nums_obs = re.findall(r"\b\d+\b", obs_name)
        nums_cat = re.findall(r"\b\d+\b", cat_name)
        if nums_obs and nums_cat:
            if nums_obs != nums_cat:
                return self.num_mismatch_penalty
            return 2.0
        return 0.0

    def rank_candidates(
        self,
        obs_row: Dict[str, Any],
        candidate_vids: List[str],
    ) -> List[Tuple[float, str, Dict[str, float]]]:
        """Rank candidates using purely factored Bayesian likelihoods.

        Returns list of (total_log_posterior, vessel_id, component_scores).
        """
        obs_name = str(obs_row.get("vessel_name", "")).strip()

        # Parse tonnage
        obs_gt = -1.0
        obs_igt = -1.0
        try:
            if pd.notna(obs_row.get("gross_tonnage")):
                obs_gt = float(obs_row["gross_tonnage"])
            if pd.notna(obs_row.get("international_gross_tonnage")):
                obs_igt = float(obs_row["international_gross_tonnage"])
        except (ValueError, TypeError):
            pass

        ton = obs_igt if obs_igt > 0 else (obs_gt if obs_gt > 0 else -1.0)
        alt_ton = obs_gt if (obs_gt > 0 and obs_gt != ton) else -1.0

        obs_flag = str(obs_row.get("flag_name", "")).strip()
        obs_type = str(obs_row.get("vessel_type", "")).strip()

        scored = []
        for vid in candidate_vids:
            c_info = self.vessel_info.get(vid)
            if not c_info:
                continue

            c_name = c_info["vessel_name"]
            c_ton = c_info["gross_tonnage"]
            c_flag = c_info["flag_name"]
            c_type = c_info["vessel_type"]

            # 1. LLM Name Likelihood: P(obs_name | cat_name)
            ch_res = self.scorer.score_pair(c_info["canonical_name"], obs_name)
            llm_name_logp = ch_res.mean_log_probability

            # 2. Spec Likelihoods
            ton_logp = self.score_tonnage(ton, alt_ton, c_ton)
            flag_logp = self.score_flag(obs_flag, c_flag)
            type_logp = self.score_type(obs_type, c_type)
            num_logp = self.score_number_match(obs_name, c_name)

            # 3. Total Factored Posterior
            total_logp = (
                self.w_name * llm_name_logp
                + self.w_ton * ton_logp
                + self.w_flag * flag_logp
                + self.w_type * type_logp
                + num_logp
            )

            components = {
                "llm_name_logp": round(llm_name_logp, 4),
                "ton_logp": round(ton_logp, 4),
                "flag_logp": round(flag_logp, 4),
                "type_logp": round(type_logp, 4),
                "num_logp": round(num_logp, 4),
                "total_logp": round(total_logp, 4),
            }
            scored.append((total_logp, vid, components))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored
