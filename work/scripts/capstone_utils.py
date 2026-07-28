"""Public-safe scoring and evaluation helpers for the capstone."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


FEATURE_COLUMNS = [
    "log_current_impressions",
    "impression_change_pct",
    "click_change_pct",
    "current_ctr",
    "current_avg_position",
    "position_change",
    "current_active_days",
]

FORBIDDEN_COLUMNS = {
    "client_hash_id",
    "content_hash_id",
    "cohort_id",
    "is_declining_proxy",
    "next_impressions",
    "future_impression_ratio",
}


def validate_feature_columns(columns: list[str]) -> None:
    """Raise if a proposed model input directly identifies or reveals the outcome."""
    blocked = sorted(set(columns) & FORBIDDEN_COLUMNS)
    if blocked:
        raise ValueError(f"Forbidden feature columns: {', '.join(blocked)}")


def add_baseline_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Score visible pages with recent downward momentum using a transparent rule."""
    scored = frame.copy()
    decline = (-scored["impression_change_pct"].fillna(0)).clip(lower=0)
    demand = np.log1p(scored["current_impressions"]).clip(lower=0)
    position_worsening = scored["position_change"].fillna(0).clip(lower=0)
    scored["baseline_score"] = decline * demand + 0.10 * position_worsening
    return scored


def precision_at_k(labels: pd.Series, scores: pd.Series, k: int) -> float:
    """Return the positive-label rate among the k highest scoring rows."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if len(labels) == 0:
        raise ValueError("labels and scores must not be empty")

    order = np.argsort(-np.asarray(scores))[: min(k, len(scores))]
    return float(np.asarray(labels)[order].mean())


def evaluate_ranking(labels: pd.Series, scores: pd.Series, k: int = 100) -> dict[str, float]:
    """Compute the decision metric, base rate, and discrimination metric together."""
    label_values = np.asarray(labels)
    score_values = np.asarray(scores)
    if len(np.unique(label_values)) < 2:
        raise ValueError("ROC-AUC requires both label classes")

    return {
        "base_rate": float(label_values.mean()),
        "precision_at_k": precision_at_k(labels, scores, k),
        "roc_auc": float(roc_auc_score(label_values, score_values)),
    }


def reason_codes(row: pd.Series) -> list[str]:
    """Explain a score with human-review signals that were known at decision time."""
    codes: list[str] = []
    if row["impression_change_pct"] <= -0.20:
        codes.append("recent_search_momentum_down")
    if row["position_change"] >= 1.0:
        codes.append("average_position_worsened")
    if row["current_impressions"] >= 500:
        codes.append("meaningful_search_visibility")
    if row["current_ctr"] < 1.0:
        codes.append("low_ctr_review_candidate")
    return codes or ["monitor_for_more_evidence"]


def public_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Create aggregate counts suitable for a public paper without raw ranked rows."""
    action_counts = frame["action"].value_counts().to_dict()
    reason_counts = frame.explode("reason_codes")["reason_codes"].value_counts().to_dict()
    return {
        "recommendations": int(len(frame)),
        "action_counts": {str(key): int(value) for key, value in action_counts.items()},
        "reason_code_counts": {str(key): int(value) for key, value in reason_counts.items()},
    }
