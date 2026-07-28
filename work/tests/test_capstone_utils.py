import unittest

import numpy as np
import pandas as pd
from work.scripts import capstone_utils

from work.scripts.capstone_utils import (
    add_baseline_score,
    precision_at_k,
    validate_feature_columns,
)


class CapstoneUtilsTest(unittest.TestCase):
    def test_imputed_feature_frame_uses_reference_medians_and_is_finite(self):
        reference = pd.DataFrame(
            {
                "current_ctr": [1.0, 3.0, np.nan],
                "position_change": [0.0, np.inf, 2.0],
            }
        )
        target = pd.DataFrame(
            {
                "current_ctr": [np.nan, np.inf],
                "position_change": [-np.inf, np.nan],
            }
        )

        self.assertTrue(hasattr(capstone_utils, "imputed_feature_frame"))
        result = capstone_utils.imputed_feature_frame(
            target, reference, ["current_ctr", "position_change"]
        )

        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertEqual(result["current_ctr"].tolist(), [2.0, 2.0])
        self.assertEqual(result["position_change"].tolist(), [1.0, 1.0])

    def test_private_action_queue_ranks_scores_and_excludes_identifiers(self):
        frame = pd.DataFrame(
            {
                "client_hash_id": ["client_a", "client_b", "client_c"],
                "content_hash_id": ["content_a", "content_b", "content_c"],
                "impression_change_pct": [-0.30, -0.05, -0.22],
                "position_change": [1.5, 0.2, 0.1],
                "current_impressions": [900, 700, 650],
                "current_ctr": [0.7, 0.5, 1.3],
            }
        )

        self.assertTrue(hasattr(capstone_utils, "build_private_action_queue"))
        queue = capstone_utils.build_private_action_queue(
            frame, pd.Series([0.40, 0.91, 0.72]), top_n=2
        )

        self.assertEqual(queue["priority_rank"].tolist(), [1, 2])
        self.assertEqual(queue["opportunity_score"].tolist(), [0.91, 0.72])
        self.assertEqual(queue["recommended_action"].tolist(), ["snippet_review", "refresh_review"])
        self.assertEqual(
            queue.loc[0, "reason_codes"],
            ["meaningful_search_visibility", "low_ctr_review_candidate"],
        )
        self.assertNotIn("client_hash_id", queue.columns)
        self.assertNotIn("content_hash_id", queue.columns)

    def test_precision_at_k_uses_top_scores(self):
        labels = pd.Series([0, 1, 1, 0])
        scores = pd.Series([0.1, 0.9, 0.8, 0.2])

        self.assertEqual(precision_at_k(labels, scores, 2), 1.0)

    def test_feature_validator_rejects_future_column(self):
        with self.assertRaises(ValueError):
            validate_feature_columns(["current_ctr", "next_impressions"])

    def test_baseline_prioritises_visible_recent_declines(self):
        frame = pd.DataFrame(
            {
                "current_impressions": [1000.0, 1000.0],
                "impression_change_pct": [-0.50, 0.10],
                "position_change": [1.0, -1.0],
                "current_ctr": [1.0, 1.0],
            }
        )

        scored = add_baseline_score(frame)

        self.assertGreater(scored.loc[0, "baseline_score"], scored.loc[1, "baseline_score"])

    def test_baseline_score_treats_missing_momentum_as_no_signal(self):
        frame = pd.DataFrame(
            {
                "current_impressions": [1000.0],
                "impression_change_pct": [np.nan],
                "position_change": [np.nan],
                "current_ctr": [1.0],
            }
        )

        scored = add_baseline_score(frame)

        self.assertEqual(scored.loc[0, "baseline_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
