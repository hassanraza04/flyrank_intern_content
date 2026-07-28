import unittest

import pandas as pd

from work.scripts.capstone_utils import (
    add_baseline_score,
    precision_at_k,
    validate_feature_columns,
)


class CapstoneUtilsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
