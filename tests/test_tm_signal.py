import unittest

import numpy as np
import pandas as pd

from scripts.validate_tm_signal import (
    ADAPTIVE_WEIGHT_MAX,
    ADAPTIVE_WEIGHT_MIN,
    BASELINE_GAINS,
    FAMILY_BLEND_WEIGHTS,
    apply_matched_blend,
    build_family_contract,
    confidence_bins,
    percentile_weight,
    _feature_max_diff,
)


class TrackManSignalTest(unittest.TestCase):
    def test_fixed_weights_and_baseline_are_frozen(self):
        self.assertEqual(FAMILY_BLEND_WEIGHTS, (0.05, 0.10, 0.15, 0.20))
        self.assertEqual((ADAPTIVE_WEIGHT_MIN, ADAPTIVE_WEIGHT_MAX), (0.05, 0.15))
        self.assertAlmostEqual(BASELINE_GAINS["pooled"], 1.9849831855711653e-05)

    def test_family_contract_is_disjoint_except_declared_unions(self):
        columns = [
            "tm_history_n",
            "tm_recent_n",
            "career_rel_speed_mean",
            "career_induced_vert_break_mean",
            "career_extension_mean",
            "career_spin_rate_mean",
            "fastball_offspeed_velocity_gap",
            "fastball_breaking_movement_separation",
            "tm_matched",
            "tm_confidence",
            "fastball_n",
            "breaking_n",
            "offspeed_n",
            "count_three_ball_n",
            "count_two_strike_n",
            "count_neutral_n",
        ]
        contract = build_family_contract(columns)
        self.assertIn("tm_history_n", contract["usage_history"])
        self.assertNotIn("fastball_offspeed_velocity_gap", contract["velocity"])
        self.assertIn("fastball_offspeed_velocity_gap", contract["pitch_separation"])
        self.assertEqual(
            set(contract["all_physical"]),
            set().union(
                contract["velocity"],
                contract["movement"],
                contract["release_mechanics"],
                contract["spin"],
                contract["pitch_separation"],
            ),
        )

    def test_unmatched_prediction_is_bit_exact_champion(self):
        fold = {
            "champion": np.array([0.1, 0.2, 0.3]),
            "matched": np.array([True, False, True]),
            "prediction": np.array([0.9, 0.8]),
        }
        candidate = apply_matched_blend(fold, 0.10)
        self.assertEqual(candidate[1], fold["champion"][1])

    def test_adaptive_weight_is_bounded_and_monotone(self):
        reference = np.array([1.0, 2.0, 3.0, 4.0])
        weights = percentile_weight(reference, reference)
        self.assertGreaterEqual(weights.min(), ADAPTIVE_WEIGHT_MIN)
        self.assertLessEqual(weights.max(), ADAPTIVE_WEIGHT_MAX)
        self.assertTrue(np.all(np.diff(weights) >= 0.0))

    def test_confidence_terciles_use_training_reference(self):
        values = np.array([0.1, 0.5, 0.9, 0.0])
        reference = np.array([0.0, 0.3, 0.6, 1.0])
        matched = np.array([True, True, True, False])
        bins = confidence_bins(values, reference, matched)
        self.assertEqual(bins.tolist(), ["high-C", "high-B", "high-A", "unmatched"])

    def test_mixed_feature_parity_handles_categories_and_nan(self):
        left = pd.DataFrame(
            {"numeric": [1.0, np.nan], "category": pd.Categorical(["R", "F"])}
        )
        right = left.copy()
        self.assertEqual(_feature_max_diff(left, right), 0.0)
        right.loc[1, "category"] = "R"
        self.assertEqual(_feature_max_diff(left, right), np.inf)


if __name__ == "__main__":
    unittest.main()
