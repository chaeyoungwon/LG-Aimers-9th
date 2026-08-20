import unittest

import numpy as np
import pandas as pd

from scripts.validate_context_specialist import (
    BLEND_WEIGHTS,
    LARGE_OVERALL_GAIN,
    MIN_TRAIN_ROWS,
    STAGES,
    build_hgb52_features,
    current_champion_oof,
    raw_gate,
    routed_prediction,
    two_strike_mask,
)


class ContextSpecialistTest(unittest.TestCase):
    def test_priority_and_fixed_blend_grid(self):
        self.assertEqual(
            [[spec.name for spec in stage] for stage in STAGES],
            [
                ["two_strike"],
                ["three_ball"],
                ["full_count"],
                ["game_type_r", "game_type_f"],
                ["pitcher_established", "pitcher_unseen_low"],
            ],
        )
        self.assertEqual(BLEND_WEIGHTS, (0.10, 0.25, 0.50))
        self.assertEqual(MIN_TRAIN_ROWS, 50_000)
        self.assertEqual(LARGE_OVERALL_GAIN, 2e-5)

    def test_two_strike_route_is_row_local(self):
        frame = pd.DataFrame({"strikes_before": [0, 2, 1, 2]})
        np.testing.assert_array_equal(
            two_strike_mask(frame, cutoff=2023, training=False),
            np.array([False, True, False, True]),
        )

    def test_routed_prediction_leaves_non_target_exact(self):
        champion = np.array([0.1, 0.2, 0.3, 0.4])
        specialist = np.array([0.8, 0.6])
        mask = np.array([False, True, False, True])
        actual = routed_prediction(champion, specialist, mask, 0.25)
        self.assertEqual(actual[0], champion[0])
        self.assertEqual(actual[2], champion[2])
        np.testing.assert_allclose(actual[mask], np.array([0.35, 0.45]))

    def test_raw_gate_requires_latest_pooled_and_sample_size(self):
        gains = {"2022": -1e-5, "2023": 2e-5, "2024": 3e-5, "pooled": 1e-5}
        self.assertTrue(raw_gate(gains, True)["passed"])
        self.assertFalse(raw_gate(gains, False)["passed"])
        gains["2024"] = 0.0
        self.assertFalse(raw_gate(gains, True)["passed"])

    def test_feature_builder_excludes_raw_ids_and_target(self):
        frame = pd.DataFrame(
            {
                "row_id": ["r1"],
                "control_success": [1],
                "pitcher_id": [11],
                "batter_id": [22],
                "top_bottom": ["T"],
                "game_type": ["R"],
                "base_state": ["___"],
                "balls_before": [3],
                "strikes_before": [2],
                "pitcher_hand": [1],
                "batter_hand": [2],
                "asof_pitcher_strike_rate": [0.4],
                "asof_pitcher_ball_rate": [0.2],
                "asof_pitcher_success_rate": [0.5],
                "asof_pitcher_middle_rate": [0.1],
                "asof_pitcher_prev5_game_success_rate": [0.6],
                "asof_pitcher_breaking_rate": [0.3],
                "asof_pitcher_offspeed_rate": [0.2],
                "asof_pitcher_fastball_rate": [0.5],
            }
        )
        levels = {
            "top_bottom": ["T"],
            "game_type": ["R"],
            "base_state": ["___"],
            "count_state": [11],
            "hand_combo": ["1-2"],
        }
        features = build_hgb52_features(frame, levels)
        self.assertFalse(
            {"row_id", "control_success", "pitcher_id", "batter_id"}
            & set(features.columns)
        )
        self.assertEqual(int(features["count_state"].iloc[0]), 11)


if __name__ == "__main__":
    unittest.main()
