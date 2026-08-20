import unittest

import numpy as np
import pandas as pd

from scripts.validate_domain_features import (
    BLEND_WEIGHTS,
    FAMILY_FEATURES,
    STRONG_GAIN,
    add_family_features,
    build_domain_columns,
    count_pressure,
    family_decision,
    fit_domain_state,
    runner_pressure,
    temporal_split,
)


def sample_frame():
    return pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3", "r4"],
            "season": [2022, 2023, 2024, 2024],
            "balls_before": [0, 3, 3, 1],
            "strikes_before": [0, 1, 2, 2],
            "runner_on_1b": [0, 1, 1, 0],
            "runner_on_2b": [0, 0, 1, 1],
            "runner_on_3b": [0, 0, 1, 0],
            "pitcher_hand": [1, 2, 1, 2],
            "batter_hand": [2, 2, 1, 1],
            "asof_pitcher_n": [0, 100, 1000, 10000],
            "asof_pitcher_success_rate": [0.5, 0.5, 0.5, 0.5],
            "asof_pitcher_prev5_game_success_rate": [0.4, 0.5, 0.6, np.nan],
            "control_success": [0, 1, 0, 1],
        }
    )


class DomainFeaturesTest(unittest.TestCase):
    def test_pressure_states_are_row_local(self):
        frame = sample_frame()
        self.assertEqual(
            count_pressure(frame).tolist(),
            ["zero_zero", "three_ball", "full_count", "two_strike"],
        )
        self.assertEqual(
            runner_pressure(frame).tolist(),
            ["bases_empty", "runners_on", "bases_loaded", "risp_proxy"],
        )

    def test_state_is_fit_from_training_only(self):
        training = sample_frame().iloc[:2]
        state = fit_domain_state(training)
        changed_validation = sample_frame().iloc[2:].copy()
        changed_validation["control_success"] = 1 - changed_validation["control_success"]
        before = build_domain_columns(sample_frame().iloc[2:], state)
        after = build_domain_columns(changed_validation, state)
        pd.testing.assert_frame_equal(before, after)

    def test_temporal_split_excludes_validation_and_future(self):
        training, validation = temporal_split(sample_frame(), 2024)
        self.assertTrue((training["season"] < 2024).all())
        self.assertTrue((validation["season"] == 2024).all())

    def test_row_order_and_batch_shape_do_not_change_features(self):
        frame = sample_frame()
        state = fit_domain_state(frame.iloc[:2])
        full = build_domain_columns(frame.iloc[2:], state)
        reverse_rows = frame.iloc[2:].iloc[::-1]
        reverse = build_domain_columns(reverse_rows, state).loc[full.index]
        pd.testing.assert_frame_equal(full, reverse)
        single = build_domain_columns(frame.iloc[[2]], state)
        pd.testing.assert_frame_equal(full.iloc[[0]], single)

    def test_family_contracts_and_fixed_blends(self):
        self.assertEqual(
            list(FAMILY_FEATURES),
            ["pressure_state", "pitcher_state_pressure", "handedness_interaction"],
        )
        self.assertEqual(BLEND_WEIGHTS, (0.01, 0.02, 0.05, 0.10))
        self.assertEqual(STRONG_GAIN, 1e-5)

    def test_family_gate_requires_2023_2024_and_threshold(self):
        rows = [
            {"validation_season": 2022, "brier_gain": -1e-6},
            {"validation_season": 2023, "brier_gain": 2e-5},
            {"validation_season": 2024, "brier_gain": 3e-5},
            {"validation_season": "pooled", "brier_gain": STRONG_GAIN},
        ]
        decision = family_decision(rows)
        self.assertTrue(decision["stable"])
        self.assertTrue(decision["strong"])
        rows[2]["brier_gain"] = 0.0
        self.assertFalse(family_decision(rows)["stable"])


if __name__ == "__main__":
    unittest.main()
