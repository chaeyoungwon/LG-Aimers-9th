import unittest

import numpy as np
import pandas as pd

from scripts.validate_control_profile import (
    BLEND_WEIGHTS,
    FAMILIES,
    MAX_FEATURES,
    MIN_SAMPLES_LEAF,
    N_ESTIMATORS,
    RANDOM_STATE,
    build_family_features,
    family_grade,
    feature_audit,
    fit_control_profile_state,
    make_model,
    temporal_split,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3", "r4"],
            "season": [2021, 2022, 2024, 2024],
            "inning": [1, 7, 8, 9],
            "balls_before": [0, 3, 3, 1],
            "strikes_before": [0, 1, 2, 2],
            "score_diff_pitcher_team": [0, 1, -2, np.nan],
            "li": [0.2, 1.0, 2.0, np.nan],
            "asof_pitcher_n": [0, 100, 1_000, 2_000],
            "asof_pitcher_success_rate": [np.nan, 0.50, 0.55, 0.60],
            "asof_pitcher_reverse_rate": [np.nan, 0.20, 0.18, 0.17],
            "asof_pitcher_middle_rate": [np.nan, 0.15, 0.14, 0.13],
            "asof_pitcher_ball_rate": [np.nan, 0.35, 0.32, 0.30],
            "asof_pitcher_strike_rate": [np.nan, 0.45, 0.48, 0.50],
            "asof_pitcher_prev1_game_success_rate": [np.nan, 0.60, 0.65, 0.50],
            "asof_pitcher_prev3_game_success_rate": [np.nan, 0.55, 0.60, 0.55],
            "asof_pitcher_prev5_game_success_rate": [np.nan, 0.52, 0.58, 0.58],
            "asof_pitcher_prev1_game_middle_rate": [np.nan, 0.10, 0.11, 0.20],
            "asof_pitcher_prev3_game_middle_rate": [np.nan, 0.12, 0.12, 0.17],
            "asof_pitcher_prev5_game_middle_rate": [np.nan, 0.14, 0.13, 0.15],
            "asof_batter_n": [0, 200, 300, 400],
            "asof_batter_success_rate": [np.nan, 0.48, 0.50, 0.61],
            "asof_batter_middle_rate": [np.nan, 0.16, 0.15, 0.12],
            "asof_pitcher_pitchmix_n": [0, 80, 900, 1_500],
            "asof_pitcher_fastball_rate": [np.nan, 0.50, 0.60, 0.40],
            "asof_pitcher_breaking_rate": [np.nan, 0.30, 0.25, 0.35],
            "asof_pitcher_offspeed_rate": [np.nan, 0.20, 0.15, 0.25],
            "control_success": [0, 1, 0, 1],
        }
    )


class ControlProfileTest(unittest.TestCase):
    def test_contract_is_exactly_one_et25_and_fixed_blends(self):
        self.assertEqual(N_ESTIMATORS, 25)
        self.assertEqual(MIN_SAMPLES_LEAF, 8)
        self.assertEqual(MAX_FEATURES, "sqrt")
        self.assertEqual(RANDOM_STATE, 42)
        self.assertEqual(BLEND_WEIGHTS, (0.005, 0.010, 0.020, 0.050))
        self.assertEqual(len(FAMILIES), 7)
        self.assertEqual(make_model(n_jobs=1).n_estimators, 25)

    def test_all_families_are_target_independent_and_row_local(self):
        frame = sample_frame()
        state = fit_control_profile_state(frame.iloc[:2])
        validation = frame.iloc[2:]
        changed = validation.copy()
        changed["control_success"] = 1 - changed["control_success"]
        for family in FAMILIES:
            expected = build_family_features(validation, family, state)
            flipped = build_family_features(changed, family, state)
            pd.testing.assert_frame_equal(expected, flipped)
            reversed_frame = validation.iloc[::-1]
            reversed_features = build_family_features(
                reversed_frame, family, state
            ).loc[expected.index]
            pd.testing.assert_frame_equal(expected, reversed_features)
            pd.testing.assert_frame_equal(
                expected.iloc[[0]],
                build_family_features(validation.iloc[[0]], family, state),
            )

    def test_form_and_profile_formulas(self):
        frame = sample_frame().iloc[[2]]
        state = fit_control_profile_state(sample_frame().iloc[:2])
        long_term = build_family_features(frame, "long_term_control", state)
        self.assertAlmostEqual(long_term.iloc[0]["ctrl_danger_failure_balance"], -0.18)
        self.assertAlmostEqual(long_term.iloc[0]["ctrl_reverse_adjusted_control"], 0.37)
        form = build_family_features(frame, "multi_timescale_form", state)
        self.assertAlmostEqual(form.iloc[0]["ctrl_success_dev_3"], 0.05)
        self.assertAlmostEqual(form.iloc[0]["ctrl_success_1v3"], 0.05)
        self.assertAlmostEqual(form.iloc[0]["ctrl_middle_dev_3"], -0.02)

    def test_cold_start_features_have_no_infinity(self):
        frame = sample_frame().iloc[[0]]
        state = fit_control_profile_state(sample_frame().iloc[:2])
        for family in FAMILIES:
            values = build_family_features(frame, family, state)
            numeric = values.select_dtypes(exclude=["category", "object"])
            self.assertFalse(np.isinf(numeric.to_numpy(dtype="float64")).any())

    def test_leverage_threshold_is_training_only(self):
        training = sample_frame().iloc[:2]
        state = fit_control_profile_state(training)
        changed_validation = sample_frame().iloc[2:].copy()
        changed_validation["li"] = [1_000, 2_000]
        self.assertEqual(state, fit_control_profile_state(training))
        features = build_family_features(
            changed_validation, "control_state_leverage", state
        )
        self.assertTrue((features["ctrl_high_leverage"].astype(str) == "HIGH").all())

    def test_temporal_split_excludes_validation_and_future(self):
        training, validation = temporal_split(sample_frame(), 2024)
        self.assertTrue((training["season"] < 2024).all())
        self.assertTrue((validation["season"] == 2024).all())

    def test_grade_policy(self):
        strong_rows = [
            {"validation_season": 2022, "feature_gain_vs_raw_et25": -3e-5},
            {"validation_season": 2023, "feature_gain_vs_raw_et25": 2e-5},
            {"validation_season": 2024, "feature_gain_vs_raw_et25": 2e-5},
            {"validation_season": "pooled", "feature_gain_vs_raw_et25": 1.1e-5},
        ]
        self.assertEqual(family_grade(strong_rows)["grade"], "STRONG")
        weak_rows = [
            {"validation_season": 2022, "feature_gain_vs_raw_et25": 1e-5},
            {"validation_season": 2023, "feature_gain_vs_raw_et25": 2e-5},
            {"validation_season": 2024, "feature_gain_vs_raw_et25": -1e-5},
            {"validation_season": "pooled", "feature_gain_vs_raw_et25": 5e-6},
        ]
        self.assertEqual(family_grade(weak_rows)["grade"], "WEAK_STABLE")
        weak_rows[2]["feature_gain_vs_raw_et25"] = -2e-5
        self.assertEqual(family_grade(weak_rows)["grade"], "UNSTABLE")

    def test_exact_champion_duplicates_are_not_added(self):
        meta = {
            "members": [
                {
                    "name": "hgb",
                    "feature_columns": [
                        "success_minus_middle",
                        "strike_minus_ball",
                        "prev_vs_career",
                    ],
                },
                {
                    "name": "team",
                    "feature_columns": [
                        "pitcher_rate_gap_1_5",
                        "pitcher_rate_gap_career_5",
                        "log1p_asof_pitcher_n",
                        "log1p_asof_pitcher_pitchmix_n",
                    ],
                },
            ]
        }
        audit = feature_audit(meta)
        duplicate = set(
            audit[audit["status"] == "EXACT_DUPLICATE_NOT_ADDED"]["candidate"]
        )
        self.assertEqual(
            duplicate,
            {
                "control_margin",
                "strike_ball_balance",
                "success_dev_5",
                "success_1v5",
                "log_pitcher_n",
                "log_pitchmix_n",
            },
        )


if __name__ == "__main__":
    unittest.main()
