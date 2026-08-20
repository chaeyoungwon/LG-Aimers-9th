import unittest

import numpy as np
import pandas as pd

from scripts.build_regime_blend_oof import (
    build_ours_features,
    fit_platt,
    forecast_next_rate,
    form_offsets,
)


class OofRecipeTest(unittest.TestCase):
    @staticmethod
    def _rows():
        n = 3
        data = {
            "row_id": [f"r{i}" for i in range(n)],
            "season": [2020, 2020, 2021],
            "game_month": [3, 4, 5],
            "game_dayofweek": [1, 2, 3],
            "inning": [1, 5, 9],
            "top_bottom": ["T", "B", "T"],
            "game_type": ["R", "F", "R"],
            "balls_before": [0, 3, 2],
            "strikes_before": [2, 0, 2],
            "outs_before": [0, 1, 2],
            "run_top_before": [0, 1, 2],
            "run_bot_before": [0, 0, 1],
            "run_total_before": [0, 1, 3],
            "score_diff_home": [0, -1, 1],
            "score_diff_pitcher_team": [0, 1, -1],
            "runner_on_1b": [0, 1, 0],
            "runner_on_2b": [0, 0, 1],
            "runner_on_3b": [0, 0, 0],
            "num_runners_on": [0, 1, 1],
            "base_state": ["___", "1__", "_2_"],
            "home_win_expectancy": [50.0, 48.0, 55.0],
            "away_win_expectancy": [50.0, 52.0, 45.0],
            "li": [0.8, 1.0, 1.2],
            "pitcher_id": [10, 11, 10],
            "batter_id": [20, 21, 22],
            "pitcher_hand": [1, 2, 1],
            "batter_hand": [2, 2, 1],
            "pitcher_team_id": [12, 13, 12],
            "batter_team_id": [13, 12, 14],
            "asof_pitcher_n": [0, 20, 30],
            "asof_batter_n": [0, 15, 25],
            "asof_pitcher_pitchmix_n": [0, 20, 30],
            "control_success": [0, 1, 1],
        }
        rate_columns = [
            "asof_pitcher_success_rate",
            "asof_pitcher_reverse_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_ball_rate",
            "asof_pitcher_strike_rate",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_prev5_game_middle_rate",
            "asof_batter_success_rate",
            "asof_batter_middle_rate",
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
        for offset, column in enumerate(rate_columns):
            data[column] = [np.nan, 0.2 + offset * 0.001, 0.3 + offset * 0.001]
        return pd.DataFrame(data)

    def test_recovered_feature_builder_is_row_local(self):
        frame = self._rows()
        levels = {
            "top_bottom": ["T", "B"],
            "game_type": ["F", "R"],
            "base_state": ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"],
        }
        full = build_ours_features(frame, levels).iloc[[1]].reset_index(drop=True)
        single = build_ours_features(frame.iloc[[1]], levels).reset_index(drop=True)
        pd.testing.assert_frame_equal(full, single, check_exact=True)

    def test_platt_fit_recovers_known_logit_transform(self):
        prediction = np.linspace(0.05, 0.95, 1000)
        logits = np.log(prediction / (1.0 - prediction))
        target_probability = 1.0 / (1.0 + np.exp(-(1.7 * logits - 0.2)))
        scale, bias = fit_platt(prediction, target_probability)
        self.assertAlmostEqual(scale, 1.7, places=10)
        self.assertAlmostEqual(bias, -0.2, places=10)

    def test_rate_forecast_uses_training_seasons_only(self):
        frame = pd.DataFrame(
            {
                "season": [2019, 2019, 2020, 2020],
                "control_success": [0.0, 0.0, 1.0, 1.0],
            }
        )
        self.assertAlmostEqual(forecast_next_rate(frame, 2021), 1.0 - 1e-6)

    def test_target_label_flip_does_not_change_form_offsets(self):
        history = pd.DataFrame(
            {
                "season": [2019, 2020],
                "game_type": ["R", "R"],
                "pitcher_id": [10, 10],
                "batter_id": [20, 20],
                "asof_pitcher_n": [0, 1],
                "asof_pitcher_success_rate": [np.nan, 1.0],
                "asof_batter_n": [0, 1],
                "asof_batter_success_rate": [np.nan, 1.0],
                "control_success": [1, 0],
            }
        )
        target = pd.DataFrame(
            {
                "season": [2021, 2021],
                "game_type": ["R", "R"],
                "pitcher_id": [10, 10],
                "batter_id": [20, 20],
                "asof_pitcher_n": [2, 3],
                "asof_pitcher_success_rate": [0.5, 2.0 / 3.0],
                "asof_batter_n": [2, 3],
                "asof_batter_success_rate": [0.5, 2.0 / 3.0],
                "control_success": [0, 1],
            }
        )
        first = form_offsets(history, 2020, target)
        flipped = target.copy()
        flipped["control_success"] = 1 - flipped["control_success"]
        second = form_offsets(history, 2020, flipped)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
