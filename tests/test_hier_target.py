import unittest

import numpy as np
import pandas as pd

from scripts.validate_hier_target import (
    GROUPS,
    M_GRID,
    fit_lookups,
    leakage_audit,
    prepare_frame,
    transform_hierarchy,
)


def example_frame():
    rows = []
    for index, (season, pitcher, batter, success) in enumerate(
        [
            (2019, 1, 10, 1),
            (2019, 1, 11, 0),
            (2020, 1, 10, 1),
            (2020, 2, 11, 0),
            (2021, 1, 10, 1),
            (2021, 2, 11, 0),
            (2022, 1, 10, 1),
            (2022, 2, 11, 0),
        ]
    ):
        rows.append(
            {
                "row_id": f"R{index}",
                "season": season,
                "game_month": 5,
                "game_dayofweek": 2,
                "inning": 4,
                "top_bottom": "T",
                "game_type": "R",
                "balls_before": index % 3,
                "strikes_before": index % 2,
                "outs_before": 1,
                "run_top_before": 0,
                "run_bot_before": 0,
                "run_total_before": 0,
                "score_diff_home": 0,
                "score_diff_pitcher_team": 0,
                "runner_on_1b": 0,
                "runner_on_2b": 0,
                "runner_on_3b": 0,
                "num_runners_on": 0,
                "base_state": "___",
                "home_win_expectancy": 50.0,
                "away_win_expectancy": 50.0,
                "li": 1.0,
                "pitcher_id": pitcher,
                "batter_id": batter,
                "pitcher_hand": 1,
                "batter_hand": 2,
                "control_success": success,
            }
        )
    return prepare_frame(pd.DataFrame(rows))


class HierTargetTest(unittest.TestCase):
    def test_grid_and_hierarchies_are_frozen(self):
        self.assertEqual(M_GRID, (25.0, 75.0, 200.0))
        self.assertEqual(GROUPS[0].name, "pitcher")
        self.assertEqual(GROUPS[-1].name, "pitcher_batter")

    def test_validation_labels_do_not_change_features(self):
        frame = example_frame()
        history = frame[frame.season < 2022]
        validation = frame[frame.season == 2022]
        lookups = fit_lookups(history, frame[frame.season == 2021])
        result = leakage_audit(validation, lookups, 75.0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["validation_target_flip_max_abs_diff"], 0.0)

    def test_unseen_group_shrinks_exactly_to_parent(self):
        frame = example_frame()
        history = frame[frame.season < 2022]
        validation = frame[frame.season == 2022].copy()
        unseen = validation.iloc[[0]].copy()
        unseen["pitcher_id"] = 999
        lookups = fit_lookups(history, frame[frame.season == 2021])
        features = transform_hierarchy(unseen, lookups, 75.0)
        self.assertEqual(float(features.iloc[0]["pitcher_n"]), 0.0)
        self.assertAlmostEqual(
            float(features.iloc[0]["pitcher_rate"]),
            float(features.iloc[0]["global_rate"]),
        )
        self.assertEqual(float(features.iloc[0]["pitcher_reliability"]), 0.0)

    def test_posterior_is_between_parent_and_local_rate(self):
        frame = example_frame()
        history = frame[frame.season < 2022]
        validation = frame[(frame.season == 2022) & (frame.pitcher_id == 1)]
        lookups = fit_lookups(history, frame[frame.season == 2021])
        features = transform_hierarchy(validation, lookups, 75.0)
        parent = float(features.iloc[0]["global_rate"])
        posterior = float(features.iloc[0]["pitcher_rate"])
        local_rate = float(history[history.pitcher_id == 1].control_success.mean())
        self.assertLessEqual(min(parent, local_rate), posterior)
        self.assertGreaterEqual(max(parent, local_rate), posterior)


if __name__ == "__main__":
    unittest.main()
