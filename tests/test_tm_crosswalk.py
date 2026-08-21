import unittest

import numpy as np
import pandas as pd

from scripts.validate_tm_crosswalk import (
    attach_physical_features,
    build_physical_profiles,
    confidence_tier,
    build_main_fingerprints,
    build_tm_fingerprints,
    infer_hand_map,
    pair_distance,
)


class TrackManCrosswalkTest(unittest.TestCase):
    def test_main_cumulative_mix_is_differenced_within_season(self):
        frame = pd.DataFrame(
            {
                "season": [2021, 2021, 2022, 2022],
                "pitcher_id": [1, 1, 1, 1],
                "pitcher_hand": [2, 2, 2, 2],
                "asof_pitcher_pitchmix_n": [0, 100, 100, 200],
                "asof_pitcher_fastball_rate": [np.nan, 0.6, 0.6, 0.5],
                "asof_pitcher_breaking_rate": [np.nan, 0.3, 0.3, 0.4],
                "asof_pitcher_offspeed_rate": [np.nan, 0.1, 0.1, 0.1],
            }
        )
        result = build_main_fingerprints(frame).set_index("season")
        self.assertEqual(result.loc[2021, "season_pitchmix_n"], 100)
        self.assertAlmostEqual(result.loc[2021, "season_fastball_rate"], 0.6)
        self.assertEqual(result.loc[2022, "season_pitchmix_n"], 100)
        self.assertAlmostEqual(result.loc[2022, "season_fastball_rate"], 0.4)
        self.assertAlmostEqual(result.loc[2022, "season_breaking_rate"], 0.5)

    def test_trackman_group_rates_sum_to_one(self):
        frame = pd.DataFrame(
            {
                "season": [2021] * 4,
                "pitcher_trackman_id": [10] * 4,
                "pitcher_hand": ["Right"] * 4,
                "pitch_type_group": ["fastball", "fastball", "breaking", None],
            }
        )
        result = build_tm_fingerprints(frame).iloc[0]
        self.assertEqual(result["tm_n"], 4)
        self.assertAlmostEqual(
            sum(result[f"tm_{group}_rate"] for group in ("fastball", "breaking", "offspeed", "other")),
            1.0,
        )

    def test_hand_map_is_inferred_only_from_official_prevalence(self):
        main = pd.DataFrame(
            {
                "season": [2021] * 10,
                "pitcher_hand": [1] * 2 + [2] * 8,
            }
        )
        trackman = pd.DataFrame(
            {
                "season": [2021] * 10,
                "pitcher_hand": ["Left"] * 2 + ["Right"] * 8,
            }
        )
        mapping, _ = infer_hand_map(main, trackman)
        self.assertEqual(mapping, {1: "Left", 2: "Right"})

    def test_multiseason_distance_prefers_matching_trajectory(self):
        main = pd.DataFrame(
            {
                "season": [2021, 2022],
                "season_pitchmix_n": [100, 200],
                "season_fastball_rate": [0.7, 0.4],
                "season_breaking_rate": [0.2, 0.5],
                "season_offspeed_rate": [0.1, 0.1],
                "season_other_rate": [0.0, 0.0],
            }
        )
        matching = pd.DataFrame(
            {
                "season": [2021, 2022],
                "tm_n": [90, 180],
                "tm_fastball_rate": [0.69, 0.41],
                "tm_breaking_rate": [0.21, 0.49],
                "tm_offspeed_rate": [0.1, 0.1],
                "tm_other_rate": [0.0, 0.0],
            }
        )
        wrong = matching.copy()
        wrong[["tm_fastball_rate", "tm_breaking_rate"]] = wrong[
            ["tm_breaking_rate", "tm_fastball_rate"]
        ].to_numpy()
        self.assertLess(
            pair_distance(main, matching)["distance"],
            pair_distance(main, wrong)["distance"],
        )

    def test_high_confidence_requires_overlap_margin_and_absolute_distance(self):
        self.assertEqual(
            confidence_tier(0.10, 0.03, 2, True, "nearest_neighbor"), "HIGH"
        )
        self.assertNotEqual(
            confidence_tier(0.10, 0.01, 2, True, "nearest_neighbor"), "HIGH"
        )
        self.assertNotEqual(
            confidence_tier(0.10, 0.03, 1, True, "nearest_neighbor"), "HIGH"
        )
        self.assertNotEqual(
            confidence_tier(0.10, 0.03, 2, False, "nearest_neighbor"), "HIGH"
        )

    def test_physical_lookup_is_row_local_and_order_independent(self):
        lookup = pd.DataFrame(
            {
                "pitcher_id": [1, 2],
                "pitcher_trackman_id": [10, 20],
                "best_distance": [0.1, 0.1],
                "margin": [0.1, 0.1],
                "seasons_overlap": [2, 2],
                "tm_matched": [1.0, 1.0],
                "tm_confidence": [0.5, 0.5],
                "tm_history_n": [100.0, 200.0],
                "career_rel_speed_mean": [140.0, 145.0],
            }
        )
        rows = pd.DataFrame({"pitcher_id": [2, 3, 1]}, index=[5, 6, 7])
        full = attach_physical_features(rows, lookup)
        reverse = attach_physical_features(rows.iloc[::-1], lookup).iloc[::-1]
        pd.testing.assert_frame_equal(full, reverse)
        self.assertEqual(full.loc[6, "tm_matched"], 0.0)
        self.assertEqual(full.loc[6, "tm_history_n"], 0.0)

    def test_physical_profile_excludes_future_trackman_seasons(self):
        rows = []
        for season, speed in ((2021, 140.0), (2022, 160.0)):
            for _ in range(2):
                rows.append(
                    {
                        "season": season,
                        "pitcher_trackman_id": 10,
                        "pitcher_hand": "Right",
                        "pitch_type_group": "fastball",
                        "balls_before": 0,
                        "strikes_before": 0,
                        "game_date": f"01/01/{season}",
                        "rel_speed": speed,
                        "spin_rate": 2000.0,
                        "induced_vert_break": 10.0,
                        "horz_break": 5.0,
                        "extension": 1.5,
                        "rel_height": 1.8,
                        "rel_side": 0.5,
                        "zone_speed": speed - 10.0,
                    }
                )
        profile = build_physical_profiles(pd.DataFrame(rows), 2021).iloc[0]
        self.assertEqual(profile["tm_history_n"], 2)
        self.assertEqual(profile["career_rel_speed_mean"], 140.0)
        self.assertEqual(profile["recent_rel_speed_mean"], 140.0)


if __name__ == "__main__":
    unittest.main()
