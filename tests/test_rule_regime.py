import unittest

import numpy as np
import pandas as pd

from scripts.analyze_rule_regime import (
    BOUNDARIES,
    CURRENT_ET_WEIGHT,
    SHRINKAGE_TAU,
    add_static_contexts,
    current_champion_prediction,
    pitcher_form_offset,
    shrink_delta,
    stage_a_decision,
)


class RuleRegimeTest(unittest.TestCase):
    def test_placebo_boundaries_are_fixed(self):
        self.assertEqual(
            BOUNDARIES,
            ((2021, 2022), (2022, 2023), (2023, 2024)),
        )

    def test_count_and_inning_contexts_are_row_local(self):
        frame = pd.DataFrame(
            {
                "balls_before": [3, 1],
                "strikes_before": [2, 0],
                "outs_before": [2, 0],
                "pitcher_hand": [1, 2],
                "batter_hand": [2, 2],
                "game_type": ["R", "F"],
                "base_state": ["___", "1__"],
                "top_bottom": ["T", "B"],
                "inning": [9, 10],
            }
        )
        actual = add_static_contexts(frame)
        self.assertEqual(actual["count_state"].tolist(), ["3-2", "1-0"])
        self.assertEqual(actual["inning_bucket"].tolist(), ["late_7_9", "extras_10_plus"])
        self.assertEqual(actual["hand_matchup"].tolist(), ["H1_vs_H2", "H2_vs_H2"])
        self.assertEqual(actual["outs_before_x_count"].tolist(), ["outs_2|3-2", "outs_0|1-0"])

    def test_shrinkage_moves_cell_delta_toward_global_delta(self):
        raw = np.array([0.10])
        shrunk, factor, effective = shrink_delta(
            raw,
            np.array([1_000]),
            np.array([1_000]),
            parent_delta=-0.02,
        )
        self.assertEqual(SHRINKAGE_TAU, 1_000.0)
        np.testing.assert_allclose(effective, np.array([1_000.0]))
        np.testing.assert_allclose(factor, np.array([0.5]))
        np.testing.assert_allclose(shrunk, np.array([0.04]))

    def test_current_champion_is_direct_original_plus_et25(self):
        original = np.array([0.2, 0.8])
        extra = np.array([0.6, 0.4])
        expected = 0.98 * original + 0.02 * extra
        np.testing.assert_allclose(
            current_champion_prediction(original, extra), expected, rtol=0.0, atol=0.0
        )
        self.assertEqual(CURRENT_ET_WEIGHT, 0.0200)

    def test_pitcher_form_uses_signed_offset_not_probability_endpoint(self):
        source = {
            "champion_base": np.array([0.5, 0.5, 0.5]),
            "pitcher_form_pred": np.array([0.52, 0.5, 0.47]),
        }
        np.testing.assert_allclose(
            pitcher_form_offset(source), np.array([0.02, 0.0, -0.03]), atol=1e-15
        )

    def test_stage_a_requires_structural_and_persistent_family(self):
        placebo = pd.DataFrame(
            [
                {
                    "family": "count_state",
                    "boundary": "2023_to_2024",
                    "n_eligible_cells": 12,
                    "weighted_mean_abs_conditional_delta": 0.01,
                    "distinct_ratio_vs_max_placebo": 1.5,
                    "significant_cell_fraction": 0.5,
                    "champion_residual_direction_alignment": 0.7,
                }
            ]
        )
        persistence = pd.DataFrame(
            [
                {
                    "family": "count_state",
                    "later_boundary": "2023_to_2024",
                    "n_common_cells": 12,
                    "pearson_shrunk_delta": 0.4,
                    "sign_agreement": 0.7,
                }
            ]
        )
        verdict, structural, localized, persistent = stage_a_decision(
            placebo, persistence
        )
        self.assertEqual(verdict, "A. STRUCTURAL 2024 REGIME SHIFT FOUND")
        self.assertEqual(len(structural), 1)
        self.assertEqual(len(localized), 1)
        self.assertEqual(len(persistent), 1)

    def test_non_distinct_family_is_c(self):
        placebo = pd.DataFrame(
            [
                {
                    "family": "count_state",
                    "boundary": "2023_to_2024",
                    "n_eligible_cells": 12,
                    "weighted_mean_abs_conditional_delta": 0.01,
                    "distinct_ratio_vs_max_placebo": 1.1,
                    "significant_cell_fraction": 0.5,
                    "champion_residual_direction_alignment": 0.7,
                }
            ]
        )
        persistence = pd.DataFrame(
            columns=[
                "family",
                "later_boundary",
                "n_common_cells",
                "pearson_shrunk_delta",
                "sign_agreement",
            ]
        )
        verdict, structural, localized, persistent = stage_a_decision(
            placebo, persistence
        )
        self.assertEqual(verdict, "C. NO DISTINCT RULE-REGIME SIGNAL")
        self.assertFalse(structural)
        self.assertFalse(localized)
        self.assertFalse(persistent)

    def test_small_but_distinct_count_effect_is_localized_b(self):
        placebo = pd.DataFrame(
            [
                {
                    "family": "count_state",
                    "boundary": "2023_to_2024",
                    "n_eligible_cells": 12,
                    "weighted_mean_abs_conditional_delta": 0.0046,
                    "distinct_ratio_vs_max_placebo": 1.89,
                    "significant_cell_fraction": 7 / 12,
                    "champion_residual_direction_alignment": 0.625,
                }
            ]
        )
        persistence = pd.DataFrame(
            [
                {
                    "family": "count_state",
                    "later_boundary": "2023_to_2024",
                    "n_common_cells": 12,
                    "pearson_shrunk_delta": 0.45,
                    "sign_agreement": 1.0,
                }
            ]
        )
        verdict, structural, localized, persistent = stage_a_decision(
            placebo, persistence
        )
        self.assertEqual(verdict, "B. WEAK / LOCALIZED SHIFT")
        self.assertFalse(structural)
        self.assertEqual([row["family"] for row in localized], ["count_state"])
        self.assertFalse(persistent)


if __name__ == "__main__":
    unittest.main()
