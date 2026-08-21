import unittest

import numpy as np
import pandas as pd

from scripts.validate_lupi_distill_oof import (
    ALPHAS,
    BLEND_WEIGHTS,
    CURRENT_CHAMPION_WEIGHT,
    CURRENT_ET25_WEIGHT,
    cutoff_safe_anchors,
    distillation_target,
    verdict_from_results,
)


class LupiDistillOofTest(unittest.TestCase):
    def test_frozen_grids_and_champion_weight_are_exact(self):
        self.assertEqual(ALPHAS, (0.0, 0.1, 0.25, 0.5))
        self.assertEqual(BLEND_WEIGHTS, (0.01, 0.025, 0.05, 0.10))
        self.assertEqual(CURRENT_CHAMPION_WEIGHT, 0.9775)
        self.assertEqual(CURRENT_ET25_WEIGHT, 0.0225)

    def test_distillation_changes_only_linked_rows(self):
        real = np.array([0.0, 1.0, 0.0, 1.0])
        positions = np.array([1, 3])
        teacher = np.array([0.4, 0.8])
        mixed = distillation_target(real, positions, teacher, 0.25)
        np.testing.assert_array_equal(mixed[[0, 2]], real[[0, 2]])
        np.testing.assert_allclose(mixed[[1, 3]], [0.85, 0.95])

    def test_alpha_zero_is_exact_real_label_baseline(self):
        real = np.array([0.0, 1.0, 0.0])
        mixed = distillation_target(real, np.array([0, 2]), np.array([0.9, 0.8]), 0.0)
        np.testing.assert_array_equal(mixed, real)

    def test_cutoff_safe_anchor_intersects_existing_high_mapping(self):
        linked = pd.DataFrame(
            {
                "row_id": ["A", "B", "C"],
                "season": [2021, 2021, 2022],
                "pitcher_id": [10, 20, 10],
                "pitcher_trackman_id": [100, 999, 100],
            }
        )
        crosswalk = pd.DataFrame(
            {
                "cutoff": [2021, 2021],
                "method": ["hungarian", "hungarian"],
                "accepted": [True, False],
                "confidence": ["HIGH", "LOW"],
                "main_pitcher_id": [10, 20],
                "tm_pitcher_id": [100, 999],
            }
        )
        safe, audit = cutoff_safe_anchors(linked, crosswalk, 2021)
        self.assertEqual(safe["row_id"].tolist(), ["A"])
        self.assertEqual(audit["cutoff_safe_anchor_rows"], 1)
        self.assertEqual(audit["anchor_seasons"], [2021])

    def test_two_worse_seasons_is_kill(self):
        temporal = pd.DataFrame(
            {
                "validation_season": [2022, 2023, 2024],
                "gain_vs_student_baseline": [-1e-6, -2e-6, 1e-6],
                "best_blend_gain_vs_champion": [1e-7, 1e-7, 1e-7],
            }
        )
        verdict, reasons = verdict_from_results(
            temporal,
            {
                "gain_vs_student_baseline": -1e-6,
                "best_blend_gain_vs_champion": 1e-7,
            },
            teacher_pooled_gain=1e-5,
            noncovered_pooled_gain=-1e-6,
        )
        self.assertEqual(verdict, "KILL")
        self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()
