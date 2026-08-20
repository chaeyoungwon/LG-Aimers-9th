import unittest

import pandas as pd

from scripts.finalize_et25_upper_weight import (
    CANDIDATE_WEIGHTS,
    CURRENT_CHAMPION_LB_BSS,
    CURRENT_CHAMPION_SHA256,
    REFERENCE_WEIGHT,
    build_comparison,
    select_smallest_stable,
)
from scripts.validate_et25_upper_weight import (
    patch_runtime_weights,
    patch_script_weight_doc,
)


def frozen_frames(failing_weight=None):
    weights = (REFERENCE_WEIGHT, *CANDIDATE_WEIGHTS)
    results = []
    subsets = []
    for weight in weights:
        increment = weight - REFERENCE_WEIGHT
        for season, multiplier in (
            (2022, 0.2),
            (2023, 2.0),
            (2024, 1.0),
            ("pooled", 1.0),
        ):
            results.append(
                {
                    "weight": weight,
                    "validation_season": season,
                    "brier_gain": 5e-6 + multiplier * increment * 1e-3,
                }
            )
        for season in (2022, 2023, 2024, "pooled"):
            for axis, subset in (("game_type", "F"), ("full_count", "full")):
                gain = 1e-6 + increment * 1e-3
                if weight == failing_weight and axis == "game_type":
                    gain = -2e-5
                subsets.append(
                    {
                        "weight": weight,
                        "validation_season": season,
                        "axis": axis,
                        "subset": subset,
                        "eligible_for_gate": True,
                        "gain_vs_original_champion": gain,
                    }
                )
    return pd.DataFrame(results), pd.DataFrame(subsets)


class FinalizeEt25UpperWeightTest(unittest.TestCase):
    def test_frozen_grid_and_promoted_champion_are_exact(self):
        self.assertEqual(REFERENCE_WEIGHT, 0.0175)
        self.assertEqual(CANDIDATE_WEIGHTS, (0.0200, 0.0225, 0.0250))
        self.assertEqual(CURRENT_CHAMPION_LB_BSS, 1017.0016684619)
        self.assertEqual(
            CURRENT_CHAMPION_SHA256,
            "7c884834651d38fa4e734e3a28846367a861fe406f05054398adba4ddbd825da",
        )

    def test_smallest_stable_existing_candidate_is_selected(self):
        comparison = build_comparison(*frozen_frames())
        self.assertEqual(comparison["weight"].tolist(), list(CANDIDATE_WEIGHTS))
        self.assertTrue(comparison["passed"].all())
        self.assertEqual(select_smallest_stable(comparison), 0.0200)

    def test_subset_floor_can_reject_smallest_candidate(self):
        comparison = build_comparison(*frozen_frames(failing_weight=0.0200))
        self.assertEqual(select_smallest_stable(comparison), 0.0225)

    def test_w0175_package_anchors_patch_to_w020(self):
        runtime = "EXTRA_WEIGHT = 0.0175\nCHAMPION_WEIGHT = 0.9825\n"
        self.assertEqual(
            patch_runtime_weights(runtime, 0.0200, source_weight=0.0175),
            "EXTRA_WEIGHT = 0.02\nCHAMPION_WEIGHT = 0.98\n",
        )
        script = (
            '"""Frozen endpoint: original champion first, then '
            '0.9825/0.0175 ET25 blend."""'
        )
        self.assertIn(
            "0.98/0.02 ET25 blend",
            patch_script_weight_doc(script, 0.0200, source_weight=0.0175),
        )


if __name__ == "__main__":
    unittest.main()
