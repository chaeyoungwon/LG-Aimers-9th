import unittest

import pandas as pd

from scripts.finalize_et25_w0225 import (
    CANDIDATE_NAME,
    CANDIDATE_WEIGHT,
    CURRENT_CHAMPION_LB_BSS,
    CURRENT_CHAMPION_SHA256,
    CURRENT_WEIGHT,
    build_direct_comparison,
    build_subset_comparison,
    safety_gate,
)
from scripts.validate_et25_upper_weight import (
    MAX_SUBMISSION_FILENAME_LENGTH,
    patch_runtime_weights,
    patch_script_weight_doc,
)


def frozen_frames(candidate_subset_gain=-5e-6):
    results = []
    for weight, increment in ((CURRENT_WEIGHT, 0.0), (CANDIDATE_WEIGHT, 8e-7)):
        for season in (2022, 2023, 2024, "pooled"):
            results.append(
                {
                    "weight": weight,
                    "validation_season": season,
                    "brier_gain": 5e-6 + increment,
                }
            )
    subsets = []
    for weight, gain in (
        (CURRENT_WEIGHT, -4.5e-6),
        (CANDIDATE_WEIGHT, candidate_subset_gain),
    ):
        for season in (2022, 2023, 2024, "pooled"):
            for axis, subset in (
                ("game_type", "R"),
                ("game_type", "F"),
                ("pitcher_seen", "seen"),
                ("pitcher_seen", "unseen"),
                ("full_count", "full"),
                ("full_count", "non_full"),
                ("pitcher_form", "positive"),
                ("pitcher_form", "negative"),
            ):
                subsets.append(
                    {
                        "weight": weight,
                        "validation_season": season,
                        "axis": axis,
                        "subset": subset,
                        "n": 10_000,
                        "gain_vs_original_champion": gain,
                        "eligible_for_gate": True,
                    }
                )
    return pd.DataFrame(results), pd.DataFrame(subsets)


class FinalizeEt25W0225Test(unittest.TestCase):
    def test_fixed_candidate_and_current_champion(self):
        self.assertEqual(CURRENT_WEIGHT, 0.0200)
        self.assertEqual(CANDIDATE_WEIGHT, 0.0225)
        self.assertEqual(CANDIDATE_NAME, "sub_et25_w0225.zip")
        self.assertLessEqual(len(CANDIDATE_NAME), MAX_SUBMISSION_FILENAME_LENGTH)
        self.assertEqual(CURRENT_CHAMPION_LB_BSS, 1017.0233029621)
        self.assertEqual(
            CURRENT_CHAMPION_SHA256,
            "316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf",
        )

    def test_existing_candidate_passes_direct_gate(self):
        results, raw_subsets = frozen_frames()
        direct = build_direct_comparison(results)
        subsets = build_subset_comparison(raw_subsets)
        passed, diagnostics = safety_gate(direct, subsets)
        self.assertTrue(passed)
        self.assertAlmostEqual(
            diagnostics["worst_major_subset_delta_vs_w020"], -0.5e-6
        )

    def test_original_subset_floor_blocks_package(self):
        results, raw_subsets = frozen_frames(candidate_subset_gain=-1.1e-5)
        passed, diagnostics = safety_gate(
            build_direct_comparison(results), build_subset_comparison(raw_subsets)
        )
        self.assertFalse(passed)
        self.assertFalse(diagnostics["conditions"]["major_subset_gain_floor"])

    def test_w020_anchors_patch_to_direct_w0225_formula(self):
        runtime = "EXTRA_WEIGHT = 0.02\nCHAMPION_WEIGHT = 0.98\n"
        self.assertEqual(
            patch_runtime_weights(runtime, 0.0225, source_weight=0.0200),
            "EXTRA_WEIGHT = 0.0225\nCHAMPION_WEIGHT = 0.9775\n",
        )
        script = (
            '"""Frozen endpoint: original champion first, then '
            '0.98/0.02 ET25 blend."""'
        )
        self.assertIn(
            "0.9775/0.0225 ET25 blend",
            patch_script_weight_doc(script, 0.0225, source_weight=0.0200),
        )


if __name__ == "__main__":
    unittest.main()
