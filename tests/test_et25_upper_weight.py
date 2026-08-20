import unittest

import pandas as pd

from scripts.validate_et25_upper_weight import (
    MAX_SUBMISSION_FILENAME_LENGTH,
    REFERENCE_WEIGHT,
    WEIGHTS,
    candidate_filename,
    patch_runtime_weights,
    patch_script_weight_doc,
    select_upper_weight,
)


def decision_frames(pass_weights=(), severe_subset_weight=None):
    results = []
    deltas = []
    subsets = []
    for weight in WEIGHTS:
        is_pass = weight in pass_weights
        delta_by_season = {
            2022: 2e-7 if is_pass else -1e-7,
            2023: 8e-7 if is_pass else -1e-7,
            2024: 7e-7 if is_pass else -1e-7,
            "pooled": 7e-7 if is_pass else -1e-7,
        }
        if weight == REFERENCE_WEIGHT:
            delta_by_season = {key: 0.0 for key in delta_by_season}
        for season in (2022, 2023, 2024, "pooled"):
            results.append(
                {
                    "weight": weight,
                    "validation_season": season,
                    "brier_gain": 3e-6 + delta_by_season[season],
                }
            )
            deltas.append(
                {
                    "weight": weight,
                    "validation_season": season,
                    "brier_gain_delta_vs_w015": delta_by_season[season],
                }
            )
        subsets.append(
            {
                "weight": weight,
                "eligible_for_gate": True,
                "gain_vs_original_champion": 0.0,
                "gain_delta_vs_w015": (
                    -6e-6 if weight == severe_subset_weight else 0.0
                ),
                "passes_original_loss_floor": True,
                "passes_delta_floor_vs_w015": weight != severe_subset_weight,
            }
        )
    return pd.DataFrame(results), pd.DataFrame(deltas), pd.DataFrame(subsets)


class Et25UpperWeightTest(unittest.TestCase):
    def test_grid_is_exact_and_reference_is_w015(self):
        self.assertEqual(WEIGHTS, (0.015, 0.0175, 0.020, 0.0225, 0.025))
        self.assertEqual(REFERENCE_WEIGHT, 0.015)

    def test_filename_encoding_and_length(self):
        expected = {
            0.0175: "sub_et25_w0175.zip",
            0.0200: "sub_et25_w020.zip",
            0.0225: "sub_et25_w0225.zip",
            0.0250: "sub_et25_w025.zip",
        }
        for weight, name in expected.items():
            self.assertEqual(candidate_filename(weight), name)
            self.assertLessEqual(len(name), MAX_SUBMISSION_FILENAME_LENGTH)

    def test_smallest_stable_upper_weight_is_selected(self):
        frames = decision_frames(pass_weights=(0.0175, 0.020, 0.0225, 0.025))
        verdict, selected, _ = select_upper_weight(*frames)
        self.assertEqual(
            verdict, "A. NEW UPPER WEIGHT READY FOR ONE LB SUBMISSION"
        )
        self.assertEqual(selected, 0.0175)

    def test_severe_subset_degradation_rejects_candidate(self):
        frames = decision_frames(
            pass_weights=(0.0175, 0.020), severe_subset_weight=0.0175
        )
        _, selected, _ = select_upper_weight(*frames)
        self.assertEqual(selected, 0.020)

    def test_package_patches_current_w015_anchors(self):
        runtime = "EXTRA_WEIGHT = 0.015\nCHAMPION_WEIGHT = 0.985\n"
        self.assertEqual(
            patch_runtime_weights(runtime, 0.0175),
            "EXTRA_WEIGHT = 0.0175\nCHAMPION_WEIGHT = 0.9825\n",
        )
        script = '"""Frozen endpoint: original champion first, then 0.985/0.015 ET25 blend."""'
        self.assertIn(
            "0.9825/0.0175 ET25 blend",
            patch_script_weight_doc(script, 0.0175),
        )


if __name__ == "__main__":
    unittest.main()
