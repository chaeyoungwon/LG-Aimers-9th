import unittest

import numpy as np
import pandas as pd

from scripts.validate_et25_weight import (
    MAX_SUBMISSION_FILENAME_LENGTH,
    REFERENCE_WEIGHT,
    WEIGHTS,
    blend_original_champion,
    candidate_filename,
    patch_script_weight_doc,
    patch_runtime_weights,
    select_weight,
)


def selection_frames(pooled_by_weight, fold_delta_by_weight=None):
    fold_delta_by_weight = fold_delta_by_weight or {}
    reference_pooled = pooled_by_weight[REFERENCE_WEIGHT]
    results = []
    deltas = []
    subsets = []
    for weight in WEIGHTS:
        fold_deltas = fold_delta_by_weight.get(weight, (0.0, 0.0, 0.0))
        for season, fold_delta in zip((2022, 2023, 2024), fold_deltas):
            gain = 3e-6 + fold_delta
            results.append(
                {"weight": weight, "validation_season": season, "brier_gain": gain}
            )
            deltas.append(
                {
                    "weight": weight,
                    "validation_season": season,
                    "brier_gain_delta_vs_w010": fold_delta,
                }
            )
        pooled = pooled_by_weight[weight]
        results.append(
            {"weight": weight, "validation_season": "pooled", "brier_gain": pooled}
        )
        deltas.append(
            {
                "weight": weight,
                "validation_season": "pooled",
                "brier_gain_delta_vs_w010": pooled - reference_pooled,
            }
        )
        subsets.append(
            {
                "weight": weight,
                "eligible_for_gate": True,
                "passes_loss_floor": True,
                "brier_gain": 0.0,
            }
        )
    return pd.DataFrame(results), pd.DataFrame(subsets), pd.DataFrame(deltas)


class Et25WeightTest(unittest.TestCase):
    def test_weights_and_composition_are_frozen(self):
        self.assertEqual(WEIGHTS, (0.005, 0.008, 0.010, 0.012, 0.015))
        champion = np.asarray([0.2, 0.8])
        extra = np.asarray([0.7, 0.3])
        actual = blend_original_champion(champion, extra, 0.012)
        np.testing.assert_allclose(actual, 0.988 * champion + 0.012 * extra)

    def test_filename_is_short_and_encodes_weight(self):
        self.assertEqual(candidate_filename(0.008), "sub_et25_w008.zip")
        self.assertEqual(candidate_filename(0.012), "sub_et25_w012.zip")
        self.assertLessEqual(
            len(candidate_filename(0.015)), MAX_SUBMISSION_FILENAME_LENGTH
        )
        with self.assertRaises(ValueError):
            candidate_filename(0.010)

    def test_runtime_patch_changes_only_frozen_constants(self):
        source = "EXTRA_WEIGHT = 0.01\nCHAMPION_WEIGHT = 0.99\n"
        self.assertEqual(
            patch_runtime_weights(source, 0.012),
            "EXTRA_WEIGHT = 0.012\nCHAMPION_WEIGHT = 0.988\n",
        )
        script = '"""Frozen endpoint: original champion first, then 99/1 ExtraTrees blend."""'
        self.assertIn(
            "0.988/0.012 ET25 blend",
            patch_script_weight_doc(script, 0.012),
        )

    def test_clear_cross_fold_improvement_selects_new_weight(self):
        pooled = {weight: 3e-6 for weight in WEIGHTS}
        pooled[0.010] = 3.5e-6
        pooled[0.012] = 4.2e-6
        frames = selection_frames(
            pooled,
            {0.012: (2e-7, 3e-7, -1e-7)},
        )
        verdict, selected, _ = select_weight(*frames)
        self.assertEqual(verdict, "A. NEW WEIGHT READY FOR ONE LB SUBMISSION")
        self.assertEqual(selected, 0.012)

    def test_small_or_concentrated_gain_is_flat(self):
        pooled = {weight: 3e-6 for weight in WEIGHTS}
        pooled[0.010] = 3.5e-6
        pooled[0.012] = 3.8e-6
        frames = selection_frames(
            pooled,
            {0.012: (8e-7, -7e-7, 2e-7)},
        )
        verdict, selected, _ = select_weight(*frames)
        self.assertEqual(verdict, "C. OOF SIGNAL TOO FLAT — STOP WEIGHT AXIS")
        self.assertIsNone(selected)

    def test_reference_numeric_best_keeps_current_champion(self):
        pooled = {weight: 3e-6 for weight in WEIGHTS}
        pooled[0.010] = 4e-6
        frames = selection_frames(pooled)
        verdict, selected, _ = select_weight(*frames)
        self.assertEqual(verdict, "B. w=.010 REMAINS BEST — KEEP CURRENT CHAMPION")
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
