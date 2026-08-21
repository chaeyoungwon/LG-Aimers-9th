import unittest

import numpy as np
import pandas as pd

from scripts.build_tm_candidate import (
    EXPECTED_FILENAME,
    MAX_FILENAME_LENGTH,
    mapping_signature,
    patch_champion_script,
)
from scripts.trackman_runtime import (
    CHAMPION_WEIGHT,
    MATCHED_OUTPUT_DECIMALS,
    TRACKMAN_WEIGHT,
    attach_trackman_features,
)


class TrackManProductionTest(unittest.TestCase):
    def test_filename_and_weights_are_frozen(self):
        self.assertEqual(EXPECTED_FILENAME, "sub_tm10.zip")
        self.assertLessEqual(len(EXPECTED_FILENAME), MAX_FILENAME_LENGTH)
        self.assertEqual((CHAMPION_WEIGHT, TRACKMAN_WEIGHT), (0.90, 0.10))
        self.assertEqual(MATCHED_OUTPUT_DECIMALS, 14)

    def test_lookup_is_row_local_and_unmatched_is_explicit(self):
        metadata = {
            "trackman_feature_columns": [
                "tm_history_n",
                "tm_matched",
                "tm_confidence",
            ]
        }
        pitcher_ids = np.array([10, 20])
        values = np.array([[100.0, 1.0, 0.7], [200.0, 1.0, 0.8]])
        rows = pd.DataFrame({"pitcher_id": [20, 99, 10]}, index=[4, 5, 6])
        full, matched = attach_trackman_features(
            rows, metadata, pitcher_ids, values
        )
        reverse, reverse_matched = attach_trackman_features(
            rows.iloc[::-1], metadata, pitcher_ids, values
        )
        pd.testing.assert_frame_equal(full, reverse.loc[rows.index])
        np.testing.assert_array_equal(matched, reverse_matched[::-1])
        self.assertEqual(full.loc[5, "tm_matched"], 0.0)
        self.assertEqual(full.loc[5, "tm_history_n"], 0.0)

    def test_mapping_signature_is_order_independent(self):
        mapping = pd.DataFrame(
            {
                "main_pitcher_id": [2, 1],
                "tm_pitcher_id": [20, 10],
                "confidence": ["HIGH", "LOW"],
            }
        )
        self.assertEqual(mapping_signature(mapping), mapping_signature(mapping.iloc[::-1]))

    def test_champion_patch_wraps_current_endpoint_once(self):
        source = (
            "from extratrees_runtime import apply_extratrees_blend\n"
            "def predict(test, model, meta, verbose=True):\n"
            "    return apply_extratrees_blend([], test, './model')\n"
            "\ndef main():\n"
            "    pass\n"
        )
        patched = patch_champion_script(source)
        self.assertEqual(patched.count("def predict_current_champion"), 1)
        self.assertEqual(patched.count("def predict(test"), 1)
        self.assertIn("apply_trackman_blend(champion", patched)


if __name__ == "__main__":
    unittest.main()
