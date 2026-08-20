import json
import unittest
import zipfile

import numpy as np
import pandas as pd

from scripts.build_extratrees_candidate import (
    BOOTSTRAP,
    CHAMPION,
    CHAMPION_WEIGHT,
    EXTRA_WEIGHT,
    MAX_FEATURES,
    MIN_SAMPLES_LEAF,
    N_ESTIMATORS,
    RANDOM_STATE,
    frozen_model,
    patch_champion_script,
    validate_recipe,
)
from scripts.extratrees_runtime import (
    apply_extratrees_blend,
    build_hgb52_features,
    fit_preprocessor,
    transform_features,
)


class ExtraTreesProductionTest(unittest.TestCase):
    def test_frozen_weights_and_recipe(self):
        self.assertEqual(CHAMPION_WEIGHT, 0.99)
        self.assertEqual(EXTRA_WEIGHT, 0.01)
        params = frozen_model().get_params()
        self.assertEqual(params["n_estimators"], N_ESTIMATORS)
        self.assertEqual(params["random_state"], RANDOM_STATE)
        self.assertEqual(params["min_samples_leaf"], MIN_SAMPLES_LEAF)
        self.assertEqual(params["max_features"], MAX_FEATURES)
        self.assertEqual(params["bootstrap"], BOOTSTRAP)
        self.assertIsNone(params["max_depth"])

    def test_feature_builder_is_exactly_id_free(self):
        frame = pd.DataFrame(
            {
                "row_id": ["A"],
                "control_success": [1],
                "pitcher_id": [10],
                "batter_id": [20],
                "balls_before": [3],
                "strikes_before": [2],
                "pitcher_hand": [1],
                "batter_hand": [2],
                "asof_pitcher_strike_rate": [0.5],
                "asof_pitcher_ball_rate": [0.2],
                "asof_pitcher_success_rate": [0.6],
                "asof_pitcher_middle_rate": [0.1],
                "asof_pitcher_prev5_game_success_rate": [0.55],
                "asof_pitcher_breaking_rate": [0.3],
                "asof_pitcher_offspeed_rate": [0.2],
                "asof_pitcher_fastball_rate": [0.5],
            }
        )
        features = build_hgb52_features(frame, {})
        self.assertNotIn("row_id", features)
        self.assertNotIn("control_success", features)
        self.assertNotIn("pitcher_id", features)
        self.assertNotIn("batter_id", features)
        self.assertEqual(features.loc[0, "count_state"], 11)
        self.assertEqual(features.loc[0, "hand_combo"], "1-2")

    def test_preprocessing_uses_training_only_constants(self):
        train = pd.DataFrame(
            {
                "category": pd.Categorical(["A", "B", None]),
                "numeric": [1.0, 3.0, np.nan],
            }
        )
        validation = pd.DataFrame(
            {
                "category": pd.Categorical(
                    ["FUTURE"], categories=["A", "B", "FUTURE"]
                ),
                "numeric": [999.0],
            }
        )
        metadata = fit_preprocessor(train)
        self.assertEqual(metadata["categorical_levels"]["category"], ["A", "B"])
        self.assertEqual(metadata["numeric_medians"]["numeric"], 2.0)
        transformed = transform_features(validation, metadata)
        self.assertEqual(transformed[0, 0], -1.0)
        self.assertEqual(transformed[0, 1], 999.0)

    def test_candidate_script_wraps_original_champion_once(self):
        with zipfile.ZipFile(CHAMPION) as archive:
            original = archive.read("script.py").decode("utf-8")
        patched = patch_champion_script(original)
        self.assertEqual(patched.count("def predict_champion("), 1)
        self.assertEqual(patched.count("def predict(test, model, meta, verbose=True):"), 1)
        self.assertIn("apply_extratrees_blend(champion, test, \"./model\")", patched)
        self.assertLess(
            patched.index("champion = predict_champion"),
            patched.index("apply_extratrees_blend(champion"),
        )

    def test_recipe_matches_frozen_oof_summary(self):
        with zipfile.ZipFile(CHAMPION) as archive:
            champion_meta = json.loads(archive.read("model/meta.json"))
        with open("artifacts/extratrees_complement/summary.json", encoding="utf-8") as handle:
            reference = json.load(handle)
        self.assertEqual(validate_recipe(champion_meta, reference), [])


if __name__ == "__main__":
    unittest.main()
