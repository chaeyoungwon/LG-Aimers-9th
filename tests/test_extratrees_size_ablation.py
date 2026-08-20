import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from scripts.extratrees_runtime import CHAMPION_WEIGHT, EXTRA_WEIGHT
from scripts.validate_extratrees_size_ablation import (
    MIN_REFERENCE_GAIN_RETENTION,
    TREE_COUNTS,
    prefix_model,
    prefix_predictions,
    select_smallest_candidate,
)


class ExtraTreesSizeAblationTest(unittest.TestCase):
    def test_only_tree_count_changes_and_weight_is_fixed(self):
        self.assertEqual(TREE_COUNTS, (25, 50, 100, 200, 300))
        self.assertEqual(CHAMPION_WEIGHT, 0.99)
        self.assertEqual(EXTRA_WEIGHT, 0.01)
        self.assertEqual(MIN_REFERENCE_GAIN_RETENTION, 0.5)

    def test_prefixes_use_first_estimators_without_mutating_source(self):
        rng = np.random.RandomState(7)
        features = rng.normal(size=(500, 4)).astype("float32")
        target = (features[:, 0] > 0).astype("int8")
        model = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=8,
            max_features="sqrt",
            bootstrap=False,
            n_jobs=1,
            random_state=42,
        ).fit(features, target)
        original_ids = [id(tree) for tree in model.estimators_]
        prefix = prefix_model(model, 25)
        self.assertEqual(len(prefix.estimators_), 25)
        self.assertEqual(
            [id(tree) for tree in prefix.estimators_], original_ids[:25]
        )
        predictions = prefix_predictions(model, features)
        self.assertEqual(set(predictions), set(TREE_COUNTS))
        self.assertEqual(len(model.estimators_), 300)
        self.assertEqual(model.n_estimators, 300)
        np.testing.assert_allclose(
            predictions[25], prefix.predict_proba(features)[:, 1], atol=0.0
        )

    def test_selection_chooses_smallest_candidate_passing_every_gate(self):
        rows = []
        gains = {
            25: (1e-6, 1e-6, -1e-7, 1e-6),
            50: (2e-6, 2e-6, 2e-6, 2e-6),
            100: (3e-6, 3e-6, 3e-6, 3e-6),
            200: (4e-6, 4e-6, 4e-6, 4e-6),
            300: (4e-6, 4e-6, 4e-6, 4e-6),
        }
        for count, values in gains.items():
            for season, gain in zip((2022, 2023, 2024, "pooled"), values):
                rows.append(
                    {
                        "tree_count": count,
                        "validation_season": season,
                        "brier_gain": gain,
                    }
                )
        subsets = pd.DataFrame(
            {
                "tree_count": np.repeat(TREE_COUNTS, 2),
                "eligible_for_gate": True,
                "passes_loss_floor": True,
                "brier_gain": 0.0,
            }
        )
        selected, decisions = select_smallest_candidate(
            pd.DataFrame(rows), subsets
        )
        self.assertEqual(selected, 50)
        self.assertFalse(decisions[0]["passed"])
        self.assertTrue(decisions[1]["passed"])

    def test_subset_loss_rejects_otherwise_passing_prefix(self):
        rows = []
        for count in TREE_COUNTS:
            for season in (2022, 2023, 2024, "pooled"):
                rows.append(
                    {
                        "tree_count": count,
                        "validation_season": season,
                        "brier_gain": 4e-6,
                    }
                )
        subsets = pd.DataFrame(
            {
                "tree_count": np.repeat(TREE_COUNTS, 1),
                "eligible_for_gate": True,
                "passes_loss_floor": True,
                "brier_gain": 0.0,
            }
        )
        subsets.loc[subsets["tree_count"] == 25, "passes_loss_floor"] = False
        subsets.loc[subsets["tree_count"] == 25, "brier_gain"] = -2e-5
        selected, _ = select_smallest_candidate(pd.DataFrame(rows), subsets)
        self.assertEqual(selected, 50)


if __name__ == "__main__":
    unittest.main()
