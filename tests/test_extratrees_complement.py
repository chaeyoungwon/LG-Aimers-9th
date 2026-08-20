import unittest

import numpy as np
import pandas as pd

from scripts.validate_extratrees_complement import (
    BLEND_WEIGHTS,
    DEFAULT_N_ESTIMATORS,
    MAX_FEATURES,
    MIN_SAMPLES_LEAF,
    RANDOM_STATE,
    blend_rows,
    complementarity_gate,
    complementarity_rows,
    fit_feature_encoding,
    make_extra_trees,
    make_random_forest,
    passing_weight_summary,
    transform_features,
    validate_feature_contract,
)


class ExtraTreesComplementTest(unittest.TestCase):
    def test_single_preregistered_model_configuration(self):
        model = make_extra_trees()
        params = model.get_params()
        self.assertEqual(DEFAULT_N_ESTIMATORS, 300)
        self.assertEqual(params["n_estimators"], 300)
        self.assertIsNone(params["max_depth"])
        self.assertEqual(params["min_samples_leaf"], MIN_SAMPLES_LEAF)
        self.assertEqual(params["max_features"], MAX_FEATURES)
        self.assertFalse(params["bootstrap"])
        self.assertEqual(params["random_state"], RANDOM_STATE)

    def test_random_forest_comparison_changes_only_bagging_split_structure(self):
        extra = make_extra_trees().get_params()
        forest = make_random_forest().get_params()
        for name in (
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "n_jobs",
            "random_state",
        ):
            self.assertEqual(extra[name], forest[name])
        self.assertFalse(extra["bootstrap"])
        self.assertTrue(forest["bootstrap"])

    def test_feature_contract_is_id_free_and_exact(self):
        columns = ["season", "game_type", "asof_pitcher_n"]
        validate_feature_contract(columns, columns)
        for forbidden in ("pitcher_id", "batter_id", "control_success", "row_id"):
            with self.assertRaises(ValueError):
                validate_feature_contract([*columns, forbidden], [*columns, forbidden])
        with self.assertRaises(ValueError):
            validate_feature_contract(columns, list(reversed(columns)))

    def test_encoding_statistics_use_training_rows_only(self):
        train = pd.DataFrame(
            {
                "category": pd.Categorical(["A", "B", None]),
                "numeric": [1.0, 3.0, np.nan],
            }
        )
        validation = pd.DataFrame(
            {
                "category": pd.Categorical(["FUTURE"], categories=["A", "B", "FUTURE"]),
                "numeric": [999999.0],
            }
        )
        encoding = fit_feature_encoding(train)
        self.assertEqual(encoding.categorical_levels["category"], ("A", "B"))
        self.assertEqual(encoding.numeric_medians["numeric"], 2.0)
        transformed = transform_features(validation, encoding)
        self.assertEqual(transformed[0, 0], -1.0)
        self.assertEqual(transformed[0, 1], 999999.0)

    def test_complementarity_gate_requires_every_fold(self):
        correlations = pd.DataFrame(
            {
                "validation_season": [2022, 2023, 2024, "pooled"],
                "corr_extra_champion": [0.7, 0.8, 0.9, 0.8],
            }
        )
        records = []
        target = np.asarray([0.0, 1.0] * 50)
        champion = np.full(100, 0.5)
        extra = np.where(target == 1.0, 0.6, 0.4)
        for season in (2022, 2023, 2024):
            records.extend(complementarity_rows(season, target, extra, champion))
        table = pd.DataFrame(records)
        passed, reasons = complementarity_gate(correlations, table)
        self.assertTrue(passed, reasons)
        table.loc[
            table["validation_season"].eq(2023)
            & table["scope"].eq("champion_error_top10pct"),
            "extra_win_rate",
        ] = 0.5
        passed, reasons = complementarity_gate(correlations, table)
        self.assertFalse(passed)
        self.assertTrue(reasons)

    def test_blend_weights_are_fixed_and_gain_sign_is_champion_minus_candidate(self):
        folds = {}
        for season in (2022, 2023, 2024):
            folds[season] = {
                "target": np.asarray([0.0, 1.0]),
                "champion": np.asarray([0.6, 0.4]),
                "extra": np.asarray([0.0, 1.0]),
            }
        results = blend_rows(folds)
        self.assertEqual(sorted(results["weight"].unique()), list(BLEND_WEIGHTS))
        self.assertTrue((results["brier_gain"] > 0.0).all())

    def test_stable_weight_requires_latest_pooled_and_cross_fold_safety(self):
        rows = []
        for weight in BLEND_WEIGHTS:
            for season, gain in ((2022, 1e-5), (2023, 1e-5), (2024, 1e-5), ("pooled", 1e-5)):
                rows.append(
                    {
                        "validation_season": season,
                        "weight": weight,
                        "n": 10000,
                        "champion_brier": 0.25,
                        "blend_brier": 0.25 - gain,
                        "brier_gain": gain,
                        "evaluated": True,
                        "evaluation_status": "test",
                    }
                )
        blends = pd.DataFrame(rows)
        subsets = pd.DataFrame(
            {
                "weight": np.repeat(BLEND_WEIGHTS, 2),
                "n": 10000,
                "brier_gain": 1e-6,
                "row_fraction": 0.5,
                "positive_gain_share": 0.5,
            }
        )
        self.assertEqual(len(passing_weight_summary(blends, subsets)), 5)
        blends.loc[
            np.isclose(blends["weight"], 0.01)
            & blends["validation_season"].astype(str).eq("2024"),
            "brier_gain",
        ] = 0.0
        passing = passing_weight_summary(blends, subsets)
        self.assertNotIn(0.01, [row["weight"] for row in passing])


if __name__ == "__main__":
    unittest.main()
