import unittest

import numpy as np
import pandas as pd

from scripts.validate_model_diversity import (
    BLEND_WEIGHTS,
    CATASTROPHIC_STANDALONE_FLOOR,
    CLEAR_STANDALONE_GAIN,
    FT_RECIPE,
    MAX_DIVERSITY_CORRELATION,
    MIN_CHAMPION_TOP10_WIN_RATE,
    TABM_RECIPE,
    XGB_RECIPE,
    complementarity_gate,
    fit_tabm_encoding,
    make_ft_model,
    make_tabm_model,
    make_xgb,
    transform_tabm,
)


def row(season, gain, corr=0.95, top10=0.50):
    return {
        "validation_season": season,
        "standalone_gain_vs_champion": gain,
        "corr_model_champion": corr,
        "champion_top10_error_model_win_rate": top10,
    }


class ModelDiversityTest(unittest.TestCase):
    def test_xgb_recipe_is_exact_and_single_seed(self):
        self.assertEqual(
            XGB_RECIPE,
            {
                "n_estimators": 600,
                "max_depth": 6,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_lambda": 5,
                "reg_alpha": 0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "tree_method": "hist",
            },
        )
        model = make_xgb(n_jobs=1)
        self.assertEqual(model.get_params()["n_estimators"], 600)
        self.assertEqual(model.get_params()["tree_method"], "hist")
        self.assertEqual(BLEND_WEIGHTS, (0.005, 0.010, 0.020, 0.050))

    def test_path_a_requires_clear_win_and_latest_safety(self):
        rows = [
            row(2022, CLEAR_STANDALONE_GAIN),
            row(2023, CATASTROPHIC_STANDALONE_FLOOR),
            row(2024, -1e-5),
            row("pooled", -1e-5),
        ]
        gate = complementarity_gate(rows)
        self.assertTrue(gate["path_a_passed"])
        rows[1]["standalone_gain_vs_champion"] = (
            CATASTROPHIC_STANDALONE_FLOOR - 1e-8
        )
        self.assertFalse(complementarity_gate(rows)["path_a_passed"])

    def test_path_b_requires_low_correlation_and_both_latest_top10_rates(self):
        rows = [
            row(2022, -1e-4, top10=0.70),
            row(2023, -1e-4, top10=MIN_CHAMPION_TOP10_WIN_RATE),
            row(2024, -1e-4, top10=MIN_CHAMPION_TOP10_WIN_RATE + 0.05),
            row("pooled", -1e-4, corr=MAX_DIVERSITY_CORRELATION, top10=0.63),
        ]
        gate = complementarity_gate(rows)
        self.assertTrue(gate["path_b_passed"])
        rows[-1]["corr_model_champion"] = MAX_DIVERSITY_CORRELATION + 1e-8
        self.assertFalse(complementarity_gate(rows)["path_b_passed"])

    def test_no_gate_means_no_implicit_weight_or_architecture_search(self):
        rows = [
            row(2022, -1e-3, corr=0.99, top10=0.40),
            row(2023, -1e-3, corr=0.99, top10=0.40),
            row(2024, -1e-3, corr=0.99, top10=0.40),
            row("pooled", -1e-3, corr=0.99, top10=0.40),
        ]
        self.assertFalse(complementarity_gate(rows)["passed"])

    def test_tabm_small_recipe_and_probability_head_count_are_fixed(self):
        self.assertEqual(
            TABM_RECIPE,
            {
                "arch_type": "tabm",
                "k": 8,
                "n_blocks": 2,
                "d_block": 128,
                "dropout": 0.1,
                "learning_rate": 0.002,
                "weight_decay": 0.0003,
                "batch_size": 4096,
                "eval_batch_size": 16384,
                "epochs": 8,
                "random_state": 42,
            },
        )
        frame = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, np.nan],
                "category": pd.Categorical(["a", "b", "a"]),
            }
        )
        encoding = fit_tabm_encoding(frame)
        self.assertEqual(make_tabm_model(encoding).k, 8)

    def test_tabm_preprocessing_is_train_only_and_unknown_safe(self):
        training = pd.DataFrame(
            {
                "numeric": [1.0, 3.0, np.nan],
                "category": pd.Categorical(["a", "b", "a"]),
            }
        )
        encoding = fit_tabm_encoding(training)
        validation = pd.DataFrame(
            {
                "numeric": [np.nan, 1000.0],
                "category": pd.Categorical(["new", "a"]),
            }
        )
        numeric, categorical = transform_tabm(validation, encoding)
        self.assertTrue(np.isfinite(numeric).all())
        self.assertEqual(categorical[0, 0], 2)
        self.assertEqual(categorical[1, 0], 0)

    def test_ft_transformer_uses_one_small_official_default_backbone(self):
        self.assertEqual(
            FT_RECIPE,
            {
                "n_blocks": 2,
                "batch_size": 2048,
                "eval_batch_size": 4096,
                "epochs": 6,
                "learning_rate": 0.0001,
                "weight_decay": 0.00001,
                "random_state": 42,
            },
        )
        frame = pd.DataFrame(
            {
                "numeric": [1.0, 2.0],
                "category": pd.Categorical(["a", "b"]),
            }
        )
        model = make_ft_model(fit_tabm_encoding(frame))
        self.assertEqual(len(model.backbone.blocks), 2)
        self.assertIsNotNone(model.backbone.blocks[0]["attention"])


if __name__ == "__main__":
    unittest.main()
