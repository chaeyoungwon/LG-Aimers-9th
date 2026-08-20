import copy
import unittest

import numpy as np

from scripts.validate_pitcher_form_shape import (
    PRIMARY_CAP,
    apply_asymmetric_correction,
    exact_row_independence_check,
    fit_asymmetric_beta,
    identity_check,
    nested_beta,
)


class PitcherFormShapeTest(unittest.TestCase):
    def test_zero_beta_is_bit_exact_champion_identity(self):
        champion = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0])
        form = np.asarray([-0.1, -0.01, 0.0, 0.01, 0.1])
        candidate, delta, raw = apply_asymmetric_correction(
            champion, form, np.zeros(2), PRIMARY_CAP
        )
        np.testing.assert_array_equal(candidate, champion)
        np.testing.assert_array_equal(delta, np.zeros_like(delta))
        np.testing.assert_array_equal(raw, np.zeros_like(raw))
        seasons = {
            2022: {
                "champion_final": champion,
                "pitcher_form_adjustment": form,
            }
        }
        self.assertTrue(identity_check(seasons))

    def test_prediction_is_exact_across_batch_shapes_and_orders(self):
        champion = np.linspace(0.2, 0.8, 101)
        form = np.linspace(-0.04, 0.04, 101)
        beta = np.asarray([-0.1, 0.4])
        self.assertTrue(
            exact_row_independence_check(champion, form, beta, PRIMARY_CAP)
        )

    def test_validation_labels_cannot_change_nested_beta(self):
        seasons = {
            year: {
                "pitcher_form_adjustment": np.asarray([-0.02, 0.0, 0.03]),
                "champion_final": np.asarray([0.4, 0.5, 0.6]),
                "y": np.asarray([0.0, 1.0, 1.0]),
            }
            for year in (2022, 2023, 2024)
        }
        beta_2023, _ = nested_beta(2023, seasons)
        beta_2024, _ = nested_beta(2024, seasons)
        flipped = copy.deepcopy(seasons)
        flipped[2023]["y"] = 1.0 - flipped[2023]["y"]
        flipped[2024]["y"] = 1.0 - flipped[2024]["y"]
        beta_2023_after, _ = nested_beta(2023, flipped)
        beta_2024_after, _ = nested_beta(2024, flipped)
        np.testing.assert_array_equal(beta_2023, beta_2023_after)
        self.assertFalse(np.array_equal(beta_2024, beta_2024_after))
        flipped_only_validation = copy.deepcopy(seasons)
        flipped_only_validation[2024]["y"] = 1.0 - flipped_only_validation[2024]["y"]
        beta_2024_validation_flip, _ = nested_beta(2024, flipped_only_validation)
        np.testing.assert_array_equal(beta_2024, beta_2024_validation_flip)

    def test_ridge_fit_has_no_intercept(self):
        form = np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02])
        residual = np.full(5, 0.25)
        beta = fit_asymmetric_beta(form, residual)
        candidate, delta, _ = apply_asymmetric_correction(
            np.full(5, 0.5), form, beta, PRIMARY_CAP
        )
        self.assertEqual(delta[2], 0.0)
        self.assertEqual(candidate[2], 0.5)


if __name__ == "__main__":
    unittest.main()
