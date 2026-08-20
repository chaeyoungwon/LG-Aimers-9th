import unittest

import numpy as np
import pandas as pd

from scripts.validate_temporal_recency import (
    DECAYS,
    STRONG_GAIN,
    WINDOWS,
    model_audit,
    recency_weights,
    scheme_decision,
    window_mask,
)


class TemporalRecencyTest(unittest.TestCase):
    def test_fixed_grids(self):
        self.assertEqual(DECAYS, (1.0, 0.85, 0.70, 0.50))
        self.assertEqual(WINDOWS, (None, 4, 3, 2))

    def test_exponential_weight_uses_age_minus_one(self):
        seasons = np.array([2020, 2021, 2022, 2023])
        actual = recency_weights(seasons, 2024, 0.70)
        expected = np.array([0.70**3, 0.70**2, 0.70, 1.0])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_window_is_relative_to_validation_season(self):
        seasons = np.array([2019, 2020, 2021, 2022, 2023])
        np.testing.assert_array_equal(
            window_mask(seasons, 2024, 2),
            np.array([False, False, False, True, True]),
        )
        self.assertTrue(window_mask(seasons, 2024, None).all())

    def test_future_or_validation_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            recency_weights(np.array([2023, 2024]), 2024, 0.85)
        with self.assertRaises(ValueError):
            window_mask(np.array([2023, 2024]), 2024, 2)

    def test_strong_gate_requires_2023_2024_and_pooled(self):
        rows = []
        for season, gain in (
            (2022, -1e-6),
            (2023, 3e-5),
            (2024, 2e-5),
            ("pooled", STRONG_GAIN),
        ):
            rows.append(
                {
                    "family": "decay",
                    "scheme": "decay_070",
                    "validation_season": season,
                    "brier_gain": gain,
                }
            )
        decision = scheme_decision(pd.DataFrame(rows), "decay", "decay_070")
        self.assertTrue(decision["stable"])
        self.assertTrue(decision["strong"])
        self.assertEqual(decision["worst_fold"], "2022")

    def test_only_team_lightgbm_has_existing_season_weights(self):
        audit = model_audit()
        weighted = audit[audit["sample_weight_usage"]]
        self.assertEqual(weighted["model"].tolist(), ["team LightGBM"])
        self.assertFalse(audit["recent_only"].any())


if __name__ == "__main__":
    unittest.main()
