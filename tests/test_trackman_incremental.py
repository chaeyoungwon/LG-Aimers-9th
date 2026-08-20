import copy
import unittest

import numpy as np
import pandas as pd

from scripts.validate_trackman_incremental import (
    OFFSET_CAP,
    PROFILE_FEATURES,
    TM_KEY,
    TM_NUMERIC_COLUMNS,
    TRAIN_KEY,
    aggregate_trackman_profiles,
    apply_tm_correction,
    attach_trackman_profiles,
    exact_lookup_row_independence,
    nested_offset_beta,
)


def synthetic_trackman():
    return pd.DataFrame(
        {
            "season": [2020, 2021, 2022, 2023],
            TM_KEY: [10, 10, 10, 20],
            "pitch_type_group": ["fastball", "breaking", "offspeed", "fastball"],
            **{
                column: np.asarray([1.0, 3.0, 999.0, 5.0])
                for column in TM_NUMERIC_COLUMNS
            },
        }
    )


class TrackmanIncrementalTest(unittest.TestCase):
    def test_cutoff_and_future_season_exclusion(self):
        trackman = synthetic_trackman()
        profile = aggregate_trackman_profiles(trackman, 2021, "career_prior")
        self.assertEqual(profile[TM_KEY].tolist(), [10])
        self.assertEqual(profile.loc[0, "tm_pitch_count"], 2.0)
        self.assertEqual(profile.loc[0, "rel_speed_mean"], 2.0)
        changed_future = trackman.copy()
        changed_future.loc[changed_future["season"].gt(2021), "rel_speed"] = -9999.0
        profile_after = aggregate_trackman_profiles(
            changed_future, 2021, "career_prior"
        )
        pd.testing.assert_frame_equal(profile, profile_after, check_exact=True)

    def test_recent_weights_are_fixed_and_cutoff_safe(self):
        profile = aggregate_trackman_profiles(
            synthetic_trackman(), 2022, "recent_weighted"
        )
        # (1*0.25 + 3*0.5 + 999*1.0) / 1.75
        expected = (0.25 + 1.5 + 999.0) / 1.75
        self.assertAlmostEqual(profile.loc[0, "rel_speed_mean"], expected)
        self.assertEqual(profile.loc[0, "tm_last_season"], 2022.0)

    def test_unseen_pitcher_fallback_is_explicit(self):
        profile = aggregate_trackman_profiles(
            synthetic_trackman(), 2021, "career_prior"
        )
        rows = pd.DataFrame({TRAIN_KEY: [10, 999]})
        attached = attach_trackman_profiles(rows, profile)
        self.assertEqual(attached.loc[0, "tm_available"], 1.0)
        self.assertEqual(attached.loc[1, "tm_available"], 0.0)
        self.assertEqual(attached.loc[1, "tm_pitch_count"], 0.0)
        self.assertTrue(np.isnan(attached.loc[1, "rel_speed_mean"]))

    def test_lookup_is_exact_across_single_shuffle_subset_reverse(self):
        profile = aggregate_trackman_profiles(
            synthetic_trackman(), 2023, "career_prior"
        )
        rows = pd.DataFrame({TRAIN_KEY: [10, 20, 999, 10, 20, 999, 10]})
        self.assertTrue(exact_lookup_row_independence(rows, profile))

    def test_zero_correction_is_bit_exact_champion_identity(self):
        champion = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0])
        design = np.arange(15, dtype="float64").reshape(5, 3)
        candidate, delta = apply_tm_correction(
            champion, design, np.zeros(3), OFFSET_CAP
        )
        np.testing.assert_array_equal(candidate, champion)
        np.testing.assert_array_equal(delta, np.zeros_like(delta))

    def test_validation_labels_cannot_change_nested_beta(self):
        designs = {
            2022: np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            2023: np.asarray([[2.0, 0.0], [0.0, 2.0]]),
            2024: np.asarray([[3.0, 0.0], [0.0, 3.0]]),
        }
        targets = {
            year: np.asarray([0.0, 1.0]) for year in (2022, 2023, 2024)
        }
        champions = {
            year: np.asarray([0.4, 0.6]) for year in (2022, 2023, 2024)
        }
        beta_2023 = nested_offset_beta(2023, designs, targets, champions)
        beta_2024 = nested_offset_beta(2024, designs, targets, champions)
        flipped = copy.deepcopy(targets)
        flipped[2023] = 1.0 - flipped[2023]
        flipped[2024] = 1.0 - flipped[2024]
        np.testing.assert_array_equal(
            beta_2023,
            nested_offset_beta(2023, designs, flipped, champions),
        )
        flipped_2024_only = copy.deepcopy(targets)
        flipped_2024_only[2024] = 1.0 - flipped_2024_only[2024]
        np.testing.assert_array_equal(
            beta_2024,
            nested_offset_beta(2024, designs, flipped_2024_only, champions),
        )

    def test_nan_inf_rejected_and_prediction_bounds_enforced(self):
        champion = np.asarray([0.01, 0.99])
        design = np.asarray([[1.0], [1.0]])
        candidate, delta = apply_tm_correction(
            champion, design, np.asarray([100.0]), OFFSET_CAP
        )
        self.assertTrue(np.isfinite(candidate).all())
        self.assertTrue(np.all((candidate >= 0.0) & (candidate <= 1.0)))
        self.assertTrue(np.all(np.abs(delta) <= OFFSET_CAP))
        with self.assertRaises(ValueError):
            apply_tm_correction(
                champion, np.asarray([[np.nan], [1.0]]), np.asarray([1.0]), OFFSET_CAP
            )
        with self.assertRaises(ValueError):
            apply_tm_correction(
                champion, np.asarray([[np.inf], [1.0]]), np.asarray([1.0]), OFFSET_CAP
            )

    def test_profile_schema_is_low_dimensional_and_complete(self):
        profile = aggregate_trackman_profiles(
            synthetic_trackman(), 2021, "career_prior"
        )
        self.assertEqual(profile.columns.tolist(), [TM_KEY, *PROFILE_FEATURES])
        self.assertLessEqual(len(PROFILE_FEATURES), 25)
        numeric = profile[list(PROFILE_FEATURES)].to_numpy(dtype="float64")
        self.assertFalse(np.isinf(numeric).any())


if __name__ == "__main__":
    unittest.main()
