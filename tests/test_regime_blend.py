import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_regime_blend import run
from src.regime_blend import (
    RegimeBlend,
    constrained_brier_weights,
    fit_regime_blend,
    member_diagnostics,
    regime_labels,
)


class ConstrainedBlendTest(unittest.TestCase):
    def test_solver_finds_simplex_boundary_optimum(self):
        target = np.asarray([0, 1, 0, 1], dtype="float64")
        predictions = np.asarray(
            [[0.1, 0.8], [0.9, 0.2], [0.2, 0.7], [0.8, 0.3]],
            dtype="float64",
        )
        weights = constrained_brier_weights(predictions, target)
        np.testing.assert_array_equal(weights, np.asarray([1.0, 0.0]))
        self.assertGreaterEqual(weights.min(), 0.0)
        self.assertEqual(float(weights.sum()), 1.0)

    def test_ridge_uses_anchor_when_members_are_identical(self):
        target = np.asarray([0, 1, 0, 1], dtype="float64")
        column = np.asarray([0.4, 0.6, 0.4, 0.6], dtype="float64")
        predictions = np.column_stack([column, column, column])
        anchor = np.asarray([0.2, 0.3, 0.5], dtype="float64")
        weights = constrained_brier_weights(
            predictions, target, anchor=anchor, ridge=1e-3
        )
        np.testing.assert_allclose(weights, anchor, rtol=0.0, atol=1e-10)

    def test_regime_blend_learns_complementary_cells(self):
        n = 800
        target = (np.arange(n) % 2).astype("float64")
        game_type = np.where(np.arange(n) < n // 2, "R", "F")
        good = 0.1 + 0.8 * target
        neutral = np.full(n, 0.5)
        first = np.where(game_type == "R", good, neutral)
        second = np.where(game_type == "F", good, neutral)
        predictions = np.column_stack([first, second])
        features = {"game_type": game_type}
        model = fit_regime_blend(
            predictions,
            target,
            features,
            ("first", "second"),
            "game_type",
            tau=0.0,
            min_samples=100,
        )
        self.assertGreater(model.local_weights["R"][0], 0.99)
        self.assertGreater(model.local_weights["F"][1], 0.99)
        prediction = model.predict(predictions, features)
        self.assertLess(np.mean((prediction - target) ** 2), 0.011)


class RowIndependenceTest(unittest.TestCase):
    def setUp(self):
        self.predictions = np.asarray(
            [
                [0.20, 0.40],
                [0.70, 0.50],
                [0.45, 0.55],
                [0.80, 0.60],
                [0.30, 0.25],
            ],
            dtype="float64",
        )
        self.features = {
            "game_type": np.asarray(["R", "F", "R", "F", "R"]),
        }
        self.model = RegimeBlend(
            member_names=("a", "b"),
            regime="game_type",
            global_weights=np.asarray([0.4, 0.6]),
            local_weights={
                "R": np.asarray([0.75, 0.25]),
                "F": np.asarray([0.20, 0.80]),
            },
            local_sample_sizes={"R": 3000, "F": 1000},
            tau=2000.0,
            min_samples=500,
            ridge=1e-6,
        )

    @staticmethod
    def _take(features, index):
        return {name: values[index] for name, values in features.items()}

    def test_same_row_is_exact_across_batch_shapes_and_orders(self):
        row = 2
        full = self.model.predict(self.predictions, self.features)[row]
        single = self.model.predict(
            self.predictions[[row]], self._take(self.features, [row])
        )[0]

        shuffle = np.asarray([4, 2, 0, 3, 1])
        shuffled = self.model.predict(
            self.predictions[shuffle], self._take(self.features, shuffle)
        )[int(np.flatnonzero(shuffle == row)[0])]

        subset = np.asarray([0, 2, 4])
        subset_value = self.model.predict(
            self.predictions[subset], self._take(self.features, subset)
        )[int(np.flatnonzero(subset == row)[0])]

        reverse = np.arange(len(self.predictions) - 1, -1, -1)
        reversed_value = self.model.predict(
            self.predictions[reverse], self._take(self.features, reverse)
        )[int(np.flatnonzero(reverse == row)[0])]

        self.assertEqual(full, single)
        self.assertEqual(full, shuffled)
        self.assertEqual(full, subset_value)
        self.assertEqual(full, reversed_value)

    def test_serialized_lookup_preserves_predictions_exactly(self):
        restored = RegimeBlend.from_dict(self.model.to_dict())
        expected = self.model.predict(self.predictions, self.features)
        actual = restored.predict(self.predictions, self.features)
        np.testing.assert_array_equal(actual, expected)

    def test_regime_label_is_row_local(self):
        features = {
            "game_type": np.asarray(["R", "F", "R"]),
            "balls_before": np.asarray([0, 3, 2]),
            "strikes_before": np.asarray([2, 0, 2]),
        }
        full = regime_labels(features, "game_type_count_bucket", 3)
        for row in range(3):
            single = regime_labels(self._take(features, [row]), "game_type_count_bucket", 1)
            self.assertEqual(full[row], single[0])


class DiagnosticsTest(unittest.TestCase):
    def test_required_segments_and_diversity_are_reported(self):
        target = np.asarray([0, 1, 0, 1], dtype="float64")
        predictions = {
            "a": np.asarray([0.1, 0.8, 0.3, 0.7]),
            "b": np.asarray([0.2, 0.6, 0.4, 0.9]),
        }
        features = {
            "game_type": np.asarray(["R", "R", "F", "F"]),
            "balls_before": np.asarray([0, 1, 2, 3]),
            "strikes_before": np.asarray([0, 1, 2, 2]),
            "pitcher_hand": np.asarray([1, 1, 2, 2]),
            "batter_hand": np.asarray([1, 2, 1, 2]),
            "pitcher_seen": np.asarray([True, False, True, False]),
        }
        report = member_diagnostics(predictions, target, features)
        self.assertEqual(report["n"], 4)
        self.assertIn("pitcher_seen", report["segments"])
        self.assertIn("count_state", report["segments"])
        self.assertIn("a_over_b", report["pairwise_error_advantage"])


class TemporalRunnerTest(unittest.TestCase):
    @staticmethod
    def _write_fold(path: Path, validation_target: np.ndarray) -> None:
        n = len(validation_target)
        inner_target = (np.arange(n) % 2).astype("float64")
        ours_inner = 0.15 + 0.70 * inner_target
        team_inner = 0.25 + 0.50 * inner_target
        ours_val = 0.15 + 0.70 * validation_target
        team_val = 0.25 + 0.50 * validation_target
        game_type = np.where(np.arange(n) % 3, "R", "F")
        balls = np.arange(n) % 4
        strikes = np.arange(n) % 3
        pitcher_hand = 1 + np.arange(n) % 2
        batter_hand = 1 + (np.arange(n) // 2) % 2
        seen = np.arange(n) % 5 != 0
        payload = {
            "y_inner": inner_target,
            "y_val": validation_target,
            "p_inner_ours_stage": ours_inner,
            "p_inner_team_stage": team_inner,
            "p_val_ours_stage": ours_val,
            "p_val_team_stage": team_val,
            "p_inner_champion_base": 0.4 * ours_inner + 0.6 * team_inner,
            "p_val_champion_base": 0.4 * ours_val + 0.6 * team_val,
            "form_offset_inner": np.zeros(n),
            "form_offset_val": np.zeros(n),
        }
        member_offsets = {
            "hgb": -0.02,
            "cat": 0.01,
            "nn": 0.03,
            "team": -0.01,
            "team_nn": 0.02,
        }
        for name, offset in member_offsets.items():
            payload[f"p_inner_{name}"] = np.clip(ours_inner + offset, 0.0, 1.0)
            payload[f"p_val_{name}"] = np.clip(ours_val + offset, 0.0, 1.0)
        feature_values = {
            "game_type": game_type,
            "balls_before": balls,
            "strikes_before": strikes,
            "pitcher_hand": pitcher_hand,
            "batter_hand": batter_hand,
            "pitcher_seen": seen,
        }
        for split in ("inner", "val"):
            for name, values in feature_values.items():
                payload[f"x_{split}_{name}"] = values
        np.savez_compressed(path, **payload)

    def test_validation_labels_cannot_change_inner_fitted_weights(self):
        with tempfile.TemporaryDirectory(prefix="regime_blend_test_") as tmp:
            root = Path(tmp)
            manifest = {
                "members": ["hgb", "cat", "nn", "team", "team_nn"],
                "blend_members": ["ours_stage", "team_stage"],
                "anchor_weights": [0.4, 0.6],
                "feature_columns": [
                    "game_type",
                    "balls_before",
                    "strikes_before",
                    "pitcher_hand",
                    "batter_hand",
                    "pitcher_seen",
                ],
                "folds": [],
            }
            original_targets = {}
            for inner, validation in ((2021, 2022), (2022, 2023), (2023, 2024)):
                target = (np.arange(120) % 2).astype("float64")
                original_targets[validation] = target
                fold_path = root / f"fold_{inner}.npz"
                self._write_fold(fold_path, target)
                manifest["folds"].append(
                    {
                        "inner_season": inner,
                        "validation_season": validation,
                        "path": fold_path.name,
                    }
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            first = run(
                manifest_path,
                root / "first",
                ridges=(0.0,),
                taus=(2000.0,),
                min_samples=20,
            )
            for fold_spec in manifest["folds"]:
                validation = fold_spec["validation_season"]
                self._write_fold(
                    root / fold_spec["path"], 1.0 - original_targets[validation]
                )
            second = run(
                manifest_path,
                root / "second",
                ridges=(0.0,),
                taus=(2000.0,),
                min_samples=20,
            )
            self.assertEqual(first["fitted_by_fold"], second["fitted_by_fold"])

            for fold_spec in manifest["folds"]:
                fold_path = root / fold_spec["path"]
                with np.load(fold_path, allow_pickle=False) as loaded:
                    payload = {key: loaded[key] for key in loaded.files}
                for key in tuple(payload):
                    if key == "y_val" or key.startswith("p_val_") or key.startswith("x_val_") or key == "form_offset_val":
                        payload[key] = payload[key][::2]
                np.savez_compressed(fold_path, **payload)
            third = run(
                manifest_path,
                root / "third",
                ridges=(0.0,),
                taus=(2000.0,),
                min_samples=20,
            )
            self.assertEqual(first["fitted_by_fold"], third["fitted_by_fold"])
            parsed = json.loads(
                (root / "first" / "regime_blend_results.json").read_text()
            )
            self.assertIs(parsed["rule_compliance"]["validation_labels_used_for_fit"], False)


if __name__ == "__main__":
    unittest.main()
