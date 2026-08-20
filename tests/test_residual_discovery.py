import unittest

import numpy as np
import pandas as pd

from scripts.run_residual_discovery import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    MatrixEncoder,
    assign_bins,
    build_discovery_frame,
    quantile_edges,
)


def _oof(length):
    base = np.linspace(0.4, 0.6, length, dtype="float64")
    payload = {
        "y": np.asarray([0.0, 1.0][:length]),
        "x_pitcher_seen": np.asarray([1, 0][:length]),
        "coldstart_mask": np.asarray([0, 1][:length]),
    }
    for index, name in enumerate(
        ("hgb", "cat", "nn", "team", "team_nn", "ours_stage", "team_stage")
    ):
        payload[name] = np.clip(base + index * 0.001, 0.0, 1.0)
    payload["champion_base"] = base
    payload["pitcher_form_pred"] = base + 0.001
    payload["batter_form_pred"] = base + 0.002
    payload["coldstart_expert_pred"] = base - 0.003
    payload["champion_final"] = base + 0.0015
    return payload


class ResidualDiscoveryTest(unittest.TestCase):
    def test_discovery_features_are_exactly_row_local(self):
        train = pd.DataFrame(
            {
                "control_success": [0, 1],
                "balls_before": [1, 3],
                "strikes_before": [2, 1],
                "pitcher_hand": [1, 2],
                "batter_hand": [1, 1],
                "inning": [2, 8],
            }
        )
        full = build_discovery_frame(train, _oof(2))
        one = build_discovery_frame(train.iloc[[0]], {k: v[:1] for k, v in _oof(2).items()})
        derived = [
            "count_state",
            "same_hand",
            "inning_bucket",
            "member_std",
            "member_range",
            "form_adjustment",
            "cold_expert_gap",
            "signed_residual",
            "squared_error",
        ]
        for name in derived:
            self.assertEqual(full.loc[0, name], one.loc[0, name])

    def test_quantile_edges_depend_only_on_training_values(self):
        edges = quantile_edges(np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]))
        before = assign_bins(np.asarray([-100.0, 2.0, 100.0]), edges)
        after = assign_bins(np.asarray([-100.0, 2.0, 100.0, 1e12]), edges)[:3]
        np.testing.assert_array_equal(before, after)

    def test_encoder_ignores_labels_and_batch_context(self):
        data = {name: [0.0, 1.0] for name in MODEL_NUMERIC_FEATURES}
        data.update({name: ["a", "b"] for name in MODEL_CATEGORICAL_FEATURES})
        data["control_success"] = [0, 1]
        frame = pd.DataFrame(data)
        encoder = MatrixEncoder.fit(frame)
        full = encoder.transform(frame)
        reversed_labels = frame.copy()
        reversed_labels["control_success"] = [1, 0]
        np.testing.assert_array_equal(full, encoder.transform(reversed_labels))
        np.testing.assert_array_equal(full[[0]], encoder.transform(frame.iloc[[0]]))


if __name__ == "__main__":
    unittest.main()
