import unittest

import numpy as np

from scripts.report_regime_blend_weights import (
    COLDSTART_WEIGHT,
    _best_method_per_family,
    _endpoint_metrics,
)


class RegimeWeightReportTest(unittest.TestCase):
    def test_selects_largest_pooled_gain_within_each_family(self):
        summaries = [
            {"method": "global_rho=0", "pooled_brier_gain": 9.0},
            {
                "method": "regime=count_state,rho=0,tau=2000",
                "pooled_brier_gain": 0.1,
            },
            {
                "method": "regime=count_state,rho=1e-5,tau=10000",
                "pooled_brier_gain": 0.2,
            },
            {
                "method": "regime=game_type,rho=0,tau=2000",
                "pooled_brier_gain": -0.1,
            },
        ]
        selected = _best_method_per_family(summaries)
        self.assertEqual(
            selected["count_state"]["method"],
            "regime=count_state,rho=1e-5,tau=10000",
        )
        self.assertEqual(
            selected["game_type"]["method"],
            "regime=game_type,rho=0,tau=2000",
        )
        self.assertNotIn("global", selected)

    def test_endpoint_composition_has_zero_gain_at_champion_parity(self):
        champion_base = np.asarray([0.2, 0.8], dtype="float64")
        form_offset = np.asarray([0.1, -0.1], dtype="float64")
        champion_form = champion_base + form_offset
        cold_expert = np.asarray([0.5, 0.5], dtype="float64")
        cold_mask = np.asarray([True, False])
        champion_final = champion_form.copy()
        champion_final[cold_mask] = (
            (1.0 - COLDSTART_WEIGHT) * champion_final[cold_mask]
            + COLDSTART_WEIGHT * cold_expert[cold_mask]
        )
        artifact = {
            "y_val": np.asarray([0.0, 1.0]),
            "p_val_champion_base": champion_base,
            "p_val_champion_form": champion_form,
            "p_val_champion_final": champion_final,
            "form_offset_val": form_offset,
            "p_val_coldstart_expert": cold_expert,
            "coldstart_mask_val": cold_mask,
        }
        metrics = _endpoint_metrics(artifact, champion_base)
        for stage in ("base", "form", "final"):
            self.assertEqual(metrics[stage]["brier_gain"], 0.0)


if __name__ == "__main__":
    unittest.main()
