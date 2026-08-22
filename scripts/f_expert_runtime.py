"""Blend a fixed train-only prior into game_type=F rows, row independently."""
from __future__ import annotations

import numpy as np


def apply_f_expert(preds, test, spec):
    if not spec:
        return np.asarray(preds, dtype="float64")
    out = np.asarray(preds, dtype="float64").copy()
    active = test[spec.get("group_col", "game_type")].eq(
        spec.get("segment", "F")
    ).to_numpy()
    weight = float(spec["weight"])
    prior = float(spec["prior"])
    out[active] = (1.0 - weight) * out[active] + weight * prior
    return np.clip(out, 0.0, 1.0)
