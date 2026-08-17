"""Apply a train-OOF residual lookup independently to each inference row."""
import json
import os

import numpy as np


def apply_residual_cells(preds, test, model_dir):
    with open(os.path.join(model_dir, "residual_cells.json"), encoding="utf-8") as f:
        spec = json.load(f)
    keys = (test["balls_before"].astype(str) + "-"
            + test["strikes_before"].astype(str) + "|"
            + test["pitcher_hand"].astype(str) + "-"
            + test["batter_hand"].astype(str))
    adjustment = keys.map(spec["adjustments"]).fillna(0.0).to_numpy(dtype="float64")
    active = test["game_type"].eq(spec["segment"]).to_numpy()
    adjustment = np.where(active, adjustment, 0.0)
    return np.clip(np.asarray(preds, dtype="float64") + adjustment, 0.0, 1.0)
