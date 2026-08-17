"""Standalone, row-independent season-form inference used in the submission."""
import json
import os

import numpy as np
import pandas as pd


def apply_season_form(preds, test, model_dir):
    with open(os.path.join(model_dir, "season_form.json"), encoding="utf-8") as f:
        spec = json.load(f)
    ids = np.asarray(spec["pitcher_ids"], dtype="int64")
    n0_table = np.asarray(spec["n0"], dtype="float64")
    s0_table = np.asarray(spec["s0"], dtype="float64")
    pid_float = pd.Series(test["pitcher_id"]).to_numpy(dtype="float64")
    missing_pid = ~np.isfinite(pid_float)
    pid = np.where(missing_pid, -1, pid_float).astype("int64")
    pos = np.searchsorted(ids, pid)
    clipped = np.clip(pos, 0, len(ids) - 1)
    seen = ((pos < len(ids)) & (ids[clipped] == pid) & ~missing_pid)
    n = test["asof_pitcher_n"].to_numpy(dtype="float64")
    rate = test["asof_pitcher_success_rate"].to_numpy(dtype="float64")
    success = np.rint(n * np.where(np.isfinite(rate), rate, 0.0))
    n0, s0 = n0_table[clipped], s0_table[clipped]
    p0 = s0 / n0
    n_season, s_season = n - n0, success - s0
    active = (seen & np.isfinite(n_season) & (n_season > 0)
              & test["game_type"].eq(spec["segment"]).to_numpy())
    adjustment = np.zeros(len(test), dtype="float64")
    adjustment[active] = (
        (s_season[active] + float(spec["m"]) * p0[active])
        / (n_season[active] + float(spec["m"])) - p0[active]
        - float(spec["mu"]))
    return np.clip(np.asarray(preds, dtype="float64")
                   + float(spec["alpha"]) * adjustment, 0.0, 1.0)
