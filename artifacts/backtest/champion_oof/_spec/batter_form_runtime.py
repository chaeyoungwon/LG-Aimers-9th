"""Row-independent batter season-to-date correction for submission inference."""
import json
import os

import numpy as np
import pandas as pd


def apply_batter_form(preds, test, model_dir):
    with open(os.path.join(model_dir, "batter_form.json"), encoding="utf-8") as f:
        spec = json.load(f)
    ids = np.asarray(spec["batter_ids"], dtype="int64")
    n0_table = np.asarray(spec["n0"], dtype="float64")
    s0_table = np.asarray(spec["s0"], dtype="float64")
    bid_float = pd.Series(test["batter_id"]).to_numpy(dtype="float64")
    missing = ~np.isfinite(bid_float)
    bid = np.where(missing, -1, bid_float).astype("int64")
    pos = np.searchsorted(ids, bid)
    clipped = np.clip(pos, 0, len(ids) - 1)
    seen = (pos < len(ids)) & (ids[clipped] == bid) & ~missing

    n = test["asof_batter_n"].to_numpy(dtype="float64")
    rate = test["asof_batter_success_rate"].to_numpy(dtype="float64")
    success = np.rint(n * np.where(np.isfinite(rate), rate, 0.0))
    n0, s0 = n0_table[clipped], s0_table[clipped]
    p0 = s0 / n0
    n_season, s_season = n - n0, success - s0
    active = (seen & np.isfinite(n_season) & (n_season > 0)
              & test["game_type"].eq(spec["segment"]).to_numpy())
    form = np.zeros(len(test), dtype="float64")
    m = float(spec["m"])
    form[active] = ((s_season[active] + m * p0[active])
                    / (n_season[active] + m) - p0[active])
    return np.clip(np.asarray(preds, dtype="float64")
                   + float(spec["alpha"]) * form, 0.0, 1.0)
