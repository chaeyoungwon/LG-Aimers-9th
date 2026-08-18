"""Row-independent ID-free expert for pitchers with no train-period R history."""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd


def _features(df, spec):
    df = df.copy()
    df["count_state"] = (df["balls_before"] * 3 + df["strikes_before"]).astype("int8")
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype("int8")
    df["late_inning"] = (df["inning"] >= 7).astype("int8")
    df["two_strikes"] = (df["strikes_before"] == 2).astype("int8")
    df["three_balls"] = (df["balls_before"] == 3).astype("int8")
    df["pitcher_rate_gap_1_5"] = (df["asof_pitcher_prev1_game_success_rate"]
                                    - df["asof_pitcher_prev5_game_success_rate"])
    df["pitcher_rate_gap_career_5"] = (df["asof_pitcher_prev5_game_success_rate"]
                                         - df["asof_pitcher_success_rate"])
    for col in ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"):
        df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))
    for col in spec["cat_cols"]:
        df[col] = df[col].astype(str).map(spec["category_maps"][col]).astype("float32")
    return df[spec["feature_cols"]]


def apply_coldstart_expert(preds, test, model_dir):
    with open(os.path.join(model_dir, "coldstart_meta.json"), encoding="utf-8") as f:
        spec = json.load(f)
    known = np.asarray(spec["known_r_pitcher_ids"], dtype="int64")
    pid = test["pitcher_id"].to_numpy(dtype="int64")
    cold = test["game_type"].eq("R").to_numpy() & ~np.isin(pid, known)
    out = np.asarray(preds, dtype="float64").copy()
    if cold.any():
        model = lgb.Booster(model_file=os.path.join(model_dir, spec["model_file"]))
        expert = model.predict(_features(test.loc[cold], spec))
        w = float(spec["weight"])
        out[cold] = (1.0 - w) * out[cold] + w * expert
    return np.clip(out, 0.0, 1.0)
