"""Train the rule-compliant normalized player-embedding MLP candidate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.nn_embed import (IdVocab, TabularPrep, prep_meta, state_dict_arrays,
                          train_embed_nn)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "nn_candidate" / "model"
DROP = ["pitcher_id", "batter_id"]
GROUPS = ("matchup", "count", "rates")


def build_nn_features(df):
    """Legacy champion features; every transformation uses only its own row."""
    exclude = {"row_id", "control_success", *DROP}
    x = df[[c for c in df.columns if c not in exclude]].copy()
    x["count_state"] = pd.Categorical(
        df["balls_before"] * 3 + df["strikes_before"], categories=list(range(12)))
    x["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype("int8")
    combo = df["pitcher_hand"].astype(str) + "-" + df["batter_hand"].astype(str)
    x["hand_combo"] = pd.Categorical(combo, categories=["1-1", "1-2", "2-1", "2-2"])
    x["strike_minus_ball"] = (df["asof_pitcher_strike_rate"]
                               - df["asof_pitcher_ball_rate"]).astype("float32")
    x["success_minus_middle"] = (df["asof_pitcher_success_rate"]
                                  - df["asof_pitcher_middle_rate"]).astype("float32")
    x["prev_vs_career"] = (df["asof_pitcher_prev5_game_success_rate"]
                            - df["asof_pitcher_success_rate"]).astype("float32")
    second = np.fmax(df["asof_pitcher_breaking_rate"],
                     df["asof_pitcher_offspeed_rate"])
    x["fastball_dominance"] = (df["asof_pitcher_fastball_rate"] - second).astype("float32")
    return x


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    X = build_nn_features(df)
    prep = TabularPrep().fit(X)
    pv = IdVocab.fit(df["pitcher_id"])
    bv = IdVocab.fit(df["batter_id"])
    Xn = prep.transform(X)
    pid = pv.transform(df["pitcher_id"])
    bid = bv.transform(df["batter_id"])
    y = df["control_success"].to_numpy(dtype="float32")

    params = dict(emb_dim=8, id_dropout=0.08, hidden=(128, 64), dropout=0.1,
                  lr=1e-3, batch_size=4096, max_epochs=30, patience=3,
                  val_frac=0.1, seed=42)
    model, info = train_embed_nn(
        Xn, pid, bid, y, n_pitcher=pv.size, n_batter=bv.size,
        device="cpu", verbose=True, **params)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "nn_model.npz", **state_dict_arrays(model))
    meta = {
        "groups": list(GROUPS),
        "drop_cols": DROP,
        "prep": prep_meta(prep),
        "vocab": {
            "pitcher_col": "pitcher_id", "batter_col": "batter_id",
            "pitcher_ids": [int(v) for v in pv.ids_],
            "batter_ids": [int(v) for v in bv.ids_],
        },
        "arch": {
            "n_features": prep.n_features, "n_pitcher": pv.size,
            "n_batter": bv.size, "emb_dim": params["emb_dim"],
            "hidden": list(params["hidden"]), "dropout": params["dropout"],
        },
        "runtime": {"batch_size": 256, "device": "cpu"},
        "train_params": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in params.items()},
        "train_info": info,
        "blend_weight": 0.10,
    }
    with open(OUT / "nn_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
