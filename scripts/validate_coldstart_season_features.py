"""Validate explicit, row-local baseball season-stage features."""
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_coldstart_expert as v

sys.path.insert(0, str(ROOT))
from src.train_base import (CAT_COLS, PARAMS, add_features,
                            apply_category_maps, build_category_maps)

DROP = {"row_id", "control_success", "pitcher_id", "batter_id"}


def add_season_features(df):
    x = add_features(df)
    month = x.game_month.to_numpy(dtype="float64")
    progress = np.clip((month - 3.0) / 7.0, 0.0, 1.0)
    x["season_progress"] = progress
    x["early_season"] = (month <= 5).astype("int8")
    x["late_season"] = (month >= 8).astype("int8")
    pn = x.asof_pitcher_n.to_numpy(dtype="float64")
    mixn = x.asof_pitcher_pitchmix_n.to_numpy(dtype="float64")
    reliability = pn / (pn + 75.0)
    mix_reliability = mixn / (mixn + 75.0)
    recent = (x.asof_pitcher_prev1_game_success_rate
              - x.asof_pitcher_prev5_game_success_rate).fillna(0.0)
    middle = (x.asof_pitcher_prev1_game_middle_rate
              - x.asof_pitcher_prev5_game_middle_rate).fillna(0.0)
    x["season_x_pitcher_reliability"] = progress * reliability
    x["season_x_pitchmix_reliability"] = progress * mix_reliability
    x["season_x_recent_control"] = progress * recent
    x["late_x_recent_control"] = (month >= 8) * recent
    x["late_x_recent_middle"] = (month >= 8) * middle
    return x


def train_predict(train, val, engineered):
    tr = add_season_features(train) if engineered else add_features(train)
    va = add_season_features(val) if engineered else add_features(val)
    features = [c for c in tr.columns if c not in DROP]
    cats = [c for c in CAT_COLS if c in features]
    maps = build_category_maps(tr, cats)
    tr = apply_category_maps(tr, cats, maps)
    va = apply_category_maps(va, cats, maps)
    params = dict(PARAMS, seed=42, num_leaves=31, min_data_in_leaf=800,
                  num_threads=6)
    ds = lgb.Dataset(tr[features], label=tr.control_success,
                     categorical_feature=cats, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=236)
    return model.predict(va[features])


def main():
    df = pd.read_csv(v.DATA, encoding="utf-8-sig")
    pooled = []
    for train_max, val_season in v.SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = val.game_type.eq("R").to_numpy() & ~val.pitcher_id.isin(known).to_numpy()
        p0 = v.baseline(v.CACHE / f"fold_{train_max}.npz", inner, val)
        raw = train_predict(train, val, False)
        eng = train_predict(train, val, True)
        y = val.control_success.to_numpy(dtype="float64")
        p_raw, p_eng = .5*p0 + .5*raw, .5*p0 + .5*eng
        d = (p_raw[cold]-y[cold])**2 - (p_eng[cold]-y[cold])**2
        pooled.append(d)
        print(f"{train_max}->{val_season} cold={cold.sum():,} "
              f"raw={np.mean((p_raw[cold]-y[cold])**2):.9f} "
              f"season={np.mean((p_eng[cold]-y[cold])**2):.9f} "
              f"gain={d.mean():+.3e}")
    d = np.concatenate(pooled)
    se = d.std(ddof=1)/np.sqrt(len(d))
    print(f"pooled n={len(d):,} gain={d.mean():+.3e} z={d.mean()/se:+.2f}")


if __name__ == "__main__":
    main()
