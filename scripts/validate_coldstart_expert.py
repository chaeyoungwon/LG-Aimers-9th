"""Forward validation for an ID-free expert on pitchers without prior R history."""
from pathlib import Path
import importlib.util
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OTHER = ROOT.parent / "LG-AIMERS_9TH"
DATA = ROOT.parent / "open" / "data" / "train.csv"
CACHE = ROOT / "artifacts" / "validation_hetero" / "exp23_cache"
sys.path.insert(0, str(ROOT))
from src.train_base import (CAT_COLS, PARAMS, add_features,
                            apply_category_maps, build_category_maps)

_spec = importlib.util.spec_from_file_location(
    "team_lgbm_external", OTHER / "src" / "team_lgbm.py")
tl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tl)

SPLITS = ((2021, 2022), (2022, 2023), (2023, 2024))
DROP = {"row_id", "control_success", "pitcher_id", "batter_id"}
BLEND_WEIGHTS = (0.0, 0.25, 0.42, 0.5, 0.75, 1.0)
SEEDS = (42, 43, 44)


def baseline(cache_path, inner, val):
    z = np.load(cache_path)
    p_inner = {n: z[f"p_inner_{n}"].astype("float64")
               for n in tl.MODEL_ORDER}
    p_val = {n: z[f"p_val_{n}"].astype("float64")
             for n in tl.MODEL_ORDER}
    y_inner = inner.control_success.to_numpy(dtype="float64")
    calib = [tl.fit_platt(tl.group_raw(p_inner, g), y_inner)
             for g in tl.ENSEMBLE_GROUPS]
    return tl.blend(p_val, calib=calib)


def train_expert(train, val, seed):
    train = add_features(train)
    val = add_features(val)
    features = [c for c in train.columns if c not in DROP]
    cats = [c for c in CAT_COLS if c in features]
    maps = build_category_maps(train, cats)
    tr = apply_category_maps(train, cats, maps)
    va = apply_category_maps(val, cats, maps)
    params = dict(PARAMS, seed=seed, num_leaves=31, min_data_in_leaf=800,
                  num_threads=6)
    ds = lgb.Dataset(tr[features], label=tr.control_success,
                     categorical_feature=cats, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=236)
    return model.predict(va[features])


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    pooled = {w: [] for w in BLEND_WEIGHTS}
    seed_ensemble = []
    print("cold-start = validation R pitcher with no R row at or before cutoff")
    for train_max, val_season in SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = val.game_type.eq("R").to_numpy() & ~val.pitcher_id.isin(known).to_numpy()
        p0 = baseline(CACHE / f"fold_{train_max}.npz", inner, val)
        experts = [train_expert(train, val, seed) for seed in SEEDS]
        pe = experts[0]
        pe3 = np.mean(experts, axis=0)
        y = val.control_success.to_numpy(dtype="float64")
        print(f"\n{train_max}->{val_season}: cold rows={cold.sum():,}, "
              f"pitchers={val.loc[cold, 'pitcher_id'].nunique()}")
        for w in BLEND_WEIGHTS:
            p = (1.0 - w) * p0 + w * pe
            delta = (p0[cold] - y[cold]) ** 2 - (p[cold] - y[cold]) ** 2
            pooled[w].append(delta)
            print(f"  expert_w={w:.2f} brier={np.mean((p[cold]-y[cold])**2):.9f} "
                  f"gain={delta.mean():+.3e}")
        p_single = 0.5 * p0 + 0.5 * pe
        p_seed3 = 0.5 * p0 + 0.5 * pe3
        delta3 = ((p_single[cold] - y[cold]) ** 2
                  - (p_seed3[cold] - y[cold]) ** 2)
        seed_ensemble.append(delta3)
        print(f"  seed3 vs seed42 at w50: gain={delta3.mean():+.3e}")
    print("\npooled cold-start")
    for w in BLEND_WEIGHTS:
        d = np.concatenate(pooled[w])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  expert_w={w:.2f} n={len(d):,} gain={d.mean():+.3e} "
              f"z={d.mean()/se:+.2f}")
    d = np.concatenate(seed_ensemble)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\nseed3 vs seed42 at w50: n={len(d):,} gain={d.mean():+.3e} "
          f"z={d.mean()/se:+.2f}")


if __name__ == "__main__":
    main()
