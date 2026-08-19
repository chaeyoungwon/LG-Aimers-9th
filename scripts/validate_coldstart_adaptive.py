"""Forward validation of row-local experience-adaptive cold-start blending."""
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
# Fixed before validation: expert influence decays as the row's own official
# as-of sample count gives the base ensemble increasingly useful pitcher form.
SCHEDULES = {
    "champion_w50": ((np.inf, .50),),
    "gentle": ((20, .65), (100, .55), (np.inf, .40)),
    "medium": ((20, .75), (100, .55), (np.inf, .30)),
    "strong": ((20, .85), (100, .50), (np.inf, .20)),
}


def baseline(cache_path, inner, val):
    z = np.load(cache_path)
    p_inner = {n: z[f"p_inner_{n}"].astype("float64") for n in tl.MODEL_ORDER}
    p_val = {n: z[f"p_val_{n}"].astype("float64") for n in tl.MODEL_ORDER}
    y_inner = inner.control_success.to_numpy(dtype="float64")
    calib = [tl.fit_platt(tl.group_raw(p_inner, g), y_inner)
             for g in tl.ENSEMBLE_GROUPS]
    return tl.blend(p_val, calib=calib)


def train_expert(train, val):
    train, val = add_features(train), add_features(val)
    features = [c for c in train.columns if c not in DROP]
    cats = [c for c in CAT_COLS if c in features]
    maps = build_category_maps(train, cats)
    tr = apply_category_maps(train, cats, maps)
    va = apply_category_maps(val, cats, maps)
    params = dict(PARAMS, seed=42, num_leaves=31, min_data_in_leaf=800,
                  num_threads=6)
    ds = lgb.Dataset(tr[features], label=tr.control_success,
                     categorical_feature=cats, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=236).predict(va[features])


def weights(n, schedule):
    out = np.empty(len(n), dtype="float64")
    lower = -np.inf
    for upper, weight in schedule:
        hit = (n > lower) & (n <= upper)
        out[hit] = weight
        lower = upper
    return out


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    pooled = {name: [] for name in SCHEDULES}
    diagnostics = {b: [] for b in ((-1, 20), (20, 100), (100, np.inf))}
    for train_max, val_season in SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = val.game_type.eq("R").to_numpy() & ~val.pitcher_id.isin(known).to_numpy()
        p0 = baseline(CACHE / f"fold_{train_max}.npz", inner, val)
        pe = train_expert(train, val)
        y = val.control_success.to_numpy(dtype="float64")
        n = val.asof_pitcher_n.to_numpy(dtype="float64")
        ref = .5 * p0 + .5 * pe
        print(f"\n{train_max}->{val_season} cold={cold.sum():,}")
        for name, schedule in SCHEDULES.items():
            w = weights(n, schedule)
            p = (1 - w) * p0 + w * pe
            d = (ref[cold] - y[cold]) ** 2 - (p[cold] - y[cold]) ** 2
            pooled[name].append(d)
            print(f"  {name:12s} vs w50={d.mean():+.3e}")
        # Diagnostic only; shows whether the predeclared monotone premise holds.
        for lo, hi in diagnostics:
            mask = cold & (n > lo) & (n <= hi)
            d = ((p0[mask] - y[mask]) ** 2 - (pe[mask] - y[mask]) ** 2)
            diagnostics[(lo, hi)].append(d)
            print(f"  n=({lo},{hi}] rows={mask.sum():,} expert-vs-base={d.mean():+.3e}")

    print("\npooled vs champion w50")
    for name, parts in pooled.items():
        d = np.concatenate(parts)
        se = d.std(ddof=1) / np.sqrt(len(d))
        z = d.mean() / se if se else 0.0
        print(f"  {name:12s} n={len(d):,} gain={d.mean():+.3e} z={z:+.2f}")
    print("\npooled expert-vs-base diagnostics")
    for bounds, parts in diagnostics.items():
        d = np.concatenate(parts)
        print(f"  n={bounds} rows={len(d):,} gain={d.mean():+.3e}")


if __name__ == "__main__":
    main()
