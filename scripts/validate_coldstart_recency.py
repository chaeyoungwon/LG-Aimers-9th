"""Validate fixed 0.85 yearly decay for the cold-start expert."""
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
DECAY = 0.85


def baseline(cache_path, inner):
    z = np.load(cache_path)
    pi = {n: z[f"p_inner_{n}"].astype("float64") for n in tl.MODEL_ORDER}
    pv = {n: z[f"p_val_{n}"].astype("float64") for n in tl.MODEL_ORDER}
    y = inner.control_success.to_numpy(dtype="float64")
    cal = [tl.fit_platt(tl.group_raw(pi, g), y) for g in tl.ENSEMBLE_GROUPS]
    return tl.blend(pv, calib=cal)


def train_predict(train, val, decay):
    tr, va = add_features(train), add_features(val)
    features = [c for c in tr.columns if c not in DROP]
    cats = [c for c in CAT_COLS if c in features]
    maps = build_category_maps(tr, cats)
    tr, va = apply_category_maps(tr, cats, maps), apply_category_maps(va, cats, maps)
    params = dict(PARAMS, seed=42, num_leaves=31, min_data_in_leaf=800,
                  num_threads=6)
    weight = None
    if decay is not None:
        weight = decay ** (int(train.season.max()) - train.season.to_numpy())
    ds = lgb.Dataset(tr[features], label=tr.control_success, weight=weight,
                     categorical_feature=cats, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=236)
    return model.predict(va[features])


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    pooled = []
    for train_max, val_season in SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = val.game_type.eq("R").to_numpy() & ~val.pitcher_id.isin(known).to_numpy()
        p0 = baseline(CACHE / f"fold_{train_max}.npz", inner)
        raw = train_predict(train, val, decay=None)
        rec = train_predict(train, val, decay=DECAY)
        y = val.control_success.to_numpy(dtype="float64")
        p_raw, p_rec = 0.5*p0 + 0.5*raw, 0.5*p0 + 0.5*rec
        d = (p_raw[cold]-y[cold])**2 - (p_rec[cold]-y[cold])**2
        pooled.append(d)
        print(f"{train_max}->{val_season} cold={cold.sum():,} "
              f"raw={np.mean((p_raw[cold]-y[cold])**2):.9f} "
              f"decay85={np.mean((p_rec[cold]-y[cold])**2):.9f} "
              f"gain={d.mean():+.3e}")
    d = np.concatenate(pooled)
    se = d.std(ddof=1)/np.sqrt(len(d))
    print(f"pooled n={len(d):,} gain={d.mean():+.3e} z={d.mean()/se:+.2f}")


if __name__ == "__main__":
    main()
