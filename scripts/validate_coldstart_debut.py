"""Compare the all-R expert with an expert trained only on debut R seasons."""
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


def baseline(path, inner):
    z = np.load(path)
    pi = {n: z[f"p_inner_{n}"].astype("float64") for n in tl.MODEL_ORDER}
    pv = {n: z[f"p_val_{n}"].astype("float64") for n in tl.MODEL_ORDER}
    y = inner.control_success.to_numpy(dtype="float64")
    cal = [tl.fit_platt(tl.group_raw(pi, g), y) for g in tl.ENSEMBLE_GROUPS]
    return tl.blend(pv, calib=cal)


def debut_rows(train):
    first = train.groupby("pitcher_id", sort=False)["season"].transform("min")
    return train[train.season.eq(first)].copy()


def train_predict(train, val):
    tr, va = add_features(train), add_features(val)
    features = [c for c in tr.columns if c not in DROP]
    cats = [c for c in CAT_COLS if c in features]
    maps = build_category_maps(tr, cats)
    tr, va = apply_category_maps(tr, cats, maps), apply_category_maps(va, cats, maps)
    params = dict(PARAMS, seed=42, num_leaves=31, min_data_in_leaf=800,
                  num_threads=6)
    ds = lgb.Dataset(tr[features], label=tr.control_success,
                     categorical_feature=cats, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=236)
    return model.predict(va[features])


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    pooled = []
    for train_max, val_season in SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        debut = debut_rows(train)
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = val.game_type.eq("R").to_numpy() & ~val.pitcher_id.isin(known).to_numpy()
        p0 = baseline(CACHE / f"fold_{train_max}.npz", inner)
        all_r = train_predict(train, val)
        first_r = train_predict(debut, val)
        y = val.control_success.to_numpy(dtype="float64")
        p_all, p_debut = 0.5*p0 + 0.5*all_r, 0.5*p0 + 0.5*first_r
        d = (p_all[cold]-y[cold])**2 - (p_debut[cold]-y[cold])**2
        pooled.append(d)
        print(f"{train_max}->{val_season} train_all={len(train):,} "
              f"train_debut={len(debut):,} cold={cold.sum():,} "
              f"all={np.mean((p_all[cold]-y[cold])**2):.9f} "
              f"debut={np.mean((p_debut[cold]-y[cold])**2):.9f} "
              f"gain={d.mean():+.3e}")
    d = np.concatenate(pooled)
    se = d.std(ddof=1)/np.sqrt(len(d))
    print(f"pooled n={len(d):,} gain={d.mean():+.3e} z={d.mean()/se:+.2f}")


if __name__ == "__main__":
    main()
