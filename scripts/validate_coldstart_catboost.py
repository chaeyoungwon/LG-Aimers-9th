"""Validate a heterogeneous CatBoost cold-start expert at the fixed w50 gate."""
from pathlib import Path
import importlib.util
import sys

from catboost import CatBoostClassifier
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


def frames(train, val):
    tr, va = add_features(train), add_features(val)
    features = [c for c in tr.columns if c not in DROP]
    cats = [c for c in CAT_COLS if c in features]
    return tr, va, features, cats


def predict_lgbm(train, val):
    tr, va, features, cats = frames(train, val)
    maps = build_category_maps(tr, cats)
    tr, va = apply_category_maps(tr, cats, maps), apply_category_maps(va, cats, maps)
    params = dict(PARAMS, seed=42, num_leaves=31, min_data_in_leaf=800,
                  num_threads=6)
    ds = lgb.Dataset(tr[features], label=tr.control_success,
                     categorical_feature=cats, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=236)
    return model.predict(va[features])


def predict_cat(train, val):
    tr, va, features, cats = frames(train, val)
    for col in cats:
        tr[col] = tr[col].astype("object").where(tr[col].notna(), "__NA__").astype(str)
        va[col] = va[col].astype("object").where(va[col].notna(), "__NA__").astype(str)
    indices = [features.index(c) for c in cats]
    model = CatBoostClassifier(
        loss_function="Logloss", iterations=500, learning_rate=0.05,
        depth=6, l2_leaf_reg=20.0, random_seed=42, verbose=False,
        allow_writing_files=False, thread_count=6)
    model.fit(tr[features], tr.control_success, cat_features=indices)
    return model.predict_proba(va[features])[:, 1]


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    pooled = {"cat": [], "hetero": []}
    for train_max, val_season in SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = val.game_type.eq("R").to_numpy() & ~val.pitcher_id.isin(known).to_numpy()
        p0 = baseline(CACHE / f"fold_{train_max}.npz", inner)
        pl, pc = predict_lgbm(train, val), predict_cat(train, val)
        y = val.control_success.to_numpy(dtype="float64")
        pred = {
            "lgbm": 0.5*p0 + 0.5*pl,
            "cat": 0.5*p0 + 0.5*pc,
            "hetero": 0.5*p0 + 0.25*pl + 0.25*pc,
        }
        print(f"\n{train_max}->{val_season} cold={cold.sum():,}")
        for name in ("cat", "hetero"):
            d = ((pred["lgbm"][cold]-y[cold])**2
                 - (pred[name][cold]-y[cold])**2)
            pooled[name].append(d)
            print(f"  {name:6s} vs lgbm gain={d.mean():+.3e} "
                  f"brier={np.mean((pred[name][cold]-y[cold])**2):.9f}")
    print("\npooled vs champion lgbm expert")
    for name, parts in pooled.items():
        d = np.concatenate(parts)
        se = d.std(ddof=1)/np.sqrt(len(d))
        print(f"  {name:6s} n={len(d):,} gain={d.mean():+.3e} z={d.mean()/se:+.2f}")


if __name__ == "__main__":
    main()
