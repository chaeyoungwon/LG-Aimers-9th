"""Forward validation of stronger recent-regime cold-start experts."""
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
SPECS = {
    "all": (None, 1.0),
    "last3": (3, 1.0),
    "last2": (2, 1.0),
    "latest2x": (None, 2.0),
    "latest3x": (None, 3.0),
}


def predict(train, val, years, latest_weight):
    if years is not None:
        cutoff = int(train.season.max()) - years + 1
        train = train[train.season >= cutoff].copy()
    tr, va = add_features(train), add_features(val)
    features = [c for c in tr.columns if c not in DROP]
    cats = [c for c in CAT_COLS if c in features]
    maps = build_category_maps(tr, cats)
    tr = apply_category_maps(tr, cats, maps)
    va = apply_category_maps(va, cats, maps)
    weight = np.ones(len(tr), dtype="float64")
    weight[tr.season.to_numpy() == tr.season.max()] = latest_weight
    params = dict(PARAMS, seed=42, num_leaves=31, min_data_in_leaf=800,
                  num_threads=6)
    ds = lgb.Dataset(tr[features], label=tr.control_success, weight=weight,
                     categorical_feature=cats, free_raw_data=False)
    model = lgb.train(params, ds, num_boost_round=236)
    return model.predict(va[features])


def main():
    df = pd.read_csv(v.DATA, encoding="utf-8-sig")
    pooled = {name: [] for name in SPECS if name != "all"}
    for train_max, val_season in v.SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = val.game_type.eq("R").to_numpy() & ~val.pitcher_id.isin(known).to_numpy()
        p0 = v.baseline(v.CACHE / f"fold_{train_max}.npz", inner, val)
        preds = {name: predict(train, val, *spec) for name, spec in SPECS.items()}
        y = val.control_success.to_numpy(dtype="float64")
        ref = .5*p0 + .5*preds["all"]
        print(f"\n{train_max}->{val_season} cold={cold.sum():,}")
        for name in pooled:
            candidate = .5*p0 + .5*preds[name]
            d = ((ref[cold]-y[cold])**2
                 - (candidate[cold]-y[cold])**2)
            pooled[name].append(d)
            print(f"  {name:10s} gain={d.mean():+.3e} "
                  f"brier={np.mean((candidate[cold]-y[cold])**2):.9f}")
    print("\npooled vs all-history expert")
    for name, parts in pooled.items():
        d = np.concatenate(parts)
        se = d.std(ddof=1)/np.sqrt(len(d))
        print(f"  {name:10s} n={len(d):,} gain={d.mean():+.3e} "
              f"z={d.mean()/se:+.2f}")


if __name__ == "__main__":
    main()
