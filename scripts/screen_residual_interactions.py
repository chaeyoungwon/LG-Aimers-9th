"""Screen new train-only residual interactions beyond the 1031 corrections."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_coldstart_expert as v


def key(rows, name):
    b = rows.balls_before.to_numpy(dtype="int64")
    s = rows.strikes_before.to_numpy(dtype="int64")
    if name == "batter_hand": return rows.batter_hand.to_numpy(dtype="int64")
    if name == "pitcher_hand": return rows.pitcher_hand.to_numpy(dtype="int64")
    if name == "count": return b * 3 + s
    if name == "leverage": return np.where(s > b, 0, np.where(b > s + 1, 2, 1))
    if name == "inning_role":
        inning = rows.inning.to_numpy(dtype="int64")
        return np.where(inning <= 3, 0, np.where(inning <= 6, 1, 2))
    if name == "base_pressure":
        n = rows.num_runners_on.to_numpy(dtype="int64")
        return np.where(n == 0, 0, np.where(n == 1, 1, 2))
    if name == "top_bottom": return rows.top_bottom.astype(str).to_numpy()
    raise ValueError(name)


def flat_table(rows, residual, group, k):
    d = pd.DataFrame({"g": key(rows, group), "r": residual})
    z = d.groupby("g").r.agg(["sum", "size"])
    return {str(g): float(x["sum"] / (x["size"] + k)) for g, x in z.iterrows()}


def interaction_table(rows, residual, entity, group, k):
    d = pd.DataFrame({"e": rows[entity].to_numpy(dtype="int64"),
                      "g": key(rows, group), "r": residual})
    cell = d.groupby(["e", "g"]).r.agg(["sum", "size"])
    main = d.groupby("e").r.agg(["sum", "size"]).rename(
        columns={"sum": "es", "size": "en"})
    joined = cell.join(main, on="e")
    expected = joined.es * joined["size"] / joined.en
    value = (joined["sum"] - expected) / (joined["size"] + k)
    return {f"{e}|{g}": float(x) for (e, g), x in value.items()}


def lookup(rows, spec, table):
    entity, group = spec
    g = key(rows, group)
    if entity is None:
        labels = [str(x) for x in g]
    else:
        e = rows[entity].to_numpy(dtype="int64")
        labels = [f"{a}|{b}" for a, b in zip(e, g)]
    return np.fromiter((table.get(x, 0.0) for x in labels),
                       dtype="float64", count=len(rows))


EXISTING = {
    "beta": (("pitcher_id", "batter_hand"), 2000.0),
    "gamma": ((None, "count"), 5000.0),
    "delta": (("pitcher_id", "leverage"), 2000.0),
}
CANDIDATES = {
    "batter_x_hand_k2k": (("batter_id", "pitcher_hand"), 2000.0),
    "batter_x_hand_k5k": (("batter_id", "pitcher_hand"), 5000.0),
    "batter_x_hand_k10k": (("batter_id", "pitcher_hand"), 10000.0),
    "batter_x_leverage": (("batter_id", "leverage"), 2000.0),
    "pitcher_x_inning_role": (("pitcher_id", "inning_role"), 2000.0),
    "pitcher_x_base_k500": (("pitcher_id", "base_pressure"), 500.0),
    "pitcher_x_base_k1k": (("pitcher_id", "base_pressure"), 1000.0),
    "pitcher_x_base_k2k": (("pitcher_id", "base_pressure"), 2000.0),
    "pitcher_x_base_k5k": (("pitcher_id", "base_pressure"), 5000.0),
    "pitcher_x_base_k10k": (("pitcher_id", "base_pressure"), 10000.0),
    "pitcher_x_top_bottom": (("pitcher_id", "top_bottom"), 2000.0),
}


def make_table(rows, residual, spec, k):
    entity, group = spec
    return (flat_table(rows, residual, group, k) if entity is None else
            interaction_table(rows, residual, entity, group, k))


def fold_predictions(cache_path, inner, val):
    z = np.load(cache_path)
    pi = {n: z[f"p_inner_{n}"].astype("float64") for n in v.tl.MODEL_ORDER}
    pv = {n: z[f"p_val_{n}"].astype("float64") for n in v.tl.MODEL_ORDER}
    y = inner.control_success.to_numpy(dtype="float64")
    cal = [v.tl.fit_platt(v.tl.group_raw(pi, g), y)
           for g in v.tl.ENSEMBLE_GROUPS]
    return v.tl.blend(pi, calib=cal), v.tl.blend(pv, calib=cal)


def main():
    df = pd.read_csv(v.DATA, encoding="utf-8-sig")
    pooled = {name: [] for name in CANDIDATES}
    for train_max, val_season in v.SPLITS:
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        pi, pv = fold_predictions(v.CACHE / f"fold_{train_max}.npz", inner, val)
        train_r = inner.game_type.eq("R").to_numpy()
        val_r = val.game_type.eq("R").to_numpy()
        residual = inner.control_success.to_numpy(dtype="float64") - pi
        residual_r = residual[train_r] - residual[train_r].mean()
        source = inner.loc[train_r].reset_index(drop=True)
        target = val.loc[val_r].reset_index(drop=True)
        y = val.loc[val_r, "control_success"].to_numpy(dtype="float64")
        base = pv[val_r].copy()
        for spec, k in EXISTING.values():
            table = make_table(source, residual_r, spec, k)
            base = np.clip(base + lookup(target, spec, table), 0, 1)
        print(f"\n{train_max}->{val_season} R={len(target):,}")
        for name, (spec, k) in CANDIDATES.items():
            table = make_table(source, residual_r, spec, k)
            candidate = np.clip(base + lookup(target, spec, table), 0, 1)
            d = (base-y)**2 - (candidate-y)**2
            pooled[name].append(d)
            print(f"  {name:26s} gain={d.mean():+.3e} hit="
                  f"{np.count_nonzero(lookup(target, spec, table)):,}")
    print("\npooled incremental gain beyond beta+gamma+delta proxy")
    for name, parts in pooled.items():
        d = np.concatenate(parts)
        se = d.std(ddof=1)/np.sqrt(len(d))
        signs = "/".join("+" if x.mean() > 0 else "-" for x in parts)
        print(f"  {name:26s} gain={d.mean():+.3e} z={d.mean()/se:+.2f} "
              f"folds={signs}")


if __name__ == "__main__":
    main()
