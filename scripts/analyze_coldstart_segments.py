"""Discover row-local segments where the cold-start expert is repeatable."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_coldstart_expert as v


def labels(df):
    out = {
        "count": df.balls_before.astype(str) + "-" + df.strikes_before.astype(str),
        "inning": pd.cut(df.inning, [0, 3, 6, 9, np.inf], labels=["1-3", "4-6", "7-9", "10+"]),
        "outs": df.outs_before.astype(str),
        "runners": df.num_runners_on.astype(str),
        "hand": df.pitcher_hand.astype(str) + "-" + df.batter_hand.astype(str),
        "batter_n": pd.cut(df.asof_batter_n, [-1, 50, 200, 1000, np.inf],
                           labels=["0-50", "51-200", "201-1000", "1001+"]),
        "leverage": pd.cut(df.li, [-np.inf, .7, 1.5, np.inf], labels=["low", "mid", "high"]),
    }
    return out


def main():
    df = pd.read_csv(v.DATA, encoding="utf-8-sig")
    rows = []
    for train_max, val_season in v.SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = val.game_type.eq("R").to_numpy() & ~val.pitcher_id.isin(known).to_numpy()
        p0 = v.baseline(v.CACHE / f"fold_{train_max}.npz", inner, val)
        pe = v.train_expert(train, val, 42)
        y = val.control_success.to_numpy(dtype="float64")
        # Positive means the champion's expert component helps versus base.
        d = (p0 - y) ** 2 - ((.5*p0 + .5*pe) - y) ** 2
        for feature, lab in labels(val).items():
            for level in pd.Series(lab[cold]).dropna().unique():
                mask = cold & (np.asarray(lab.astype(str)) == str(level))
                if mask.sum() >= 300:
                    rows.append((feature, str(level), val_season, mask.sum(), d[mask].mean()))
    tab = pd.DataFrame(rows, columns=["feature", "level", "season", "n", "gain"])
    summary = (tab.groupby(["feature", "level"])
               .agg(folds=("gain", "size"), min_gain=("gain", "min"),
                    max_gain=("gain", "max"), mean_gain=("gain", "mean"),
                    total_n=("n", "sum"))
               .reset_index())
    stable = summary[(summary.folds == 3) & (summary.min_gain > 0)].sort_values("mean_gain", ascending=False)
    print("segments where w50 beats base in every forward fold")
    print(stable.to_string(index=False, float_format=lambda x: f"{x:+.3e}"))
    print("\nall segment fold gains")
    print(tab.sort_values(["feature", "level", "season"]).to_string(index=False,
          float_format=lambda x: f"{x:+.3e}"))


if __name__ == "__main__":
    main()
