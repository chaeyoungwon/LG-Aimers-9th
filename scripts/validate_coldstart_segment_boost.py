"""Validate conservative expert boosts on stable row-local cold-start segments."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_coldstart_expert as v


def masks(df):
    count30 = df.balls_before.eq(3).to_numpy() & df.strikes_before.eq(0).to_numpy()
    runners2 = df.num_runners_on.eq(2).to_numpy()
    hand11 = df.pitcher_hand.eq(1).to_numpy() & df.batter_hand.eq(1).to_numpy()
    # Broad, independently interpretable consensus: at least two stable states.
    consensus = (count30.astype(int) + runners2.astype(int) + hand11.astype(int)) >= 2
    return {"count30": count30, "runners2": runners2,
            "hand11": hand11, "either": runners2 | hand11,
            "consensus": consensus}


def main():
    df = pd.read_csv(v.DATA, encoding="utf-8-sig")
    candidates = [(name, boost) for name in ("count30", "runners2", "hand11", "either", "consensus")
                  for boost in (.60, .70)]
    pooled = {c: [] for c in candidates}
    for train_max, val_season in v.SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = val.game_type.eq("R").to_numpy() & ~val.pitcher_id.isin(known).to_numpy()
        p0 = v.baseline(v.CACHE / f"fold_{train_max}.npz", inner, val)
        pe = v.train_expert(train, val, 42)
        y = val.control_success.to_numpy(dtype="float64")
        ref = .5*p0 + .5*pe
        segs = masks(val)
        print(f"\n{train_max}->{val_season} cold={cold.sum():,}")
        for key in candidates:
            name, boost = key
            hit = cold & segs[name]
            p = ref.copy()
            p[hit] = (1-boost)*p0[hit] + boost*pe[hit]
            d = (ref[cold]-y[cold])**2 - (p[cold]-y[cold])**2
            pooled[key].append(d)
            print(f"  {name:9s} w={boost:.2f} rows={hit.sum():,} gain={d.mean():+.3e}")
    print("\npooled vs champion w50")
    for (name, boost), parts in pooled.items():
        d = np.concatenate(parts)
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {name:9s} w={boost:.2f} gain={d.mean():+.3e} z={d.mean()/se:+.2f}")


if __name__ == "__main__":
    main()
