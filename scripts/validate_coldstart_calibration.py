"""Forward-only Platt calibration check for the champion cold-start path."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "open" / "data" / "train.csv"
CACHE = ROOT / "artifacts" / "validation_hetero" / "exp23_cache"
sys.path.insert(0, str(ROOT / "scripts"))
import validate_coldstart_expert as vce

SPLITS = ((2021, 2022), (2022, 2023), (2023, 2024))


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    folds = []
    for train_max, val_season in SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = val.game_type.eq("R").to_numpy() & ~val.pitcher_id.isin(known).to_numpy()
        p0 = vce.baseline(CACHE / f"fold_{train_max}.npz", inner, val)
        expert = vce.train_expert(train, val, seed=42)
        p = 0.5 * p0 + 0.5 * expert
        folds.append({"name": f"{train_max}->{val_season}", "p": p[cold],
                      "y": val.control_success.to_numpy(dtype="float64")[cold]})

    earlier_p, earlier_y = [], []
    pooled_delta = []
    for fold in folds:
        if earlier_p:
            scale, bias = vce.tl.fit_platt(np.concatenate(earlier_p),
                                           np.concatenate(earlier_y))
            calibrated = vce.tl.apply_platt(fold["p"], scale, bias)
        else:
            scale, bias, calibrated = 1.0, 0.0, fold["p"]
        d = ((fold["p"] - fold["y"]) ** 2
             - (calibrated - fold["y"]) ** 2)
        pooled_delta.append(d)
        print(f"{fold['name']} n={len(d):,} scale={scale:.6f} bias={bias:+.6f} "
              f"raw={np.mean((fold['p']-fold['y'])**2):.9f} "
              f"cal={np.mean((calibrated-fold['y'])**2):.9f} "
              f"gain={d.mean():+.3e}")
        earlier_p.append(fold["p"])
        earlier_y.append(fold["y"])
    d = np.concatenate(pooled_delta[1:])
    se = d.std(ddof=1) / np.sqrt(len(d))
    final = vce.tl.fit_platt(np.concatenate(earlier_p), np.concatenate(earlier_y))
    print(f"forward evaluated folds 2+3: n={len(d):,} gain={d.mean():+.3e} "
          f"z={d.mean()/se:+.2f}")
    print(f"deployment fit on all OOF cold rows: scale={final[0]:.9f} "
          f"bias={final[1]:+.9f}")


if __name__ == "__main__":
    main()
