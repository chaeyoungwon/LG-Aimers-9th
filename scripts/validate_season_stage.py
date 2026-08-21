"""Forward validation for rule-safe season-stage corrections.

Every candidate uses only values present in the current row.  Parameters below
are fixed before looking at validation labels; the report includes per-fold
results so pooled improvements cannot hide an unstable season.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_coldstart_expert as v


# Base expert weight is the conservative, forward-validation setting rather
# than a value inferred from leaderboard feedback.
BASE_W = 0.42
MONTH_SCHEDULES = {
    "constant42": (0.42, 0.42, 0.42),
    # Baseball priors: unfamiliar pitchers are hardest to identify early;
    # accumulated official as-of information should make the ID model safer.
    "early_high": (0.55, 0.42, 0.32),
    "early_gentle": (0.48, 0.42, 0.36),
    # Alternative fatigue/roster-expansion prior.
    "late_high": (0.34, 0.42, 0.52),
    "late_gentle": (0.38, 0.42, 0.48),
}


def stage(month):
    """0=Mar-May, 1=Jun-Jul, 2=Aug-Oct; row-local."""
    m = np.asarray(month, dtype="float64")
    return np.where(m <= 5, 0, np.where(m <= 7, 1, 2))


def recent_signal(df):
    """Stable recent-control signal already available before the pitch."""
    r1 = df.asof_pitcher_prev1_game_success_rate.to_numpy(dtype="float64")
    r5 = df.asof_pitcher_prev5_game_success_rate.to_numpy(dtype="float64")
    return np.clip(np.nan_to_num(r1 - r5, nan=0.0), -0.20, 0.20)


def summarize(name, parts):
    d = np.concatenate(parts)
    se = d.std(ddof=1) / np.sqrt(len(d))
    z = d.mean() / se if se > 0 else 0.0
    print(f"  {name:18s} n={len(d):,} gain={d.mean():+.3e} "
          f"z={z:+.2f}")


def main():
    df = pd.read_csv(v.DATA, encoding="utf-8-sig")
    month_gain = {name: [] for name in MONTH_SCHEDULES}
    form_alphas = (-0.04, -0.02, 0.0, 0.02, 0.04)
    form_gain = {a: [] for a in form_alphas}
    diagnostics = {s: [] for s in range(3)}

    for train_max, val_season in v.SPLITS:
        train = df[(df.season <= train_max) & df.game_type.eq("R")].copy()
        inner = df[df.season == train_max].reset_index(drop=True)
        val = df[df.season == val_season].reset_index(drop=True)
        known = set(train.pitcher_id.unique())
        cold = (val.game_type.eq("R").to_numpy()
                & ~val.pitcher_id.isin(known).to_numpy())
        regular = val.game_type.eq("R").to_numpy()
        p0 = v.baseline(v.CACHE / f"fold_{train_max}.npz", inner, val)
        pe = v.train_expert(train, val, 42)
        y = val.control_success.to_numpy(dtype="float64")
        st = stage(val.game_month)
        ref = p0.copy()
        ref[cold] = (1.0 - BASE_W) * p0[cold] + BASE_W * pe[cold]

        print(f"\n{train_max}->{val_season}: R={regular.sum():,} cold={cold.sum():,}")
        for name, weights in MONTH_SCHEDULES.items():
            w = np.choose(st, weights)
            pred = p0.copy()
            pred[cold] = ((1.0 - w[cold]) * p0[cold]
                          + w[cold] * pe[cold])
            d = (ref[cold] - y[cold])**2 - (pred[cold] - y[cold])**2
            month_gain[name].append(d)
            print(f"  {name:18s} cold gain={d.mean():+.3e}")

        # Diagnostic only: expert-vs-base by stage explains directionality.
        for s in range(3):
            mask = cold & (st == s)
            d = (p0[mask] - y[mask])**2 - (pe[mask] - y[mask])**2
            diagnostics[s].append(d)
            print(f"  stage={s} rows={mask.sum():,} expert-base={d.mean():+.3e}")

        sig = recent_signal(val)
        late = st == 2
        eligible = regular & late & np.isfinite(sig)
        for alpha in form_alphas:
            pred = np.clip(ref + alpha * sig * late, 0.0, 1.0)
            d = ((ref[eligible] - y[eligible])**2
                 - (pred[eligible] - y[eligible])**2)
            form_gain[alpha].append(d)
            print(f"  late_form {alpha:+.2f}  gain={d.mean():+.3e}")

    print("\npooled month x cold-start vs constant w=.42")
    for name, parts in month_gain.items():
        summarize(name, parts)
    print("\npooled late-season recent-form correction")
    for alpha, parts in form_gain.items():
        summarize(f"alpha={alpha:+.2f}", parts)
    print("\nstage diagnostics: expert vs base")
    for s, parts in diagnostics.items():
        summarize(f"stage={s}", parts)


if __name__ == "__main__":
    main()
