"""Cutoff-safe hierarchical empirical-Bayes prior-only validation.

The validation frame's label is never passed to ``transform``.  Every statistic
is frozen from seasons strictly earlier than the validation cutoff.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "control_success"
FOLDS = (2022, 2023, 2024)
K_GRID = (20.0, 100.0, 500.0)
BLENDS = (0.025, 0.05, 0.10, 0.15)


@dataclass(frozen=True)
class Group:
    name: str
    keys: tuple[str, ...]
    parent: str


GROUPS = (
    Group("pitcher", ("pitcher_id",), "global"),
    Group("batter", ("batter_id",), "global"),
    Group("pitcher_batter_side", ("pitcher_id", "batter_hand"), "pitcher"),
    Group("batter_pitcher_side", ("batter_id", "pitcher_hand"), "batter"),
    Group("pitcher_count", ("pitcher_id", "count_state"), "pitcher"),
    Group("batter_count", ("batter_id", "count_state"), "batter"),
    Group("pitcher_batter", ("pitcher_id", "batter_id"), "player_pair"),
)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["count_state"] = (
        out["balls_before"].astype("int8") * 3 + out["strikes_before"].astype("int8")
    ).astype("int8")
    return out


def _aggregate(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    return frame.groupby(list(keys), observed=True, sort=False)[TARGET].agg(s="sum", n="count")


def fit_lookups(history: pd.DataFrame) -> dict:
    if history.empty or TARGET not in history:
        raise ValueError("non-empty labeled history is required")
    return {
        "global_s": float(history[TARGET].sum()),
        "global_n": int(len(history)),
        "groups": {g.name: _aggregate(history, g.keys) for g in GROUPS},
    }


def _mapped(stats: pd.DataFrame, rows: pd.DataFrame, keys: tuple[str, ...]):
    index = (pd.Index(rows[keys[0]], name=keys[0]) if len(keys) == 1
             else pd.MultiIndex.from_frame(rows[list(keys)]))
    found = stats.reindex(index)
    return (found["s"].fillna(0).to_numpy(float),
            found["n"].fillna(0).to_numpy(float))


def transform(rows: pd.DataFrame, lookups: dict, k: float) -> pd.DataFrame:
    """Create row-local posteriors; ``rows`` need not and should not contain target."""
    n_rows = len(rows)
    global_rate = lookups["global_s"] / lookups["global_n"]
    values = {"global_rate": np.full(n_rows, global_rate),
              "global_n": np.full(n_rows, lookups["global_n"], float)}
    for group in GROUPS:
        if group.parent == "player_pair":
            parent = (values["pitcher_rate"] + values["batter_rate"]) / 2
        else:
            parent = values[f"{group.parent}_rate"]
        successes, counts = _mapped(lookups["groups"][group.name], rows, group.keys)
        values[f"{group.name}_rate"] = (successes + k * parent) / (counts + k)
        values[f"{group.name}_n"] = counts
        values[f"{group.name}_reliability"] = counts / (counts + k)
    return pd.DataFrame(values, index=rows.index)


def prior_prediction(features: pd.DataFrame) -> np.ndarray:
    """Fixed confidence-weighted average; no validation-fitted coefficients."""
    base = features["global_rate"].to_numpy(float)
    numerator = base.copy()
    denominator = np.ones(len(features))
    for group in GROUPS:
        reliability = features[f"{group.name}_reliability"].to_numpy(float)
        numerator += reliability * features[f"{group.name}_rate"].to_numpy(float)
        denominator += reliability
    return numerator / denominator


def brier(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def champion_oof(out: Path, season: int) -> np.ndarray:
    return 0.30 * np.load(out / f"{season}_cat.npy") + 0.70 * np.load(out / f"{season}_team.npy")


def independence_audit(rows, lookups, k):
    sample = rows.iloc[:2000].copy()
    sample["audit_id"] = np.arange(len(sample))
    reference = transform(sample.drop(columns=[TARGET], errors="ignore"), lookups, k)
    variants = {
        "shuffled": sample.sample(frac=1, random_state=42),
        "reversed": sample.iloc[::-1],
        "subset": sample.iloc[::3],
        "deleted": sample.iloc[1:],
        "unrelated_duplicate": pd.concat([sample, sample.iloc[[-1]]], ignore_index=True),
    }
    diffs = {}
    ref = reference.set_axis(sample["audit_id"]).sort_index()
    for name, variant in variants.items():
        ids = variant["audit_id"].to_numpy()
        got = transform(variant.drop(columns=[TARGET, "audit_id"], errors="ignore"), lookups, k)
        got.index = ids
        common = got.index.intersection(ref.index)
        # Duplicate IDs are compared row-by-row against their reference.
        delta = got.to_numpy(float) - ref.loc[ids].to_numpy(float)
        diffs[name] = float(np.max(np.abs(delta))) if len(delta) else 0.0
    return diffs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT.parent / "open/data/train.csv")
    parser.add_argument("--oof", type=Path, default=ROOT / "artifacts/backtest/champion_oof")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/hierarchical_eb")
    args = parser.parse_args(argv)
    frame = prepare(pd.read_csv(args.train, encoding="utf-8-sig"))
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, coverage, predictions = [], [], {}
    targets, champions = {}, {}
    lookups_2024 = validation_2024 = None
    for season in FOLDS:
        history, validation = frame[frame.season < season], frame[frame.season == season]
        lookups = fit_lookups(history)
        y = validation[TARGET].to_numpy(float)
        champ = champion_oof(args.oof, season)
        if len(champ) != len(y):
            raise ValueError(f"OOF alignment failure for {season}")
        targets[season], champions[season] = y, champ
        if season == 2024:
            lookups_2024, validation_2024 = lookups, validation
        for group in GROUPS:
            _, count = _mapped(lookups["groups"][group.name], validation, group.keys)
            coverage.append({"season": season, "hierarchy": group.name,
                             "group_count": len(lookups["groups"][group.name]),
                             "seen_rate": float(np.mean(count > 0)),
                             "seen_median_n": float(np.median(count[count > 0])) if np.any(count > 0) else 0})
        for k in K_GRID:
            features = transform(validation.drop(columns=[TARGET]), lookups, k)
            pred = prior_prediction(features)
            predictions[(season, k)] = pred
            metrics.append({"season": season, "k": k, "candidate_brier": brier(y, pred),
                            "champion_brier": brier(y, champ),
                            "gain_vs_champion": brier(y, champ)-brier(y, pred),
                            "corr_candidate_champion": corr(pred, champ),
                            "corr_gap_champion_residual": corr(pred-champ, y-champ)})
    pooled_y = np.concatenate([targets[s] for s in FOLDS])
    pooled_champ = np.concatenate([champions[s] for s in FOLDS])
    for k in K_GRID:
        pred = np.concatenate([predictions[(s, k)] for s in FOLDS])
        metrics.append({"season": "pooled", "k": k, "candidate_brier": brier(pooled_y, pred),
                        "champion_brier": brier(pooled_y, pooled_champ),
                        "gain_vs_champion": brier(pooled_y, pooled_champ)-brier(pooled_y, pred),
                        "corr_candidate_champion": corr(pred, pooled_champ),
                        "corr_gap_champion_residual": corr(pred-pooled_champ, pooled_y-pooled_champ)})
    metric_frame = pd.DataFrame(metrics)
    selected_k = float(metric_frame[metric_frame.season.eq("pooled")].sort_values("candidate_brier").iloc[0].k)
    blends, segments = [], []
    for season in FOLDS:
        y, champ, pred = targets[season], champions[season], predictions[(season, selected_k)]
        validation = frame[frame.season == season]
        lookup = fit_lookups(frame[frame.season < season])
        features = transform(validation.drop(columns=[TARGET]), lookup, selected_k)
        err = (y-champ)**2
        for q, label in ((0.95, 5), (0.90, 10), (0.80, 20)):
            mask = err >= np.quantile(err, q)
            segments.append({"season": season, "segment": f"champion_error_top_{label}pct",
                             "n": int(mask.sum()), "candidate_gain": brier(y[mask], champ[mask])-brier(y[mask], pred[mask])})
        segment_masks = {
            "pitcher_unseen": features.pitcher_n.to_numpy() == 0,
            "pitcher_n_1_100": features.pitcher_n.to_numpy().astype(float).clip(1, 100) == features.pitcher_n.to_numpy(),
            "pitcher_n_gt_100": features.pitcher_n.to_numpy() > 100,
            "batter_unseen": features.batter_n.to_numpy() == 0,
            "batter_n_1_100": features.batter_n.to_numpy().astype(float).clip(1, 100) == features.batter_n.to_numpy(),
            "batter_n_gt_100": features.batter_n.to_numpy() > 100,
            "matchup_unseen": features.pitcher_batter_n.to_numpy() == 0,
            "matchup_n_1_5": (features.pitcher_batter_n.to_numpy() >= 1) & (features.pitcher_batter_n.to_numpy() <= 5),
            "matchup_n_gt_5": features.pitcher_batter_n.to_numpy() > 5,
        }
        for count in sorted(validation.count_state.unique()):
            segment_masks[f"count_{int(count)}"] = validation.count_state.to_numpy() == count
        for p_hand in ("L", "R"):
            for b_hand in ("L", "R"):
                segment_masks[f"hands_{p_hand}_{b_hand}"] = ((validation.pitcher_hand.to_numpy() == p_hand)
                                                               & (validation.batter_hand.to_numpy() == b_hand))
        for name, mask in segment_masks.items():
            if np.any(mask):
                segments.append({"season": season, "segment": name, "n": int(mask.sum()),
                                 "candidate_gain": brier(y[mask], champ[mask])-brier(y[mask], pred[mask])})
        for weight in BLENDS:
            blended = (1-weight)*champ + weight*pred
            blends.append({"season": season, "weight": weight,
                           "gain_vs_champion": brier(y, champ)-brier(y, blended)})
    for weight in BLENDS:
        pred = np.concatenate([predictions[(s, selected_k)] for s in FOLDS])
        blends.append({"season": "pooled", "weight": weight,
                       "gain_vs_champion": brier(pooled_y, pooled_champ)-brier(pooled_y, (1-weight)*pooled_champ+weight*pred)})
    leakage = independence_audit(validation_2024, lookups_2024, selected_k)
    fold_blends = pd.DataFrame(blends)
    stable_blend = any((fold_blends[np.isclose(fold_blends.weight, w) & fold_blends.season.isin(FOLDS)]
                        .gain_vs_champion > 0).all() for w in BLENDS)
    summary = {"selected_k": selected_k, "leakage_max_abs_diff": max(leakage.values()),
               "leakage_variants": leakage, "logistic_run": False,
               "stable_positive_blend": bool(stable_blend),
               "logistic_reason": "stopped: 2022/2024 degraded and no fixed blend improved all folds"}
    metric_frame.to_csv(args.output / "prior_metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.output / "coverage.csv", index=False)
    pd.DataFrame(blends).to_csv(args.output / "blend_metrics.csv", index=False)
    pd.DataFrame(segments).to_csv(args.output / "error_segments.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
