"""Cutoff-safe hard-example and uncertainty-weighted LightGBM validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.champion import Champion
from src.train_base import PARAMS, add_features, apply_category_maps, build_category_maps

ROOT = Path(__file__).resolve().parents[1]
TARGET = "control_success"
OUTER = (2022, 2023, 2024)
SEEDS = (42, 43, 44)
BLENDS = (.025, .05, .10, .15)


def champion_oof(path: Path, season: int) -> np.ndarray:
    return .30 * np.load(path / f"{season}_cat.npy") + .70 * np.load(path / f"{season}_team.npy")


def inner_oof_vector(df: pd.DataFrame, out: Path, cutoff: int):
    """Return past-only OOF values for train rows; earliest season stays missing."""
    values = np.full(len(df), np.nan)
    for season in sorted(int(s) for s in df.loc[df.season.lt(cutoff), "season"].unique()):
        cat, team = out / f"{season}_cat.npy", out / f"{season}_team.npy"
        if cat.is_file() and team.is_file():
            mask = df.season.eq(season).to_numpy()
            pred = champion_oof(out, season)
            if len(pred) != int(mask.sum()):
                raise ValueError(f"inner OOF alignment failure: {season}")
            values[mask] = pred
    return values


def objective_weights(y: np.ndarray, p_oof: np.ndarray, kind: str) -> np.ndarray:
    """Weights use labels only where a same-row past-only OOF prediction exists."""
    valid = np.isfinite(p_oof)
    weight = np.ones(len(y), dtype="float64")
    if kind == "hard":
        error = (y[valid] - p_oof[valid]) ** 2
        q50, q75, q90 = np.quantile(error, [.50, .75, .90])
        weight[valid] = np.select(
            [error >= q90, error >= q75, error >= q50], [3., 2., 1.5], default=1.)
    elif kind == "uncertainty_0p5":
        weight[valid] = 1 + .5 * (4 * p_oof[valid] * (1-p_oof[valid]))
    elif kind == "uncertainty_1p0":
        weight[valid] = 1 + 4 * p_oof[valid] * (1-p_oof[valid])
    else:
        raise ValueError(kind)
    return weight


def prepare_features(champion: Champion, df: pd.DataFrame):
    spec = champion.members["team"]["team"]
    cols = list(champion.members["team"]["feature_columns"])
    d = add_features(df)
    if any(c.startswith("cur_") and c not in d for c in cols):
        d = pd.concat([d, champion.state_frame(df)], axis=1)
    return d, cols, list(spec["cat_cols"])


def fit_weighted_team(champion, df, prepared, cols, cats, cutoff, seed, sample_weight):
    spec = champion.members["team"]["team"]
    cfgs = champion.meta["model_params"]["team"]
    tr = df.season.lt(cutoff).to_numpy(); te = df.season.eq(cutoff).to_numpy()
    maps = build_category_maps(prepared[tr], cats)
    X = apply_category_maps(prepared, cats, maps)[cols]
    y = df[TARGET].to_numpy(float); season = df.season.to_numpy(int)
    pos = {"asof_pitcher_success_rate", "asof_pitcher_strike_rate",
           "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
           "asof_pitcher_prev5_game_success_rate", "asof_batter_success_rate"}
    neg = {"asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
           "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
           "asof_pitcher_prev5_game_middle_rate", "asof_batter_middle_rate"}
    pred = {}
    for index, (name, cfg) in enumerate(cfgs.items()):
        params = dict(PARAMS, seed=seed + index, num_leaves=cfg["num_leaves"],
                      min_data_in_leaf=cfg["min_data_in_leaf"], num_threads=6)
        if cfg.get("monotone"):
            params["monotone_constraints"] = [1 if c in pos else -1 if c in neg else 0 for c in cols]
            params["monotone_constraints_method"] = "advanced"
        weight = sample_weight[tr].copy()
        if cfg.get("season_decay"):
            weight *= cfg["season_decay"] ** (cutoff - season[tr])
        ds = lgb.Dataset(X[tr], label=y[tr], weight=weight,
                         categorical_feature=cats, free_raw_data=False)
        pred[name] = lgb.train(params, ds, num_boost_round=cfg["rounds"]).predict(X[te])
    groups = []
    for i, group in enumerate(spec["groups"]):
        raw = np.average([pred[m] for m in group["members"]], axis=0, weights=group["weights"])
        cal = spec["calib"][i]
        groups.append(champion.script.apply_logit_shift(raw, cal["bias"], slope=cal["scale"]))
    return np.average(groups, axis=0, weights=spec["group_weights"])


def brier(y, p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def corr(a, b): return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins+1); total = 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any(): total += mask.mean() * abs(float(y[mask].mean()-p[mask].mean()))
    return total


def diagnostics(y, champion, candidate, season, kind, seed):
    row = {"season": season, "candidate": kind, "seed": seed,
           "champion_brier": brier(y, champion), "candidate_brier": brier(y, candidate),
           "gain": brier(y, champion)-brier(y, candidate),
           "target_mean": float(y.mean()), "champion_mean": float(champion.mean()),
           "candidate_mean": float(candidate.mean()), "candidate_logloss": log_loss(y, candidate),
           "champion_logloss": log_loss(y, champion), "champion_ece": ece(y, champion),
           "candidate_ece": ece(y, candidate), "prediction_corr": corr(candidate, champion),
           "residual_corr": corr(y-candidate, y-champion),
           "corr_gap_residual": corr(candidate-champion, y-champion)}
    error = (y-champion)**2
    sub = []
    for pct in (1, 5, 10, 20):
        mask = error >= np.quantile(error, 1-pct/100)
        sub.append({"season": season, "candidate": kind, "seed": seed,
                    "segment": f"top_{pct}pct", "n": int(mask.sum()),
                    "gain": brier(y[mask], champion[mask])-brier(y[mask], candidate[mask])})
    # Fixed, predeclared confidence sides; no validation-fitted threshold.
    pos = (y == 1) & (champion < .40); neg = (y == 0) & (champion > .60)
    for name, mask in (("false_confident_positive", pos), ("false_confident_negative", neg)):
        if mask.any(): sub.append({"season": season, "candidate": kind, "seed": seed,
                                   "segment": name, "n": int(mask.sum()),
                                   "gain": brier(y[mask], champion[mask])-brier(y[mask], candidate[mask])})
    return row, sub


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT.parent / "open/data/train.csv")
    parser.add_argument("--zip", type=Path, default=ROOT / "artifacts/submit_season_state.zip")
    parser.add_argument("--oof", type=Path, default=ROOT / "artifacts/backtest/champion_oof")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/backtest/hard_example")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args(argv); args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.train, encoding="utf-8-sig")
    champion = Champion(args.zip, workdir=str(args.out / "_spec"))
    prepared, cols, cats = prepare_features(champion, df)
    rows, subgroup, blend_rows, coverage = [], [], [], []
    candidates = ("hard", "uncertainty_0p5", "uncertainty_1p0")
    stop = None
    for seed_index, seed in enumerate(args.seeds):
        for season in OUTER:
            inner = inner_oof_vector(df, args.oof, season)
            tr = df.season.lt(season).to_numpy(); valid = np.isfinite(inner) & tr
            coverage.append({"season": season, "seed": seed, "train_rows": int(tr.sum()),
                             "inner_oof_rows": int(valid.sum()), "coverage": float(valid.sum()/tr.sum()),
                             "source_seasons": sorted(df.loc[valid, "season"].unique().astype(int).tolist())})
            y_all = df[TARGET].to_numpy(float)
            y = y_all[df.season.eq(season)]; champ = champion_oof(args.oof, season)
            for kind in candidates:
                path = args.out / f"{season}_{kind}_s{seed}.npy"
                if path.exists(): pred = np.load(path)
                else:
                    weight = objective_weights(y_all, inner, kind)
                    print(f"fit season={season} seed={seed} {kind} coverage={valid.mean():.3f}", flush=True)
                    pred = fit_weighted_team(champion, df, prepared, cols, cats, season, seed, weight)
                    np.save(path, pred)
                metric, subs = diagnostics(y, champ, pred, season, kind, seed)
                rows.append(metric); subgroup.extend(subs)
                for w in BLENDS:
                    blend_rows.append({"season": season, "candidate": kind, "seed": seed, "weight": w,
                                       "gain": brier(y, champ)-brier(y, (1-w)*champ+w*pred)})
                print(metric, flush=True)
        # Fast stop after seeds 42/43 if both hard and uncertainty families fail recent folds.
        if seed_index == 1:
            recent = pd.DataFrame(rows)
            check = recent[recent.seed.isin(args.seeds[:2]) & recent.season.isin((2022, 2024))]
            if all((check[check.candidate.eq(kind)].gain < 0).all() for kind in candidates):
                stop = "seeds 42/43: every candidate worsened both 2022 and 2024"
                break
    pd.DataFrame(rows).to_csv(args.out / "candidate_metrics.csv", index=False)
    pd.DataFrame(subgroup).to_csv(args.out / "hard_error_subgroups.csv", index=False)
    pd.DataFrame(blend_rows).to_csv(args.out / "blend_metrics.csv", index=False)
    pd.DataFrame(coverage).drop_duplicates(["season", "seed"]).to_csv(args.out / "inner_oof_coverage.csv", index=False)
    summary = {"fast_stop": stop, "specialist_run": False,
               "specialist_reason": "run only if A/B are not complete failures",
               "outer_validation_used_for_weights": False, "row_independent_inference": True}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
