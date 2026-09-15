"""Validate one new official-data axis: row-local pitcher usage/role state.

The actual deployed LightGBM-team recipe is kept fixed.  Candidate models add
only deterministic features from the current row and official as-of values;
no evaluation aggregation, ordering, or frequency is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.champion import Champion
from src.train_base import PARAMS, add_features, apply_category_maps, build_category_maps


ROOT = Path(__file__).resolve().parents[1]
SEASONS = (2022, 2023, 2024)
SEED_OFFSETS = (0, 1, 2)
WEIGHTS = (0.025, 0.05, 0.10, 0.15)
TARGET = "control_success"
ROLE_COLS = (
    "role_inning_phase", "role_experience", "role_pitch_mix",
    "role_recent_stability", "role_team_context",
)


def role_features(df: pd.DataFrame) -> pd.DataFrame:
    inning = pd.to_numeric(df.inning, errors="coerce").fillna(-1).to_numpy()
    phase = np.select([inning <= 3, inning <= 6], ["early", "middle"], default="late")
    n = pd.to_numeric(df.asof_pitcher_n, errors="coerce").fillna(0).to_numpy()
    exp = np.select(
        [n == 0, n < 200, n < 1000, n < 3000],
        ["new", "low", "medium", "high"], default="veteran",
    )
    mix_cols = [
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    mix = df[mix_cols].to_numpy(dtype="float64")
    valid = np.isfinite(mix).any(axis=1)
    dominant = np.array(["unknown"] * len(df), dtype=object)
    dominant[valid] = np.array(["fastball", "breaking", "offspeed"])[
        np.nanargmax(np.where(np.isfinite(mix[valid]), mix[valid], -np.inf), axis=1)
    ]
    gap = np.abs(
        pd.to_numeric(df.asof_pitcher_prev1_game_success_rate, errors="coerce")
        - pd.to_numeric(df.asof_pitcher_prev5_game_success_rate, errors="coerce")
    ).fillna(np.inf).to_numpy()
    stability = np.select(
        [gap <= .02, gap <= .05, gap <= .10],
        ["stable", "small", "medium"], default="volatile_or_unknown",
    )
    team = pd.to_numeric(df.pitcher_team_id, errors="coerce").fillna(-1).astype(int).astype(str)
    context = team + "|" + pd.Series(phase, index=df.index) + "|" + df.game_type.astype(str)
    out = pd.DataFrame(
        {
            "role_inning_phase": phase,
            "role_experience": exp,
            "role_pitch_mix": dominant,
            "role_recent_stability": stability,
            "role_team_context": context,
        }, index=df.index,
    )
    for col in out:
        out[col] = out[col].astype("category")
    return out


def fit_team(champion, df, train_seasons, target, seed_offset, add_role):
    spec = champion.members["team"]["team"]
    cfgs = champion.meta["model_params"]["team"]
    cols = list(champion.members["team"]["feature_columns"])
    cats = list(spec["cat_cols"])
    d = add_features(df)
    if any(c.startswith("cur_") and c not in d for c in cols):
        d = pd.concat([d, champion.state_frame(df)], axis=1)
    if add_role:
        d = pd.concat([d, role_features(df)], axis=1)
        cols += list(ROLE_COLS)
        cats += list(ROLE_COLS)
    train_mask = df.season.isin(train_seasons)
    maps = build_category_maps(d[train_mask], cats)
    X = apply_category_maps(d, cats, maps)[cols]
    y = df[TARGET].to_numpy(dtype="float64")
    season = df.season.astype(int).to_numpy()
    tr = train_mask.to_numpy(); te = df.season.eq(target).to_numpy(); nxt = max(train_seasons) + 1
    pos = {"asof_pitcher_success_rate", "asof_pitcher_strike_rate",
           "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
           "asof_pitcher_prev5_game_success_rate", "asof_batter_success_rate"}
    neg = {"asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
           "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
           "asof_pitcher_prev5_game_middle_rate", "asof_batter_middle_rate"}
    preds = {}
    for name, cfg in cfgs.items():
        par = dict(PARAMS, seed=cfg["seed"] + seed_offset,
                   num_leaves=cfg["num_leaves"], min_data_in_leaf=cfg["min_data_in_leaf"],
                   num_threads=6)
        if cfg.get("monotone"):
            par["monotone_constraints"] = [1 if f in pos else -1 if f in neg else 0 for f in cols]
            par["monotone_constraints_method"] = "advanced"
        weight = cfg["season_decay"] ** (nxt - season[tr]) if cfg.get("season_decay") else None
        ds = lgb.Dataset(X[tr], label=y[tr], weight=weight,
                         categorical_feature=cats, free_raw_data=False)
        preds[name] = lgb.train(par, ds, num_boost_round=cfg["rounds"]).predict(X[te])
    groups = []
    for i, group in enumerate(spec["groups"]):
        raw = np.average([preds[m] for m in group["members"]], axis=0, weights=group["weights"])
        cal = spec["calib"][i]
        groups.append(champion.script.apply_logit_shift(raw, cal["bias"], slope=cal["scale"]))
    return np.average(groups, axis=0, weights=spec["group_weights"])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=ROOT.parent / "open/data/train.csv")
    p.add_argument("--zip", type=Path, default=ROOT / "artifacts/submit_season_state.zip")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/backtest/role_state")
    p.add_argument("--seed-offsets", type=int, nargs="+", default=list(SEED_OFFSETS))
    args = p.parse_args(argv)
    df = pd.read_csv(args.train, encoding="utf-8-sig")
    args.out.mkdir(parents=True, exist_ok=True)
    champion = Champion(args.zip, workdir=str(args.out / "_spec"))
    report = {"axis": "pitcher_usage_role_state", "row_independent": True, "folds": []}
    for seed in args.seed_offsets:
        for target in SEASONS:
            train_seasons = sorted(df.loc[df.season.lt(target), "season"].unique().astype(int))
            values = {}
            for label, add_role in (("control", False), ("candidate", True)):
                path = args.out / f"{target}_{label}_s{seed}.npy"
                if path.exists(): pred = np.load(path)
                else:
                    print(f"fit seed={seed} fold={target} {label}", flush=True)
                    pred = fit_team(champion, df, train_seasons, target, seed, add_role)
                    np.save(path, pred)
                values[label] = pred
            y = df.loc[df.season.eq(target), TARGET].to_numpy(float)
            base = np.mean((values["control"] - y) ** 2)
            cand = np.mean((values["candidate"] - y) ** 2)
            report["folds"].append({"seed_offset": seed, "season": target,
                                    "control_brier": base, "candidate_brier": cand,
                                    "gain": base - cand})
            print(report["folds"][-1], flush=True)
        # Fast stop after two seeds when 2022 and 2024 both fail in both seeds.
        done = [r for r in report["folds"] if r["seed_offset"] in args.seed_offsets[:2]]
        if len(done) == 6:
            by = {(r["seed_offset"], r["season"]): r["gain"] for r in done}
            if all(by[(s, y)] < 0 for s in args.seed_offsets[:2] for y in (2022, 2024)):
                report["fast_stop"] = "first two seeds both worsen 2022 and 2024"
                break
    (args.out / "verdict.json").write_text(json.dumps(report, indent=2), "utf-8")


if __name__ == "__main__":
    main()
