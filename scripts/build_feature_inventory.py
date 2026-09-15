"""Build a machine-readable inventory of every official distributed column."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "open/data"
FILES = ("train.csv", "test.csv", "sample_submission.csv", "trackman_history.csv")


def family(column):
    if column.startswith("asof_pitcher_prev"): return "pitcher_recent_asof"
    if column.startswith("asof_pitcher"): return "pitcher_asof"
    if column.startswith("asof_batter"): return "batter_asof"
    if column in {"balls_before", "strikes_before", "outs_before"}: return "count_out"
    if column.startswith("runner_") or column in {"num_runners_on", "base_state"}: return "base_state"
    if "run_" in column or "score_diff" in column: return "score_state"
    if column in {"home_win_expectancy", "away_win_expectancy", "li"}: return "leverage"
    if "team" in column or "hand" in column: return "team_hand"
    if column.endswith("_id") or column == "row_id": return "identifier"
    if column in {"season", "game_month", "game_dayofweek"}: return "calendar"
    if column in {"inning", "top_bottom", "game_type"}: return "game_context"
    if column == "control_success": return "target"
    return "trackman_physics_or_event"


def main():
    with zipfile.ZipFile(ROOT / "artifacts/sub_tree_reblend_w0p300.zip") as z:
        meta = json.loads(z.read("model/meta.json"))
    used = set()
    for member in meta["members"]:
        if member["name"] in {"cat", "team"}:
            used.update(member["feature_columns"])
    rows = []
    for filename in FILES:
        path = DATA / filename
        sample = pd.read_csv(path, encoding="utf-8-sig", nrows=1000)
        for column in sample.columns:
            predictive = column not in {"row_id", "control_success"}
            is_trackman = filename == "trackman_history.csv"
            direct = predictive and not is_trackman and column in used
            if column == "control_success": safety = "TRAIN-ONLY"
            elif is_trackman: safety = "TRAIN-ONLY"
            elif column == "row_id": safety = "SAFE"
            else: safety = "SAFE"
            rows.append({
                "dataset": filename, "column": column, "dtype": str(sample[column].dtype),
                "semantic_family": family(column), "predictive": predictive,
                "champion_direct": direct, "champion_derived_source": direct,
                "existing_experiment": "TrackMan closed axis" if is_trackman else "raw/derived champion coverage",
                "rule_safety": safety,
                "notes": "current-pitch TrackMan unavailable at inference" if is_trackman else "current-row official column",
            })
    out = ROOT / "artifacts/feature_inventory.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
