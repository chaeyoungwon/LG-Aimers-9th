"""Reconstruct and validate an official-data-only anonymous pitcher crosswalk.

Only ``train.csv`` and ``trackman_history.csv`` are used.  Main-data cumulative
pitch-mix counters are differenced within pitcher-season so that their profile
is comparable with the season-level TrackMan log.  Mapping is fitted separately
at each historical cutoff and is never informed by the validation season.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extratrees_runtime import (  # noqa: E402
    build_hgb52_features,
    fit_preprocessor,
    transform_features,
)

DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_TRACKMAN = Path("/Users/wooh/Documents/dev/open/data/trackman_history.csv")
DEFAULT_OUTPUT = ROOT / "artifacts" / "tm_crosswalk"
CHAMPION = ROOT / "artifacts" / "sub_et25_w020.zip"
CHAMPION_SHA256 = (
    "316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf"
)

PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")
MAIN_RATE_COLUMNS = {
    "fastball": "asof_pitcher_fastball_rate",
    "breaking": "asof_pitcher_breaking_rate",
    "offspeed": "asof_pitcher_offspeed_rate",
}
MAIN_USECOLS = (
    "season",
    "pitcher_id",
    "pitcher_hand",
    "asof_pitcher_pitchmix_n",
    *MAIN_RATE_COLUMNS.values(),
)
TM_USECOLS = (
    "season",
    "pitcher_trackman_id",
    "pitcher_hand",
    "pitch_type_group",
)
TM_PHYSICAL_COLUMNS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
TM_MODEL_USECOLS = (
    *TM_USECOLS,
    "balls_before",
    "strikes_before",
    "game_date",
    *TM_PHYSICAL_COLUMNS,
)
CUTOFFS = (2021, 2022, 2023, 2024)
FUTURE_VALIDATION = {2021: 2022, 2022: 2023, 2023: 2024}
MIN_SEASON_PITCHES = 20

# Fixed, interpretable distance weights.  Pitch mix remains the dominant term.
COUNT_PATTERN_WEIGHT = 0.10
PRESENCE_WEIGHT = 0.05
COVERAGE_RATIO_WEIGHT = 0.05
OVERLAP_WEIGHT = 0.05

# Fixed before inspecting crosswalk results.  Only HIGH can unlock modeling.
CONFIDENCE_RULES = {
    "HIGH": {"min_overlap": 2, "max_distance": 0.16, "min_margin": 0.025},
    "MEDIUM": {"min_overlap": 2, "max_distance": 0.24, "min_margin": 0.010},
    "LOW": {"min_overlap": 1, "max_distance": 0.35, "min_margin": 0.000},
}
MAPPING_GATE = {
    "min_pooled_high_row_coverage": 0.20,
    "min_latest_high_row_coverage": 0.15,
    "min_future_pairs": 20,
    "max_future_median_mix_l1": 0.15,
    "min_future_top10_rate": 0.60,
    "min_high_stability": 0.80,
    "max_high_assignment_fraction": 0.95,
}
FEATURE_SIGNAL_MIN_ABS_POOLED_CORR = 0.015
FEATURE_SIGNAL_MIN_ABS_LATEST_CORR = 0.005
MODEL_NAMES = ("accepted52_plus_tm", "tm_plus_context")
MODEL_CONTEXT_COLUMNS = (
    "season",
    "game_month",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "li",
    "pitcher_hand",
    "batter_hand",
)
MODEL_RECIPE = {
    "n_estimators": 25,
    "max_depth": None,
    "min_samples_leaf": 8,
    "max_features": "sqrt",
    "bootstrap": False,
    "n_jobs": -1,
    "random_state": 42,
}
MODEL_CACHE_VERSION = "tm-crosswalk-et25-v1"
BLEND_WEIGHTS = (0.01, 0.02, 0.05, 0.10)
LARGE_SIGNAL_GAIN = 2e-5
COUNT_SHRINKAGE = 50.0
OOF_CACHE_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
ET25_CACHE_DIR = ROOT / "artifacts" / "extratrees_size_ablation" / "cache"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def read_main(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=MAIN_USECOLS)


def read_trackman(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=TM_USECOLS)


def _main_group_counts(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    n = result["asof_pitcher_pitchmix_n"].astype("float64")
    for group, rate_column in MAIN_RATE_COLUMNS.items():
        result[f"cumulative_{group}_n"] = (n * result[rate_column]).fillna(0.0)
    known = result[[f"cumulative_{group}_n" for group in MAIN_RATE_COLUMNS]].sum(
        axis=1
    )
    result["cumulative_other_n"] = n - known
    return result


def build_main_fingerprints(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(MAIN_USECOLS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"main schema missing {missing}")
    work = _main_group_counts(frame)
    keys = ["pitcher_id", "season"]
    grouped = work.groupby(keys, observed=True)
    min_n = grouped["asof_pitcher_pitchmix_n"].transform("min")
    max_n = grouped["asof_pitcher_pitchmix_n"].transform("max")
    first = (
        work.loc[work["asof_pitcher_pitchmix_n"].eq(min_n)]
        .groupby(keys, as_index=False, observed=True)
        .first()
    )
    terminal = (
        work.loc[work["asof_pitcher_pitchmix_n"].eq(max_n)]
        .groupby(keys, as_index=False, observed=True)
        .last()
    )
    count_columns = [f"cumulative_{group}_n" for group in PITCH_GROUPS]
    result = terminal[
        [
            *keys,
            "pitcher_hand",
            "asof_pitcher_pitchmix_n",
            *MAIN_RATE_COLUMNS.values(),
            *count_columns,
        ]
    ].merge(
        first[[*keys, "asof_pitcher_pitchmix_n", *count_columns]],
        on=keys,
        suffixes=("_terminal", "_start"),
        validate="one_to_one",
    )
    result = result.rename(
        columns={
            "asof_pitcher_pitchmix_n_terminal": "terminal_pitchmix_n",
            "asof_pitcher_pitchmix_n_start": "start_pitchmix_n",
            **{
                column: f"terminal_{group}_rate"
                for group, column in MAIN_RATE_COLUMNS.items()
            },
        }
    )
    result["terminal_other_rate"] = 1.0 - result[
        [f"terminal_{group}_rate" for group in MAIN_RATE_COLUMNS]
    ].sum(axis=1)
    result["season_pitchmix_n"] = (
        result["terminal_pitchmix_n"] - result["start_pitchmix_n"]
    )
    denominator = result["season_pitchmix_n"].replace(0, np.nan)
    for group in PITCH_GROUPS:
        delta = (
            result[f"cumulative_{group}_n_terminal"]
            - result[f"cumulative_{group}_n_start"]
        )
        # Rates originate from integer group counts; remove floating roundoff only.
        delta = delta.where(delta.abs() > 1e-9, 0.0)
        result[f"season_{group}_rate"] = delta / denominator
    keep = [
        "pitcher_id",
        "season",
        "pitcher_hand",
        "start_pitchmix_n",
        "terminal_pitchmix_n",
        "season_pitchmix_n",
        *[f"terminal_{group}_rate" for group in PITCH_GROUPS],
        *[f"season_{group}_rate" for group in PITCH_GROUPS],
    ]
    result = result[keep].sort_values(["pitcher_id", "season"]).reset_index(drop=True)
    valid = result["season_pitchmix_n"].gt(0)
    rate_sum = result.loc[valid, [f"season_{g}_rate" for g in PITCH_GROUPS]].sum(axis=1)
    if not np.allclose(rate_sum, 1.0, atol=1e-8):
        raise ValueError("derived main season pitch-mix rates do not sum to one")
    if (result["season_pitchmix_n"] < 0).any():
        raise ValueError("main cumulative pitch-mix count decreased within season")
    return result


def build_tm_fingerprints(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(TM_USECOLS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"TrackMan schema missing {missing}")
    work = frame.copy()
    work["pitch_type_group"] = work["pitch_type_group"].fillna("other")
    work.loc[~work["pitch_type_group"].isin(PITCH_GROUPS), "pitch_type_group"] = "other"
    counts = (
        work.groupby(
            ["pitcher_trackman_id", "season", "pitcher_hand", "pitch_type_group"],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )
    result = counts.pivot_table(
        index=["pitcher_trackman_id", "season", "pitcher_hand"],
        columns="pitch_type_group",
        values="n",
        fill_value=0,
        observed=True,
    ).reset_index()
    result.columns.name = None
    for group in PITCH_GROUPS:
        if group not in result:
            result[group] = 0
    result["tm_n"] = result[list(PITCH_GROUPS)].sum(axis=1)
    for group in PITCH_GROUPS:
        result[f"tm_{group}_rate"] = result[group] / result["tm_n"]
    result = result[
        [
            "pitcher_trackman_id",
            "season",
            "pitcher_hand",
            "tm_n",
            *[f"tm_{group}_rate" for group in PITCH_GROUPS],
        ]
    ].sort_values(["pitcher_trackman_id", "season"]).reset_index(drop=True)
    return result


def infer_hand_map(
    main: pd.DataFrame, trackman: pd.DataFrame
) -> tuple[dict[object, object], list[dict]]:
    main_values = sorted(main["pitcher_hand"].dropna().unique().tolist())
    tm_values = sorted(trackman["pitcher_hand"].dropna().unique().tolist())
    if len(main_values) != len(tm_values):
        raise ValueError("hand namespaces have different cardinality")
    seasons = sorted(set(main["season"]) & set(trackman["season"]))
    evaluations = []
    for permutation in itertools.permutations(tm_values):
        mapping = dict(zip(main_values, permutation))
        error = 0.0
        for season in seasons:
            main_season = main[main["season"].eq(season)]
            tm_season = trackman[trackman["season"].eq(season)]
            for value in main_values:
                main_share = float(main_season["pitcher_hand"].eq(value).mean())
                tm_share = float(tm_season["pitcher_hand"].eq(mapping[value]).mean())
                error += (main_share - tm_share) ** 2
        evaluations.append({"mapping": mapping, "prevalence_squared_error": error})
    evaluations.sort(key=lambda row: row["prevalence_squared_error"])
    if len(evaluations) > 1 and not (
        evaluations[0]["prevalence_squared_error"]
        < evaluations[1]["prevalence_squared_error"]
    ):
        raise ValueError("official data cannot identify the hand-code permutation")
    return evaluations[0]["mapping"], evaluations


def active_main(main: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    return main[
        main["season"].le(cutoff)
        & main["season_pitchmix_n"].ge(MIN_SEASON_PITCHES)
    ].copy()


def active_tm(trackman: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    return trackman[
        trackman["season"].le(cutoff) & trackman["tm_n"].ge(MIN_SEASON_PITCHES)
    ].copy()


def pair_distance(
    main_profile: pd.DataFrame,
    tm_profile: pd.DataFrame,
) -> dict:
    overlap = sorted(set(main_profile["season"]) & set(tm_profile["season"]))
    union = sorted(set(main_profile["season"]) | set(tm_profile["season"]))
    if not overlap:
        return {
            "distance": math.inf,
            "pitchmix_distance": math.inf,
            "count_distance": math.inf,
            "season_presence_penalty": 1.0,
            "coverage_ratio_penalty": math.inf,
            "seasons_overlap": 0,
            "seasons_union": len(union),
        }
    left = main_profile.set_index("season").loc[overlap]
    right = tm_profile.set_index("season").loc[overlap]
    main_mix = left[[f"season_{g}_rate" for g in PITCH_GROUPS]].to_numpy()
    tm_mix = right[[f"tm_{g}_rate" for g in PITCH_GROUPS]].to_numpy()
    weights = np.sqrt(
        np.minimum(
            left["season_pitchmix_n"].to_numpy(dtype="float64"),
            right["tm_n"].to_numpy(dtype="float64"),
        )
    )
    mix_by_season = np.abs(main_mix - tm_mix).sum(axis=1)
    pitchmix_distance = float(np.average(mix_by_season, weights=weights))
    log_ratio = np.log1p(left["season_pitchmix_n"].to_numpy()) - np.log1p(
        right["tm_n"].to_numpy()
    )
    centered = log_ratio - np.median(log_ratio)
    count_distance = float(np.sqrt(np.average(centered**2, weights=weights)))
    coverage_ratio_penalty = float(min(abs(np.median(log_ratio)), 2.0))
    presence_penalty = 1.0 - len(overlap) / len(union)
    overlap_penalty = 1.0 / len(overlap)
    distance = (
        pitchmix_distance
        + COUNT_PATTERN_WEIGHT * count_distance
        + PRESENCE_WEIGHT * presence_penalty
        + COVERAGE_RATIO_WEIGHT * coverage_ratio_penalty
        + OVERLAP_WEIGHT * overlap_penalty
    )
    return {
        "distance": distance,
        "pitchmix_distance": pitchmix_distance,
        "count_distance": count_distance,
        "season_presence_penalty": presence_penalty,
        "coverage_ratio_penalty": coverage_ratio_penalty,
        "seasons_overlap": len(overlap),
        "seasons_union": len(union),
    }


def build_distance_table(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    cutoff: int,
    hand_map: Mapping,
) -> pd.DataFrame:
    main_active = active_main(main, cutoff)
    tm_active = active_tm(trackman, cutoff)
    main_groups = {
        key: group for key, group in main_active.groupby("pitcher_id", observed=True)
    }
    tm_groups = {
        key: group
        for key, group in tm_active.groupby("pitcher_trackman_id", observed=True)
    }
    tm_hand = tm_active.groupby("pitcher_trackman_id", observed=True)[
        "pitcher_hand"
    ].first()
    rows = []
    for main_id, main_profile in main_groups.items():
        main_hand = main_profile["pitcher_hand"].iloc[0]
        expected_tm_hand = hand_map[main_hand]
        for tm_id, tm_profile in tm_groups.items():
            if tm_hand.loc[tm_id] != expected_tm_hand:
                continue
            metrics = pair_distance(main_profile, tm_profile)
            if math.isfinite(metrics["distance"]):
                rows.append(
                    {
                        "cutoff": cutoff,
                        "main_pitcher_id": main_id,
                        "tm_pitcher_id": tm_id,
                        "main_hand": main_hand,
                        "tm_hand": expected_tm_hand,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def confidence_tier(
    distance: float,
    margin: float,
    seasons_overlap: int,
    mutual_nearest: bool,
    method: str,
) -> str:
    for tier in ("HIGH", "MEDIUM", "LOW"):
        rule = CONFIDENCE_RULES[tier]
        passed = (
            seasons_overlap >= rule["min_overlap"]
            and distance <= rule["max_distance"]
            and margin >= rule["min_margin"]
        )
        if method == "nearest_neighbor" and tier in {"HIGH", "MEDIUM"}:
            passed = passed and mutual_nearest
        if passed:
            return tier
    return "UNMATCHED"


def assign_candidates(distances: pd.DataFrame, method: str) -> pd.DataFrame:
    if method not in {"nearest_neighbor", "hungarian"}:
        raise ValueError(method)
    ranked = distances.sort_values(
        ["main_pitcher_id", "distance", "tm_pitcher_id"]
    ).copy()
    ranked["rank"] = ranked.groupby("main_pitcher_id").cumcount() + 1
    best = ranked[ranked["rank"].eq(1)].set_index("main_pitcher_id")
    second = ranked[ranked["rank"].eq(2)].set_index("main_pitcher_id")["distance"]
    tm_best_main = (
        distances.sort_values(["tm_pitcher_id", "distance", "main_pitcher_id"])
        .groupby("tm_pitcher_id", observed=True)
        .first()["main_pitcher_id"]
    )
    if method == "nearest_neighbor":
        assigned = best.reset_index()
    else:
        main_ids = sorted(distances["main_pitcher_id"].unique())
        tm_ids = sorted(distances["tm_pitcher_id"].unique())
        main_index = {value: index for index, value in enumerate(main_ids)}
        tm_index = {value: index for index, value in enumerate(tm_ids)}
        cost = np.full((len(main_ids), len(tm_ids)), 1e6, dtype="float64")
        for row in distances.itertuples(index=False):
            cost[main_index[row.main_pitcher_id], tm_index[row.tm_pitcher_id]] = row.distance
        row_indices, column_indices = linear_sum_assignment(cost)
        pairs = pd.DataFrame(
            {
                "main_pitcher_id": [main_ids[index] for index in row_indices],
                "tm_pitcher_id": [tm_ids[index] for index in column_indices],
            }
        )
        assigned = pairs.merge(
            distances,
            on=["main_pitcher_id", "tm_pitcher_id"],
            how="inner",
            validate="one_to_one",
        )
    assigned["best_distance"] = assigned["distance"]
    assigned["second_best_distance"] = assigned["main_pitcher_id"].map(second)
    assigned["margin"] = assigned["second_best_distance"] - assigned["best_distance"]
    assigned["confidence_score"] = np.clip(
        assigned["margin"] / (assigned["second_best_distance"] + 1e-12),
        0.0,
        1.0,
    ) * np.minimum(assigned["seasons_overlap"] / 3.0, 1.0)
    assigned["mutual_nearest"] = [
        tm_best_main.get(tm_id) == main_id
        for main_id, tm_id in zip(
            assigned["main_pitcher_id"], assigned["tm_pitcher_id"]
        )
    ]
    assigned["method"] = method
    assigned["confidence"] = [
        confidence_tier(distance, margin, overlap, mutual, method)
        for distance, margin, overlap, mutual in zip(
            assigned["best_distance"],
            assigned["margin"],
            assigned["seasons_overlap"],
            assigned["mutual_nearest"],
        )
    ]
    assigned["accepted"] = assigned["confidence"].isin(["HIGH", "MEDIUM"])
    columns = [
        "cutoff",
        "method",
        "main_pitcher_id",
        "tm_pitcher_id",
        "best_distance",
        "second_best_distance",
        "margin",
        "confidence_score",
        "confidence",
        "accepted",
        "mutual_nearest",
        "seasons_overlap",
        "seasons_union",
        "pitchmix_distance",
        "count_distance",
        "season_presence_penalty",
        "coverage_ratio_penalty",
        "main_hand",
        "tm_hand",
    ]
    return assigned[columns].sort_values("main_pitcher_id").reset_index(drop=True)


def future_validate(
    mappings: pd.DataFrame,
    main: pd.DataFrame,
    trackman: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for mapping in mappings.itertuples(index=False):
        if mapping.cutoff not in FUTURE_VALIDATION or mapping.confidence == "UNMATCHED":
            continue
        season = FUTURE_VALIDATION[mapping.cutoff]
        main_future = main[
            main["pitcher_id"].eq(mapping.main_pitcher_id)
            & main["season"].eq(season)
            & main["season_pitchmix_n"].ge(MIN_SEASON_PITCHES)
        ]
        tm_future = trackman[
            trackman["pitcher_trackman_id"].eq(mapping.tm_pitcher_id)
            & trackman["season"].eq(season)
            & trackman["tm_n"].ge(MIN_SEASON_PITCHES)
        ]
        if len(main_future) != 1 or len(tm_future) != 1:
            continue
        main_mix = main_future[
            [f"season_{group}_rate" for group in PITCH_GROUPS]
        ].to_numpy()[0]
        tm_mix = tm_future[[f"tm_{group}_rate" for group in PITCH_GROUPS]].to_numpy()[0]
        mix_l1 = float(np.abs(main_mix - tm_mix).sum())
        alternatives = trackman[
            trackman["season"].eq(season)
            & trackman["pitcher_hand"].eq(mapping.tm_hand)
            & trackman["tm_n"].ge(MIN_SEASON_PITCHES)
        ]
        alt_mix = alternatives[
            [f"tm_{group}_rate" for group in PITCH_GROUPS]
        ].to_numpy()
        distances = np.abs(alt_mix - main_mix).sum(axis=1)
        rank = 1 + int(np.sum(distances < mix_l1 - 1e-15))
        rows.append(
            {
                "method": mapping.method,
                "cutoff": mapping.cutoff,
                "validation_season": season,
                "main_pitcher_id": mapping.main_pitcher_id,
                "tm_pitcher_id": mapping.tm_pitcher_id,
                "confidence": mapping.confidence,
                "future_mix_l1": mix_l1,
                "future_log_volume_abs_diff": abs(
                    math.log1p(float(main_future["season_pitchmix_n"].iloc[0]))
                    - math.log1p(float(tm_future["tm_n"].iloc[0]))
                ),
                "future_rank": rank,
                "future_candidate_count": len(alternatives),
                "future_rank_percentile": rank / len(alternatives),
                "future_top1": rank == 1,
                "future_top5": rank <= 5,
                "future_top10_percent": rank <= max(1, math.ceil(0.10 * len(alternatives))),
            }
        )
    return pd.DataFrame(rows)


def mapping_stability(mappings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in sorted(mappings["method"].unique()):
        subset = mappings[
            mappings["method"].eq(method) & mappings["confidence"].eq("HIGH")
        ]
        for left_cutoff, right_cutoff in zip(CUTOFFS[:-1], CUTOFFS[1:]):
            left = subset[subset["cutoff"].eq(left_cutoff)].set_index(
                "main_pitcher_id"
            )["tm_pitcher_id"]
            right = subset[subset["cutoff"].eq(right_cutoff)].set_index(
                "main_pitcher_id"
            )["tm_pitcher_id"]
            common = left.index.intersection(right.index)
            stable = left.loc[common].to_numpy() == right.loc[common].to_numpy()
            rows.append(
                {
                    "method": method,
                    "left_cutoff": left_cutoff,
                    "right_cutoff": right_cutoff,
                    "common_high_pitchers": len(common),
                    "stable_pitchers": int(stable.sum()),
                    "stability_rate": float(stable.mean()) if len(stable) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def coverage_table(
    mappings: pd.DataFrame, train: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for validation_season in (2022, 2023, 2024):
        cutoff = validation_season - 1
        validation = train[train["season"].eq(validation_season)]
        for method in sorted(mappings["method"].unique()):
            current = mappings[
                mappings["method"].eq(method) & mappings["cutoff"].eq(cutoff)
            ]
            matched_ids = set(
                current.loc[current["accepted"], "main_pitcher_id"].tolist()
            )
            high_ids = set(
                current.loc[
                    current["confidence"].eq("HIGH"), "main_pitcher_id"
                ].tolist()
            )
            matched_rows = validation["pitcher_id"].isin(matched_ids)
            high_rows = validation["pitcher_id"].isin(high_ids)
            validation_ids = set(validation["pitcher_id"].unique())
            rows.append(
                {
                    "method": method,
                    "validation_season": validation_season,
                    "cutoff": cutoff,
                    "validation_rows": len(validation),
                    "matched_validation_rows": int(matched_rows.sum()),
                    "high_confidence_matched_rows": int(high_rows.sum()),
                    "validation_pitchers": len(validation_ids),
                    "matched_pitchers": len(validation_ids & matched_ids),
                    "high_confidence_matched_pitchers": len(validation_ids & high_ids),
                    "row_coverage": float(matched_rows.mean()),
                    "high_row_coverage": float(high_rows.mean()),
                    "pitcher_coverage": len(validation_ids & matched_ids)
                    / len(validation_ids),
                    "high_pitcher_coverage": len(validation_ids & high_ids)
                    / len(validation_ids),
                }
            )
    return pd.DataFrame(rows)


def aggregate_future(future: pd.DataFrame, method: str) -> dict:
    current = future[
        future["method"].eq(method) & future["confidence"].eq("HIGH")
    ]
    return {
        "n": len(current),
        "median_future_mix_l1": float(current["future_mix_l1"].median())
        if len(current)
        else None,
        "median_future_rank_percentile": float(
            current["future_rank_percentile"].median()
        )
        if len(current)
        else None,
        "future_top1_rate": float(current["future_top1"].mean())
        if len(current)
        else None,
        "future_top5_rate": float(current["future_top5"].mean())
        if len(current)
        else None,
        "future_top10_percent_rate": float(
            current["future_top10_percent"].mean()
        )
        if len(current)
        else None,
    }


def evaluate_mapping_gate(
    mappings: pd.DataFrame,
    stability: pd.DataFrame,
    future: pd.DataFrame,
    coverage: pd.DataFrame,
    method: str,
) -> dict:
    current_coverage = coverage[coverage["method"].eq(method)]
    pooled_high = current_coverage["high_confidence_matched_rows"].sum() / current_coverage[
        "validation_rows"
    ].sum()
    latest_high = float(
        current_coverage.loc[
            current_coverage["validation_season"].eq(2024), "high_row_coverage"
        ].iloc[0]
    )
    future_summary = aggregate_future(future, method)
    current_stability = stability[
        stability["method"].eq(method) & stability["stability_rate"].notna()
    ]
    weighted_stability = (
        current_stability["stable_pitchers"].sum()
        / current_stability["common_high_pitchers"].sum()
        if current_stability["common_high_pitchers"].sum()
        else 0.0
    )
    current_mappings = mappings[mappings["method"].eq(method)]
    high_fraction = float(current_mappings["confidence"].eq("HIGH").mean())
    checks = {
        "pooled_high_row_coverage": pooled_high
        >= MAPPING_GATE["min_pooled_high_row_coverage"],
        "latest_high_row_coverage": latest_high
        >= MAPPING_GATE["min_latest_high_row_coverage"],
        "future_pair_count": future_summary["n"] >= MAPPING_GATE["min_future_pairs"],
        "future_median_mix_l1": future_summary["median_future_mix_l1"] is not None
        and future_summary["median_future_mix_l1"]
        <= MAPPING_GATE["max_future_median_mix_l1"],
        "future_top10_rate": future_summary["future_top10_percent_rate"] is not None
        and future_summary["future_top10_percent_rate"]
        >= MAPPING_GATE["min_future_top10_rate"],
        "high_mapping_stability": weighted_stability
        >= MAPPING_GATE["min_high_stability"],
        "ambiguity_excluded": high_fraction
        <= MAPPING_GATE["max_high_assignment_fraction"],
    }
    return {
        "method": method,
        "passed": all(checks.values()),
        "checks": checks,
        "pooled_high_row_coverage": pooled_high,
        "latest_high_row_coverage": latest_high,
        "weighted_high_stability": weighted_stability,
        "high_assignment_fraction": high_fraction,
        "future_validation": future_summary,
    }


def read_trackman_physical(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=TM_MODEL_USECOLS)
    frame["pitch_type_group"] = frame["pitch_type_group"].fillna("other")
    frame.loc[~frame["pitch_type_group"].isin(PITCH_GROUPS), "pitch_type_group"] = "other"
    for column in TM_PHYSICAL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _flatten_aggregate(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    aggregate = frame.groupby("pitcher_trackman_id", observed=True)[
        list(TM_PHYSICAL_COLUMNS)
    ].agg(["mean", "std"])
    aggregate.columns = [
        f"{prefix}_{feature}_{stat}" for feature, stat in aggregate.columns
    ]
    return aggregate.reset_index()


def build_physical_profiles(trackman: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    history = trackman[trackman["season"].le(cutoff)].copy()
    if history["season"].gt(cutoff).any():
        raise AssertionError("future TrackMan row entered physical profile")
    career = _flatten_aggregate(history, "career")
    counts = history.groupby("pitcher_trackman_id", observed=True).size().rename(
        "tm_history_n"
    )
    result = career.merge(counts.reset_index(), on="pitcher_trackman_id")

    recent = history[history["season"].eq(cutoff)]
    recent_profile = _flatten_aggregate(recent, "recent")
    recent_counts = recent.groupby("pitcher_trackman_id", observed=True).size().rename(
        "tm_recent_n"
    )
    result = result.merge(recent_profile, on="pitcher_trackman_id", how="left")
    result = result.merge(
        recent_counts.reset_index(), on="pitcher_trackman_id", how="left"
    )
    for feature in TM_PHYSICAL_COLUMNS:
        for stat in ("mean", "std"):
            result[f"delta_{feature}_{stat}"] = (
                result[f"recent_{feature}_{stat}"]
                - result[f"career_{feature}_{stat}"]
            )

    pitch_specs = {
        "fastball": (
            "rel_speed",
            "spin_rate",
            "induced_vert_break",
            "horz_break",
        ),
        "breaking": (
            "rel_speed",
            "spin_rate",
            "induced_vert_break",
            "horz_break",
        ),
        "offspeed": ("rel_speed", "induced_vert_break", "horz_break"),
    }
    for group, features in pitch_specs.items():
        current = history[history["pitch_type_group"].eq(group)]
        means = current.groupby("pitcher_trackman_id", observed=True)[
            list(features)
        ].mean()
        means.columns = [f"{group}_{feature}_mean" for feature in means.columns]
        group_n = current.groupby("pitcher_trackman_id", observed=True).size().rename(
            f"{group}_n"
        )
        result = result.merge(
            means.reset_index(), on="pitcher_trackman_id", how="left"
        ).merge(group_n.reset_index(), on="pitcher_trackman_id", how="left")

    result["fastball_breaking_velocity_gap"] = (
        result["fastball_rel_speed_mean"] - result["breaking_rel_speed_mean"]
    )
    result["fastball_offspeed_velocity_gap"] = (
        result["fastball_rel_speed_mean"] - result["offspeed_rel_speed_mean"]
    )
    result["fastball_breaking_movement_separation"] = np.sqrt(
        (result["fastball_induced_vert_break_mean"] - result["breaking_induced_vert_break_mean"]) ** 2
        + (result["fastball_horz_break_mean"] - result["breaking_horz_break_mean"]) ** 2
    )
    result["fastball_offspeed_movement_separation"] = np.sqrt(
        (result["fastball_induced_vert_break_mean"] - result["offspeed_induced_vert_break_mean"]) ** 2
        + (result["fastball_horz_break_mean"] - result["offspeed_horz_break_mean"]) ** 2
    )

    # Cutoff-only normalization for interpretable consistency magnitudes.
    consistency = {
        "release_spread": ("career_rel_height_std", "career_rel_side_std"),
        "movement_variability": (
            "career_induced_vert_break_std",
            "career_horz_break_std",
        ),
    }
    for output, inputs in consistency.items():
        squared = np.zeros(len(result), dtype="float64")
        valid = np.ones(len(result), dtype=bool)
        for column in inputs:
            values = result[column].to_numpy(dtype="float64")
            finite = np.isfinite(values)
            center = float(np.nanmean(values))
            scale = float(np.nanstd(values))
            scale = scale if scale > 1e-12 else 1.0
            squared += ((values - center) / scale) ** 2
            valid &= finite
        result[output] = np.where(valid, np.sqrt(squared), np.nan)
    speed_std = result["career_rel_speed_std"].to_numpy(dtype="float64")
    result["velocity_variability"] = (
        speed_std - np.nanmean(speed_std)
    ) / max(float(np.nanstd(speed_std)), 1e-12)

    # Mutually exclusive current-count buckets; sparse cells shrink to career.
    history["count_bucket"] = np.select(
        [history["balls_before"].eq(3), history["strikes_before"].eq(2)],
        ["three_ball", "two_strike"],
        default="neutral",
    )
    count_features = (
        "rel_speed",
        "induced_vert_break",
        "horz_break",
        "rel_height",
        "rel_side",
    )
    for bucket in ("three_ball", "two_strike", "neutral"):
        current = history[history["count_bucket"].eq(bucket)]
        grouped = current.groupby("pitcher_trackman_id", observed=True)
        means = grouped[list(count_features)].mean()
        bucket_n = grouped.size().rename(f"count_{bucket}_n")
        means = means.merge(bucket_n, left_index=True, right_index=True)
        for feature in count_features:
            local = means[feature]
            career_mean = result.set_index("pitcher_trackman_id")[
                f"career_{feature}_mean"
            ].reindex(means.index)
            weight = means[f"count_{bucket}_n"] / (
                means[f"count_{bucket}_n"] + COUNT_SHRINKAGE
            )
            means[f"count_{bucket}_{feature}_mean"] = (
                weight * local + (1.0 - weight) * career_mean
            )
        keep = [
            f"count_{bucket}_n",
            *[f"count_{bucket}_{feature}_mean" for feature in count_features],
        ]
        result = result.merge(
            means[keep].reset_index(), on="pitcher_trackman_id", how="left"
        )
    count_columns = [column for column in result if column.endswith("_n")]
    result[count_columns] = result[count_columns].fillna(0.0)
    return result.sort_values("pitcher_trackman_id").reset_index(drop=True)


def build_main_physical_lookup(
    mappings: pd.DataFrame,
    profiles: pd.DataFrame,
    cutoff: int,
    method: str,
) -> pd.DataFrame:
    selected = mappings[
        mappings["cutoff"].eq(cutoff)
        & mappings["method"].eq(method)
        & mappings["confidence"].eq("HIGH")
    ][
        [
            "main_pitcher_id",
            "tm_pitcher_id",
            "best_distance",
            "margin",
            "seasons_overlap",
        ]
    ].copy()
    selected = selected.rename(
        columns={"main_pitcher_id": "pitcher_id", "tm_pitcher_id": "pitcher_trackman_id"}
    )
    lookup = selected.merge(
        profiles, on="pitcher_trackman_id", how="inner", validate="one_to_one"
    )
    lookup["tm_matched"] = 1.0
    lookup["tm_confidence"] = np.clip(
        lookup["margin"] / (lookup["best_distance"] + lookup["margin"] + 1e-12),
        0.0,
        1.0,
    )
    return lookup


def attach_physical_features(rows: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    metadata = {
        "pitcher_id",
        "pitcher_trackman_id",
        "best_distance",
        "margin",
        "seasons_overlap",
    }
    feature_columns = [column for column in lookup.columns if column not in metadata]
    attached = rows[["pitcher_id"]].merge(
        lookup[["pitcher_id", *feature_columns]],
        on="pitcher_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    attached.index = rows.index
    attached["tm_matched"] = attached["tm_matched"].fillna(0.0)
    attached["tm_confidence"] = attached["tm_confidence"].fillna(0.0)
    attached["tm_history_n"] = attached["tm_history_n"].fillna(0.0)
    return attached[feature_columns]


def load_current_champion_oof(season: int, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oof = np.load(OOF_CACHE_DIR / f"season_{season}.npz", allow_pickle=False)
    et = np.load(
        ET25_CACHE_DIR / f"season_{season}_prefixes.npz", allow_pickle=False
    )
    if not np.array_equal(oof["y"].astype("float64"), target):
        raise ValueError(f"champion OOF target/order mismatch for {season}")
    original = oof["champion_final"].astype("float64")
    et25 = et["prediction_25"].astype("float64")
    return 0.98 * original + 0.02 * et25, et25


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((target - prediction) ** 2))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.std(left[finite]) <= 1e-15 or np.std(right[finite]) <= 1e-15:
        return np.nan
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def feature_family(column: str) -> str:
    if column in {"release_spread", "career_rel_height_std", "career_rel_side_std"}:
        return "release_consistency"
    if "rel_speed" in column or "velocity" in column or "zone_speed" in column:
        return "velocity"
    if "break" in column or "movement" in column:
        return "movement"
    if "extension" in column:
        return "extension"
    if "spin" in column:
        return "spin"
    if column.startswith("recent_") or column.startswith("delta_"):
        return "recency"
    if column.startswith("count_"):
        return "count_conditioned"
    if column.startswith(("fastball_", "breaking_", "offspeed_")):
        return "pitch_type"
    return "history_or_mapping"


def feature_diagnostics(
    train: pd.DataFrame,
    trackman: pd.DataFrame,
    mappings: pd.DataFrame,
    method: str,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    rows = []
    lookups = {}
    validation_features = {}
    pooled_values: dict[str, list[np.ndarray]] = {}
    pooled_residuals: dict[str, list[np.ndarray]] = {}
    for season in (2022, 2023, 2024):
        cutoff = season - 1
        profile = build_physical_profiles(trackman, cutoff)
        lookup = build_main_physical_lookup(mappings, profile, cutoff, method)
        lookups[season] = lookup
        validation = train[train["season"].eq(season)]
        physical = attach_physical_features(validation, lookup)
        validation_features[season] = physical
        target = validation["control_success"].to_numpy(dtype="float64")
        champion, _ = load_current_champion_oof(season, target)
        residual = target - champion
        matched = physical["tm_matched"].eq(1.0).to_numpy()
        for column in physical.columns:
            if column == "tm_matched":
                continue
            values = physical[column].to_numpy(dtype="float64")
            mask = matched & np.isfinite(values)
            correlation = safe_corr(values[mask], residual[mask])
            rows.append(
                {
                    "family": feature_family(column),
                    "feature": column,
                    "validation_season": season,
                    "n": int(mask.sum()),
                    "residual_correlation": correlation,
                    "status": "evaluated" if mask.sum() >= 100 else "insufficient_rows",
                }
            )
            if mask.sum() >= 100:
                pooled_values.setdefault(column, []).append(values[mask])
                pooled_residuals.setdefault(column, []).append(residual[mask])
    for column, values in pooled_values.items():
        residual = pooled_residuals[column]
        rows.append(
            {
                "family": feature_family(column),
                "feature": column,
                "validation_season": "pooled",
                "n": sum(len(value) for value in values),
                "residual_correlation": safe_corr(
                    np.concatenate(values), np.concatenate(residual)
                ),
                "status": "evaluated",
            }
        )
    return pd.DataFrame(rows), lookups, validation_features


def feature_signal_gate(results: pd.DataFrame) -> dict:
    pivot = results.pivot_table(
        index="feature", columns="validation_season", values="residual_correlation"
    )
    passing = []
    for feature, row in pivot.iterrows():
        if not all(key in row.index for key in (2023, 2024, "pooled")):
            continue
        values = [row[2023], row[2024], row["pooled"]]
        if not np.isfinite(values).all():
            continue
        stable_sign = np.sign(row[2023]) == np.sign(row[2024]) != 0
        if (
            stable_sign
            and abs(row[2023]) >= FEATURE_SIGNAL_MIN_ABS_LATEST_CORR
            and abs(row[2024]) >= FEATURE_SIGNAL_MIN_ABS_LATEST_CORR
            and abs(row["pooled"]) >= FEATURE_SIGNAL_MIN_ABS_POOLED_CORR
        ):
            passing.append(
                {
                    "feature": feature,
                    "corr_2023": row[2023],
                    "corr_2024": row[2024],
                    "corr_pooled": row["pooled"],
                }
            )
    passing.sort(key=lambda item: abs(item["corr_pooled"]), reverse=True)
    return {
        "passed": bool(passing),
        "thresholds": {
            "min_abs_latest_corr": FEATURE_SIGNAL_MIN_ABS_LATEST_CORR,
            "min_abs_pooled_corr": FEATURE_SIGNAL_MIN_ABS_POOLED_CORR,
            "same_sign_2023_2024": True,
        },
        "passing_features": passing,
    }


def load_champion_meta() -> dict:
    with zipfile.ZipFile(CHAMPION) as archive:
        return json.loads(archive.read("model/meta.json"))


def model_cache_path(output: Path, learner: str, season: int) -> Path:
    return output / "cache" / f"{learner}_{season}.npz"


def frame_signature(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    hashed = pd.util.hash_pandas_object(
        frame[list(columns)].reset_index(drop=True), index=False
    ).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def train_model_fold(
    learner: str,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    train_physical: pd.DataFrame,
    validation_physical: pd.DataFrame,
    champion_meta: Mapping,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    train_matched = train_physical["tm_matched"].eq(1.0).to_numpy()
    validation_matched = validation_physical["tm_matched"].eq(1.0).to_numpy()
    if train_matched.sum() < 50_000:
        raise RuntimeError(f"{learner}: fewer than 50,000 matched training rows")
    train_rows = training.loc[train_matched]
    validation_rows = validation.loc[validation_matched]
    physical_columns = list(train_physical.columns)
    if learner == "accepted52_plus_tm":
        train_base = build_hgb52_features(train_rows, champion_meta["cat_levels"])
        validation_base = build_hgb52_features(
            validation_rows, champion_meta["cat_levels"]
        )
    elif learner == "tm_plus_context":
        train_base = train_rows[list(MODEL_CONTEXT_COLUMNS)].copy()
        validation_base = validation_rows[list(MODEL_CONTEXT_COLUMNS)].copy()
    else:
        raise ValueError(learner)
    train_features = pd.concat(
        [
            train_base.reset_index(drop=True),
            train_physical.loc[train_matched, physical_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    validation_features = pd.concat(
        [
            validation_base.reset_index(drop=True),
            validation_physical.loc[
                validation_matched, physical_columns
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    preprocessor = fit_preprocessor(train_features)
    x_train = transform_features(train_features, preprocessor)
    x_validation = transform_features(validation_features, preprocessor)
    target = train_rows["control_success"].to_numpy(dtype="int8")
    model = ExtraTreesClassifier(**MODEL_RECIPE)
    started = time.monotonic()
    model.fit(x_train, target)
    fit_seconds = time.monotonic() - started
    started = time.monotonic()
    prediction = model.predict_proba(x_validation)[:, 1].astype("float64")
    predict_seconds = time.monotonic() - started
    if not np.isfinite(prediction).all():
        raise ValueError(f"{learner} produced non-finite prediction")
    del model, x_train, x_validation, train_features, validation_features
    gc.collect()
    return prediction, validation_matched, fit_seconds, predict_seconds


def error_top10_win_rate(
    target: np.ndarray, champion: np.ndarray, challenger: np.ndarray
) -> float:
    champion_error = (target - champion) ** 2
    threshold = np.quantile(champion_error, 0.90)
    mask = champion_error >= threshold
    return float(
        np.mean((target[mask] - challenger[mask]) ** 2 < champion_error[mask])
    )


def run_trackman_models(
    train: pd.DataFrame,
    trackman: pd.DataFrame,
    mappings: pd.DataFrame,
    method: str,
    output: Path,
    lookups: Mapping[int, pd.DataFrame],
    validation_features: Mapping[int, pd.DataFrame],
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    champion_meta = load_champion_meta()
    all_results = []
    all_complement = []
    prediction_folds: dict[str, dict[int, dict]] = {name: {} for name in MODEL_NAMES}
    (output / "cache").mkdir(parents=True, exist_ok=True)
    for season in (2022, 2023, 2024):
        cutoff = season - 1
        training = train[train["season"].le(cutoff)]
        validation = train[train["season"].eq(season)]
        target = validation["control_success"].to_numpy(dtype="float64")
        champion, et25 = load_current_champion_oof(season, target)
        profile = build_physical_profiles(trackman, cutoff)
        lookup = lookups.get(season)
        if lookup is None:
            lookup = build_main_physical_lookup(mappings, profile, cutoff, method)
        train_physical = attach_physical_features(training, lookup)
        validation_physical = validation_features.get(season)
        if validation_physical is None:
            validation_physical = attach_physical_features(validation, lookup)
        for learner in MODEL_NAMES:
            cache = model_cache_path(output, learner, season)
            validation_signature = frame_signature(validation, ["row_id"])
            lookup_signature = frame_signature(
                lookup.sort_values("pitcher_id"),
                ["pitcher_id", "pitcher_trackman_id"],
            )
            cache_is_valid = False
            if cache.is_file() and not force:
                payload = np.load(cache, allow_pickle=False)
                cache_is_valid = (
                    "cache_version" in payload
                    and str(payload["cache_version"].item()) == MODEL_CACHE_VERSION
                    and int(payload["season"].item()) == season
                    and str(payload["learner"].item()) == learner
                    and str(payload["validation_signature"].item())
                    == validation_signature
                    and str(payload["lookup_signature"].item()) == lookup_signature
                )
            if cache_is_valid:
                prediction = payload["prediction"].astype("float64")
                matched = payload["matched"].astype(bool)
                fit_seconds = float(payload["fit_seconds"])
                predict_seconds = float(payload["predict_seconds"])
                if len(matched) != len(validation) or len(prediction) != matched.sum():
                    raise ValueError(f"invalid model cache shape {cache}")
                print(f"[{learner} {season}] cached", flush=True)
            else:
                prediction, matched, fit_seconds, predict_seconds = train_model_fold(
                    learner,
                    training,
                    validation,
                    train_physical,
                    validation_physical,
                    champion_meta,
                )
                np.savez_compressed(
                    cache,
                    cache_version=np.asarray(MODEL_CACHE_VERSION),
                    season=np.asarray(season),
                    learner=np.asarray(learner),
                    validation_signature=np.asarray(validation_signature),
                    lookup_signature=np.asarray(lookup_signature),
                    prediction=prediction,
                    matched=matched,
                    fit_seconds=np.asarray(fit_seconds),
                    predict_seconds=np.asarray(predict_seconds),
                )
                print(
                    f"[{learner} {season}] matched_validation={matched.sum():,} "
                    f"fit={fit_seconds:.1f}s predict={predict_seconds:.1f}s",
                    flush=True,
                )
            routed = champion.copy()
            routed[matched] = prediction
            matched_target = target[matched]
            matched_champion = champion[matched]
            matched_et25 = et25[matched]
            result = {
                "learner": learner,
                "validation_season": season,
                "n": len(target),
                "matched_n": int(matched.sum()),
                "matched_row_coverage": float(matched.mean()),
                "standalone_brier_matched": brier(matched_target, prediction),
                "champion_brier_matched": brier(matched_target, matched_champion),
                "matched_subset_gain": brier(matched_target, matched_champion)
                - brier(matched_target, prediction),
                "overall_brier": brier(target, routed),
                "champion_brier": brier(target, champion),
                "overall_gain": brier(target, champion) - brier(target, routed),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
            complement = {
                "learner": learner,
                "validation_season": season,
                "n": int(matched.sum()),
                "corr_champion": safe_corr(prediction, matched_champion),
                "corr_et25": safe_corr(prediction, matched_et25),
                "champion_top10_error_model_win_rate": error_top10_win_rate(
                    matched_target, matched_champion, prediction
                ),
            }
            all_results.append(result)
            all_complement.append(complement)
            prediction_folds[learner][season] = {
                "target": target,
                "champion": champion,
                "et25": et25,
                "matched": matched,
                "prediction": prediction,
                "routed": routed,
            }
        del profile, train_physical
        gc.collect()

    for learner in MODEL_NAMES:
        folds = prediction_folds[learner]
        target = np.concatenate([folds[season]["target"] for season in (2022, 2023, 2024)])
        champion = np.concatenate([folds[season]["champion"] for season in (2022, 2023, 2024)])
        matched = np.concatenate([folds[season]["matched"] for season in (2022, 2023, 2024)])
        prediction = np.concatenate([folds[season]["prediction"] for season in (2022, 2023, 2024)])
        routed = np.concatenate([folds[season]["routed"] for season in (2022, 2023, 2024)])
        et25 = np.concatenate([folds[season]["et25"][folds[season]["matched"]] for season in (2022, 2023, 2024)])
        matched_champion = champion[matched]
        matched_target = target[matched]
        season_rows = [row for row in all_results if row["learner"] == learner]
        all_results.append(
            {
                "learner": learner,
                "validation_season": "pooled",
                "n": len(target),
                "matched_n": int(matched.sum()),
                "matched_row_coverage": float(matched.mean()),
                "standalone_brier_matched": brier(matched_target, prediction),
                "champion_brier_matched": brier(matched_target, matched_champion),
                "matched_subset_gain": brier(matched_target, matched_champion) - brier(matched_target, prediction),
                "overall_brier": brier(target, routed),
                "champion_brier": brier(target, champion),
                "overall_gain": brier(target, champion) - brier(target, routed),
                "fit_seconds": sum(row["fit_seconds"] for row in season_rows),
                "predict_seconds": sum(row["predict_seconds"] for row in season_rows),
            }
        )
        all_complement.append(
            {
                "learner": learner,
                "validation_season": "pooled",
                "n": int(matched.sum()),
                "corr_champion": safe_corr(prediction, matched_champion),
                "corr_et25": safe_corr(prediction, et25),
                "champion_top10_error_model_win_rate": error_top10_win_rate(
                    matched_target, matched_champion, prediction
                ),
            }
        )

    blend_rows = []
    decisions = []
    for learner in MODEL_NAMES:
        folds = prediction_folds[learner]
        for weight in BLEND_WEIGHTS:
            gains = {}
            matched_gains = {}
            for season in (2022, 2023, 2024):
                fold = folds[season]
                candidate = fold["champion"].copy()
                matched = fold["matched"]
                candidate[matched] = (
                    (1.0 - weight) * fold["champion"][matched]
                    + weight * fold["prediction"]
                )
                gain = brier(fold["target"], fold["champion"]) - brier(
                    fold["target"], candidate
                )
                matched_gain = brier(
                    fold["target"][matched], fold["champion"][matched]
                ) - brier(fold["target"][matched], candidate[matched])
                gains[str(season)] = gain
                matched_gains[str(season)] = matched_gain
                blend_rows.append(
                    {
                        "learner": learner,
                        "weight": weight,
                        "validation_season": season,
                        "n": len(candidate),
                        "matched_n": int(matched.sum()),
                        "brier_gain": gain,
                        "matched_subset_gain": matched_gain,
                    }
                )
            target = np.concatenate([folds[s]["target"] for s in (2022, 2023, 2024)])
            champion = np.concatenate([folds[s]["champion"] for s in (2022, 2023, 2024)])
            candidates = []
            pooled_matched_candidate = []
            pooled_matched_target = []
            pooled_matched_champion = []
            for season in (2022, 2023, 2024):
                fold = folds[season]
                current = fold["champion"].copy()
                mask = fold["matched"]
                current[mask] = (1.0 - weight) * fold["champion"][mask] + weight * fold["prediction"]
                candidates.append(current)
                pooled_matched_candidate.append(current[mask])
                pooled_matched_target.append(fold["target"][mask])
                pooled_matched_champion.append(fold["champion"][mask])
            candidate = np.concatenate(candidates)
            pooled_gain = brier(target, champion) - brier(target, candidate)
            pooled_matched_gain = brier(
                np.concatenate(pooled_matched_target), np.concatenate(pooled_matched_champion)
            ) - brier(np.concatenate(pooled_matched_target), np.concatenate(pooled_matched_candidate))
            gains["pooled"] = pooled_gain
            matched_gains["pooled"] = pooled_matched_gain
            passed = gains["2023"] > 0 and gains["2024"] > 0 and pooled_gain > 0
            blend_rows.append(
                {
                    "learner": learner,
                    "weight": weight,
                    "validation_season": "pooled",
                    "n": len(candidate),
                    "matched_n": sum(int(folds[s]["matched"].sum()) for s in (2022, 2023, 2024)),
                    "brier_gain": pooled_gain,
                    "matched_subset_gain": pooled_matched_gain,
                }
            )
            decisions.append(
                {
                    "learner": learner,
                    "weight": weight,
                    "gains": gains,
                    "matched_subset_gains": matched_gains,
                    "passed": passed,
                    "strong": passed and pooled_gain >= LARGE_SIGNAL_GAIN,
                }
            )
    return (
        pd.DataFrame(all_results),
        pd.DataFrame(all_complement),
        pd.DataFrame(blend_rows),
        decisions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--trackman", type=Path, default=DEFAULT_TRACKMAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-crosswalk", action="store_true")
    parser.add_argument("--force-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256_path(CHAMPION) != CHAMPION_SHA256:
        raise SystemExit("immutable champion checksum mismatch before crosswalk audit")
    args.output.mkdir(parents=True, exist_ok=True)
    train = read_main(args.train)
    if args.reuse_crosswalk:
        preliminary = json.loads((args.output / "summary.json").read_text())
        main_fingerprints = pd.read_csv(args.output / "main_fingerprints.csv")
        tm_fingerprints = pd.read_csv(args.output / "tm_fingerprints.csv")
        mappings = pd.read_csv(args.output / "mapping_candidates.csv")
        if "confidence_score" not in mappings:
            mappings["confidence_score"] = np.clip(
                mappings["margin"] / (mappings["second_best_distance"] + 1e-12),
                0.0,
                1.0,
            ) * np.minimum(mappings["seasons_overlap"] / 3.0, 1.0)
            before = list(mappings.columns)
            confidence_index = before.index("confidence")
            before.remove("confidence_score")
            before.insert(confidence_index, "confidence_score")
            mappings = mappings[before]
            mappings.to_csv(args.output / "mapping_candidates.csv", index=False)
        stability = mapping_stability(mappings)
        future = future_validate(mappings, main_fingerprints, tm_fingerprints)
        coverage = coverage_table(mappings, train)
        stability.to_csv(args.output / "mapping_stability.csv", index=False)
        future.to_csv(args.output / "future_validation.csv", index=False)
        coverage.to_csv(args.output / "coverage.csv", index=False)
        hand_map = {
            int(key): value
            for key, value in preliminary["hand_mapping"]["selected"].items()
        }
        hand_evaluations = [
            {
                "mapping": {
                    int(key): value for key, value in row["mapping"].items()
                },
                "prevalence_squared_error": row["prevalence_squared_error"],
            }
            for row in preliminary["hand_mapping"]["evaluations"]
        ]
        print("[crosswalk] reused validated fingerprint artifacts", flush=True)
    else:
        trackman_raw = read_trackman(args.trackman)
        main_fingerprints = build_main_fingerprints(train)
        tm_fingerprints = build_tm_fingerprints(trackman_raw)
        hand_map, hand_evaluations = infer_hand_map(
            main_fingerprints, tm_fingerprints
        )
        main_fingerprints.to_csv(args.output / "main_fingerprints.csv", index=False)
        tm_fingerprints.to_csv(args.output / "tm_fingerprints.csv", index=False)

        assignments = []
        for cutoff in CUTOFFS:
            distances = build_distance_table(
                main_fingerprints, tm_fingerprints, cutoff, hand_map
            )
            for method in ("nearest_neighbor", "hungarian"):
                current = assign_candidates(distances, method)
                assignments.append(current)
                print(
                    f"[{cutoff} {method}] candidates={len(current)} "
                    f"high={(current['confidence'] == 'HIGH').sum()} "
                    f"medium={(current['confidence'] == 'MEDIUM').sum()}",
                    flush=True,
                )
        mappings = pd.concat(assignments, ignore_index=True)
        mappings.to_csv(args.output / "mapping_candidates.csv", index=False)
        stability = mapping_stability(mappings)
        stability.to_csv(args.output / "mapping_stability.csv", index=False)
        future = future_validate(mappings, main_fingerprints, tm_fingerprints)
        future.to_csv(args.output / "future_validation.csv", index=False)
        coverage = coverage_table(mappings, train)
        coverage.to_csv(args.output / "coverage.csv", index=False)

    gates = {
        method: evaluate_mapping_gate(
            mappings, stability, future, coverage, method
        )
        for method in ("nearest_neighbor", "hungarian")
    }
    accepted_methods = [method for method, gate in gates.items() if gate["passed"]]
    selected_method = "hungarian" if "hungarian" in accepted_methods else (
        accepted_methods[0] if accepted_methods else None
    )
    mapping_passed = selected_method is not None

    feature_gate = None
    model_results = pd.DataFrame()
    complementarity = pd.DataFrame()
    blends = pd.DataFrame()
    blend_decisions = []
    selected_candidate = None
    if mapping_passed:
        full_train = pd.read_csv(args.train)
        physical_trackman = read_trackman_physical(args.trackman)
        feature_results, lookups, validation_features = feature_diagnostics(
            full_train, physical_trackman, mappings, selected_method
        )
        feature_results.to_csv(args.output / "feature_results.csv", index=False)
        feature_gate = feature_signal_gate(feature_results)
        print(
            f"[feature signal] passed={feature_gate['passed']} "
            f"stable_features={len(feature_gate['passing_features'])}",
            flush=True,
        )
        if feature_gate["passed"]:
            (
                model_results,
                complementarity,
                blends,
                blend_decisions,
            ) = run_trackman_models(
                full_train,
                physical_trackman,
                mappings,
                selected_method,
                args.output,
                lookups,
                validation_features,
                args.force_models,
            )
        del full_train, physical_trackman
        gc.collect()
    else:
        pd.DataFrame(
            columns=[
                "family",
                "feature",
                "validation_season",
                "n",
                "residual_correlation",
                "status",
            ]
        ).to_csv(args.output / "feature_results.csv", index=False)
    model_results.to_csv(args.output / "oof_results.csv", index=False)
    complementarity.to_csv(args.output / "complementarity.csv", index=False)
    blends.to_csv(args.output / "blend_results.csv", index=False)
    passing_candidates = [row for row in blend_decisions if row["passed"]]
    if passing_candidates:
        selected_candidate = max(
            passing_candidates,
            key=lambda row: (row["gains"]["pooled"], -row["weight"]),
        )

    strong_signal = bool(selected_candidate and selected_candidate["strong"])
    if strong_signal:
        verdict = "A. RELIABLE CROSSWALK + STRONG TRACKMAN SIGNAL"
    elif mapping_passed:
        verdict = "B. RELIABLE CROSSWALK BUT WEAK MODEL SIGNAL"
    else:
        verdict = "C. CROSSWALK NOT RELIABLE"

    champion_sha_after = sha256_path(CHAMPION)
    if champion_sha_after != CHAMPION_SHA256:
        raise RuntimeError("immutable champion changed during crosswalk audit")
    summary = {
        "question": (
            "can official anonymous pitch-mix/time-series fingerprints support a "
            "reliable main-to-TrackMan pitcher crosswalk"
        ),
        "official_data_only": True,
        "external_identity_information_used": False,
        "test_read": False,
        "champion": {
            "artifact": str(CHAMPION),
            "sha256_before": CHAMPION_SHA256,
            "sha256_after": champion_sha_after,
            "unchanged": champion_sha_after == CHAMPION_SHA256,
        },
        "hand_mapping": {
            "selected": {str(key): value for key, value in hand_map.items()},
            "method": "minimum season-wise hand-prevalence squared error",
            "evaluations": [
                {
                    "mapping": {str(key): value for key, value in row["mapping"].items()},
                    "prevalence_squared_error": row["prevalence_squared_error"],
                }
                for row in hand_evaluations
            ],
        },
        "fingerprint": {
            "main": "within-season delta of cumulative as-of group counts",
            "trackman": "pitcher-season pitch_type_group counts and proportions",
            "minimum_season_pitches": MIN_SEASON_PITCHES,
            "distance_weights": {
                "pitchmix_l1": 1.0,
                "relative_count_pattern": COUNT_PATTERN_WEIGHT,
                "season_presence": PRESENCE_WEIGHT,
                "coverage_ratio": COVERAGE_RATIO_WEIGHT,
                "overlap": OVERLAP_WEIGHT,
            },
        },
        "confidence_rules": CONFIDENCE_RULES,
        "mapping_gate_thresholds": MAPPING_GATE,
        "mapping_gates": gates,
        "future_validation_by_confidence": (
            future.groupby(["method", "confidence"], observed=True)
            .agg(
                n=("future_mix_l1", "size"),
                median_future_mix_l1=("future_mix_l1", "median"),
                future_top1_rate=("future_top1", "mean"),
                future_top5_rate=("future_top5", "mean"),
                future_top10_percent_rate=("future_top10_percent", "mean"),
                median_future_rank_percentile=("future_rank_percentile", "median"),
            )
            .reset_index()
            .to_dict("records")
        ),
        "mapping_stability": stability.to_dict("records"),
        "coverage": coverage.to_dict("records"),
        "selected_method": selected_method,
        "mapping_gate_passed": mapping_passed,
        "trackman_modeling_allowed": mapping_passed,
        "feature_signal_gate": feature_gate,
        "model_recipe": MODEL_RECIPE,
        "model_results": model_results.to_dict("records"),
        "complementarity": complementarity.to_dict("records"),
        "blend_weights": BLEND_WEIGHTS,
        "blend_decisions": blend_decisions,
        "selected_candidate": selected_candidate,
        "trackman_modeling_status": (
            "evaluated" if not model_results.empty else (
                "not evaluated: physical feature signal gate failed"
                if mapping_passed
                else "not evaluated: crosswalk gate failed"
            )
        ),
        "production": {
            "created": False,
            "reason": (
                "strong OOF candidate requires guarded packaging"
                if strong_signal
                else "no A-grade signal"
            ),
            "filename_limit": 30,
        },
        "verdict": verdict,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(gates, indent=2, default=_json_default), flush=True)
    print(summary["verdict"], flush=True)


if __name__ == "__main__":
    main()
