"""Leakage-safe hierarchical empirical-Bayes target-statistics research.

Every target-derived value for a row in season S is fitted from seasons < S.
Validation rows are never aggregated together, and the current test set is only
mapped through lookup tables fitted on train seasons <= 2024.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import importlib
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _import_lightgbm():
    """Load a wheel-local libomp on macOS before importing LightGBM if needed."""
    try:
        return importlib.import_module("lightgbm")
    except OSError as error:
        if "libomp.dylib" not in str(error):
            raise
        prefix = Path(sys.prefix)
        candidates = list(
            prefix.glob("lib/python*/site-packages/torch/lib/libomp.dylib")
        )
        candidates += list(
            prefix.glob("lib/python*/site-packages/sklearn/.dylibs/libomp.dylib")
        )
        if not candidates:
            raise
        ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)
        return importlib.import_module("lightgbm")


lgb = _import_lightgbm()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_extratrees_candidate import sha256_path, write_json  # noqa: E402


CURRENT_CHAMPION = ROOT / "artifacts" / "sub_et25_w020.zip"
CURRENT_CHAMPION_SHA256 = (
    "316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf"
)
CURRENT_CHAMPION_LB_BSS = 1017.0233029621
CURRENT_ET_WEIGHT = 0.0200
ORIGINAL_OOF_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
ET25_OOF_DIR = ROOT / "artifacts" / "extratrees_size_ablation" / "cache"
OUTPUT_DIR = ROOT / "artifacts" / "hier_target"
DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_TEST = Path("/Users/wooh/Documents/dev/open/data/test.csv")

TARGET = "control_success"
VALIDATION_SEASONS = (2022, 2023, 2024)
M_GRID = (25.0, 75.0, 200.0)
BLEND_WEIGHTS = (0.02, 0.05, 0.10, 0.15)
STRONG_POOLED_GAIN = 2e-5
STAGE_A_GLOBAL_GAIN = 1e-4
TOP_ERROR_QUANTILE = 0.90
MAX_SUBMISSION_FILENAME_LENGTH = 30

READ_COLUMNS = (
    "row_id",
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    TARGET,
)

CONTEXT_COLUMNS = (
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom_code",
    "game_type_code",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state_code",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_hand",
    "batter_hand",
    "count_state",
    "inning_bucket",
    "same_hand",
)


@dataclass(frozen=True)
class GroupSpec:
    name: str
    keys: tuple[str, ...]
    parent: str
    priority: int


GROUPS = (
    GroupSpec("pitcher", ("pitcher_id",), "global", 1),
    GroupSpec("batter", ("batter_id",), "global", 1),
    GroupSpec("pitcher_batter_hand", ("pitcher_id", "batter_hand"), "pitcher", 2),
    GroupSpec("pitcher_count", ("pitcher_id", "count_state"), "pitcher", 2),
    GroupSpec("pitcher_game_type", ("pitcher_id", "game_type"), "pitcher", 2),
    GroupSpec("batter_pitcher_hand", ("batter_id", "pitcher_hand"), "batter", 2),
    GroupSpec("batter_count", ("batter_id", "count_state"), "batter", 2),
    GroupSpec(
        "pitcher_batter_hand_count",
        ("pitcher_id", "batter_hand", "count_state"),
        "pitcher_batter_hand",
        3,
    ),
    GroupSpec("pitcher_base", ("pitcher_id", "base_state"), "pitcher", 3),
    GroupSpec("pitcher_inning", ("pitcher_id", "inning_bucket"), "pitcher", 3),
    GroupSpec(
        "batter_pitcher_hand_count",
        ("batter_id", "pitcher_hand", "count_state"),
        "batter_pitcher_hand",
        3,
    ),
    GroupSpec("pitcher_batter", ("pitcher_id", "batter_id"), "pitcher", 4),
)


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    return float(np.mean((y - p) ** 2))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype="float64")
    y = np.asarray(right, dtype="float64")
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["count_state"] = (
        output["balls_before"].astype("int16") * 3
        + output["strikes_before"].astype("int16")
    ).astype("int8")
    inning = output["inning"].to_numpy()
    output["inning_bucket"] = np.select(
        [inning <= 3, inning <= 6], [0, 1], default=2
    ).astype("int8")
    output["same_hand"] = (
        output["pitcher_hand"].to_numpy()
        == output["batter_hand"].to_numpy()
    ).astype("int8")
    output["top_bottom_code"] = output["top_bottom"].map({"T": 0, "B": 1}).fillna(-1)
    output["game_type_code"] = output["game_type"].map({"R": 0, "F": 1}).fillna(-1)
    base_levels = {"___": 0, "1__": 1, "_2_": 2, "__3": 3, "12_": 4, "1_3": 5, "_23": 6, "123": 7}
    output["base_state_code"] = output["base_state"].map(base_levels).fillna(-1)
    return output


def _aggregate(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    return frame.groupby(list(keys), observed=True, sort=False)[TARGET].agg(
        successes="sum", n="count"
    )


def fit_lookups(history: pd.DataFrame, recent: pd.DataFrame) -> dict:
    if history.empty:
        raise ValueError("hierarchy history cannot be empty")
    return {
        "history_rows": len(history),
        "recent_rows": len(recent),
        "global_successes": float(history[TARGET].sum()),
        "global_n": int(len(history)),
        "recent_global_successes": float(recent[TARGET].sum()),
        "recent_global_n": int(len(recent)),
        "career": {spec.name: _aggregate(history, spec.keys) for spec in GROUPS},
        "recent": {spec.name: _aggregate(recent, spec.keys) for spec in GROUPS},
    }


def _map_stats(
    stats: pd.DataFrame,
    target: pd.DataFrame,
    keys: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if len(keys) == 1:
        index = pd.Index(target[keys[0]].to_numpy(), name=keys[0])
    else:
        index = pd.MultiIndex.from_frame(target[list(keys)])
    mapped = stats.reindex(index)
    successes = mapped["successes"].fillna(0.0).to_numpy(dtype="float64")
    counts = mapped["n"].fillna(0.0).to_numpy(dtype="float64")
    return successes, counts


def transform_hierarchy(
    target: pd.DataFrame,
    lookups: Mapping,
    m: float,
) -> pd.DataFrame:
    rows = len(target)
    global_n = float(lookups["global_n"])
    global_rate = float(lookups["global_successes"]) / global_n
    recent_global_n = float(lookups["recent_global_n"])
    recent_global_rate = (
        (float(lookups["recent_global_successes"]) + m * global_rate)
        / (recent_global_n + m)
    )
    values: dict[str, np.ndarray] = {
        "global_rate": np.full(rows, global_rate),
        "global_n": np.full(rows, global_n),
        "global_reliability": np.full(rows, global_n / (global_n + m)),
        "global_diff_parent": np.zeros(rows),
        "global_recent_rate": np.full(rows, recent_global_rate),
        "global_recent_n": np.full(rows, recent_global_n),
        "global_recent_reliability": np.full(
            rows, recent_global_n / (recent_global_n + m)
        ),
        "global_recent_minus_career": np.full(
            rows, recent_global_rate - global_rate
        ),
    }
    for spec in GROUPS:
        parent = values[f"{spec.parent}_rate"]
        successes, counts = _map_stats(
            lookups["career"][spec.name], target, spec.keys
        )
        posterior = (successes + m * parent) / (counts + m)
        reliability = counts / (counts + m)
        recent_successes, recent_counts = _map_stats(
            lookups["recent"][spec.name], target, spec.keys
        )
        recent_posterior = (
            recent_successes + m * posterior
        ) / (recent_counts + m)
        values[f"{spec.name}_rate"] = posterior
        values[f"{spec.name}_n"] = counts
        values[f"{spec.name}_reliability"] = reliability
        values[f"{spec.name}_diff_parent"] = posterior - parent
        values[f"{spec.name}_recent_rate"] = recent_posterior
        values[f"{spec.name}_recent_n"] = recent_counts
        values[f"{spec.name}_recent_reliability"] = recent_counts / (
            recent_counts + m
        )
        values[f"{spec.name}_recent_minus_career"] = recent_posterior - posterior
    return pd.DataFrame(
        {name: array.astype("float32") for name, array in values.items()},
        index=target.index,
    )


def context_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, CONTEXT_COLUMNS].astype("float32")


def current_champion_oof(season: int, target: np.ndarray) -> np.ndarray:
    original = load_npz(ORIGINAL_OOF_DIR / f"season_{season}.npz")
    extra = load_npz(ET25_OOF_DIR / f"season_{season}_prefixes.npz")
    if not np.array_equal(target, original["y"].astype("float64")):
        raise ValueError(f"train/OOF target order mismatch for {season}")
    return (
        (1.0 - CURRENT_ET_WEIGHT) * original["champion_final"].astype("float64")
        + CURRENT_ET_WEIGHT * extra["prediction_25"].astype("float64")
    )


def current_feature_audit() -> dict:
    with zipfile.ZipFile(CURRENT_CHAMPION) as archive:
        archive_names = set(archive.namelist())
        meta = json.loads(archive.read("model/meta.json"))
        extra = json.loads(archive.read("model/extra_trees_meta.json"))
        cold = json.loads(archive.read("model/coldstart_meta.json"))
    members = []
    for member in meta["members"]:
        columns = member.get("feature_columns", []) or []
        members.append(
            {
                "name": member["name"],
                "feature_count": len(columns),
                "raw_ids": [c for c in columns if c in {"pitcher_id", "batter_id"}],
                "asof_features": [c for c in columns if c.startswith("asof_")],
            }
        )
    interaction_tokens = [
        "pitcher_count_rate",
        "pitcher_batter_hand_count_rate",
        "batter_pitcher_hand_count_rate",
        "pitcher_base_rate",
        "pitcher_inning_rate",
    ]
    all_columns = {
        column for member in meta["members"] for column in member.get("feature_columns", [])
    } | set(extra["feature_columns"]) | set(cold.get("feature_cols", []))
    return {
        "members": members,
        "extratrees_raw_ids": [
            c for c in extra["feature_columns"] if c in {"pitcher_id", "batter_id"}
        ],
        "extratrees_asof_features": [
            c for c in extra["feature_columns"] if c.startswith("asof_")
        ],
        "pitcher_form_present": "season_form" in meta,
        "batter_form_present": "model/batter_form.json" in archive_names,
        "coldstart_present": bool(cold.get("feature_cols")),
        "hierarchical_interaction_features_present": sorted(
            set(interaction_tokens) & all_columns
        ),
        "conclusion": (
            "simple official as-of, form, and cold-start history already exist; "
            "cutoff-static player×context empirical-Bayes interactions do not"
        ),
    }


def coverage_rows(
    season: int,
    validation: pd.DataFrame,
    lookups: Mapping,
) -> list[dict]:
    rows = []
    for spec in GROUPS:
        _, counts = _map_stats(lookups["career"][spec.name], validation, spec.keys)
        stats = lookups["career"][spec.name]["n"].to_numpy(dtype="float64")
        seen_counts = counts[counts > 0]
        rows.append(
            {
                "validation_season": season,
                "hierarchy": spec.name,
                "priority": spec.priority,
                "keys": " × ".join(spec.keys),
                "parent": spec.parent,
                "history_rows": lookups["history_rows"],
                "group_count": len(stats),
                "group_median_n": float(np.median(stats)),
                "validation_rows": len(validation),
                "seen_rows": int(np.sum(counts > 0)),
                "coverage": float(np.mean(counts > 0)),
                "seen_row_median_n": (
                    float(np.median(seen_counts)) if len(seen_counts) else 0.0
                ),
                "raw_rate_used": False,
                "included_with_parent_shrinkage": True,
            }
        )
    return rows


def signal_rows(
    season: int | str,
    m: float,
    hierarchy: pd.DataFrame,
    target: np.ndarray,
    champion: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    brier_rows = []
    residual_rows = []
    global_brier = brier(target, hierarchy["global_rate"].to_numpy())
    champion_brier = brier(target, champion)
    rate_columns = [
        column
        for column in hierarchy.columns
        if column.endswith("_rate") and "reliability" not in column
    ]
    champion_error = (target - champion) ** 2
    cutoff = float(np.quantile(champion_error, TOP_ERROR_QUANTILE))
    top = champion_error >= cutoff
    for column in rate_columns:
        prediction = hierarchy[column].to_numpy(dtype="float64")
        group = column.removesuffix("_recent_rate").removesuffix("_rate")
        signal_type = "recent" if column.endswith("_recent_rate") else "career"
        score = brier(target, prediction)
        gap = prediction - champion
        residual = target - champion
        expert_error = (target - prediction) ** 2
        brier_rows.append(
            {
                "m": m,
                "validation_season": season,
                "hierarchy": group,
                "signal_type": signal_type,
                "n": len(target),
                "brier": score,
                "global_brier": global_brier,
                "gain_vs_global": global_brier - score,
                "champion_brier": champion_brier,
                "gain_vs_champion": champion_brier - score,
            }
        )
        residual_rows.append(
            {
                "m": m,
                "validation_season": season,
                "hierarchy": group,
                "signal_type": signal_type,
                "n": len(target),
                "corr_posterior_champion": safe_corr(prediction, champion),
                "corr_gap_champion_residual": safe_corr(gap, residual),
                "corr_gap_champion_squared_error": safe_corr(gap, champion_error),
                "posterior_wins_rate": float(np.mean(expert_error < champion_error)),
                "champion_top10_error_posterior_wins_rate": float(
                    np.mean(expert_error[top] < champion_error[top])
                ),
                "mean_abs_gap": float(np.mean(np.abs(gap))),
            }
        )
    return brier_rows, residual_rows


def pooled_signal_rows(
    prediction_store: Mapping[tuple[float, str], Mapping[int, np.ndarray]],
    targets: Mapping[int, np.ndarray],
    champions: Mapping[int, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    rows = []
    residual = []
    pooled_target = np.concatenate([targets[s] for s in VALIDATION_SEASONS])
    pooled_champion = np.concatenate([champions[s] for s in VALIDATION_SEASONS])
    for (m, column), folds in prediction_store.items():
        prediction = np.concatenate([folds[s] for s in VALIDATION_SEASONS])
        frame = pd.DataFrame(
            {
                "global_rate": np.concatenate(
                    [prediction_store[(m, "global_rate")][s] for s in VALIDATION_SEASONS]
                ),
                column: prediction,
            }
        )
        current_rows, current_residual = signal_rows(
            "pooled", m, frame, pooled_target, pooled_champion
        )
        selected_group = column.removesuffix("_recent_rate").removesuffix("_rate")
        selected_type = "recent" if column.endswith("_recent_rate") else "career"
        rows.extend(
            row
            for row in current_rows
            if row["hierarchy"] == selected_group
            and row["signal_type"] == selected_type
        )
        residual.extend(
            row
            for row in current_residual
            if row["hierarchy"] == selected_group
            and row["signal_type"] == selected_type
        )
    return rows, residual


def choose_m(posterior: pd.DataFrame) -> tuple[float, list[dict]]:
    pooled = posterior[
        posterior["validation_season"].astype(str).eq("pooled")
        & posterior["signal_type"].eq("career")
        & ~posterior["hierarchy"].eq("global")
    ]
    rows = []
    for m in M_GRID:
        current = pooled[np.isclose(pooled["m"], m)]
        rows.append(
            {
                "m": m,
                "mean_gain_vs_global": float(current["gain_vs_global"].mean()),
                "median_gain_vs_global": float(current["gain_vs_global"].median()),
                "best_gain_vs_global": float(current["gain_vs_global"].max()),
            }
        )
    selected = max(rows, key=lambda row: (row["mean_gain_vs_global"], -row["m"]))
    return float(selected["m"]), rows


def model_frame(frame: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    context = context_features(frame).reset_index(drop=True)
    historical = hierarchy.reset_index(drop=True)
    return pd.concat([context, historical], axis=1).astype("float32")


def historical_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def expert_metrics(
    name: str,
    season: int | str,
    target: np.ndarray,
    prediction: np.ndarray,
    champion: np.ndarray,
) -> tuple[dict, dict]:
    champion_error = (target - champion) ** 2
    expert_error = (target - prediction) ** 2
    top = champion_error >= np.quantile(champion_error, TOP_ERROR_QUANTILE)
    result = {
        "expert": name,
        "validation_season": season,
        "n": len(target),
        "expert_brier": brier(target, prediction),
        "champion_brier": brier(target, champion),
        "expert_gain_vs_champion": brier(target, champion) - brier(target, prediction),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
    }
    complement = {
        "expert": name,
        "validation_season": season,
        "n": len(target),
        "corr_expert_champion": safe_corr(prediction, champion),
        "corr_expert_error_champion_error": safe_corr(expert_error, champion_error),
        "corr_gap_champion_residual": safe_corr(prediction - champion, target - champion),
        "expert_wins_rate": float(np.mean(expert_error < champion_error)),
        "champion_top10_error_expert_wins_rate": float(
            np.mean(expert_error[top] < champion_error[top])
        ),
    }
    return result, complement


def evaluate_blends(
    expert_name: str,
    predictions: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    champions: Mapping[int, np.ndarray],
) -> list[dict]:
    rows = []
    for weight in BLEND_WEIGHTS:
        fold_values = {}
        for season in VALIDATION_SEASONS:
            candidate = (
                (1.0 - weight) * champions[season] + weight * predictions[season]
            )
            gain = brier(targets[season], champions[season]) - brier(
                targets[season], candidate
            )
            fold_values[season] = (targets[season], champions[season], candidate, gain)
            rows.append(
                {
                    "expert": expert_name,
                    "weight": weight,
                    "validation_season": season,
                    "n": len(candidate),
                    "brier_gain_vs_current_champion": gain,
                }
            )
        target = np.concatenate([fold_values[s][0] for s in VALIDATION_SEASONS])
        champion = np.concatenate([fold_values[s][1] for s in VALIDATION_SEASONS])
        candidate = np.concatenate([fold_values[s][2] for s in VALIDATION_SEASONS])
        rows.append(
            {
                "expert": expert_name,
                "weight": weight,
                "validation_season": "pooled",
                "n": len(candidate),
                "brier_gain_vs_current_champion": brier(target, champion)
                - brier(target, candidate),
            }
        )
    return rows


def leakage_audit(
    validation: pd.DataFrame,
    lookups: Mapping,
    m: float,
) -> dict:
    sample = validation.iloc[: min(2_000, len(validation))].copy()
    baseline = transform_hierarchy(sample, lookups, m)
    changed = sample.copy()
    changed[TARGET] = 1 - changed[TARGET].to_numpy()
    flipped = transform_hierarchy(changed, lookups, m)
    flip_diff = float(
        np.max(np.abs(baseline.to_numpy(dtype="float64") - flipped.to_numpy(dtype="float64")))
    )
    reduced = transform_hierarchy(sample.iloc[1:].copy(), lookups, m)
    removal_diff = float(
        np.max(
            np.abs(
                baseline.iloc[1:].to_numpy(dtype="float64")
                - reduced.to_numpy(dtype="float64")
            )
        )
    )
    return {
        "validation_rows_tested": len(sample),
        "validation_target_flip_max_abs_diff": flip_diff,
        "remove_validation_row_other_rows_max_abs_diff": removal_diff,
        "passed": flip_diff == 0.0 and removal_diff == 0.0,
    }


def predict_expert(
    model,
    frame: pd.DataFrame,
    lookups: Mapping,
    m: float,
    feature_columns: Sequence[str],
) -> np.ndarray:
    prepared = prepare_frame(frame)
    hierarchy = transform_hierarchy(prepared, lookups, m)
    features = model_frame(prepared, hierarchy).loc[:, list(feature_columns)]
    return model.predict_proba(features)[:, 1].astype("float64")


def row_independence_audit(
    model,
    test: pd.DataFrame,
    lookups: Mapping,
    m: float,
    feature_columns: Sequence[str],
) -> dict:
    variants = {
        "single": test.iloc[[0]].copy(),
        "full": test.copy(),
        "shuffle": test.sample(frac=1.0, random_state=42).reset_index(drop=True),
        "subset": test.iloc[[0, 2, 4]].reset_index(drop=True),
        "reverse": test.iloc[::-1].reset_index(drop=True),
    }
    full_prediction = predict_expert(
        model, variants["full"], lookups, m, feature_columns
    )
    reference = dict(zip(variants["full"]["row_id"], full_prediction))
    rows = {}
    maximum = 0.0
    for name, current in variants.items():
        prediction = predict_expert(model, current, lookups, m, feature_columns)
        diff = max(
            (
                abs(float(value) - float(reference[row_id]))
                for row_id, value in zip(current["row_id"], prediction)
            ),
            default=0.0,
        )
        maximum = max(maximum, diff)
        rows[name] = {"rows": len(current), "max_abs_diff": diff}
    return {"variants": rows, "max_abs_diff": maximum, "passed": maximum == 0.0}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    required = [args.train, args.test, CURRENT_CHAMPION]
    required.extend(ORIGINAL_OOF_DIR / f"season_{s}.npz" for s in VALIDATION_SEASONS)
    required.extend(ET25_OOF_DIR / f"season_{s}_prefixes.npz" for s in VALIDATION_SEASONS)
    for path in required:
        if not path.is_file():
            parser.error(f"required file not found: {path}")
    champion_sha_before = sha256_path(CURRENT_CHAMPION)
    if champion_sha_before != CURRENT_CHAMPION_SHA256:
        raise SystemExit("immutable w=.020 champion checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    print("[audit] loading leakage-safe columns", flush=True)
    train = prepare_frame(
        pd.read_csv(args.train, usecols=list(READ_COLUMNS), encoding="utf-8-sig")
    )
    feature_audit = current_feature_audit()
    print(f"[audit] rows={len(train):,}; seasons={sorted(train.season.unique())}", flush=True)

    coverage = []
    brier_output = []
    residual_output = []
    prediction_store: dict[tuple[float, str], dict[int, np.ndarray]] = {}
    targets: dict[int, np.ndarray] = {}
    champions: dict[int, np.ndarray] = {}
    print("[stage A] fitting cutoff lookups and posterior grid", flush=True)
    selected_2024_lookups = None
    for season in VALIDATION_SEASONS:
        history = train[train["season"] < season]
        recent = train[train["season"] == season - 1]
        validation = train[train["season"] == season]
        lookups = fit_lookups(history, recent)
        if season == 2024:
            selected_2024_lookups = lookups
        coverage.extend(coverage_rows(season, validation, lookups))
        target = validation[TARGET].to_numpy(dtype="float64")
        champion = current_champion_oof(season, target)
        targets[season] = target
        champions[season] = champion
        for m in M_GRID:
            hierarchy = transform_hierarchy(validation, lookups, m)
            current_brier, current_residual = signal_rows(
                season, m, hierarchy, target, champion
            )
            brier_output.extend(current_brier)
            residual_output.extend(current_residual)
            for column in hierarchy.columns:
                if column.endswith("_rate") and "reliability" not in column:
                    prediction_store.setdefault((m, column), {})[season] = hierarchy[
                        column
                    ].to_numpy(dtype="float64")
            del hierarchy
        print(f"[stage A] season={season} complete", flush=True)
        del history, recent, validation, lookups
        gc.collect()
    pooled_brier, pooled_residual = pooled_signal_rows(
        prediction_store, targets, champions
    )
    brier_output.extend(pooled_brier)
    residual_output.extend(pooled_residual)
    posterior = pd.DataFrame(brier_output).drop_duplicates(
        ["m", "validation_season", "hierarchy", "signal_type"]
    )
    residual = pd.DataFrame(residual_output).drop_duplicates(
        ["m", "validation_season", "hierarchy", "signal_type"]
    )
    selected_m, m_diagnostics = choose_m(posterior)
    pooled_candidates = posterior[
        posterior["validation_season"].astype(str).eq("pooled")
        & ~posterior["hierarchy"].eq("global")
    ].sort_values("gain_vs_global", ascending=False)
    best_signal = pooled_candidates.iloc[0]
    best_residual = residual[
        np.isclose(residual["m"], best_signal["m"])
        & residual["validation_season"].astype(str).eq("pooled")
        & residual["hierarchy"].eq(best_signal["hierarchy"])
        & residual["signal_type"].eq(best_signal["signal_type"])
    ].iloc[0]
    stage_a_passed = bool(
        best_signal["gain_vs_global"] >= STAGE_A_GLOBAL_GAIN
        and best_residual["corr_gap_champion_residual"] > 0.0
    )
    del prediction_store
    gc.collect()
    pd.DataFrame(coverage).to_csv(args.output / "hierarchy_coverage.csv", index=False)
    posterior.to_csv(args.output / "posterior_brier.csv", index=False)
    residual.to_csv(args.output / "residual_signal.csv", index=False)
    print(
        f"[stage A] selected M={selected_m:g}; best={best_signal['hierarchy']}/"
        f"{best_signal['signal_type']}; gain_vs_global={best_signal['gain_vs_global']:+.3e}; "
        f"passed={stage_a_passed}",
        flush=True,
    )

    expert_rows: list[dict] = []
    complementarity_rows: list[dict] = []
    blend_rows: list[dict] = []
    profile_rows: list[dict] = []
    expert_predictions: dict[int, np.ndarray] = {}
    profile_predictions: dict[int, np.ndarray] = {}
    leakage = None
    row_independence = None
    stage_b_passed = False
    stage_c_passed = False
    stage_d_run = False
    feature_columns: list[str] = []

    if stage_a_passed:
        print("[stage B] building past-only training representations", flush=True)
        feature_parts = []
        season_parts = []
        target_parts = []
        for season in range(2020, 2025):
            history = train[train["season"] < season]
            recent = train[train["season"] == season - 1]
            current = train[train["season"] == season]
            lookups = fit_lookups(history, recent)
            hierarchy = transform_hierarchy(current, lookups, selected_m)
            current_features = model_frame(current, hierarchy)
            feature_parts.append(current_features)
            season_parts.append(np.full(len(current), season, dtype="int16"))
            target_parts.append(current[TARGET].to_numpy(dtype="int8"))
            print(f"[stage B] representation season={season} rows={len(current):,}", flush=True)
            del history, recent, current, lookups, hierarchy, current_features
            gc.collect()
        all_features = pd.concat(feature_parts, ignore_index=True)
        all_seasons = np.concatenate(season_parts)
        all_targets = np.concatenate(target_parts)
        feature_columns = list(all_features.columns)
        del feature_parts, season_parts, target_parts
        gc.collect()
        for validation_season in VALIDATION_SEASONS:
            train_mask = all_seasons < validation_season
            validation_mask = all_seasons == validation_season
            model = historical_model()
            model.fit(
                all_features.loc[train_mask],
                all_targets[train_mask],
                callbacks=[lgb.log_evaluation(0)],
            )
            prediction = model.predict_proba(all_features.loc[validation_mask])[:, 1]
            expert_predictions[validation_season] = prediction.astype("float64")
            result, complement = expert_metrics(
                "lightgbm_historical",
                validation_season,
                targets[validation_season],
                prediction,
                champions[validation_season],
            )
            expert_rows.append(result)
            complementarity_rows.append(complement)
            print(
                f"[stage B] season={validation_season} expert_gain="
                f"{result['expert_gain_vs_champion']:+.3e}",
                flush=True,
            )
            del model, prediction
            gc.collect()
        pooled_target = np.concatenate([targets[s] for s in VALIDATION_SEASONS])
        pooled_champion = np.concatenate([champions[s] for s in VALIDATION_SEASONS])
        pooled_expert = np.concatenate([expert_predictions[s] for s in VALIDATION_SEASONS])
        result, complement = expert_metrics(
            "lightgbm_historical",
            "pooled",
            pooled_target,
            pooled_expert,
            pooled_champion,
        )
        expert_rows.append(result)
        complementarity_rows.append(complement)
        stage_b_passed = bool(
            complement["corr_gap_champion_residual"] > 0.0
            and complement["champion_top10_error_expert_wins_rate"] > 0.5
            and complement["corr_expert_champion"] < 0.9995
        )
        print(
            f"[stage B] pooled corr={complement['corr_expert_champion']:.6f}; "
            f"top10 wins={complement['champion_top10_error_expert_wins_rate']:.4f}; "
            f"passed={stage_b_passed}",
            flush=True,
        )

        if stage_b_passed:
            print("[stage C] evaluating fixed blend weights", flush=True)
            blend_rows.extend(
                evaluate_blends(
                    "lightgbm_historical", expert_predictions, targets, champions
                )
            )
            blend_frame = pd.DataFrame(blend_rows)
            stable_weights = []
            for weight in BLEND_WEIGHTS:
                current = blend_frame[
                    (blend_frame["expert"] == "lightgbm_historical")
                    & np.isclose(blend_frame["weight"], weight)
                ]
                gains = {
                    str(row.validation_season): row.brier_gain_vs_current_champion
                    for row in current.itertuples()
                }
                if all(gains[str(season)] > 0.0 for season in VALIDATION_SEASONS) and gains[
                    "pooled"
                ] > 0.0:
                    stable_weights.append((weight, gains["pooled"]))
            stage_c_passed = bool(stable_weights)
            print(f"[stage C] stable weights={stable_weights}", flush=True)

        if stage_c_passed:
            stage_d_run = True
            print("[stage D] nested stacked posterior logistic expert", flush=True)
            probability_columns = [
                column
                for column in all_features.columns
                if column.endswith("_rate")
                or column.endswith("_recent_minus_career")
            ]
            for validation_season in VALIDATION_SEASONS:
                train_mask = all_seasons < validation_season
                validation_mask = all_seasons == validation_season
                model = LogisticRegression(
                    C=0.1,
                    max_iter=500,
                    solver="lbfgs",
                    random_state=42,
                )
                model.fit(
                    all_features.loc[train_mask, probability_columns],
                    all_targets[train_mask],
                )
                prediction = model.predict_proba(
                    all_features.loc[validation_mask, probability_columns]
                )[:, 1]
                profile_predictions[validation_season] = prediction.astype("float64")
                result, complement = expert_metrics(
                    "stacked_logistic",
                    validation_season,
                    targets[validation_season],
                    prediction,
                    champions[validation_season],
                )
                profile_rows.append({**result, **{
                    "feature_count": len(probability_columns),
                    "nested_temporal": True,
                }})
                complementarity_rows.append(complement)
                del model
                gc.collect()
            pooled_profile = np.concatenate(
                [profile_predictions[s] for s in VALIDATION_SEASONS]
            )
            result, complement = expert_metrics(
                "stacked_logistic",
                "pooled",
                pooled_target,
                pooled_profile,
                pooled_champion,
            )
            profile_rows.append({**result, "feature_count": len(probability_columns), "nested_temporal": True})
            complementarity_rows.append(complement)
            blend_rows.extend(
                evaluate_blends(
                    "stacked_logistic", profile_predictions, targets, champions
                )
            )

        if selected_2024_lookups is None:
            raise RuntimeError("missing 2024 leakage lookup")
        validation_2024 = train[train["season"] == 2024]
        leakage = leakage_audit(validation_2024, selected_2024_lookups, selected_m)

        print("[audit] fitting full historical expert for test row independence", flush=True)
        final_model = historical_model()
        final_model.fit(all_features, all_targets, callbacks=[lgb.log_evaluation(0)])
        test = pd.read_csv(args.test, encoding="utf-8-sig")
        missing_test = sorted(set(READ_COLUMNS) - {TARGET} - set(test.columns))
        if missing_test:
            raise ValueError(f"test is missing hierarchy columns: {missing_test}")
        full_lookups = fit_lookups(train, train[train["season"] == 2024])
        row_independence = row_independence_audit(
            final_model, test, full_lookups, selected_m, feature_columns
        )
        del final_model, all_features, all_targets, all_seasons
        gc.collect()
    else:
        profile_rows.append(
            {
                "expert": "not_run",
                "validation_season": "all",
                "reason": "Stage A failed",
                "nested_temporal": True,
            }
        )

    if not stage_d_run and not profile_rows:
        profile_rows.append(
            {
                "expert": "not_run",
                "validation_season": "all",
                "reason": "Stage B/C gate failed",
                "nested_temporal": True,
            }
        )
    expert_frame = pd.DataFrame(
        expert_rows,
        columns=[
            "expert",
            "validation_season",
            "n",
            "expert_brier",
            "champion_brier",
            "expert_gain_vs_champion",
            "prediction_mean",
            "prediction_std",
            "prediction_min",
            "prediction_max",
        ],
    )
    complementarity_frame = pd.DataFrame(
        complementarity_rows,
        columns=[
            "expert",
            "validation_season",
            "n",
            "corr_expert_champion",
            "corr_expert_error_champion_error",
            "corr_gap_champion_residual",
            "expert_wins_rate",
            "champion_top10_error_expert_wins_rate",
        ],
    )
    blend_frame = pd.DataFrame(
        blend_rows,
        columns=[
            "expert",
            "weight",
            "validation_season",
            "n",
            "brier_gain_vs_current_champion",
        ],
    )
    profile_frame = pd.DataFrame(
        profile_rows,
        columns=[
            "expert",
            "validation_season",
            "n",
            "expert_brier",
            "champion_brier",
            "expert_gain_vs_champion",
            "prediction_mean",
            "prediction_std",
            "prediction_min",
            "prediction_max",
            "feature_count",
            "nested_temporal",
            "reason",
        ],
    )
    expert_frame.to_csv(args.output / "expert_results.csv", index=False)
    complementarity_frame.to_csv(args.output / "complementarity.csv", index=False)
    blend_frame.to_csv(args.output / "blend_results.csv", index=False)
    profile_frame.to_csv(args.output / "profile_results.csv", index=False)

    stable_blends = []
    if not blend_frame.empty:
        for (expert, weight), current in blend_frame.groupby(["expert", "weight"]):
            gains = {
                str(row.validation_season): float(row.brier_gain_vs_current_champion)
                for row in current.itertuples()
            }
            stable = all(gains.get(str(s), float("-inf")) > 0.0 for s in VALIDATION_SEASONS)
            stable = stable and gains.get("pooled", float("-inf")) > 0.0
            stable_blends.append(
                {
                    "expert": expert,
                    "weight": weight,
                    "gains": gains,
                    "stable": stable,
                }
            )
    passing = [row for row in stable_blends if row["stable"]]
    best_blend = max(passing, key=lambda row: row["gains"]["pooled"]) if passing else None
    if best_blend and best_blend["gains"]["pooled"] >= STRONG_POOLED_GAIN:
        verdict = "A. LARGE NEW SIGNAL FOUND"
    elif best_blend:
        verdict = "B. incremental but too small for 1150 target"
    else:
        verdict = "C. no stable historical interaction signal"
    champion_sha_after = sha256_path(CURRENT_CHAMPION)
    if champion_sha_after != champion_sha_before:
        raise RuntimeError("immutable w=.020 champion changed during hierarchy research")
    production = {
        "created": False,
        "reason": (
            "production packaging is only allowed for verdict A"
            if verdict != "A. LARGE NEW SIGNAL FOUND"
            else "large signal found; production runtime packaging is the next gated step"
        ),
        "filename_hard_gate": MAX_SUBMISSION_FILENAME_LENGTH,
    }
    summary = {
        "verdict": verdict,
        "current_immutable_champion": {
            "lb_bss": CURRENT_CHAMPION_LB_BSS,
            "artifact": str(CURRENT_CHAMPION),
            "sha256_before": champion_sha_before,
            "sha256_after": champion_sha_after,
            "unchanged": champion_sha_before == champion_sha_after == CURRENT_CHAMPION_SHA256,
        },
        "feature_audit": feature_audit,
        "temporal_contract": {
            "folds": [
                "train <= 2021 -> validation 2022",
                "train <= 2022 -> validation 2023",
                "train <= 2023 -> validation 2024",
            ],
            "training_row_features": "season S uses target aggregates from seasons < S",
            "test_features": "train <= 2024 lookup only; no test aggregation",
        },
        "m_grid": list(M_GRID),
        "selected_m": selected_m,
        "m_diagnostics": m_diagnostics,
        "stage_a": {
            "passed": stage_a_passed,
            "threshold_gain_vs_global": STAGE_A_GLOBAL_GAIN,
            "best_signal": json_safe(best_signal.to_dict()),
            "best_residual_signal": json_safe(best_residual.to_dict()),
        },
        "stage_b": {"run": stage_a_passed, "passed": stage_b_passed},
        "stage_c": {"run": stage_b_passed, "passed": stage_c_passed},
        "stage_d": {
            "run": stage_d_run,
            "reason": None if stage_d_run else "Stage B/C gate failed",
        },
        "strong_signal_threshold": STRONG_POOLED_GAIN,
        "best_stable_blend": best_blend,
        "leakage_audit": leakage,
        "test_row_independence": row_independence,
        "production": production,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(json_safe({
        "verdict": verdict,
        "selected_m": selected_m,
        "stage_a_passed": stage_a_passed,
        "stage_b_passed": stage_b_passed,
        "stage_c_passed": stage_c_passed,
        "stage_d_run": stage_d_run,
        "best_stable_blend": best_blend,
        "leakage_audit": leakage,
        "test_row_independence": row_independence,
        "champion_unchanged": champion_sha_before == champion_sha_after,
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
