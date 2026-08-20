"""Evaluate fixed global recency weighting on one source-exact LightGBM member.

The only tuning axis is training-season age.  Test rows and test distribution
are never read by model selection, preprocessing, drift analysis, or weights.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import importlib
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _import_lightgbm():
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
from src.train_base import (  # noqa: E402
    CAT_COLS,
    PARAMS,
    add_features,
    apply_category_maps,
    build_category_maps,
)


CURRENT_CHAMPION = ROOT / "artifacts" / "sub_et25_w020.zip"
CURRENT_CHAMPION_SHA256 = (
    "316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf"
)
CURRENT_CHAMPION_LB_BSS = 1017.0233029621
CURRENT_ET_WEIGHT = 0.0200
OOF_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
ET25_OOF_DIR = ROOT / "artifacts" / "extratrees_size_ablation" / "cache"
OUTPUT_DIR = ROOT / "artifacts" / "temporal_recency"
DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")

TARGET = "control_success"
ID_COLUMN = "row_id"
VALIDATION_SEASONS = (2022, 2023, 2024)
DECAYS = (1.00, 0.85, 0.70, 0.50)
WINDOWS: tuple[int | None, ...] = (None, 4, 3, 2)
BLEND_WEIGHTS = (0.02, 0.05, 0.10, 0.15)
STRONG_GAIN = 2e-5
MAX_SUBMISSION_FILENAME_LENGTH = 30
MODEL_NAME = "team_raw_mono63"
N_ROUNDS = 220

NUMERIC_DRIFT_COLUMNS = (
    "season",
    "game_month",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
)
CATEGORICAL_DRIFT_COLUMNS = (
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save_prediction(path: Path, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, prediction=np.asarray(prediction, dtype="float64"))
    os.replace(temporary, path)


def recency_weights(train_seasons: np.ndarray, validation_season: int, decay: float) -> np.ndarray:
    if decay not in DECAYS:
        raise ValueError(f"decay must be one of {DECAYS}")
    seasons = np.asarray(train_seasons, dtype="int16")
    if np.any(seasons >= validation_season):
        raise ValueError("validation/future season leaked into recency weights")
    age = validation_season - seasons
    return np.power(decay, age - 1, dtype="float64")


def window_mask(train_seasons: np.ndarray, validation_season: int, window: int | None) -> np.ndarray:
    seasons = np.asarray(train_seasons, dtype="int16")
    if np.any(seasons >= validation_season):
        raise ValueError("validation/future season leaked into window rows")
    if window is None:
        return np.ones(len(seasons), dtype=bool)
    if window not in (4, 3, 2):
        raise ValueError("window must be all/4/3/2")
    return seasons >= validation_season - window


def model_audit() -> pd.DataFrame:
    rows = [
        {
            "model": "HGB",
            "training_seasons": "all available <= cutoff; production 2019-2024",
            "season_weights": "uniform",
            "sample_weight_usage": False,
            "recent_only": False,
            "evidence": "build_regime_blend_oof.train_hgb model.fit(X,y)",
        },
        {
            "model": "CatBoost",
            "training_seasons": "all available <= cutoff; production 2019-2024",
            "season_weights": "uniform",
            "sample_weight_usage": False,
            "recent_only": False,
            "evidence": "train_catboost model.fit without sample_weight",
        },
        {
            "model": "ours NN",
            "training_seasons": "all available <= cutoff; production 2019-2024",
            "season_weights": "uniform",
            "sample_weight_usage": False,
            "recent_only": False,
            "evidence": "train_embed_nn has no season/sample weight input",
        },
        {
            "model": "team LightGBM",
            "training_seasons": "all available <= cutoff; production 2019-2024",
            "season_weights": "eng31/leaf63/mono63 uniform; decay85 exponential 0.85",
            "sample_weight_usage": True,
            "recent_only": False,
            "evidence": "decay85 is 15.8753% of group1 and 4.5001% of group2",
        },
        {
            "model": "team NN",
            "training_seasons": "all available <= cutoff; production 2019-2024",
            "season_weights": "uniform",
            "sample_weight_usage": False,
            "recent_only": False,
            "evidence": "source-exact train_nn has no sample_weight",
        },
        {
            "model": "cold-start expert",
            "training_seasons": "all regular-season rows <= cutoff; production 2019-2024",
            "season_weights": "uniform",
            "sample_weight_usage": False,
            "recent_only": False,
            "evidence": "train_coldstart_expert LightGBM Dataset has no weight",
        },
        {
            "model": "ExtraTrees ET25",
            "training_seasons": "all available <= cutoff; production 2019-2024",
            "season_weights": "uniform",
            "sample_weight_usage": False,
            "recent_only": False,
            "evidence": "ExtraTrees.fit called without sample_weight",
        },
    ]
    return pd.DataFrame(rows)


def monotone_params(feature_columns: Sequence[str], threads: int) -> dict:
    positive = {
        "asof_pitcher_success_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_batter_success_rate",
    }
    negative = {
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_middle_rate",
    }
    return dict(
        PARAMS,
        seed=42,
        num_leaves=63,
        min_data_in_leaf=1200,
        num_threads=threads,
        monotone_constraints=[
            1 if column in positive else -1 if column in negative else 0
            for column in feature_columns
        ],
        monotone_constraints_method="advanced",
    )


def train_predict(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    sample_weight: np.ndarray | None,
    threads: int,
) -> np.ndarray:
    feature_columns = [
        column for column in train_features.columns if column not in (ID_COLUMN, TARGET)
    ]
    categorical = [column for column in CAT_COLS if column in feature_columns]
    maps = build_category_maps(train_features, categorical)
    train_encoded = apply_category_maps(train_features, categorical, maps)
    validation_encoded = apply_category_maps(validation_features, categorical, maps)
    dataset = lgb.Dataset(
        train_encoded[feature_columns],
        label=train_encoded[TARGET],
        weight=sample_weight,
        categorical_feature=categorical,
        free_raw_data=False,
    )
    model = lgb.train(
        monotone_params(feature_columns, threads), dataset, num_boost_round=N_ROUNDS
    )
    prediction = model.predict(validation_encoded[feature_columns])
    del dataset, model, train_encoded, validation_encoded
    gc.collect()
    return np.asarray(prediction, dtype="float64")


def current_champion_prediction(season: int, target: np.ndarray) -> np.ndarray:
    source = load_npz(OOF_DIR / f"season_{season}.npz")
    extra = load_npz(ET25_OOF_DIR / f"season_{season}_prefixes.npz")
    if not np.array_equal(target, source["y"].astype("float64")):
        raise ValueError(f"train/OOF order mismatch for {season}")
    return (
        (1.0 - CURRENT_ET_WEIGHT) * source["champion_final"].astype("float64")
        + CURRENT_ET_WEIGHT * extra["prediction_25"].astype("float64")
    )


def result_rows(
    family: str,
    scheme: str,
    values: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict]:
    rows = []
    for season in VALIDATION_SEASONS:
        target, original, candidate = values[season]
        rows.append(
            {
                "family": family,
                "scheme": scheme,
                "validation_season": season,
                "n": len(target),
                "original_brier": brier(target, original),
                "recency_brier": brier(target, candidate),
                "brier_gain": brier(target, original) - brier(target, candidate),
                "actual_positive_rate": float(np.mean(target)),
                "original_prediction_mean": float(np.mean(original)),
                "recency_prediction_mean": float(np.mean(candidate)),
                "original_calibration_gap": float(np.mean(original) - np.mean(target)),
                "recency_calibration_gap": float(np.mean(candidate) - np.mean(target)),
            }
        )
    target = np.concatenate([values[s][0] for s in VALIDATION_SEASONS])
    original = np.concatenate([values[s][1] for s in VALIDATION_SEASONS])
    candidate = np.concatenate([values[s][2] for s in VALIDATION_SEASONS])
    rows.append(
        {
            "family": family,
            "scheme": scheme,
            "validation_season": "pooled",
            "n": len(target),
            "original_brier": brier(target, original),
            "recency_brier": brier(target, candidate),
            "brier_gain": brier(target, original) - brier(target, candidate),
            "actual_positive_rate": float(np.mean(target)),
            "original_prediction_mean": float(np.mean(original)),
            "recency_prediction_mean": float(np.mean(candidate)),
            "original_calibration_gap": float(np.mean(original) - np.mean(target)),
            "recency_calibration_gap": float(np.mean(candidate) - np.mean(target)),
        }
    )
    return rows


def scheme_decision(results: pd.DataFrame, family: str, scheme: str) -> dict:
    current = results[(results["family"] == family) & (results["scheme"] == scheme)]
    gains = {
        str(row.validation_season): float(row.brier_gain)
        for row in current.itertuples()
    }
    worst_fold = min((str(s) for s in VALIDATION_SEASONS), key=lambda s: gains[s])
    stable = gains["2023"] > 0.0 and gains["2024"] > 0.0 and gains["pooled"] > 0.0
    strong = stable and gains["pooled"] >= STRONG_GAIN
    return {
        "family": family,
        "scheme": scheme,
        "gains": gains,
        "worst_fold": worst_fold,
        "worst_fold_gain": gains[worst_fold],
        "2023_positive": gains["2023"] > 0.0,
        "2024_positive": gains["2024"] > 0.0,
        "stable": stable,
        "strong": strong,
    }


def complementarity_rows(
    family: str,
    scheme: str,
    predictions: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    champions: Mapping[int, np.ndarray],
) -> list[dict]:
    rows = []
    for season in (*VALIDATION_SEASONS, "pooled"):
        if season == "pooled":
            target = np.concatenate([targets[s] for s in VALIDATION_SEASONS])
            prediction = np.concatenate([predictions[s] for s in VALIDATION_SEASONS])
            champion = np.concatenate([champions[s] for s in VALIDATION_SEASONS])
        else:
            target = targets[int(season)]
            prediction = predictions[int(season)]
            champion = champions[int(season)]
        recency_error = (target - prediction) ** 2
        champion_error = (target - champion) ** 2
        top = champion_error >= np.quantile(champion_error, 0.90)
        rows.append(
            {
                "record_type": "overall",
                "family": family,
                "scheme": scheme,
                "validation_season": season,
                "axis": "all",
                "subset": "all",
                "n": len(target),
                "corr_recency_champion": safe_corr(prediction, champion),
                "corr_recency_error_champion_error": safe_corr(
                    recency_error, champion_error
                ),
                "champion_top10_error_recency_wins_rate": float(
                    np.mean(recency_error[top] < champion_error[top])
                ),
                "recency_wins_rate": float(np.mean(recency_error < champion_error)),
                "brier_gain_vs_champion": brier(target, champion)
                - brier(target, prediction),
            }
        )
    return rows


def subset_complementarity_rows(
    family: str,
    scheme: str,
    predictions: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    originals: Mapping[int, np.ndarray],
    subset_masks: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
) -> list[dict]:
    rows = []
    for season in VALIDATION_SEASONS:
        for (axis, subset), mask in subset_masks[season].items():
            target = targets[season][mask]
            original = originals[season][mask]
            candidate = predictions[season][mask]
            rows.append(
                {
                    "record_type": "subset",
                    "family": family,
                    "scheme": scheme,
                    "validation_season": season,
                    "axis": axis,
                    "subset": subset,
                    "n": int(mask.sum()),
                    "corr_recency_champion": np.nan,
                    "corr_recency_error_champion_error": np.nan,
                    "champion_top10_error_recency_wins_rate": np.nan,
                    "recency_wins_rate": float(
                        np.mean((target - candidate) ** 2 < (target - original) ** 2)
                    ),
                    "brier_gain_vs_champion": brier(target, original)
                    - brier(target, candidate),
                }
            )
    return rows


def blend_rows(
    family: str,
    scheme: str,
    predictions: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    champions: Mapping[int, np.ndarray],
) -> list[dict]:
    rows = []
    for weight in BLEND_WEIGHTS:
        candidates = {}
        for season in VALIDATION_SEASONS:
            candidate = (
                (1.0 - weight) * champions[season] + weight * predictions[season]
            )
            candidates[season] = candidate
            rows.append(
                {
                    "family": family,
                    "scheme": scheme,
                    "weight": weight,
                    "validation_season": season,
                    "n": len(candidate),
                    "brier_gain_vs_current_champion": brier(
                        targets[season], champions[season]
                    )
                    - brier(targets[season], candidate),
                }
            )
        target = np.concatenate([targets[s] for s in VALIDATION_SEASONS])
        champion = np.concatenate([champions[s] for s in VALIDATION_SEASONS])
        candidate = np.concatenate([candidates[s] for s in VALIDATION_SEASONS])
        rows.append(
            {
                "family": family,
                "scheme": scheme,
                "weight": weight,
                "validation_season": "pooled",
                "n": len(candidate),
                "brier_gain_vs_current_champion": brier(target, champion)
                - brier(target, candidate),
            }
        )
    return rows


def js_divergence(left: pd.Series, right: pd.Series) -> float:
    left_frequency = left.fillna("__NA__").astype(str).value_counts(normalize=True)
    right_frequency = right.fillna("__NA__").astype(str).value_counts(normalize=True)
    levels = left_frequency.index.union(right_frequency.index)
    p = left_frequency.reindex(levels, fill_value=0.0).to_numpy(dtype="float64")
    q = right_frequency.reindex(levels, fill_value=0.0).to_numpy(dtype="float64")
    midpoint = 0.5 * (p + q)
    positive_p = p > 0.0
    positive_q = q > 0.0
    left_term = np.sum(p[positive_p] * np.log(p[positive_p] / midpoint[positive_p]))
    right_term = np.sum(q[positive_q] * np.log(q[positive_q] / midpoint[positive_q]))
    return float(0.5 * (left_term + right_term))


def drift_rows(train: pd.DataFrame) -> list[dict]:
    rows = []
    for old_season, new_season in ((2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)):
        old = train[train["season"] == old_season]
        new = train[train["season"] == new_season]
        for column in NUMERIC_DRIFT_COLUMNS:
            old_values = old[column].to_numpy(dtype="float64")
            new_values = new[column].to_numpy(dtype="float64")
            pooled_std = float(np.nanstd(np.concatenate([old_values, new_values])))
            mean_shift = float(np.nanmean(new_values) - np.nanmean(old_values))
            rows.append(
                {
                    "from_season": old_season,
                    "to_season": new_season,
                    "feature": column,
                    "feature_type": "numeric",
                    "mean_from": float(np.nanmean(old_values)),
                    "mean_to": float(np.nanmean(new_values)),
                    "std_from": float(np.nanstd(old_values)),
                    "std_to": float(np.nanstd(new_values)),
                    "normalized_mean_shift": mean_shift / pooled_std if pooled_std else 0.0,
                    "q10_shift": float(
                        np.nanquantile(new_values, 0.10) - np.nanquantile(old_values, 0.10)
                    ),
                    "q50_shift": float(
                        np.nanquantile(new_values, 0.50) - np.nanquantile(old_values, 0.50)
                    ),
                    "q90_shift": float(
                        np.nanquantile(new_values, 0.90) - np.nanquantile(old_values, 0.90)
                    ),
                    "js_divergence": np.nan,
                    "new_row_unseen_rate": np.nan,
                }
            )
        for column in CATEGORICAL_DRIFT_COLUMNS:
            rows.append(
                {
                    "from_season": old_season,
                    "to_season": new_season,
                    "feature": column,
                    "feature_type": "categorical",
                    "mean_from": np.nan,
                    "mean_to": np.nan,
                    "std_from": np.nan,
                    "std_to": np.nan,
                    "normalized_mean_shift": np.nan,
                    "q10_shift": np.nan,
                    "q50_shift": np.nan,
                    "q90_shift": np.nan,
                    "js_divergence": js_divergence(old[column], new[column]),
                    "new_row_unseen_rate": np.nan,
                }
            )
        for column in ("pitcher_id", "batter_id"):
            known = set(old[column].dropna())
            unseen_rate = float((~new[column].isin(known)).mean())
            rows.append(
                {
                    "from_season": old_season,
                    "to_season": new_season,
                    "feature": column,
                    "feature_type": "player_id",
                    "mean_from": np.nan,
                    "mean_to": np.nan,
                    "std_from": np.nan,
                    "std_to": np.nan,
                    "normalized_mean_shift": np.nan,
                    "q10_shift": np.nan,
                    "q50_shift": np.nan,
                    "q90_shift": np.nan,
                    "js_divergence": js_divergence(old[column], new[column]),
                    "new_row_unseen_rate": unseen_rate,
                }
            )
        rows.append(
            {
                "from_season": old_season,
                "to_season": new_season,
                "feature": TARGET,
                "feature_type": "target_rate_diagnostic",
                "mean_from": float(old[TARGET].mean()),
                "mean_to": float(new[TARGET].mean()),
                "std_from": float(old[TARGET].std()),
                "std_to": float(new[TARGET].std()),
                "normalized_mean_shift": float(new[TARGET].mean() - old[TARGET].mean()),
                "q10_shift": np.nan,
                "q50_shift": np.nan,
                "q90_shift": np.nan,
                "js_divergence": np.nan,
                "new_row_unseen_rate": np.nan,
            }
        )
    return rows


def leakage_audit(train: pd.DataFrame, validation_season: int, decay: float) -> dict:
    training = train[train["season"] < validation_season]
    validation = train[train["season"] == validation_season].copy()
    rows_before = training[ID_COLUMN].to_numpy(copy=True)
    weights_before = recency_weights(training["season"], validation_season, decay)
    features = add_features(training)
    categorical = [column for column in CAT_COLS if column in features.columns]
    maps_before = build_category_maps(features, categorical)
    validation[TARGET] = 1 - validation[TARGET].to_numpy()
    training_after = train[train["season"] < validation_season]
    weights_after = recency_weights(training_after["season"], validation_season, decay)
    maps_after = build_category_maps(add_features(training_after), categorical)
    validation_rows_excluded = set(validation[ID_COLUMN]).isdisjoint(rows_before)
    return {
        "validation_season": validation_season,
        "validation_rows_excluded": validation_rows_excluded,
        "training_rows_same_after_validation_target_flip": np.array_equal(
            rows_before, training_after[ID_COLUMN].to_numpy()
        ),
        "sample_weights_max_abs_diff_after_validation_target_flip": float(
            np.max(np.abs(weights_before - weights_after))
        ),
        "preprocessing_maps_same_after_validation_target_flip": maps_before == maps_after,
        "passed": bool(
            validation_rows_excluded
            and np.array_equal(rows_before, training_after[ID_COLUMN].to_numpy())
            and np.array_equal(weights_before, weights_after)
            and maps_before == maps_after
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    required = [args.train, CURRENT_CHAMPION]
    required.extend(OOF_DIR / f"season_{s}.npz" for s in VALIDATION_SEASONS)
    required.extend(ET25_OOF_DIR / f"season_{s}_prefixes.npz" for s in VALIDATION_SEASONS)
    for path in required:
        if not path.is_file():
            parser.error(f"required file not found: {path}")
    champion_sha_before = sha256_path(CURRENT_CHAMPION)
    if champion_sha_before != CURRENT_CHAMPION_SHA256:
        raise SystemExit("immutable w=.020 champion checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "cache"
    cache_dir.mkdir(exist_ok=True)
    audit = model_audit()
    audit.to_csv(args.output / "model_audit.csv", index=False)
    print("[audit] selected source-exact team_raw_mono63 LightGBM", flush=True)
    train = pd.read_csv(args.train, encoding="utf-8-sig")
    print(f"[audit] train rows={len(train):,}; test data not read", flush=True)
    drift = pd.DataFrame(drift_rows(train))
    drift.to_csv(args.output / "drift_results.csv", index=False)

    targets: dict[int, np.ndarray] = {}
    originals: dict[int, np.ndarray] = {}
    champions: dict[int, np.ndarray] = {}
    validation_subset_masks: dict[int, dict[tuple[str, str], np.ndarray]] = {}
    predictions: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for season in VALIDATION_SEASONS:
        source = load_npz(OOF_DIR / f"season_{season}.npz")
        target = source["y"].astype("float64")
        validation = train[train["season"] == season].copy()
        if not np.array_equal(target, validation[TARGET].to_numpy(dtype="float64")):
            raise ValueError(f"train/OOF target mismatch for {season}")
        training = train[train["season"] < season].copy()
        known_pitchers = set(training["pitcher_id"].dropna().astype("int64"))
        known_batters = set(training["batter_id"].dropna().astype("int64"))
        seen_pitcher = validation["pitcher_id"].astype("int64").isin(known_pitchers).to_numpy()
        seen_batter = validation["batter_id"].astype("int64").isin(known_batters).to_numpy()
        history_threshold = float(training["asof_pitcher_n"].median())
        low_history = (
            validation["asof_pitcher_n"].to_numpy(dtype="float64")
            < history_threshold
        )
        validation_subset_masks[season] = {
            ("pitcher_seen", "seen"): seen_pitcher,
            ("pitcher_history", "low"): low_history,
            ("pitcher_history", "high"): ~low_history,
            ("batter_seen", "seen"): seen_batter,
        }
        targets[season] = target
        originals[season] = source[MODEL_NAME].astype("float64")
        champions[season] = current_champion_prediction(season, target)
        training_features = add_features(training)
        validation_features = add_features(validation)
        train_seasons = training["season"].to_numpy(dtype="int16")
        predictions.setdefault(("decay", "decay_100"), {})[season] = originals[season]
        predictions.setdefault(("window", "all_history"), {})[season] = originals[season]
        for decay in DECAYS[1:]:
            scheme = f"decay_{int(round(decay * 100)):03d}"
            cache = cache_dir / f"{scheme}_season_{season}.npz"
            if cache.is_file() and not args.force:
                prediction = load_npz(cache)["prediction"]
            else:
                weights = recency_weights(train_seasons, season, decay)
                prediction = train_predict(
                    training_features, validation_features, weights, args.threads
                )
                save_prediction(cache, prediction)
            predictions.setdefault(("decay", scheme), {})[season] = prediction
            print(f"[decay] season={season} scheme={scheme} complete", flush=True)
        for window in WINDOWS[1:]:
            scheme = f"last_{window}_seasons"
            mask = window_mask(train_seasons, season, window)
            if bool(mask.all()):
                prediction = originals[season]
            else:
                cache = cache_dir / f"{scheme}_season_{season}.npz"
                if cache.is_file() and not args.force:
                    prediction = load_npz(cache)["prediction"]
                else:
                    prediction = train_predict(
                        training_features.loc[mask],
                        validation_features,
                        None,
                        args.threads,
                    )
                    save_prediction(cache, prediction)
            predictions.setdefault(("window", scheme), {})[season] = prediction
            print(f"[window] season={season} scheme={scheme} complete", flush=True)
        del training, validation, training_features, validation_features, train_seasons
        gc.collect()

    grid_rows = []
    window_rows = []
    decisions = []
    for (family, scheme), fold_predictions in predictions.items():
        values = {
            season: (targets[season], originals[season], fold_predictions[season])
            for season in VALIDATION_SEASONS
        }
        rows = result_rows(family, scheme, values)
        if family == "decay":
            grid_rows.extend(rows)
        else:
            window_rows.extend(rows)
        if scheme not in {"decay_100", "all_history"}:
            decisions.append(scheme_decision(pd.DataFrame(rows), family, scheme))
    grid = pd.DataFrame(grid_rows)
    windows = pd.DataFrame(window_rows)
    grid.to_csv(args.output / "recency_grid.csv", index=False)
    windows.to_csv(args.output / "window_results.csv", index=False)
    stable = [row for row in decisions if row["stable"]]
    selected = max(stable, key=lambda row: row["gains"]["pooled"]) if stable else None
    print(f"[gate] stable schemes={[(x['scheme'], x['gains']['pooled']) for x in stable]}", flush=True)

    complement_rows_output = []
    blend_output = []
    if selected is not None:
        key = (selected["family"], selected["scheme"])
        selected_prediction = predictions[key]
        complement_rows_output.extend(
            complementarity_rows(
                selected["family"], selected["scheme"], selected_prediction, targets, champions
            )
        )
        complement_rows_output.extend(
            subset_complementarity_rows(
                selected["family"],
                selected["scheme"],
                selected_prediction,
                targets,
                originals,
                validation_subset_masks,
            )
        )
        blend_output.extend(
            blend_rows(
                selected["family"], selected["scheme"], selected_prediction, targets, champions
            )
        )
    complement = pd.DataFrame(
        complement_rows_output,
        columns=[
            "record_type",
            "family",
            "scheme",
            "validation_season",
            "axis",
            "subset",
            "n",
            "corr_recency_champion",
            "corr_recency_error_champion_error",
            "champion_top10_error_recency_wins_rate",
            "recency_wins_rate",
            "brier_gain_vs_champion",
        ],
    )
    blends = pd.DataFrame(
        blend_output,
        columns=[
            "family",
            "scheme",
            "weight",
            "validation_season",
            "n",
            "brier_gain_vs_current_champion",
        ],
    )
    complement.to_csv(args.output / "complementarity.csv", index=False)
    blends.to_csv(args.output / "blend_results.csv", index=False)

    blend_decisions = []
    if not blends.empty:
        for weight in BLEND_WEIGHTS:
            current = blends[np.isclose(blends["weight"], weight)]
            gains = {
                str(row.validation_season): float(row.brier_gain_vs_current_champion)
                for row in current.itertuples()
            }
            stable_blend = gains["2023"] > 0 and gains["2024"] > 0 and gains["pooled"] > 0
            blend_decisions.append(
                {
                    "weight": weight,
                    "gains": gains,
                    "stable": stable_blend,
                    "strong": stable_blend and gains["pooled"] >= STRONG_GAIN,
                }
            )
    strong_base = [row for row in decisions if row["strong"]]
    strong_blend = [row for row in blend_decisions if row["strong"]]
    if strong_base or strong_blend:
        verdict = "A. LARGE TEMPORAL SIGNAL FOUND"
    elif stable or any(row["stable"] for row in blend_decisions):
        verdict = "B. stable but incremental only"
    else:
        verdict = "C. no temporal recency signal"
    leakage = leakage_audit(train, 2024, 0.70)
    champion_sha_after = sha256_path(CURRENT_CHAMPION)
    if champion_sha_after != champion_sha_before:
        raise RuntimeError("immutable w=.020 champion changed")
    production = {
        "created": False,
        "reason": (
            "verdict is not A; no production package allowed"
            if verdict != "A. LARGE TEMPORAL SIGNAL FOUND"
            else "large signal found; one production package must be built in Stage C"
        ),
        "filename_length_hard_gate": MAX_SUBMISSION_FILENAME_LENGTH,
    }
    target_drift = drift[drift["feature"].eq(TARGET)][
        ["from_season", "to_season", "mean_from", "mean_to", "normalized_mean_shift"]
    ].to_dict(orient="records")
    latest_player_drift = drift[
        drift["feature_type"].eq("player_id")
        & drift["from_season"].eq(2023)
        & drift["to_season"].eq(2024)
    ][["feature", "js_divergence", "new_row_unseen_rate"]].to_dict(orient="records")
    summary = {
        "verdict": verdict,
        "current_immutable_champion": {
            "lb_bss": CURRENT_CHAMPION_LB_BSS,
            "artifact": str(CURRENT_CHAMPION),
            "sha256_before": champion_sha_before,
            "sha256_after": champion_sha_after,
            "unchanged": champion_sha_before == champion_sha_after == CURRENT_CHAMPION_SHA256,
        },
        "test_data_used": False,
        "selected_model": {
            "name": MODEL_NAME,
            "learner": "LightGBM monotone-constraint tree",
            "selection_reason": "source-exact reproducible OOF and strongest stable uniform team tree recipe",
            "rounds": N_ROUNDS,
            "existing_decay_overlap": "team ensemble already contains a distinct decay85 member; this audit isolates the uniform mono63 recipe",
        },
        "decay_grid": list(DECAYS),
        "window_grid": ["all" if value is None else value for value in WINDOWS],
        "scheme_decisions": decisions,
        "selected_stable_scheme": selected,
        "blend_decisions": blend_decisions,
        "strong_gain_threshold": STRONG_GAIN,
        "drift_summary": {
            "target_rate_transitions": target_drift,
            "latest_player_id_drift": latest_player_drift,
            "interpretation": (
                "train-only drift is material, but global recency weighting does not "
                "convert it into stable OOF improvement"
            ),
        },
        "leakage_audit": leakage,
        "production": production,
    }
    write_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "selected_stable_scheme": selected,
                "blend_decisions": blend_decisions,
                "leakage_audit": leakage,
                "champion_unchanged": champion_sha_before == champion_sha_after,
                "production": production,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
