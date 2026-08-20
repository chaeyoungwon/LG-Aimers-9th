"""Validate leakage-safe baseball-state feature families with temporal OOF.

Research motivates only the hypotheses. All feature values, quantiles, model
effects, and selection decisions are learned from official training data.
Evaluation/test rows are never aggregated and test distribution is not read.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
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
        candidates = list(prefix.glob("lib/python*/site-packages/torch/lib/libomp.dylib"))
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

from src.train_base import (  # noqa: E402
    CAT_COLS,
    PARAMS,
    add_features,
    build_category_maps,
)


DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_TRACKMAN = Path("/Users/wooh/Documents/dev/open/data/trackman_history.csv")
OUTPUT_DIR = ROOT / "artifacts" / "domain_features"
OOF_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
ET25_OOF_DIR = ROOT / "artifacts" / "extratrees_size_ablation" / "cache"
CHAMPION = ROOT / "artifacts" / "sub_et25_w020.zip"
CHAMPION_SHA256 = (
    "316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf"
)
CHAMPION_LB_BSS = 1017.0233029621
CURRENT_ET_WEIGHT = 0.0200

TARGET = "control_success"
ID_COLUMN = "row_id"
VALIDATION_SEASONS = (2022, 2023, 2024)
N_ROUNDS = 220
STRONG_GAIN = 1e-5
BLEND_WEIGHTS = (0.01, 0.02, 0.05, 0.10)
CACHE_VERSION = "domain-features-mono63-v1"

RESEARCH_SOURCES = (
    {
        "topic": "within-game fatigue and kinematics",
        "title": "The Impact of Fatigue on the Kinematics of Collegiate Baseball Pitchers",
        "url": "https://pubmed.ncbi.nlm.nih.gov/26535338/",
        "use": "motivates workload hypotheses only",
    },
    {
        "topic": "pitcher-batter familiarity / time through order",
        "title": "A Bayesian analysis of the time through the order penalty in baseball",
        "url": "https://arxiv.org/abs/2210.06724",
        "use": "motivates continuous familiarity hypotheses only",
    },
    {
        "topic": "windup versus stretch",
        "title": "Biomechanical Comparison of the Fastball from Wind-up and the Fastball from Stretch",
        "url": "https://doi.org/10.1177/0363546507308938",
        "use": "supports a weak, not hard-coded, base-runner prior",
    },
)


@dataclass(frozen=True)
class DomainState:
    experience_edges: tuple[float, ...]
    form_edges: tuple[float, ...]


FAMILY_FEATURES = {
    "pressure_state": (
        "domain_count_pressure",
        "domain_runner_pressure",
        "domain_count_x_runner",
    ),
    "pitcher_state_pressure": (
        "domain_experience_bucket",
        "domain_form_bucket",
        "domain_experience_x_count",
        "domain_form_x_count",
    ),
    "handedness_interaction": (
        "domain_hand_combo",
        "domain_hand_x_count",
        "domain_hand_x_form",
    ),
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_npz(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping) -> None:
    path.write_text(
        json.dumps(json_ready(dict(payload)), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("Brier inputs must be same-length vectors")
    return float(np.mean((y - p) ** 2))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype="float64")
    y = np.asarray(right, dtype="float64")
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("correlation inputs must be same-length vectors")
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def temporal_split(frame: pd.DataFrame, validation_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = frame[frame["season"] < validation_season]
    validation = frame[frame["season"] == validation_season]
    if training.empty or validation.empty:
        raise ValueError(f"empty temporal split for {validation_season}")
    if int(training["season"].max()) >= validation_season:
        raise ValueError("validation/future season leaked into training")
    return training, validation


def finite_quantiles(values: pd.Series, probabilities: Sequence[float]) -> tuple[float, ...]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    finite = numeric[np.isfinite(numeric)]
    if not len(finite):
        return tuple()
    return tuple(float(value) for value in np.unique(np.quantile(finite, probabilities)))


def fit_domain_state(training: pd.DataFrame) -> DomainState:
    form_gap = (
        training["asof_pitcher_prev5_game_success_rate"]
        - training["asof_pitcher_success_rate"]
    )
    return DomainState(
        experience_edges=finite_quantiles(
            training["asof_pitcher_n"], (0.25, 0.50, 0.75)
        ),
        form_edges=finite_quantiles(form_gap, (1.0 / 3.0, 2.0 / 3.0)),
    )


def bucket_numeric(
    values: pd.Series, edges: Sequence[float], prefix: str
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    bucket = np.searchsorted(np.asarray(edges, dtype="float64"), numeric, side="right")
    output = np.asarray([f"{prefix}_{index}" for index in bucket], dtype="object")
    output[~np.isfinite(numeric)] = f"{prefix}_missing"
    return output


def count_pressure(frame: pd.DataFrame) -> np.ndarray:
    balls = frame["balls_before"].to_numpy(dtype="int64")
    strikes = frame["strikes_before"].to_numpy(dtype="int64")
    output = np.full(len(frame), "balanced", dtype="object")
    output[(balls == 0) & (strikes == 0)] = "zero_zero"
    output[(balls > strikes) & ~((balls == 0) & (strikes == 0))] = "hitter_ahead"
    output[(strikes > balls)] = "pitcher_ahead"
    output[strikes == 2] = "two_strike"
    output[balls == 3] = "three_ball"
    output[(balls == 3) & (strikes == 2)] = "full_count"
    return output


def runner_pressure(frame: pd.DataFrame) -> np.ndarray:
    first = frame["runner_on_1b"].to_numpy(dtype="int8").astype(bool)
    second = frame["runner_on_2b"].to_numpy(dtype="int8").astype(bool)
    third = frame["runner_on_3b"].to_numpy(dtype="int8").astype(bool)
    output = np.full(len(frame), "bases_empty", dtype="object")
    output[first | second | third] = "runners_on"
    output[second | third] = "risp_proxy"
    output[first & second & third] = "bases_loaded"
    return output


def build_domain_columns(
    frame: pd.DataFrame,
    state: DomainState,
    requested: Sequence[str] | None = None,
) -> pd.DataFrame:
    selected = set(
        requested
        if requested is not None
        else [column for columns in FAMILY_FEATURES.values() for column in columns]
    )
    output = pd.DataFrame(index=frame.index)
    need_count = bool(
        selected
        & {
            "domain_count_pressure",
            "domain_count_x_runner",
            "domain_experience_x_count",
            "domain_form_x_count",
            "domain_hand_x_count",
        }
    )
    need_runner = bool(
        selected & {"domain_runner_pressure", "domain_count_x_runner"}
    )
    need_experience = bool(
        selected & {"domain_experience_bucket", "domain_experience_x_count"}
    )
    need_form = bool(
        selected
        & {"domain_form_bucket", "domain_form_x_count", "domain_hand_x_form"}
    )
    need_hand = bool(
        selected
        & {"domain_hand_combo", "domain_hand_x_count", "domain_hand_x_form"}
    )
    count_values = count_pressure(frame) if need_count else None
    runner_values = runner_pressure(frame) if need_runner else None
    experience_values = (
        bucket_numeric(frame["asof_pitcher_n"], state.experience_edges, "experience")
        if need_experience
        else None
    )
    if need_form:
        form_gap = (
            frame["asof_pitcher_prev5_game_success_rate"]
            - frame["asof_pitcher_success_rate"]
        )
        form_values = bucket_numeric(form_gap, state.form_edges, "form")
    else:
        form_values = None
    hand_values = (
        (
            "H"
            + frame["pitcher_hand"].astype(str)
            + "_vs_H"
            + frame["batter_hand"].astype(str)
        ).to_numpy(dtype="object")
        if need_hand
        else None
    )
    if "domain_count_pressure" in selected:
        output["domain_count_pressure"] = count_values
    if "domain_runner_pressure" in selected:
        output["domain_runner_pressure"] = runner_values
    if "domain_count_x_runner" in selected:
        output["domain_count_x_runner"] = np.char.add(
            np.char.add(count_values.astype(str), "|"), runner_values.astype(str)
        )
    if "domain_experience_bucket" in selected:
        output["domain_experience_bucket"] = experience_values
    if "domain_form_bucket" in selected:
        output["domain_form_bucket"] = form_values
    if "domain_experience_x_count" in selected:
        output["domain_experience_x_count"] = np.char.add(
            np.char.add(experience_values.astype(str), "|"), count_values.astype(str)
        )
    if "domain_form_x_count" in selected:
        output["domain_form_x_count"] = np.char.add(
            np.char.add(form_values.astype(str), "|"), count_values.astype(str)
        )
    if "domain_hand_combo" in selected:
        output["domain_hand_combo"] = hand_values
    if "domain_hand_x_count" in selected:
        output["domain_hand_x_count"] = np.char.add(
            np.char.add(hand_values.astype(str), "|"), count_values.astype(str)
        )
    if "domain_hand_x_form" in selected:
        output["domain_hand_x_form"] = np.char.add(
            np.char.add(hand_values.astype(str), "|"), form_values.astype(str)
        )
    return output


def add_family_features(
    frame: pd.DataFrame,
    state: DomainState,
    family: str,
    excluded_feature: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if family not in FAMILY_FEATURES:
        raise ValueError(f"unknown domain family: {family}")
    base = add_features(frame)
    domain = build_domain_columns(frame, state, FAMILY_FEATURES[family])
    selected = [
        column
        for column in FAMILY_FEATURES[family]
        if column != excluded_feature
    ]
    for column in selected:
        base[column] = domain[column]
    return base, selected


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


def train_predict_family(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    family: str,
    excluded_feature: str | None,
    threads: int,
) -> tuple[np.ndarray, float, DomainState, list[str]]:
    state = fit_domain_state(training)
    train_features, domain_columns = add_family_features(
        training, state, family, excluded_feature
    )
    validation_features, validation_domain = add_family_features(
        validation, state, family, excluded_feature
    )
    if domain_columns != validation_domain:
        raise ValueError("training/validation domain feature contract mismatch")
    feature_columns = [
        column
        for column in train_features.columns
        if column not in (ID_COLUMN, TARGET)
    ]
    categorical = [column for column in CAT_COLS if column in feature_columns]
    categorical.extend(domain_columns)
    maps = build_category_maps(train_features, categorical)
    train_encoded = train_features
    validation_encoded = validation_features
    for column in categorical:
        train_encoded[column] = (
            train_encoded[column].astype(str).map(maps[column]).astype("float32")
        )
        validation_encoded[column] = (
            validation_encoded[column].astype(str).map(maps[column]).astype("float32")
        )
    train_matrix = train_encoded[feature_columns]
    dataset = lgb.Dataset(
        train_matrix,
        label=train_encoded[TARGET],
        categorical_feature=categorical,
        free_raw_data=True,
    )
    dataset.construct()
    validation_matrix = validation_encoded[feature_columns]
    del train_matrix, train_features, train_encoded, validation_features, validation_encoded
    gc.collect()
    started = time.monotonic()
    model = lgb.train(
        monotone_params(feature_columns, threads), dataset, num_boost_round=N_ROUNDS
    )
    prediction = model.predict(validation_matrix).astype("float64")
    fit_seconds = time.monotonic() - started
    del dataset, model, validation_matrix
    gc.collect()
    if not np.isfinite(prediction).all() or np.any(
        (prediction < 0.0) | (prediction > 1.0)
    ):
        raise ValueError("domain model produced invalid probabilities")
    return prediction, fit_seconds, state, domain_columns


def current_champion_oof(season: int, target: np.ndarray) -> np.ndarray:
    source = load_npz(OOF_DIR / f"season_{season}.npz")
    prefix = load_npz(ET25_OOF_DIR / f"season_{season}_prefixes.npz")
    if not np.array_equal(source["y"].astype("float64"), target):
        raise ValueError(f"train/OOF target order mismatch for {season}")
    return (
        (1.0 - CURRENT_ET_WEIGHT) * source["champion_final"].astype("float64")
        + CURRENT_ET_WEIGHT * prefix["prediction_25"].astype("float64")
    )


def prediction_cache_path(
    output: Path, family: str, variant: str, season: int
) -> Path:
    return output / "cache" / f"{family}_{variant}_{season}.npz"


def cached_prediction(
    path: Path,
    family: str,
    variant: str,
    season: int,
    target: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    if not path.is_file():
        return None
    payload = load_npz(path)
    actual = {
        "version": str(payload["cache_version"].item()),
        "family": str(payload["family"].item()),
        "variant": str(payload["variant"].item()),
        "season": int(payload["season"].item()),
        "target_sha256": str(payload["target_sha256"].item()),
    }
    expected = {
        "version": CACHE_VERSION,
        "family": family,
        "variant": variant,
        "season": season,
        "target_sha256": array_sha256(target),
    }
    if actual != expected:
        raise ValueError(f"stale domain cache: {actual} != {expected}")
    prediction = payload["prediction"].astype("float64")
    if prediction.shape != target.shape:
        raise ValueError("cached domain prediction length mismatch")
    return prediction, float(payload["fit_seconds"].item())


def evaluate_family_variant(
    train: pd.DataFrame,
    family: str,
    variant: str,
    excluded_feature: str | None,
    output: Path,
    threads: int,
    force: bool,
) -> tuple[list[dict], dict[int, np.ndarray], dict[int, np.ndarray]]:
    rows = []
    predictions = {}
    targets = {}
    for season in VALIDATION_SEASONS:
        training, validation = temporal_split(train, season)
        target = validation[TARGET].to_numpy(dtype="float64")
        baseline = load_npz(OOF_DIR / f"season_{season}.npz")[
            "team_raw_mono63"
        ].astype("float64")
        if baseline.shape != target.shape:
            raise ValueError(f"baseline OOF length mismatch for {season}")
        path = prediction_cache_path(output, family, variant, season)
        cached = None if force else cached_prediction(
            path, family, variant, season, target
        )
        if cached is None:
            print(
                f"[{family}/{variant}/{season}] train={len(training):,}; "
                f"validation={len(validation):,}",
                flush=True,
            )
            prediction, fit_seconds, state, domain_columns = train_predict_family(
                training, validation, family, excluded_feature, threads
            )
            save_npz(
                path,
                cache_version=np.asarray(CACHE_VERSION),
                family=np.asarray(family),
                variant=np.asarray(variant),
                season=np.asarray(season, dtype="int16"),
                target_sha256=np.asarray(array_sha256(target)),
                prediction=prediction,
                fit_seconds=np.asarray(fit_seconds),
                experience_edges=np.asarray(state.experience_edges),
                form_edges=np.asarray(state.form_edges),
                domain_columns=np.asarray(domain_columns),
            )
        else:
            prediction, fit_seconds = cached
            print(f"[{family}/{variant}/{season}] reuse {path}", flush=True)
        rows.append(
            {
                "family": family,
                "variant": variant,
                "excluded_feature": excluded_feature,
                "validation_season": season,
                "train_n": len(training),
                "validation_n": len(validation),
                "baseline_brier": brier(target, baseline),
                "domain_brier": brier(target, prediction),
                "brier_gain": brier(target, baseline) - brier(target, prediction),
                "baseline_prediction_mean": float(np.mean(baseline)),
                "domain_prediction_mean": float(np.mean(prediction)),
                "domain_prediction_std": float(np.std(prediction)),
                "fit_seconds": fit_seconds,
            }
        )
        predictions[season] = prediction
        targets[season] = target
    pooled_target = np.concatenate([targets[s] for s in VALIDATION_SEASONS])
    pooled_baseline = np.concatenate(
        [
            load_npz(OOF_DIR / f"season_{s}.npz")["team_raw_mono63"].astype(
                "float64"
            )
            for s in VALIDATION_SEASONS
        ]
    )
    pooled_prediction = np.concatenate(
        [predictions[s] for s in VALIDATION_SEASONS]
    )
    rows.append(
        {
            "family": family,
            "variant": variant,
            "excluded_feature": excluded_feature,
            "validation_season": "pooled",
            "train_n": np.nan,
            "validation_n": len(pooled_target),
            "baseline_brier": brier(pooled_target, pooled_baseline),
            "domain_brier": brier(pooled_target, pooled_prediction),
            "brier_gain": brier(pooled_target, pooled_baseline)
            - brier(pooled_target, pooled_prediction),
            "baseline_prediction_mean": float(np.mean(pooled_baseline)),
            "domain_prediction_mean": float(np.mean(pooled_prediction)),
            "domain_prediction_std": float(np.std(pooled_prediction)),
            "fit_seconds": sum(row["fit_seconds"] for row in rows),
        }
    )
    return rows, predictions, targets


def family_decision(rows: Sequence[Mapping]) -> dict:
    gains = {
        str(row["validation_season"]): float(row["brier_gain"])
        for row in rows
    }
    stable = gains["2023"] > 0.0 and gains["2024"] > 0.0 and gains["pooled"] > 0.0
    strong = stable and gains["pooled"] >= STRONG_GAIN
    return {"gains": gains, "stable": stable, "strong": strong}


def complementarity_rows(
    family: str,
    predictions: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
) -> list[dict]:
    rows = []
    for season in (*VALIDATION_SEASONS, "pooled"):
        if season == "pooled":
            target = np.concatenate([targets[s] for s in VALIDATION_SEASONS])
            prediction = np.concatenate(
                [predictions[s] for s in VALIDATION_SEASONS]
            )
            champion = np.concatenate(
                [current_champion_oof(s, targets[s]) for s in VALIDATION_SEASONS]
            )
        else:
            target = targets[int(season)]
            prediction = predictions[int(season)]
            champion = current_champion_oof(int(season), target)
        domain_error = (target - prediction) ** 2
        champion_error = (target - champion) ** 2
        top = champion_error >= np.quantile(champion_error, 0.90)
        rows.append(
            {
                "record_type": "raw_complementarity",
                "family": family,
                "weight": np.nan,
                "validation_season": season,
                "n": len(target),
                "corr_domain_champion": safe_corr(prediction, champion),
                "residual_error_correlation": safe_corr(domain_error, champion_error),
                "champion_top10_error_domain_wins_rate": float(
                    np.mean(domain_error[top] < champion_error[top])
                ),
                "brier_gain_vs_champion": brier(target, champion)
                - brier(target, prediction),
            }
        )
    return rows


def fixed_blend_results(
    family: str,
    predictions: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
) -> list[dict]:
    rows = []
    for weight in BLEND_WEIGHTS:
        for season in (*VALIDATION_SEASONS, "pooled"):
            if season == "pooled":
                target = np.concatenate([targets[s] for s in VALIDATION_SEASONS])
                domain = np.concatenate(
                    [predictions[s] for s in VALIDATION_SEASONS]
                )
                champion = np.concatenate(
                    [current_champion_oof(s, targets[s]) for s in VALIDATION_SEASONS]
                )
            else:
                target = targets[int(season)]
                domain = predictions[int(season)]
                champion = current_champion_oof(int(season), target)
            candidate = (1.0 - weight) * champion + weight * domain
            rows.append(
                {
                    "record_type": "fixed_blend",
                    "family": family,
                    "weight": weight,
                    "validation_season": season,
                    "n": len(target),
                    "corr_domain_champion": np.nan,
                    "residual_error_correlation": np.nan,
                    "champion_top10_error_domain_wins_rate": np.nan,
                    "brier_gain_vs_champion": brier(target, champion)
                    - brier(target, candidate),
                }
            )
    return rows


def schema_audit(
    train_columns: Sequence[str], trackman_columns: Sequence[str], crosswalk: Mapping
) -> pd.DataFrame:
    available = set(train_columns)
    rows = [
        ("season/year", "AVAILABLE" if "season" in available else "UNAVAILABLE", "season", "train season is explicit"),
        ("pitcher_id", "AVAILABLE", "pitcher_id", "anonymous main-table ID"),
        ("batter_id", "AVAILABLE", "batter_id", "anonymous main-table ID"),
        ("inning", "AVAILABLE", "inning", "pre-pitch row context"),
        ("top_bottom", "AVAILABLE", "top_bottom", "pre-pitch row context"),
        ("balls_before", "AVAILABLE", "balls_before", "pre-pitch row context"),
        ("strikes_before", "AVAILABLE", "strikes_before", "pre-pitch row context"),
        ("base_state", "AVAILABLE", "base_state and runner flags", "pre-pitch row context"),
        ("game_type", "AVAILABLE", "game_type", "low-cardinality code"),
        ("pitcher_hand", "AVAILABLE", "pitcher_hand", "H1/H2; no official R/L mapping"),
        ("batter_hand", "AVAILABLE", "batter_hand", "H1/H2; no official R/L mapping"),
        ("pitch/order sequence", "UNSAFE", "row_id only", "row order cannot be assumed to be game chronology"),
        ("same game identifier", "UNAVAILABLE", "none", "no game_id in train"),
        ("date/time", "UNAVAILABLE", "game_month/dayofweek only", "no date or timestamp"),
        ("pitcher cumulative history", "AVAILABLE", "asof_pitcher_*", "official pre-pitch cumulative features"),
        ("batter cumulative history", "AVAILABLE", "asof_batter_*", "official pre-pitch cumulative features"),
        ("matchup history", "UNSAFE", "pitcher_id+batter_id without order/game", "cannot create prior-only pair history"),
        ("main-table pitch type", "UNAVAILABLE", "none", "current pitch type is forbidden/unavailable"),
        ("main-table velocity/release/movement", "UNAVAILABLE", "none", "current mechanics unavailable"),
        ("TrackMan pitch/mechanics", "UNAVAILABLE", "trackman_history columns", f"no deterministic ID crosswalk; ID overlap={crosswalk['id_overlap_count']}"),
        ("weather", "UNAVAILABLE", "none", "no date, stadium, temperature, or weather"),
        ("count pressure", "DERIVABLE", "balls_before+strikes_before", "row-local categorical state"),
        ("runner pressure", "DERIVABLE", "runner flags/base_state", "row-local categorical state"),
        ("pitcher state", "DERIVABLE", "asof_pitcher_n and prior rates", "train-quantile buckets; target-free"),
        ("handedness interactions", "DERIVABLE", "pitcher_hand+batter_hand", "H1/H2 labels preserved"),
        ("season x count pressure", "DERIVABLE", "season+count", "inventory only; rule-regime axis closed"),
    ]
    return pd.DataFrame(rows, columns=["item", "status", "source", "reason"])


def feature_inventory(crosswalk: Mapping) -> pd.DataFrame:
    rows = []

    def add(family, feature, status, source, duplicate, evaluated, reason):
        rows.append(
            {
                "family": family,
                "feature": feature,
                "status": status,
                "source": source,
                "baseline_duplicate": duplicate,
                "evaluated": evaluated,
                "reason": reason,
            }
        )

    for feature in (
        "pitcher_pitches_so_far_game",
        "pitcher_pitches_so_far_inning",
        "prev_inning_pitch_count",
        "pitcher_pitches_last_1_game",
        "pitcher_pitches_last_3_games",
        "pitcher_pitches_last_5_games",
        "game_workload_sq",
        "recent_load_ratio",
    ):
        add("workload", feature, "UNSAFE", "missing game/date/order", False, False, "cannot reconstruct chronology")
    for feature in (
        "game_workload_x_count",
        "inning_workload_x_full_count",
        "recent_workload_x_runner_pressure",
    ):
        add("fatigue_pressure", feature, "UNAVAILABLE", "workload unavailable", False, False, "parent feature is unsafe")
    for feature in (
        "matchup_seen_before",
        "matchup_prior_pitch_count",
        "matchup_prior_pa_proxy",
        "same_game_matchup_count",
    ):
        add("familiarity", feature, "UNSAFE", "missing game/date/order", False, False, "prior-only pair sequence unavailable")
    add("pitcher_reliability", "log1p_asof_pitcher_n", "AVAILABLE", "asof_pitcher_n", True, False, "already in baseline")
    add("pitcher_reliability", "recent_form_minus_career", "AVAILABLE", "prev5-career", True, False, "already in baseline")
    for feature in FAMILY_FEATURES["pressure_state"]:
        add("pressure_state", feature, "DERIVABLE", "row-local count/base", False, True, "categorical state representation")
    for feature in FAMILY_FEATURES["pitcher_state_pressure"]:
        add("pitcher_state_pressure", feature, "DERIVABLE", "as-of history + row count", False, True, "fold-train quantile state")
    add("handedness_interaction", "same_hand", "AVAILABLE", "hand codes", True, False, "already in baseline")
    for feature in FAMILY_FEATURES["handedness_interaction"]:
        add("handedness_interaction", feature, "DERIVABLE", "H1/H2 + pressure/form", False, True, "row-local interaction")
    add("base_pressure_workload", "base_state_x_workload", "UNAVAILABLE", "workload unavailable", False, False, "parent feature is unsafe")
    for feature in (
        "release_mean_before",
        "release_std_before",
        "velocity_mean_before",
        "velocity_std_before",
        "recent_release_deviation",
    ):
        add("mechanics", feature, "UNAVAILABLE", "TrackMan", False, False, f"no deterministic crosswalk; overlap={crosswalk['id_overlap_count']}")
    add("weather", "temperature", "UNAVAILABLE", "no official join keys", False, False, "external weather join forbidden this round")
    add("rule_environment", "season_x_count_pressure", "DERIVABLE", "season+count", False, False, "inventory only; prior axis closed")
    return pd.DataFrame(rows)


def leakage_audit(train: pd.DataFrame) -> dict:
    training, validation = temporal_split(train, 2024)
    state = fit_domain_state(training)
    before = build_domain_columns(validation, state)
    flipped = validation.copy()
    flipped[TARGET] = 1 - flipped[TARGET]
    after_flip = build_domain_columns(flipped, state)
    subset = validation.iloc[::3]
    subset_features = build_domain_columns(subset, state)
    expected_subset = before.iloc[::3]
    expected_subset.index = subset_features.index
    return {
        "validation_target_flip_max_changed_cells": int(
            (before.astype(str) != after_flip.astype(str)).to_numpy().sum()
        ),
        "validation_row_removal_features_identical": bool(
            subset_features.astype(str).equals(expected_subset.astype(str))
        ),
        "future_season_excluded": int(training["season"].max()) < 2024,
        "preprocessing_fit_rows": len(training),
        "test_data_read": False,
        "passed": bool(
            before.astype(str).equals(after_flip.astype(str))
            and subset_features.astype(str).equals(expected_subset.astype(str))
            and int(training["season"].max()) < 2024
        ),
    }


def trackman_crosswalk_audit(train: pd.DataFrame, trackman_path: Path) -> dict:
    trackman_header = pd.read_csv(trackman_path, nrows=0).columns.tolist()
    required = "pitcher_trackman_id"
    if required not in trackman_header:
        return {"id_overlap_count": 0, "train_pitcher_coverage": 0.0, "columns": trackman_header}
    trackman_ids = set(
        pd.read_csv(trackman_path, usecols=[required])[required].dropna().astype(str)
    )
    train_ids = set(train["pitcher_id"].dropna().astype(str))
    overlap = train_ids & trackman_ids
    return {
        "id_overlap_count": len(overlap),
        "train_pitcher_coverage": len(overlap) / len(train_ids) if train_ids else 0.0,
        "columns": trackman_header,
    }


def run_fit_worker(args: argparse.Namespace) -> None:
    family = args.worker_family
    variant = args.worker_variant
    season = int(args.worker_season)
    excluded = None if args.worker_excluded == "__none__" else args.worker_excluded
    train = pd.read_csv(args.train, encoding="utf-8-sig").drop(columns=[ID_COLUMN])
    training, validation = temporal_split(train, season)
    target = validation[TARGET].to_numpy(dtype="float64")
    path = prediction_cache_path(args.output, family, variant, season)
    if path.is_file() and not args.force:
        cached_prediction(path, family, variant, season, target)
        print(f"[worker] reuse {path}", flush=True)
        return
    print(
        f"[worker {family}/{variant}/{season}] train={len(training):,}; "
        f"validation={len(validation):,}",
        flush=True,
    )
    prediction, fit_seconds, state, domain_columns = train_predict_family(
        training, validation, family, excluded, args.threads
    )
    save_npz(
        path,
        cache_version=np.asarray(CACHE_VERSION),
        family=np.asarray(family),
        variant=np.asarray(variant),
        season=np.asarray(season, dtype="int16"),
        target_sha256=np.asarray(array_sha256(target)),
        prediction=prediction,
        fit_seconds=np.asarray(fit_seconds),
        experience_edges=np.asarray(state.experience_edges),
        form_edges=np.asarray(state.form_edges),
        domain_columns=np.asarray(domain_columns),
    )


def ensure_isolated_caches(
    args: argparse.Namespace,
    jobs: Sequence[tuple[str, str, str | None, int]],
) -> None:
    for family, variant, excluded, season in jobs:
        path = prediction_cache_path(args.output, family, variant, season)
        if path.is_file() and not args.force:
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--train",
            str(args.train),
            "--output",
            str(args.output),
            "--threads",
            str(args.threads),
            "--worker-family",
            family,
            "--worker-variant",
            variant,
            "--worker-season",
            str(season),
            "--worker-excluded",
            excluded if excluded is not None else "__none__",
        ]
        if args.force:
            command.append("--force")
        subprocess.run(command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--trackman", type=Path, default=DEFAULT_TRACKMAN)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-family")
    parser.add_argument("--worker-variant", default="full")
    parser.add_argument("--worker-season", type=int)
    parser.add_argument("--worker-excluded", default="__none__")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    checksum_before = sha256_path(CHAMPION)
    if checksum_before != CHAMPION_SHA256:
        raise ValueError("immutable champion checksum mismatch before experiment")
    if args.worker_family is not None:
        if args.worker_season not in VALIDATION_SEASONS:
            raise ValueError("worker season must be one of the fixed validation seasons")
        run_fit_worker(args)
        return
    train = pd.read_csv(args.train, encoding="utf-8-sig")
    train_columns = train.columns.tolist()
    print(f"[schema] train rows={len(train):,}; test data not read", flush=True)
    crosswalk = trackman_crosswalk_audit(train, args.trackman)
    schema = schema_audit(train_columns, crosswalk["columns"], crosswalk)
    inventory = feature_inventory(crosswalk)
    args.output.mkdir(parents=True, exist_ok=True)
    schema.to_csv(args.output / "schema_audit.csv", index=False)
    inventory.to_csv(args.output / "feature_inventory.csv", index=False)
    del train
    gc.collect()
    ensure_isolated_caches(
        args,
        [
            (family, "full", None, season)
            for family in FAMILY_FEATURES
            for season in VALIDATION_SEASONS
        ],
    )
    train = pd.read_csv(args.train, encoding="utf-8-sig")
    train = train.drop(columns=[ID_COLUMN])

    all_rows = []
    decisions = []
    predictions_by_family = {}
    targets_by_family = {}
    for family in FAMILY_FEATURES:
        rows, predictions, targets = evaluate_family_variant(
            train, family, "full", None, args.output, args.threads, False
        )
        decision = family_decision(rows)
        decision.update({"family": family, "variant": "full"})
        decisions.append(decision)
        all_rows.extend(rows)
        predictions_by_family[family] = predictions
        targets_by_family[family] = targets
    strong_full = [row for row in decisions if row["strong"]]
    if strong_full:
        del train
        gc.collect()
        ablation_jobs = [
            (row["family"], f"without_{feature}", feature, season)
            for row in strong_full
            for feature in FAMILY_FEATURES[row["family"]]
            for season in VALIDATION_SEASONS
        ]
        ensure_isolated_caches(args, ablation_jobs)
        train = pd.read_csv(args.train, encoding="utf-8-sig").drop(
            columns=[ID_COLUMN]
        )
        for row in strong_full:
            family = row["family"]
            for feature in FAMILY_FEATURES[family]:
                variant = f"without_{feature}"
                ablation_rows, _, _ = evaluate_family_variant(
                    train,
                    family,
                    variant,
                    feature,
                    args.output,
                    args.threads,
                    False,
                )
                all_rows.extend(ablation_rows)

    result_frame = pd.DataFrame(all_rows)
    result_frame.to_csv(args.output / "family_oof_results.csv", index=False)
    result_frame[
        result_frame["family"].isin(["pressure_state", "pitcher_state_pressure"])
    ].to_csv(args.output / "pressure_results.csv", index=False)
    inventory[inventory["family"].isin(["workload", "fatigue_pressure"])] .to_csv(
        args.output / "workload_results.csv", index=False
    )
    inventory[inventory["family"].eq("familiarity")].to_csv(
        args.output / "familiarity_results.csv", index=False
    )

    complementarity = []
    blend_decisions = []
    for decision in decisions:
        if not decision["strong"]:
            continue
        family = decision["family"]
        complementarity.extend(
            complementarity_rows(
                family, predictions_by_family[family], targets_by_family[family]
            )
        )
        blends = fixed_blend_results(
            family, predictions_by_family[family], targets_by_family[family]
        )
        complementarity.extend(blends)
        for weight in BLEND_WEIGHTS:
            selected = [
                row
                for row in blends
                if math.isclose(row["weight"], weight)
            ]
            gains = {
                str(row["validation_season"]): float(row["brier_gain_vs_champion"])
                for row in selected
            }
            stable = gains["2023"] > 0 and gains["2024"] > 0 and gains["pooled"] > 0
            blend_decisions.append(
                {
                    "family": family,
                    "weight": weight,
                    "gains": gains,
                    "stable": stable,
                    "strong": stable and gains["pooled"] >= STRONG_GAIN,
                }
            )
    complementarity_frame = (
        pd.DataFrame(complementarity)
        if complementarity
        else pd.DataFrame(
            columns=[
                "record_type",
                "family",
                "weight",
                "validation_season",
                "n",
                "corr_domain_champion",
                "residual_error_correlation",
                "champion_top10_error_domain_wins_rate",
                "brier_gain_vs_champion",
            ]
        )
    )
    complementarity_frame.to_csv(args.output / "complementarity.csv", index=False)

    strong_families = [row for row in decisions if row["strong"]]
    stable_families = [row for row in decisions if row["stable"]]
    strong_blends = [row for row in blend_decisions if row["strong"]]
    if strong_families and strong_blends:
        verdict = "A. STRONG BASEBALL DOMAIN SIGNAL FOUND"
    elif stable_families:
        verdict = "B. stable but incremental domain signal"
    else:
        verdict = "C. no useful domain feature signal"

    leakage = leakage_audit(train)
    checksum_after = sha256_path(CHAMPION)
    if checksum_after != checksum_before:
        raise ValueError("immutable champion changed during experiment")
    summary = {
        "verdict": verdict,
        "current_immutable_champion": {
            "lb_bss": CHAMPION_LB_BSS,
            "artifact": str(CHAMPION),
            "sha256_before": checksum_before,
            "sha256_after": checksum_after,
            "unchanged": checksum_before == checksum_after,
        },
        "research_sources": list(RESEARCH_SOURCES),
        "schema": {
            "train_columns": train_columns,
            "trackman_crosswalk": crosswalk,
            "test_data_used": False,
        },
        "model_contract": {
            "baseline": "cached source-exact team_raw_mono63 temporal OOF",
            "candidate": "same mono63 LightGBM plus one independent feature family",
            "rounds": N_ROUNDS,
            "seed": 42,
            "feature_effects_hardcoded": False,
            "calibration_added": False,
        },
        "family_decisions": decisions,
        "blend_decisions": blend_decisions,
        "strong_families": [row["family"] for row in strong_families],
        "stable_families": [row["family"] for row in stable_families],
        "leakage_audit": leakage,
        "production": {
            "created": False,
            "reason": "verdict is not A" if verdict != "A. STRONG BASEBALL DOMAIN SIGNAL FOUND" else "full-train parity stage required",
            "filename_length_hard_gate": 30,
        },
    }
    write_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "family_decisions": [
                    {
                        "family": row["family"],
                        "gains": row["gains"],
                        "stable": row["stable"],
                        "strong": row["strong"],
                    }
                    for row in decisions
                ],
                "champion_unchanged": checksum_before == checksum_after,
                "test_data_used": False,
                "production_created": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
