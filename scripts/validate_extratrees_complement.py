"""Validate one ID-free ExtraTrees learner against the frozen champion OOF.

This is a research-only temporal OOF experiment.  It never reads test.csv,
never modifies the champion, and evaluates the preregistered small blend only
after a raw-prediction complementarity gate passes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime_blend_oof import build_ours_features, load_champion_meta  # noqa: E402


CHAMPION = ROOT / "artifacts" / "submit_r10_pitcher_w418194.zip"
EXPECTED_CHAMPION_SHA256 = (
    "ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47"
)
DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_OOF_DIR = ROOT / "artifacts" / "regime_blend_oof"
DEFAULT_OUTPUT = ROOT / "artifacts" / "extratrees_complement"

TARGET = "control_success"
ID_COLUMNS = ("pitcher_id", "batter_id")
VALIDATION_SEASONS = (2022, 2023, 2024)
BLEND_WEIGHTS = (0.01, 0.02, 0.03, 0.05, 0.08)
DEFAULT_N_ESTIMATORS = 300
ALLOWED_N_ESTIMATORS = (300, 500)
RANDOM_STATE = 42
MIN_SAMPLES_LEAF = 8
MAX_FEATURES = "sqrt"

# Preregistered before viewing ExtraTrees results.
MAX_COMPLEMENT_CORRELATION = 0.98
MIN_OVERALL_EXTRA_WIN_RATE = 0.20
MIN_HIGH_ERROR_EXTRA_WIN_RATE = 0.50
MAX_FOLD_OR_SUBSET_LOSS = 1e-5
MIN_SUBSET_ROWS = 5_000
TINY_SEGMENT_MAX_FRACTION = 0.20
TINY_SEGMENT_MAX_GAIN_SHARE = 0.80
TIE_ATOL = 1e-15
CACHE_VERSION = "extratrees-idfree-hgb52-v1"
RF_CACHE_VERSION = "randomforest-idfree-hgb52-v1"

MEMBER_COLUMNS = (
    "validation_season",
    "cutoff_season",
    "n",
    "n_estimators",
    "extra_brier",
    "champion_brier",
    "prediction_mean",
    "prediction_std",
    "prediction_min",
    "prediction_max",
    "fit_seconds",
    "predict_seconds",
)
CORRELATION_COLUMNS = (
    "validation_season",
    "n",
    "corr_extra_champion",
    "corr_extra_ours_stage",
    "corr_extra_team_stage",
)
COMPLEMENTARITY_COLUMNS = (
    "validation_season",
    "scope",
    "n",
    "extra_wins",
    "champion_wins",
    "ties",
    "extra_win_rate",
    "champion_win_rate",
    "tie_rate",
    "champion_error_threshold",
)
BLEND_COLUMNS = (
    "validation_season",
    "weight",
    "n",
    "champion_brier",
    "blend_brier",
    "brier_gain",
    "evaluated",
    "evaluation_status",
)
SUBSET_COLUMNS = (
    "validation_season",
    "weight",
    "axis",
    "subset",
    "n",
    "row_fraction",
    "champion_brier",
    "blend_brier",
    "brier_gain",
    "gain_contribution",
    "positive_gain_share",
    "history_threshold",
)


@dataclass(frozen=True)
class FeatureEncoding:
    feature_columns: tuple[str, ...]
    categorical_levels: Mapping[str, tuple[str, ...]]
    numeric_medians: Mapping[str, float]


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


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
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


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save_npz_atomic(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype="float64")
    y = np.asarray(right, dtype="float64")
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("correlation inputs must be same-length vectors")
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("Brier inputs must be same-length vectors")
    return float(np.mean((y - p) ** 2))


def make_extra_trees(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    n_jobs: int = -1,
) -> ExtraTreesClassifier:
    if n_estimators not in ALLOWED_N_ESTIMATORS:
        raise ValueError(
            f"n_estimators must be one of {ALLOWED_N_ESTIMATORS}, got {n_estimators}"
        )
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=False,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
    )


def make_random_forest(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    if n_estimators not in ALLOWED_N_ESTIMATORS:
        raise ValueError(
            f"n_estimators must be one of {ALLOWED_N_ESTIMATORS}, got {n_estimators}"
        )
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=True,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
    )


def fit_feature_encoding(frame: pd.DataFrame) -> FeatureEncoding:
    categorical_levels: dict[str, tuple[str, ...]] = {}
    numeric_medians: dict[str, float] = {}
    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype) or is_object_dtype(series.dtype):
            observed = series.astype("object").dropna().astype(str).unique().tolist()
            categorical_levels[column] = tuple(sorted(observed))
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
            finite = values[np.isfinite(values)]
            numeric_medians[column] = (
                float(np.median(finite)) if len(finite) else 0.0
            )
    return FeatureEncoding(
        feature_columns=tuple(frame.columns),
        categorical_levels=categorical_levels,
        numeric_medians=numeric_medians,
    )


def transform_features(frame: pd.DataFrame, encoding: FeatureEncoding) -> np.ndarray:
    if tuple(frame.columns) != encoding.feature_columns:
        raise ValueError("ExtraTrees feature order differs from fitted contract")
    matrix = np.empty((len(frame), len(frame.columns)), dtype="float32")
    for index, column in enumerate(frame.columns):
        series = frame[column]
        if column in encoding.categorical_levels:
            mapping = {
                level: position
                for position, level in enumerate(encoding.categorical_levels[column])
            }
            objects = series.astype("object")
            strings = objects.where(objects.notna(), "__MISSING__").astype(str)
            matrix[:, index] = (
                strings.map(mapping).fillna(-1).to_numpy(dtype="float32")
            )
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
            values = np.where(
                np.isfinite(values), values, encoding.numeric_medians[column]
            )
            matrix[:, index] = values.astype("float32")
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite value remains after ExtraTrees preprocessing")
    return matrix


def validate_feature_contract(
    feature_columns: Sequence[str], champion_hgb_columns: Sequence[str]
) -> None:
    if tuple(feature_columns) != tuple(champion_hgb_columns):
        raise ValueError("ExtraTrees does not exactly reuse champion HGB 52-column contract")
    forbidden = {"row_id", TARGET, *ID_COLUMNS}
    overlap = sorted(forbidden & set(feature_columns))
    if overlap:
        raise ValueError(f"forbidden ID/target columns in ExtraTrees: {overlap}")


def prediction_cache_path(
    output: Path,
    season: int,
    n_estimators: int,
    learner: str = "extratrees",
) -> Path:
    prefix = "season" if learner == "extratrees" else learner
    return output / "cache" / f"{prefix}_{season}_trees{n_estimators}.npz"


def load_cached_prediction(
    path: Path,
    season: int,
    n_estimators: int,
    target: np.ndarray,
    cache_version: str = CACHE_VERSION,
) -> tuple[np.ndarray, float, float] | None:
    if not path.is_file():
        return None
    payload = load_npz(path)
    expected = {
        "cache_version": cache_version,
        "season": season,
        "n_estimators": n_estimators,
        "target_sha256": array_sha256(np.asarray(target, dtype="float64")),
    }
    actual = {
        "cache_version": str(payload["cache_version"].item()),
        "season": int(payload["season"].item()),
        "n_estimators": int(payload["n_estimators"].item()),
        "target_sha256": str(payload["target_sha256"].item()),
    }
    if actual != expected:
        raise ValueError(f"stale ExtraTrees cache at {path}: {actual} != {expected}")
    prediction = np.asarray(payload["prediction"], dtype="float64")
    if prediction.shape != target.shape:
        raise ValueError(f"cached prediction length mismatch at {path}")
    return (
        prediction,
        float(payload["fit_seconds"].item()),
        float(payload["predict_seconds"].item()),
    )


def fit_fold_prediction(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    champion_meta: Mapping[str, object],
    n_estimators: int,
    n_jobs: int,
    learner: str = "extratrees",
) -> tuple[np.ndarray, float, float, FeatureEncoding]:
    train_features = build_ours_features(train_frame, champion_meta["cat_levels"])
    validation_features = build_ours_features(
        validation_frame, champion_meta["cat_levels"]
    )
    hgb_specs = [
        member for member in champion_meta["members"] if member["name"] == "hgb"
    ]
    if len(hgb_specs) != 1:
        raise ValueError("champion HGB feature contract is missing or ambiguous")
    validate_feature_contract(train_features.columns, hgb_specs[0]["feature_columns"])
    encoding = fit_feature_encoding(train_features)
    x_train = transform_features(train_features, encoding)
    x_validation = transform_features(validation_features, encoding)
    y_train = train_frame[TARGET].to_numpy(dtype="int8")
    del train_features, validation_features
    gc.collect()

    if learner == "extratrees":
        model = make_extra_trees(n_estimators=n_estimators, n_jobs=n_jobs)
    elif learner == "random_forest":
        model = make_random_forest(n_estimators=n_estimators, n_jobs=n_jobs)
    else:
        raise ValueError(f"unsupported learner: {learner}")
    started = time.monotonic()
    model.fit(x_train, y_train)
    fit_seconds = time.monotonic() - started
    started = time.monotonic()
    prediction = model.predict_proba(x_validation)[:, 1].astype("float64")
    predict_seconds = time.monotonic() - started
    del model, x_train, x_validation, y_train
    gc.collect()
    if not np.isfinite(prediction).all() or np.any(
        (prediction < 0.0) | (prediction > 1.0)
    ):
        raise ValueError("ExtraTrees produced invalid probabilities")
    return prediction, fit_seconds, predict_seconds, encoding


def member_result(
    season: int,
    target: np.ndarray,
    extra: np.ndarray,
    champion: np.ndarray,
    n_estimators: int,
    fit_seconds: float,
    predict_seconds: float,
) -> dict:
    return {
        "validation_season": season,
        "cutoff_season": season - 1,
        "n": len(target),
        "n_estimators": n_estimators,
        "extra_brier": brier(target, extra),
        "champion_brier": brier(target, champion),
        "prediction_mean": float(np.mean(extra)),
        "prediction_std": float(np.std(extra)),
        "prediction_min": float(np.min(extra)),
        "prediction_max": float(np.max(extra)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def correlation_result(
    season: int | str,
    extra: np.ndarray,
    champion: np.ndarray,
    ours_stage: np.ndarray,
    team_stage: np.ndarray,
) -> dict:
    return {
        "validation_season": season,
        "n": len(extra),
        "corr_extra_champion": safe_corr(extra, champion),
        "corr_extra_ours_stage": safe_corr(extra, ours_stage),
        "corr_extra_team_stage": safe_corr(extra, team_stage),
    }


def complementarity_result(
    season: int | str,
    scope: str,
    target: np.ndarray,
    extra: np.ndarray,
    champion: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict:
    y = np.asarray(target, dtype="float64")
    p_extra = np.asarray(extra, dtype="float64")
    p_champion = np.asarray(champion, dtype="float64")
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    extra_error = (y[mask] - p_extra[mask]) ** 2
    champion_error = (y[mask] - p_champion[mask]) ** 2
    advantage = champion_error - extra_error
    extra_wins = int(np.sum(advantage > TIE_ATOL))
    champion_wins = int(np.sum(advantage < -TIE_ATOL))
    ties = int(len(advantage) - extra_wins - champion_wins)
    denominator = max(len(advantage), 1)
    return {
        "validation_season": season,
        "scope": scope,
        "n": len(advantage),
        "extra_wins": extra_wins,
        "champion_wins": champion_wins,
        "ties": ties,
        "extra_win_rate": extra_wins / denominator,
        "champion_win_rate": champion_wins / denominator,
        "tie_rate": ties / denominator,
        "champion_error_threshold": threshold,
    }


def complementarity_rows(
    season: int | str,
    target: np.ndarray,
    extra: np.ndarray,
    champion: np.ndarray,
) -> list[dict]:
    champion_error = (np.asarray(target) - np.asarray(champion)) ** 2
    rows = [complementarity_result(season, "all", target, extra, champion)]
    for fraction, scope in ((0.10, "champion_error_top10pct"), (0.20, "champion_error_top20pct")):
        threshold = float(np.quantile(champion_error, 1.0 - fraction))
        mask = champion_error >= threshold
        rows.append(
            complementarity_result(
                season,
                scope,
                target,
                extra,
                champion,
                mask=mask,
                threshold=threshold,
            )
        )
    return rows


def complementarity_gate(
    correlation: pd.DataFrame, complementarity: pd.DataFrame
) -> tuple[bool, list[str]]:
    reasons = []
    fold_correlation = correlation[
        correlation["validation_season"].astype(str) != "pooled"
    ]
    if (
        fold_correlation["corr_extra_champion"].isna().any()
        or (fold_correlation["corr_extra_champion"] >= MAX_COMPLEMENT_CORRELATION).any()
    ):
        reasons.append(
            f"fold champion correlation is missing or >= {MAX_COMPLEMENT_CORRELATION}"
        )
    fold_rows = complementarity[
        complementarity["validation_season"].astype(str) != "pooled"
    ]
    overall = fold_rows[fold_rows["scope"].eq("all")]
    if len(overall) != len(VALIDATION_SEASONS) or (
        overall["extra_win_rate"] <= MIN_OVERALL_EXTRA_WIN_RATE
    ).any():
        reasons.append(
            f"a fold overall ExtraTrees win rate is <= {MIN_OVERALL_EXTRA_WIN_RATE}"
        )
    high_error = fold_rows[fold_rows["scope"].ne("all")]
    if len(high_error) != 2 * len(VALIDATION_SEASONS) or (
        high_error["extra_win_rate"] <= MIN_HIGH_ERROR_EXTRA_WIN_RATE
    ).any():
        reasons.append(
            "a fold champion top-10%/top-20% error ExtraTrees win rate is "
            f"<= {MIN_HIGH_ERROR_EXTRA_WIN_RATE}"
        )
    return not reasons, reasons


def blend_rows(
    folds: Mapping[int, Mapping[str, np.ndarray]],
    weights: Sequence[float] = BLEND_WEIGHTS,
) -> pd.DataFrame:
    rows = []
    for weight in weights:
        pooled_target = []
        pooled_champion = []
        pooled_blend = []
        for season in VALIDATION_SEASONS:
            fold = folds[season]
            candidate = (
                (1.0 - weight) * fold["champion"] + weight * fold["extra"]
            )
            rows.append(
                {
                    "validation_season": season,
                    "weight": weight,
                    "n": len(candidate),
                    "champion_brier": brier(fold["target"], fold["champion"]),
                    "blend_brier": brier(fold["target"], candidate),
                    "brier_gain": brier(fold["target"], fold["champion"])
                    - brier(fold["target"], candidate),
                    "evaluated": True,
                    "evaluation_status": "evaluated_after_complementarity_gate",
                }
            )
            pooled_target.append(fold["target"])
            pooled_champion.append(fold["champion"])
            pooled_blend.append(candidate)
        target = np.concatenate(pooled_target)
        champion = np.concatenate(pooled_champion)
        candidate = np.concatenate(pooled_blend)
        rows.append(
            {
                "validation_season": "pooled",
                "weight": weight,
                "n": len(candidate),
                "champion_brier": brier(target, champion),
                "blend_brier": brier(target, candidate),
                "brier_gain": brier(target, champion) - brier(target, candidate),
                "evaluated": True,
                "evaluation_status": "evaluated_after_complementarity_gate",
            }
        )
    return pd.DataFrame(rows, columns=BLEND_COLUMNS)


def unevaluated_blend_rows(
    weights: Sequence[float] = BLEND_WEIGHTS,
) -> pd.DataFrame:
    rows = [
        {
            "validation_season": "not_evaluated",
            "weight": weight,
            "n": 0,
            "champion_brier": np.nan,
            "blend_brier": np.nan,
            "brier_gain": np.nan,
            "evaluated": False,
            "evaluation_status": "not_evaluated_complementarity_gate_failed",
        }
        for weight in weights
    ]
    return pd.DataFrame(rows, columns=BLEND_COLUMNS)


def fold_subset_masks(fold: Mapping[str, np.ndarray]) -> list[tuple[str, str, np.ndarray]]:
    game_type = np.asarray(fold["game_type"]).astype(str)
    seen = np.asarray(fold["pitcher_seen"], dtype=bool)
    full_count = (
        np.asarray(fold["balls_before"], dtype="int64") == 3
    ) & (np.asarray(fold["strikes_before"], dtype="int64") == 2)
    form = np.asarray(fold["pitcher_form_adjustment"], dtype="float64")
    history = np.asarray(fold["pitcher_history"], dtype="float64")
    history_threshold = float(fold["history_threshold"])
    return [
        ("game_type", "R", game_type == "R"),
        ("game_type", "F", game_type == "F"),
        ("pitcher_seen", "seen", seen),
        ("pitcher_seen", "unseen", ~seen),
        ("full_count", "full", full_count),
        ("full_count", "non_full", ~full_count),
        ("pitcher_form", "positive", form > 0.0),
        ("pitcher_form", "negative", form < 0.0),
        ("pitcher_history", "low", history <= history_threshold),
        ("pitcher_history", "high", history > history_threshold),
    ]


def subset_rows(
    folds: Mapping[int, Mapping[str, np.ndarray]],
    weights: Sequence[float] = BLEND_WEIGHTS,
) -> pd.DataFrame:
    rows = []
    for weight in weights:
        pooled_parts: dict[tuple[str, str], list[tuple[np.ndarray, ...]]] = {}
        total_pooled_n = sum(len(folds[season]["target"]) for season in VALIDATION_SEASONS)
        for season in VALIDATION_SEASONS:
            fold = folds[season]
            target = fold["target"]
            champion = fold["champion"]
            candidate = (1.0 - weight) * champion + weight * fold["extra"]
            row_gain = (target - champion) ** 2 - (target - candidate) ** 2
            total_gain = float(np.mean(row_gain))
            for axis, subset, mask in fold_subset_masks(fold):
                mask = np.asarray(mask, dtype=bool)
                n = int(mask.sum())
                contribution = float(row_gain[mask].sum() / len(target)) if n else 0.0
                rows.append(
                    {
                        "validation_season": season,
                        "weight": weight,
                        "axis": axis,
                        "subset": subset,
                        "n": n,
                        "row_fraction": n / len(target),
                        "champion_brier": brier(target[mask], champion[mask]) if n else np.nan,
                        "blend_brier": brier(target[mask], candidate[mask]) if n else np.nan,
                        "brier_gain": (
                            brier(target[mask], champion[mask])
                            - brier(target[mask], candidate[mask])
                            if n
                            else np.nan
                        ),
                        "gain_contribution": contribution,
                        "positive_gain_share": (
                            contribution / total_gain
                            if total_gain > 0.0 and contribution > 0.0
                            else 0.0
                        ),
                        "history_threshold": (
                            float(fold["history_threshold"])
                            if axis == "pitcher_history"
                            else np.nan
                        ),
                    }
                )
                pooled_parts.setdefault((axis, subset), []).append(
                    (target[mask], champion[mask], candidate[mask], row_gain[mask])
                )
        pooled_target = np.concatenate(
            [folds[season]["target"] for season in VALIDATION_SEASONS]
        )
        pooled_champion = np.concatenate(
            [folds[season]["champion"] for season in VALIDATION_SEASONS]
        )
        pooled_candidate = np.concatenate(
            [
                (1.0 - weight) * folds[season]["champion"]
                + weight * folds[season]["extra"]
                for season in VALIDATION_SEASONS
            ]
        )
        pooled_total_gain = brier(pooled_target, pooled_champion) - brier(
            pooled_target, pooled_candidate
        )
        for (axis, subset), parts in pooled_parts.items():
            target = np.concatenate([part[0] for part in parts])
            champion = np.concatenate([part[1] for part in parts])
            candidate = np.concatenate([part[2] for part in parts])
            gains = np.concatenate([part[3] for part in parts])
            n = len(target)
            contribution = float(gains.sum() / total_pooled_n) if n else 0.0
            rows.append(
                {
                    "validation_season": "pooled",
                    "weight": weight,
                    "axis": axis,
                    "subset": subset,
                    "n": n,
                    "row_fraction": n / total_pooled_n,
                    "champion_brier": brier(target, champion) if n else np.nan,
                    "blend_brier": brier(target, candidate) if n else np.nan,
                    "brier_gain": (
                        brier(target, champion) - brier(target, candidate)
                        if n
                        else np.nan
                    ),
                    "gain_contribution": contribution,
                    "positive_gain_share": (
                        contribution / pooled_total_gain
                        if pooled_total_gain > 0.0 and contribution > 0.0
                        else 0.0
                    ),
                    "history_threshold": np.nan,
                }
            )
    return pd.DataFrame(rows, columns=SUBSET_COLUMNS)


def passing_weight_summary(
    blends: pd.DataFrame, subsets: pd.DataFrame
) -> list[dict]:
    passing = []
    if blends.empty or not bool(blends["evaluated"].all()):
        return passing
    for weight in BLEND_WEIGHTS:
        weight_rows = blends[np.isclose(blends["weight"], weight)]
        gain_by_season = {
            str(row.validation_season): float(row.brier_gain)
            for row in weight_rows.itertuples()
        }
        fold_gains = [gain_by_season[str(season)] for season in VALIDATION_SEASONS]
        subset_body = subsets[np.isclose(subsets["weight"], weight)]
        eligible = subset_body[subset_body["n"] >= MIN_SUBSET_ROWS]
        subset_safe = bool(
            eligible.empty
            or (eligible["brier_gain"] >= -MAX_FOLD_OR_SUBSET_LOSS).all()
        )
        concentration = eligible[
            (eligible["row_fraction"] <= TINY_SEGMENT_MAX_FRACTION)
            & (eligible["positive_gain_share"] >= TINY_SEGMENT_MAX_GAIN_SHARE)
        ]
        conditions = {
            "latest_positive": gain_by_season["2024"] > 0.0,
            "pooled_positive": gain_by_season["pooled"] > 0.0,
            "2022_safe": gain_by_season["2022"] >= -MAX_FOLD_OR_SUBSET_LOSS,
            "2023_safe": gain_by_season["2023"] >= -MAX_FOLD_OR_SUBSET_LOSS,
            "at_least_two_positive_folds": sum(gain > 0.0 for gain in fold_gains) >= 2,
            "subset_safe": subset_safe,
            "not_tiny_segment_concentrated": concentration.empty,
        }
        if all(conditions.values()):
            passing.append(
                {
                    "weight": weight,
                    "gains": gain_by_season,
                    "conditions": conditions,
                }
            )
    return passing


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def read_train(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"row_id", "season", TARGET, *ID_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"train is missing columns: {missing}")
    return frame


def run_random_forest_comparison(
    train: pd.DataFrame,
    champion_meta: Mapping[str, object],
    champion_folds: Mapping[int, Mapping[str, np.ndarray]],
    output: Path,
    n_estimators: int,
    n_jobs: int,
    force: bool,
) -> dict:
    """Run one RF only after ExtraTrees has a passing stable blend."""
    folds: dict[int, dict[str, np.ndarray]] = {}
    member_records = []
    correlation_records = []
    complementarity_records = []
    for season in VALIDATION_SEASONS:
        cutoff = season - 1
        train_frame = train[train["season"] <= cutoff]
        validation_frame = train[train["season"] == season]
        target = champion_folds[season]["target"]
        cache_path = prediction_cache_path(
            output, season, n_estimators, learner="random_forest"
        )
        cached = None if force else load_cached_prediction(
            cache_path,
            season,
            n_estimators,
            target,
            cache_version=RF_CACHE_VERSION,
        )
        if cached is None:
            print(
                f"[{season}] fit RandomForest comparison: train<={cutoff} "
                f"{len(train_frame):,} rows, validation {len(validation_frame):,}, "
                f"trees={n_estimators}",
                flush=True,
            )
            prediction, fit_seconds, predict_seconds, _ = fit_fold_prediction(
                train_frame,
                validation_frame,
                champion_meta,
                n_estimators,
                n_jobs,
                learner="random_forest",
            )
            save_npz_atomic(
                cache_path,
                cache_version=np.asarray(RF_CACHE_VERSION),
                season=np.asarray(season, dtype="int16"),
                n_estimators=np.asarray(n_estimators, dtype="int16"),
                target_sha256=np.asarray(array_sha256(target)),
                prediction=prediction,
                fit_seconds=np.asarray(fit_seconds),
                predict_seconds=np.asarray(predict_seconds),
            )
        else:
            prediction, fit_seconds, predict_seconds = cached
            print(f"[{season}] reuse {cache_path}", flush=True)
        source = champion_folds[season]
        fold = {
            key: value
            for key, value in source.items()
            if key != "extra"
        }
        fold["extra"] = prediction
        folds[season] = fold
        member_records.append(
            member_result(
                season,
                target,
                prediction,
                fold["champion"],
                n_estimators,
                fit_seconds,
                predict_seconds,
            )
        )
        correlation_records.append(
            correlation_result(
                season,
                prediction,
                fold["champion"],
                fold["ours_stage"],
                fold["team_stage"],
            )
        )
        complementarity_records.extend(
            complementarity_rows(
                season, target, prediction, fold["champion"]
            )
        )
        print(
            f"[{season}] RandomForest Brier={member_records[-1]['extra_brier']:.9f}, "
            f"champion={member_records[-1]['champion_brier']:.9f}, "
            f"corr={correlation_records[-1]['corr_extra_champion']:.6f}, "
            f"fit={fit_seconds:.1f}s",
            flush=True,
        )
        del train_frame, validation_frame
        gc.collect()

    pooled = {
        key: np.concatenate([folds[season][key] for season in VALIDATION_SEASONS])
        for key in ("target", "extra", "champion", "ours_stage", "team_stage")
    }
    correlation_records.append(
        correlation_result(
            "pooled",
            pooled["extra"],
            pooled["champion"],
            pooled["ours_stage"],
            pooled["team_stage"],
        )
    )
    complementarity_records.extend(
        complementarity_rows(
            "pooled", pooled["target"], pooled["extra"], pooled["champion"]
        )
    )
    members = pd.DataFrame(member_records, columns=MEMBER_COLUMNS)
    correlations = pd.DataFrame(correlation_records, columns=CORRELATION_COLUMNS)
    complementarity = pd.DataFrame(
        complementarity_records, columns=COMPLEMENTARITY_COLUMNS
    )
    gate_passed, gate_reasons = complementarity_gate(
        correlations, complementarity
    )
    if gate_passed:
        blends = blend_rows(folds)
        subsets = subset_rows(folds)
        passing = passing_weight_summary(blends, subsets)
    else:
        blends = unevaluated_blend_rows()
        subsets = pd.DataFrame(columns=SUBSET_COLUMNS)
        passing = []
    members.to_csv(output / "random_forest_member_results.csv", index=False)
    correlations.to_csv(output / "random_forest_correlation.csv", index=False)
    complementarity.to_csv(
        output / "random_forest_complementarity.csv", index=False
    )
    blends.to_csv(output / "random_forest_blend_results.csv", index=False)
    subsets.to_csv(output / "random_forest_subset_results.csv", index=False)
    if passing:
        comparison_verdict = "A. Stable complementary signal found"
    elif gate_passed:
        comparison_verdict = "B. Weak complementary signal, not deployment-worthy"
    else:
        comparison_verdict = "C. No complementary signal"
    return {
        "evaluated": True,
        "reason": "ExtraTrees had a passing stable blend",
        "verdict": comparison_verdict,
        "fit_seconds_total": float(members["fit_seconds"].sum()),
        "predict_seconds_total": float(members["predict_seconds"].sum()),
        "model": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "parameters": make_random_forest(n_estimators, n_jobs).get_params(),
            "parameter_grid_search": False,
            "seed_ensemble": False,
        },
        "complementarity_gate_passed": gate_passed,
        "complementarity_gate_reasons": gate_reasons,
        "blend_evaluated": gate_passed,
        "passing_weights": passing,
        "artifact_prefix": "random_forest_",
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--n-estimators",
        type=int,
        choices=ALLOWED_N_ESTIMATORS,
        default=DEFAULT_N_ESTIMATORS,
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.train.is_file():
        parser.error(f"train not found: {args.train}")
    if not CHAMPION.is_file():
        parser.error(f"champion not found: {CHAMPION}")
    for season in VALIDATION_SEASONS:
        cache = args.oof_dir / "cache" / f"season_{season}.npz"
        if not cache.is_file():
            parser.error(f"OOF cache not found: {cache}")

    checksum_before = sha256_path(CHAMPION)
    if checksum_before != EXPECTED_CHAMPION_SHA256:
        raise SystemExit(f"champion checksum mismatch before experiment: {checksum_before}")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"champion before: {checksum_before}", flush=True)
    print(f"load official train only: {args.train}", flush=True)
    train = read_train(args.train)
    champion_meta = load_champion_meta(CHAMPION)
    hgb_spec = next(
        member for member in champion_meta["members"] if member["name"] == "hgb"
    )
    if any(column in hgb_spec["feature_columns"] for column in ID_COLUMNS):
        raise ValueError("champion HGB contract unexpectedly contains raw player IDs")

    folds: dict[int, dict[str, np.ndarray]] = {}
    member_records = []
    correlation_records = []
    complementarity_records = []
    encoding_records = {}
    started_all = time.monotonic()
    for season in VALIDATION_SEASONS:
        cutoff = season - 1
        train_frame = train[train["season"] <= cutoff]
        validation_frame = train[train["season"] == season]
        oof = load_npz(args.oof_dir / "cache" / f"season_{season}.npz")
        target = validation_frame[TARGET].to_numpy(dtype="float64")
        if not np.array_equal(target, oof["y"].astype("float64")):
            raise ValueError(f"train/OOF target order mismatch for {season}")
        cache_path = prediction_cache_path(args.output, season, args.n_estimators)
        cached = None if args.force else load_cached_prediction(
            cache_path, season, args.n_estimators, target
        )
        if cached is None:
            print(
                f"[{season}] fit ExtraTrees: train<={cutoff} {len(train_frame):,} rows, "
                f"validation {len(validation_frame):,}, trees={args.n_estimators}",
                flush=True,
            )
            extra, fit_seconds, predict_seconds, encoding = fit_fold_prediction(
                train_frame,
                validation_frame,
                champion_meta,
                args.n_estimators,
                args.n_jobs,
            )
            save_npz_atomic(
                cache_path,
                cache_version=np.asarray(CACHE_VERSION),
                season=np.asarray(season, dtype="int16"),
                n_estimators=np.asarray(args.n_estimators, dtype="int16"),
                target_sha256=np.asarray(array_sha256(target)),
                prediction=extra,
                fit_seconds=np.asarray(fit_seconds),
                predict_seconds=np.asarray(predict_seconds),
            )
            encoding_records[str(season)] = {
                "categorical_levels": encoding.categorical_levels,
                "numeric_medians": encoding.numeric_medians,
            }
        else:
            extra, fit_seconds, predict_seconds = cached
            print(f"[{season}] reuse {cache_path}", flush=True)
            encoding_records[str(season)] = "reused prediction cache"
        champion = oof["champion_final"].astype("float64")
        fold = {
            "target": target,
            "extra": extra,
            "champion": champion,
            "ours_stage": oof["ours_stage"].astype("float64"),
            "team_stage": oof["team_stage"].astype("float64"),
            "game_type": validation_frame["game_type"].to_numpy(),
            "pitcher_seen": oof["x_pitcher_seen"].astype(bool),
            "balls_before": validation_frame["balls_before"].to_numpy(),
            "strikes_before": validation_frame["strikes_before"].to_numpy(),
            "pitcher_form_adjustment": (
                oof["pitcher_form_pred"].astype("float64")
                - oof["champion_base"].astype("float64")
            ),
            "pitcher_history": validation_frame["asof_pitcher_n"].to_numpy(
                dtype="float64"
            ),
            "history_threshold": np.asarray(
                float(train_frame["asof_pitcher_n"].median())
            ),
        }
        folds[season] = fold
        member_records.append(
            member_result(
                season,
                target,
                extra,
                champion,
                args.n_estimators,
                fit_seconds,
                predict_seconds,
            )
        )
        correlation_records.append(
            correlation_result(
                season,
                extra,
                champion,
                fold["ours_stage"],
                fold["team_stage"],
            )
        )
        complementarity_records.extend(
            complementarity_rows(season, target, extra, champion)
        )
        print(
            f"[{season}] ExtraTrees Brier={member_records[-1]['extra_brier']:.9f}, "
            f"champion={member_records[-1]['champion_brier']:.9f}, "
            f"corr={correlation_records[-1]['corr_extra_champion']:.6f}, "
            f"fit={fit_seconds:.1f}s",
            flush=True,
        )
        del train_frame, validation_frame, oof
        gc.collect()

    pooled = {
        key: np.concatenate([folds[season][key] for season in VALIDATION_SEASONS])
        for key in ("target", "extra", "champion", "ours_stage", "team_stage")
    }
    correlation_records.append(
        correlation_result(
            "pooled",
            pooled["extra"],
            pooled["champion"],
            pooled["ours_stage"],
            pooled["team_stage"],
        )
    )
    complementarity_records.extend(
        complementarity_rows(
            "pooled", pooled["target"], pooled["extra"], pooled["champion"]
        )
    )
    members = pd.DataFrame(member_records, columns=MEMBER_COLUMNS)
    correlations = pd.DataFrame(correlation_records, columns=CORRELATION_COLUMNS)
    complementarity = pd.DataFrame(
        complementarity_records, columns=COMPLEMENTARITY_COLUMNS
    )
    gate_passed, gate_reasons = complementarity_gate(
        correlations, complementarity
    )
    print(
        f"complementarity gate: {'pass' if gate_passed else 'fail'} "
        f"{gate_reasons}",
        flush=True,
    )
    if gate_passed:
        blends = blend_rows(folds)
        subsets = subset_rows(folds)
        passing = passing_weight_summary(blends, subsets)
    else:
        blends = unevaluated_blend_rows()
        subsets = pd.DataFrame(columns=SUBSET_COLUMNS)
        passing = []

    if passing:
        verdict = "A. Stable complementary signal found"
    elif gate_passed:
        verdict = "B. Weak complementary signal, not deployment-worthy"
    else:
        verdict = "C. No complementary signal"

    members.to_csv(args.output / "member_results.csv", index=False)
    correlations.to_csv(args.output / "correlation.csv", index=False)
    complementarity.to_csv(args.output / "complementarity.csv", index=False)
    blends.to_csv(args.output / "blend_results.csv", index=False)
    subsets.to_csv(args.output / "subset_results.csv", index=False)
    if passing:
        print(
            "ExtraTrees stable gate passed; run one RandomForest comparison",
            flush=True,
        )
        random_forest = run_random_forest_comparison(
            train,
            champion_meta,
            folds,
            args.output,
            args.n_estimators,
            args.n_jobs,
            args.force,
        )
    else:
        random_forest = {
            "evaluated": False,
            "reason": "ExtraTrees did not pass stable blend adoption criteria",
        }
    checksum_after = sha256_path(CHAMPION)
    if checksum_after != checksum_before:
        raise RuntimeError("champion changed during ExtraTrees experiment")
    summary = {
        "experiment": "ExtraTrees ID-free HGB-52 complementarity",
        "verdict": verdict,
        "champion": {
            "lb_bss": 1016.4442212358,
            "path": str(CHAMPION),
            "sha256_before": checksum_before,
            "sha256_after": checksum_after,
            "immutable": checksum_before == checksum_after,
        },
        "data": {
            "train": str(args.train),
            "test_rows_read": False,
            "validation_seasons": list(VALIDATION_SEASONS),
            "cutoffs": {str(season): season - 1 for season in VALIDATION_SEASONS},
            "validation_labels_used_for_training_statistics": False,
        },
        "feature_contract": {
            "source": "champion HGB/CatBoost row-local feature contract",
            "n_features": len(hgb_spec["feature_columns"]),
            "feature_columns": hgb_spec["feature_columns"],
            "raw_player_ids_included": False,
            "categorical_encoding": "cutoff-train-only ordinal map; unknown/missing=-1",
            "numeric_missing": "cutoff-train-only median",
            "fold_encoding": encoding_records,
        },
        "model": {
            "class": "sklearn.ensemble.ExtraTreesClassifier",
            "parameters": make_extra_trees(args.n_estimators, args.n_jobs).get_params(),
            "tree_count_reason": (
                "300-tree allowed fallback selected because the largest fold has "
                "1,221,585 training rows on a 16 GiB host"
                if args.n_estimators == 300
                else "requested 500-tree baseline"
            ),
            "parameter_grid_search": False,
            "seed_ensemble": False,
            "fit_seconds_total": float(members["fit_seconds"].sum()),
            "predict_seconds_total": float(members["predict_seconds"].sum()),
        },
        "preregistered_gates": {
            "max_champion_correlation": MAX_COMPLEMENT_CORRELATION,
            "min_overall_extra_win_rate": MIN_OVERALL_EXTRA_WIN_RATE,
            "min_high_error_extra_win_rate": MIN_HIGH_ERROR_EXTRA_WIN_RATE,
            "max_fold_or_subset_loss": MAX_FOLD_OR_SUBSET_LOSS,
            "min_subset_rows": MIN_SUBSET_ROWS,
            "tiny_segment_max_fraction": TINY_SEGMENT_MAX_FRACTION,
            "tiny_segment_max_gain_share": TINY_SEGMENT_MAX_GAIN_SHARE,
        },
        "complementarity_gate_passed": gate_passed,
        "complementarity_gate_reasons": gate_reasons,
        "blend_evaluated": gate_passed,
        "blend_weights": list(BLEND_WEIGHTS),
        "passing_weights": passing,
        "random_forest_evaluated": random_forest["evaluated"],
        "random_forest": random_forest,
        "xgboost_evaluated": False,
        "production_artifact_created": False,
        "leaderboard_submission_created": False,
        "latest_invocation_elapsed_seconds": time.monotonic() - started_all,
        "latest_invocation_peak_rss_bytes": peak_rss_bytes(),
    }
    write_json(args.output / "summary.json", summary)
    print(f"champion after: {checksum_after}", flush=True)
    print(f"passing weights: {[item['weight'] for item in passing]}", flush=True)
    print(f"verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
