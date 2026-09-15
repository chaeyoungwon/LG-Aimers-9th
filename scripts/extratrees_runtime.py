"""Frozen ID-free ExtraTrees 2% production endpoint.

The feature builder and preprocessing contract intentionally mirror the
accepted temporal OOF recipe.  Inference uses only the current row and
full-train-derived constants stored in ``extra_trees_meta.json``.
"""
from __future__ import annotations

import json
import os
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype


MODEL_FILE = "extra_trees.joblib"
META_FILE = "extra_trees_meta.json"
EXTRA_WEIGHT = 0.02
CHAMPION_WEIGHT = 0.98
ID_COL = "row_id"
TARGET_COL = "control_success"
RAW_PLAYER_IDS = ("pitcher_id", "batter_id")


def build_hgb52_features(
    frame: pd.DataFrame, cat_levels: Mapping[str, Sequence]
) -> pd.DataFrame:
    """Build the champion HGB/CatBoost ID-free 52-column row-local frame."""
    excluded = {ID_COL, TARGET_COL, *RAW_PLAYER_IDS}
    output = frame[
        [column for column in frame.columns if column not in excluded]
    ].copy()
    for column, levels in cat_levels.items():
        output[column] = pd.Categorical(output[column], categories=levels)
    count = frame["balls_before"] * 3 + frame["strikes_before"]
    output["count_state"] = pd.Categorical(count, categories=list(range(12)))
    output["same_hand"] = (
        frame["pitcher_hand"] == frame["batter_hand"]
    ).astype("int8")
    combo = (
        frame["pitcher_hand"].astype(str)
        + "-"
        + frame["batter_hand"].astype(str)
    )
    output["hand_combo"] = pd.Categorical(
        combo, categories=["1-1", "1-2", "2-1", "2-2"]
    )
    output["strike_minus_ball"] = (
        frame["asof_pitcher_strike_rate"]
        - frame["asof_pitcher_ball_rate"]
    ).astype("float32")
    output["success_minus_middle"] = (
        frame["asof_pitcher_success_rate"]
        - frame["asof_pitcher_middle_rate"]
    ).astype("float32")
    output["prev_vs_career"] = (
        frame["asof_pitcher_prev5_game_success_rate"]
        - frame["asof_pitcher_success_rate"]
    ).astype("float32")
    second = np.fmax(
        frame["asof_pitcher_breaking_rate"],
        frame["asof_pitcher_offspeed_rate"],
    )
    output["fastball_dominance"] = (
        frame["asof_pitcher_fastball_rate"] - second
    ).astype("float32")
    return output


def fit_preprocessor(features: pd.DataFrame) -> dict:
    """Fit category maps and numeric medians from training rows only."""
    categorical_levels: dict[str, list[str]] = {}
    numeric_medians: dict[str, float] = {}
    for column in features.columns:
        series = features[column]
        if isinstance(series.dtype, pd.CategoricalDtype) or is_object_dtype(
            series.dtype
        ):
            observed = (
                series.astype("object")
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            categorical_levels[column] = sorted(observed)
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(
                dtype="float64"
            )
            finite = values[np.isfinite(values)]
            numeric_medians[column] = (
                float(np.median(finite)) if len(finite) else 0.0
            )
    return {
        "feature_columns": list(features.columns),
        "categorical_levels": categorical_levels,
        "numeric_medians": numeric_medians,
        "categorical_missing_unknown_value": -1.0,
        "numeric_dtype": "float32",
    }


def transform_features(features: pd.DataFrame, preprocessor: Mapping) -> np.ndarray:
    columns = list(preprocessor["feature_columns"])
    if list(features.columns) != columns:
        raise ValueError("ExtraTrees feature order differs from frozen metadata")
    matrix = np.empty((len(features), len(columns)), dtype="float32")
    categorical = preprocessor["categorical_levels"]
    medians = preprocessor["numeric_medians"]
    for index, column in enumerate(columns):
        series = features[column]
        if column in categorical:
            mapping = {
                str(level): position
                for position, level in enumerate(categorical[column])
            }
            objects = series.astype("object")
            strings = objects.where(objects.notna(), "__MISSING__").astype(str)
            matrix[:, index] = (
                strings.map(mapping).fillna(-1).to_numpy(dtype="float32")
            )
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(
                dtype="float64"
            )
            values = np.where(np.isfinite(values), values, float(medians[column]))
            matrix[:, index] = values.astype("float32")
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite ExtraTrees input remains after preprocessing")
    return matrix


def predict_extratrees(
    test: pd.DataFrame,
    model_dir: str,
    model=None,
    metadata: Mapping | None = None,
) -> np.ndarray:
    if metadata is None:
        with open(os.path.join(model_dir, META_FILE), encoding="utf-8") as handle:
            metadata = json.load(handle)
    if model is None:
        model = joblib.load(os.path.join(model_dir, MODEL_FILE))
    features = build_hgb52_features(test, metadata["cat_levels"])
    if list(features.columns) != list(metadata["feature_columns"]):
        raise ValueError("runtime ExtraTrees feature contract mismatch")
    matrix = transform_features(features, metadata["preprocessor"])
    prediction = np.asarray(model.predict_proba(matrix)[:, 1], dtype="float64")
    if not np.isfinite(prediction).all() or np.any(
        (prediction < 0.0) | (prediction > 1.0)
    ):
        raise ValueError("ExtraTrees produced invalid probability")
    return prediction


def apply_extratrees_blend(
    champion_prediction: np.ndarray,
    test: pd.DataFrame,
    model_dir: str,
) -> np.ndarray:
    with open(os.path.join(model_dir, META_FILE), encoding="utf-8") as handle:
        metadata = json.load(handle)
    champion_weight = float(metadata["champion_weight"])
    extra_weight = float(metadata["extra_weight"])
    if champion_weight != CHAMPION_WEIGHT or extra_weight != EXTRA_WEIGHT:
        raise ValueError("ExtraTrees production weights are not frozen 0.98/0.02")
    if champion_weight + extra_weight != 1.0:
        raise ValueError("ExtraTrees production weights do not sum to one")
    champion = np.asarray(champion_prediction, dtype="float64")
    extra = predict_extratrees(test, model_dir, metadata=metadata)
    if champion.shape != extra.shape:
        raise ValueError("champion/ExtraTrees prediction length mismatch")
    output = champion_weight * champion + extra_weight * extra
    if not np.isfinite(output).all() or np.any((output < 0.0) | (output > 1.0)):
        raise ValueError("ExtraTrees convex blend produced invalid probability")
    return output
