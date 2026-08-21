"""Frozen official-data-only TrackMan expert runtime.

Inference reads an immutable full-train-derived HIGH mapping/feature lookup.
No statistic is computed across test rows; every lookup and prediction is row-local.
"""
from __future__ import annotations

import json
import os
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

try:  # flat import inside the submission ZIP
    from extratrees_runtime import build_hgb52_features, transform_features
except ModuleNotFoundError:  # repository import for build/parity tests
    from scripts.extratrees_runtime import build_hgb52_features, transform_features


MODEL_FILE = "tm_expert.joblib"
META_FILE = "tm_meta.json"
LOOKUP_FILE = "tm_lookup.npz"
TRACKMAN_WEIGHT = 0.10
CHAMPION_WEIGHT = 0.90
MATCHED_OUTPUT_DECIMALS = 14


def load_lookup(model_dir: str, metadata: Mapping) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(os.path.join(model_dir, LOOKUP_FILE), allow_pickle=False)
    pitcher_ids = payload["pitcher_ids"].astype("int64")
    values = payload["values"].astype("float64")
    if values.shape != (len(pitcher_ids), len(metadata["trackman_feature_columns"])):
        raise ValueError("TrackMan lookup shape differs from frozen metadata")
    if len(np.unique(pitcher_ids)) != len(pitcher_ids):
        raise ValueError("TrackMan lookup pitcher ids are not unique")
    return pitcher_ids, values


def attach_trackman_features(
    rows: pd.DataFrame,
    metadata: Mapping,
    pitcher_ids: np.ndarray,
    lookup_values: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Attach a frozen lookup using only each row's pitcher_id."""
    columns = list(metadata["trackman_feature_columns"])
    id_to_position = {int(value): index for index, value in enumerate(pitcher_ids)}
    positions = np.asarray(
        [id_to_position.get(int(value), -1) for value in rows["pitcher_id"]],
        dtype="int64",
    )
    matched = positions >= 0
    output = np.full((len(rows), len(columns)), np.nan, dtype="float64")
    output[matched] = lookup_values[positions[matched]]
    features = pd.DataFrame(output, columns=columns, index=rows.index)
    for column in ("tm_matched", "tm_confidence", "tm_history_n"):
        features[column] = features[column].fillna(0.0)
    if not np.array_equal(features["tm_matched"].eq(1.0).to_numpy(), matched):
        raise ValueError("TrackMan lookup matched flag parity failed")
    return features, matched


def predict_trackman(
    rows: pd.DataFrame,
    model_dir: str,
    model=None,
    metadata: Mapping | None = None,
    lookup: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if metadata is None:
        with open(os.path.join(model_dir, META_FILE), encoding="utf-8") as handle:
            metadata = json.load(handle)
    if model is None:
        model = joblib.load(os.path.join(model_dir, MODEL_FILE))
    if lookup is None:
        lookup = load_lookup(model_dir, metadata)
    physical, matched = attach_trackman_features(rows, metadata, *lookup)
    if not matched.any():
        return np.empty(0, dtype="float64"), matched
    base = build_hgb52_features(rows.loc[matched], metadata["cat_levels"])
    features = pd.concat(
        [base, physical.loc[matched, metadata["trackman_feature_columns"]]], axis=1
    )
    if list(features.columns) != list(metadata["feature_columns"]):
        raise ValueError("TrackMan production feature order mismatch")
    matrix = transform_features(features, metadata["preprocessor"])
    prediction = np.asarray(model.predict_proba(matrix)[:, 1], dtype="float64")
    if not np.isfinite(prediction).all() or np.any(
        (prediction < 0.0) | (prediction > 1.0)
    ):
        raise ValueError("TrackMan expert produced invalid probability")
    return prediction, matched


def apply_trackman_blend(
    champion_prediction: np.ndarray,
    rows: pd.DataFrame,
    model_dir: str,
) -> np.ndarray:
    with open(os.path.join(model_dir, META_FILE), encoding="utf-8") as handle:
        metadata = json.load(handle)
    champion_weight = float(metadata["champion_weight"])
    trackman_weight = float(metadata["trackman_weight"])
    if champion_weight != CHAMPION_WEIGHT or trackman_weight != TRACKMAN_WEIGHT:
        raise ValueError("TrackMan production weights are not frozen 0.90/0.10")
    if int(metadata.get("matched_output_round_decimals", -1)) != MATCHED_OUTPUT_DECIMALS:
        raise ValueError("TrackMan matched-output determinism rounding is not frozen")
    if champion_weight + trackman_weight != 1.0:
        raise ValueError("TrackMan production weights do not sum to one")
    champion = np.asarray(champion_prediction, dtype="float64")
    expert, matched = predict_trackman(rows, model_dir, metadata=metadata)
    if len(champion) != len(rows):
        raise ValueError("champion/row length mismatch")
    output = champion.copy()
    output[matched] = np.round(
        champion_weight * champion[matched] + trackman_weight * expert,
        decimals=MATCHED_OUTPUT_DECIMALS,
    )
    if not np.array_equal(output[~matched], champion[~matched]):
        raise ValueError("unmatched TrackMan rows changed")
    if not np.isfinite(output).all() or np.any((output < 0.0) | (output > 1.0)):
        raise ValueError("TrackMan blend produced invalid probability")
    return output
