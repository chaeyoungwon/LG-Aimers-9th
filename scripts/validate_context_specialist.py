"""Validate cutoff-safe context-trained LightGBM specialists.

Candidates are evaluated in a fixed priority order. A later stage is reached
only when the preceding stage has a stable raw specialist signal. Evaluation
rows and distributions are never used for feature fitting or routing rules.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib
import json
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

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

from src.train_base import PARAMS as TEAM_PARAMS  # noqa: E402


DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
OUTPUT_DIR = ROOT / "artifacts" / "context_specialist"
OOF_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
ET25_OOF_DIR = ROOT / "artifacts" / "extratrees_size_ablation" / "cache"
CHAMPION = ROOT / "artifacts" / "sub_et25_w020.zip"
CHAMPION_SHA256 = (
    "316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf"
)
CHAMPION_LB_BSS = 1017.0233029621
CURRENT_ET_WEIGHT = 0.0200

TARGET = "control_success"
ID_COLUMNS = ("pitcher_id", "batter_id")
VALIDATION_SEASONS = (2022, 2023, 2024)
BLEND_WEIGHTS = (0.10, 0.25, 0.50)
MIN_TRAIN_ROWS = 50_000
MIN_DIAGNOSTIC_ROWS = 1_000
MAX_MAJOR_SUBSET_LOSS = 1e-5
LARGE_OVERALL_GAIN = 2e-5
N_ROUNDS = 220
CACHE_VERSION = "context-specialist-mono63-hgb52-v1"

BASE_CATEGORICAL = ("top_bottom", "game_type", "base_state")
DERIVED_CATEGORICAL = ("count_state", "hand_combo")


@dataclass(frozen=True)
class SpecialistSpec:
    stage: int
    name: str
    partition: str
    subset: str
    mask: Callable[[pd.DataFrame, int, bool], np.ndarray]
    minimum_train_rows: int = MIN_TRAIN_ROWS
    low_sample_exception: bool = False


def two_strike_mask(frame: pd.DataFrame, cutoff: int, training: bool) -> np.ndarray:
    del cutoff, training
    return frame["strikes_before"].eq(2).to_numpy()


def three_ball_mask(frame: pd.DataFrame, cutoff: int, training: bool) -> np.ndarray:
    del cutoff, training
    return frame["balls_before"].eq(3).to_numpy()


def full_count_mask(frame: pd.DataFrame, cutoff: int, training: bool) -> np.ndarray:
    del cutoff, training
    return (
        frame["balls_before"].eq(3) & frame["strikes_before"].eq(2)
    ).to_numpy()


def regular_mask(frame: pd.DataFrame, cutoff: int, training: bool) -> np.ndarray:
    del cutoff, training
    return frame["game_type"].eq("R").to_numpy()


def final_mask(frame: pd.DataFrame, cutoff: int, training: bool) -> np.ndarray:
    del cutoff, training
    return frame["game_type"].eq("F").to_numpy()


def established_mask(frame: pd.DataFrame, cutoff: int, training: bool) -> np.ndarray:
    del cutoff, training
    return frame["asof_pitcher_n"].ge(1_000).to_numpy()


def low_history_mask(frame: pd.DataFrame, cutoff: int, training: bool) -> np.ndarray:
    del cutoff, training
    return frame["asof_pitcher_n"].lt(1_000).to_numpy()


STAGES = (
    (
        SpecialistSpec(1, "two_strike", "strike_bucket", "2_strike", two_strike_mask),
    ),
    (
        SpecialistSpec(2, "three_ball", "ball_pressure", "balls_3", three_ball_mask),
    ),
    (
        SpecialistSpec(
            3,
            "full_count",
            "full_count",
            "3_2",
            full_count_mask,
            low_sample_exception=True,
        ),
    ),
    (
        SpecialistSpec(4, "game_type_r", "game_type", "R", regular_mask),
        SpecialistSpec(4, "game_type_f", "game_type", "F", final_mask),
    ),
    (
        SpecialistSpec(
            5,
            "pitcher_established",
            "pitcher_history",
            "seen_established",
            established_mask,
        ),
        SpecialistSpec(
            5,
            "pitcher_unseen_low",
            "pitcher_history",
            "unseen_or_low_history",
            low_history_mask,
        ),
    ),
)


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


def load_champion_contract(path: Path = CHAMPION) -> tuple[tuple[str, ...], dict]:
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read("model/meta.json"))
    members = [member for member in meta["members"] if member["name"] == "hgb"]
    if len(members) != 1:
        raise ValueError("champion HGB feature contract is missing or ambiguous")
    columns = tuple(members[0]["feature_columns"])
    if set(ID_COLUMNS) & set(columns):
        raise ValueError("champion HGB feature contract unexpectedly contains raw IDs")
    return columns, meta


def category_levels_from_training(frame: pd.DataFrame) -> dict[str, tuple]:
    levels = {}
    for column in BASE_CATEGORICAL:
        levels[column] = tuple(
            sorted(frame[column].dropna().astype(str).unique().tolist())
        )
    count = frame["balls_before"] * 3 + frame["strikes_before"]
    levels["count_state"] = tuple(sorted(count.dropna().astype(int).unique().tolist()))
    combo = frame["pitcher_hand"].astype(str) + "-" + frame["batter_hand"].astype(str)
    levels["hand_combo"] = tuple(sorted(combo.dropna().unique().tolist()))
    return levels


def build_hgb52_features(
    frame: pd.DataFrame, category_levels: Mapping[str, Sequence]
) -> pd.DataFrame:
    exclude = {"row_id", TARGET, *ID_COLUMNS}
    out = frame[[column for column in frame.columns if column not in exclude]].copy()
    for column in BASE_CATEGORICAL:
        values = frame[column].astype("object")
        strings = values.where(values.notna(), np.nan).astype(str)
        out[column] = pd.Categorical(strings, categories=category_levels[column])
    count = frame["balls_before"] * 3 + frame["strikes_before"]
    out["count_state"] = pd.Categorical(
        count, categories=category_levels["count_state"]
    )
    out["same_hand"] = (frame["pitcher_hand"] == frame["batter_hand"]).astype("int8")
    combo = frame["pitcher_hand"].astype(str) + "-" + frame["batter_hand"].astype(str)
    out["hand_combo"] = pd.Categorical(
        combo, categories=category_levels["hand_combo"]
    )
    out["strike_minus_ball"] = (
        frame["asof_pitcher_strike_rate"] - frame["asof_pitcher_ball_rate"]
    ).astype("float32")
    out["success_minus_middle"] = (
        frame["asof_pitcher_success_rate"] - frame["asof_pitcher_middle_rate"]
    ).astype("float32")
    out["prev_vs_career"] = (
        frame["asof_pitcher_prev5_game_success_rate"]
        - frame["asof_pitcher_success_rate"]
    ).astype("float32")
    second = np.fmax(
        frame["asof_pitcher_breaking_rate"], frame["asof_pitcher_offspeed_rate"]
    )
    out["fastball_dominance"] = (
        frame["asof_pitcher_fastball_rate"] - second
    ).astype("float32")
    return out


def validate_feature_contract(
    actual: Sequence[str], expected: Sequence[str]
) -> None:
    if tuple(actual) != tuple(expected):
        raise ValueError("specialist does not exactly use champion HGB 52-column contract")
    forbidden = {"row_id", TARGET, *ID_COLUMNS}
    overlap = sorted(forbidden & set(actual))
    if overlap:
        raise ValueError(f"forbidden specialist features: {overlap}")


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
        TEAM_PARAMS,
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


def fit_specialist(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    feature_contract: Sequence[str],
    threads: int,
) -> tuple[np.ndarray, float, float, dict]:
    levels = category_levels_from_training(training)
    train_features = build_hgb52_features(training, levels)
    validation_features = build_hgb52_features(validation, levels)
    validate_feature_contract(train_features.columns, feature_contract)
    validate_feature_contract(validation_features.columns, feature_contract)
    categorical = [
        column
        for column in (*BASE_CATEGORICAL, *DERIVED_CATEGORICAL)
        if column in train_features.columns
    ]
    dataset = lgb.Dataset(
        train_features,
        label=training[TARGET].to_numpy(dtype="float64"),
        categorical_feature=categorical,
        free_raw_data=False,
    )
    started = time.monotonic()
    model = lgb.train(
        monotone_params(feature_contract, threads), dataset, num_boost_round=N_ROUNDS
    )
    fit_seconds = time.monotonic() - started
    started = time.monotonic()
    prediction = model.predict(validation_features).astype("float64")
    predict_seconds = time.monotonic() - started
    del dataset, model, train_features, validation_features
    gc.collect()
    if not np.isfinite(prediction).all() or np.any(
        (prediction < 0.0) | (prediction > 1.0)
    ):
        raise ValueError("specialist produced invalid probability")
    return prediction, fit_seconds, predict_seconds, {
        key: list(values) for key, values in levels.items()
    }


def current_champion_oof(season: int, target: np.ndarray) -> tuple[np.ndarray, dict]:
    source = load_npz(OOF_DIR / f"season_{season}.npz")
    prefix = load_npz(ET25_OOF_DIR / f"season_{season}_prefixes.npz")
    if not np.array_equal(source["y"].astype("float64"), target):
        raise ValueError(f"train/OOF target order mismatch for {season}")
    champion = (
        (1.0 - CURRENT_ET_WEIGHT) * source["champion_final"].astype("float64")
        + CURRENT_ET_WEIGHT * prefix["prediction_25"].astype("float64")
    )
    return champion, source


def cache_path(output: Path, spec: SpecialistSpec, season: int) -> Path:
    return output / "cache" / f"{spec.name}_{season}.npz"


def cached_prediction(
    path: Path, spec: SpecialistSpec, season: int, target: np.ndarray
) -> tuple[np.ndarray, float, float] | None:
    if not path.is_file():
        return None
    payload = load_npz(path)
    actual = {
        "version": str(payload["cache_version"].item()),
        "specialist": str(payload["specialist"].item()),
        "season": int(payload["season"].item()),
        "target_sha256": str(payload["target_sha256"].item()),
    }
    expected = {
        "version": CACHE_VERSION,
        "specialist": spec.name,
        "season": season,
        "target_sha256": array_sha256(target),
    }
    if actual != expected:
        raise ValueError(f"stale specialist cache: {actual} != {expected}")
    prediction = payload["prediction"].astype("float64")
    if prediction.shape != target.shape:
        raise ValueError("cached specialist prediction length mismatch")
    return (
        prediction,
        float(payload["fit_seconds"].item()),
        float(payload["predict_seconds"].item()),
    )


def raw_gate(gains: Mapping[str, float], enough_training_rows: bool) -> dict:
    conditions = {
        "2023_positive": gains["2023"] > 0.0,
        "2024_positive": gains["2024"] > 0.0,
        "pooled_positive": gains["pooled"] > 0.0,
        "minimum_training_rows": enough_training_rows,
    }
    return {"passed": all(conditions.values()), "conditions": conditions}


def complementarity_record(
    spec: SpecialistSpec,
    season: int | str,
    target: np.ndarray,
    specialist: np.ndarray,
    champion: np.ndarray,
) -> dict:
    specialist_error = (target - specialist) ** 2
    champion_error = (target - champion) ** 2
    top = champion_error >= np.quantile(champion_error, 0.90)
    return {
        "specialist": spec.name,
        "partition": spec.partition,
        "subset": spec.subset,
        "validation_season": season,
        "n": len(target),
        "corr_specialist_champion": safe_corr(specialist, champion),
        "specialist_wins_rate": float(np.mean(specialist_error < champion_error)),
        "champion_top10_error_specialist_wins_rate": float(
            np.mean(specialist_error[top] < champion_error[top])
        ),
        "specialist_error_champion_error_corr": safe_corr(
            specialist_error, champion_error
        ),
    }


def routed_prediction(
    champion: np.ndarray,
    specialist: np.ndarray,
    target_mask: np.ndarray,
    weight: float,
) -> np.ndarray:
    output = np.asarray(champion, dtype="float64").copy()
    output[target_mask] = (
        (1.0 - float(weight)) * output[target_mask]
        + float(weight) * np.asarray(specialist, dtype="float64")
    )
    return output


def major_subset_masks(frame: pd.DataFrame, source: Mapping[str, np.ndarray]) -> dict:
    form = source["pitcher_form_pred"].astype("float64") - source[
        "champion_base"
    ].astype("float64")
    return {
        ("game_type", "R"): frame["game_type"].eq("R").to_numpy(),
        ("game_type", "F"): frame["game_type"].eq("F").to_numpy(),
        ("pitcher_seen", "seen"): source["x_pitcher_seen"].astype(bool),
        ("pitcher_seen", "unseen"): ~source["x_pitcher_seen"].astype(bool),
        ("full_count", "3_2"): (
            frame["balls_before"].eq(3) & frame["strikes_before"].eq(2)
        ).to_numpy(),
        ("full_count", "non_3_2"): ~(
            frame["balls_before"].eq(3) & frame["strikes_before"].eq(2)
        ).to_numpy(),
        ("pitcher_form", "positive"): form > 1e-12,
        ("pitcher_form", "neutral"): np.abs(form) <= 1e-12,
        ("pitcher_form", "negative"): form < -1e-12,
    }


def evaluate_specialist(
    spec: SpecialistSpec,
    train: pd.DataFrame,
    feature_contract: Sequence[str],
    output: Path,
    threads: int,
    force: bool,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    folds = {}
    result_rows = []
    complementarity_rows = []
    minimum_training_rows = math.inf
    for season in VALIDATION_SEASONS:
        cutoff = season - 1
        train_all = train[train["season"] <= cutoff]
        validation_all = train[train["season"] == season]
        train_mask = spec.mask(train_all, cutoff, True)
        validation_mask = spec.mask(validation_all, cutoff, False)
        training = train_all.loc[train_mask]
        validation = validation_all.loc[validation_mask]
        target_all = validation_all[TARGET].to_numpy(dtype="float64")
        champion_all, source = current_champion_oof(season, target_all)
        target = validation[TARGET].to_numpy(dtype="float64")
        champion = champion_all[validation_mask]
        minimum_training_rows = min(minimum_training_rows, len(training))
        path = cache_path(output, spec, season)
        cached = None if force else cached_prediction(path, spec, season, target)
        if cached is None:
            print(
                f"[{spec.name}/{season}] train rows={len(training):,}; "
                f"validation rows={len(validation):,}",
                flush=True,
            )
            specialist, fit_seconds, predict_seconds, levels = fit_specialist(
                training, validation, feature_contract, threads
            )
            save_npz(
                path,
                cache_version=np.asarray(CACHE_VERSION),
                specialist=np.asarray(spec.name),
                season=np.asarray(season, dtype="int16"),
                target_sha256=np.asarray(array_sha256(target)),
                prediction=specialist,
                fit_seconds=np.asarray(fit_seconds),
                predict_seconds=np.asarray(predict_seconds),
                category_levels_json=np.asarray(json.dumps(levels, sort_keys=True)),
            )
        else:
            specialist, fit_seconds, predict_seconds = cached
            print(f"[{spec.name}/{season}] reuse {path}", flush=True)
        subset_gain = brier(target, champion) - brier(target, specialist)
        hard_route = routed_prediction(
            champion_all, specialist, validation_mask, weight=1.0
        )
        hard_route_overall_gain = brier(target_all, champion_all) - brier(
            target_all, hard_route
        )
        result_rows.append(
            {
                "specialist": spec.name,
                "stage": spec.stage,
                "partition": spec.partition,
                "subset": spec.subset,
                "validation_season": season,
                "train_n": len(training),
                "validation_n": len(validation),
                "minimum_train_gate": len(training) >= spec.minimum_train_rows,
                "low_sample_exception": spec.low_sample_exception,
                "champion_brier_subset": brier(target, champion),
                "specialist_brier_subset": brier(target, specialist),
                "subset_brier_gain": subset_gain,
                "overall_brier_gain_hard_route_diagnostic": hard_route_overall_gain,
                "champion_prediction_mean": float(np.mean(champion)),
                "specialist_prediction_mean": float(np.mean(specialist)),
                "specialist_prediction_std": float(np.std(specialist)),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
        )
        complementarity_rows.append(
            complementarity_record(spec, season, target, specialist, champion)
        )
        folds[season] = {
            "target_all": target_all,
            "champion_all": champion_all,
            "target_mask": validation_mask,
            "target": target,
            "champion": champion,
            "specialist": specialist,
            "validation": validation_all,
            "source": source,
        }

    pooled_target = np.concatenate([folds[s]["target"] for s in VALIDATION_SEASONS])
    pooled_champion = np.concatenate(
        [folds[s]["champion"] for s in VALIDATION_SEASONS]
    )
    pooled_specialist = np.concatenate(
        [folds[s]["specialist"] for s in VALIDATION_SEASONS]
    )
    pooled_gain = brier(pooled_target, pooled_champion) - brier(
        pooled_target, pooled_specialist
    )
    pooled_all_target = np.concatenate(
        [folds[s]["target_all"] for s in VALIDATION_SEASONS]
    )
    pooled_all_champion = np.concatenate(
        [folds[s]["champion_all"] for s in VALIDATION_SEASONS]
    )
    pooled_hard_route = np.concatenate(
        [
            routed_prediction(
                folds[s]["champion_all"],
                folds[s]["specialist"],
                folds[s]["target_mask"],
                weight=1.0,
            )
            for s in VALIDATION_SEASONS
        ]
    )
    pooled_hard_route_gain = brier(
        pooled_all_target, pooled_all_champion
    ) - brier(pooled_all_target, pooled_hard_route)
    result_rows.append(
        {
            "specialist": spec.name,
            "stage": spec.stage,
            "partition": spec.partition,
            "subset": spec.subset,
            "validation_season": "pooled",
            "train_n": np.nan,
            "validation_n": len(pooled_target),
            "minimum_train_gate": minimum_training_rows >= spec.minimum_train_rows,
            "low_sample_exception": spec.low_sample_exception,
            "champion_brier_subset": brier(pooled_target, pooled_champion),
            "specialist_brier_subset": brier(pooled_target, pooled_specialist),
            "subset_brier_gain": pooled_gain,
            "overall_brier_gain_hard_route_diagnostic": pooled_hard_route_gain,
            "champion_prediction_mean": float(np.mean(pooled_champion)),
            "specialist_prediction_mean": float(np.mean(pooled_specialist)),
            "specialist_prediction_std": float(np.std(pooled_specialist)),
            "fit_seconds": sum(row["fit_seconds"] for row in result_rows),
            "predict_seconds": sum(row["predict_seconds"] for row in result_rows),
        }
    )
    complementarity_rows.append(
        complementarity_record(
            spec,
            "pooled",
            pooled_target,
            pooled_specialist,
            pooled_champion,
        )
    )
    gains = {
        str(row["validation_season"]): float(row["subset_brier_gain"])
        for row in result_rows
    }
    gate = raw_gate(gains, minimum_training_rows >= spec.minimum_train_rows)
    decision = {
        "specialist": spec.name,
        "stage": spec.stage,
        "partition": spec.partition,
        "subset": spec.subset,
        "minimum_training_rows_observed": int(minimum_training_rows),
        "minimum_training_rows_required": spec.minimum_train_rows,
        "low_sample_exception": spec.low_sample_exception,
        "raw_gains": gains,
        "raw_hard_route_overall_gains": {
            str(row["validation_season"]): float(
                row["overall_brier_gain_hard_route_diagnostic"]
            )
            for row in result_rows
        },
        "raw_gate": gate,
    }

    blend_rows = []
    subset_rows = []
    if gate["passed"]:
        for weight in BLEND_WEIGHTS:
            for season in VALIDATION_SEASONS:
                fold = folds[season]
                candidate = routed_prediction(
                    fold["champion_all"],
                    fold["specialist"],
                    fold["target_mask"],
                    weight,
                )
                non_target = ~fold["target_mask"]
                non_target_max_diff = float(
                    np.max(np.abs(candidate[non_target] - fold["champion_all"][non_target]))
                ) if np.any(non_target) else 0.0
                blend_rows.append(
                    {
                        "specialist": spec.name,
                        "weight": weight,
                        "validation_season": season,
                        "target_n": int(fold["target_mask"].sum()),
                        "subset_brier_gain": brier(fold["target"], fold["champion"])
                        - brier(fold["target"], candidate[fold["target_mask"]]),
                        "overall_brier_gain": brier(
                            fold["target_all"], fold["champion_all"]
                        )
                        - brier(fold["target_all"], candidate),
                        "non_target_max_abs_diff": non_target_max_diff,
                    }
                )
                for (axis, subset), mask in major_subset_masks(
                    fold["validation"], fold["source"]
                ).items():
                    subset_rows.append(
                        {
                            "specialist": spec.name,
                            "weight": weight,
                            "validation_season": season,
                            "axis": axis,
                            "subset": subset,
                            "n": int(mask.sum()),
                            "brier_gain": brier(
                                fold["target_all"][mask], fold["champion_all"][mask]
                            )
                            - brier(fold["target_all"][mask], candidate[mask]),
                        }
                    )
            season_rows = [
                row
                for row in blend_rows
                if row["specialist"] == spec.name
                and math.isclose(row["weight"], weight)
                and row["validation_season"] != "pooled"
            ]
            pooled_all_target = np.concatenate(
                [folds[s]["target_all"] for s in VALIDATION_SEASONS]
            )
            pooled_all_champion = np.concatenate(
                [folds[s]["champion_all"] for s in VALIDATION_SEASONS]
            )
            pooled_all_candidate = np.concatenate(
                [
                    routed_prediction(
                        folds[s]["champion_all"],
                        folds[s]["specialist"],
                        folds[s]["target_mask"],
                        weight,
                    )
                    for s in VALIDATION_SEASONS
                ]
            )
            pooled_subset_candidate = np.concatenate(
                [
                    routed_prediction(
                        folds[s]["champion_all"],
                        folds[s]["specialist"],
                        folds[s]["target_mask"],
                        weight,
                    )[folds[s]["target_mask"]]
                    for s in VALIDATION_SEASONS
                ]
            )
            blend_rows.append(
                {
                    "specialist": spec.name,
                    "weight": weight,
                    "validation_season": "pooled",
                    "target_n": len(pooled_target),
                    "subset_brier_gain": brier(pooled_target, pooled_champion)
                    - brier(pooled_target, pooled_subset_candidate),
                    "overall_brier_gain": brier(
                        pooled_all_target, pooled_all_champion
                    )
                    - brier(pooled_all_target, pooled_all_candidate),
                    "non_target_max_abs_diff": max(
                        row["non_target_max_abs_diff"] for row in season_rows
                    ),
                }
            )
            for axis_subset in sorted(
                {(row["axis"], row["subset"]) for row in subset_rows}
            ):
                parts = []
                for season in VALIDATION_SEASONS:
                    fold = folds[season]
                    mask = major_subset_masks(fold["validation"], fold["source"])[
                        axis_subset
                    ]
                    candidate = routed_prediction(
                        fold["champion_all"],
                        fold["specialist"],
                        fold["target_mask"],
                        weight,
                    )
                    parts.append(
                        (
                            fold["target_all"][mask],
                            fold["champion_all"][mask],
                            candidate[mask],
                        )
                    )
                target_part = np.concatenate([part[0] for part in parts])
                champion_part = np.concatenate([part[1] for part in parts])
                candidate_part = np.concatenate([part[2] for part in parts])
                subset_rows.append(
                    {
                        "specialist": spec.name,
                        "weight": weight,
                        "validation_season": "pooled",
                        "axis": axis_subset[0],
                        "subset": axis_subset[1],
                        "n": len(target_part),
                        "brier_gain": brier(target_part, champion_part)
                        - brier(target_part, candidate_part),
                    }
                )

    return decision, result_rows, complementarity_rows, blend_rows, subset_rows


def blend_decisions(
    specialist: str,
    blends: pd.DataFrame,
    subsets: pd.DataFrame,
) -> list[dict]:
    decisions = []
    if blends.empty:
        return decisions
    for weight in BLEND_WEIGHTS:
        current = blends[
            blends["specialist"].eq(specialist)
            & np.isclose(blends["weight"], weight)
        ]
        gains = {
            str(row.validation_season): float(row.overall_brier_gain)
            for row in current.itertuples(index=False)
        }
        subset_current = subsets[
            subsets["specialist"].eq(specialist)
            & np.isclose(subsets["weight"], weight)
            & subsets["validation_season"].eq("pooled")
            & (subsets["n"] >= MIN_DIAGNOSTIC_ROWS)
        ]
        worst_subset = (
            float(subset_current["brier_gain"].min())
            if not subset_current.empty
            else float("nan")
        )
        non_target_exact = bool((current["non_target_max_abs_diff"] == 0.0).all())
        conditions = {
            "2023_overall_positive": gains["2023"] > 0.0,
            "2024_overall_positive": gains["2024"] > 0.0,
            "pooled_overall_positive": gains["pooled"] > 0.0,
            "non_target_exact": non_target_exact,
            "major_subsets_safe": math.isnan(worst_subset)
            or worst_subset >= -MAX_MAJOR_SUBSET_LOSS,
        }
        decisions.append(
            {
                "specialist": specialist,
                "weight": weight,
                "overall_gains": gains,
                "worst_major_subset_gain": worst_subset,
                "conditions": conditions,
                "accepted": all(conditions.values()),
                "large": all(conditions.values())
                and gains["pooled"] >= LARGE_OVERALL_GAIN,
            }
        )
    return decisions


def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    checksum_before = sha256_path(CHAMPION)
    if checksum_before != CHAMPION_SHA256:
        raise ValueError("immutable champion checksum mismatch before experiment")
    feature_contract, _ = load_champion_contract()
    train = pd.read_csv(args.train, encoding="utf-8-sig")
    required = {
        "row_id",
        "season",
        TARGET,
        *ID_COLUMNS,
        "balls_before",
        "strikes_before",
    }
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"train is missing columns: {missing}")
    print(
        f"[contract] HGB52 ID-free features; mono63 LightGBM; "
        f"train rows={len(train):,}; test data not read",
        flush=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    decisions = []
    result_rows = []
    complementarity_rows = []
    blend_rows = []
    subset_rows = []
    blend_gate_rows = []
    stopped_after_stage = None
    stop_reason = None
    for stage_specs in STAGES:
        stage_passed = False
        for spec in stage_specs:
            decision, results, complementarity, blends, subsets = evaluate_specialist(
                spec, train, feature_contract, args.output, args.threads, args.force
            )
            decisions.append(decision)
            result_rows.extend(results)
            complementarity_rows.extend(complementarity)
            blend_rows.extend(blends)
            subset_rows.extend(subsets)
            blend_frame = pd.DataFrame(blend_rows)
            subset_frame = pd.DataFrame(subset_rows)
            current_blend_decisions = blend_decisions(
                spec.name, blend_frame, subset_frame
            )
            decision["blend_decisions"] = current_blend_decisions
            blend_gate_rows.extend(current_blend_decisions)
            stage_passed = stage_passed or bool(decision["raw_gate"]["passed"])
        if not stage_passed:
            stopped_after_stage = stage_specs[0].stage
            stop_reason = (
                f"stage {stopped_after_stage} raw specialist gate failed; "
                "later priority candidates were not evaluated"
            )
            break

    specialist_frame = pd.DataFrame(result_rows)
    complementarity_frame = pd.DataFrame(complementarity_rows)
    blend_frame = (
        pd.DataFrame(blend_rows)
        if blend_rows
        else empty_frame(
            [
                "specialist",
                "weight",
                "validation_season",
                "target_n",
                "subset_brier_gain",
                "overall_brier_gain",
                "non_target_max_abs_diff",
            ]
        )
    )
    subset_frame = (
        pd.DataFrame(subset_rows)
        if subset_rows
        else empty_frame(
            [
                "specialist",
                "weight",
                "validation_season",
                "axis",
                "subset",
                "n",
                "brier_gain",
            ]
        )
    )
    specialist_frame.to_csv(args.output / "specialist_results.csv", index=False)
    subset_frame.to_csv(args.output / "subset_results.csv", index=False)
    blend_frame.to_csv(args.output / "blend_results.csv", index=False)
    complementarity_frame.to_csv(args.output / "complementarity.csv", index=False)

    large = [row for row in blend_gate_rows if row["large"]]
    accepted = [row for row in blend_gate_rows if row["accepted"]]
    stable_raw = [row for row in decisions if row["raw_gate"]["passed"]]
    if large:
        verdict = "A. LARGE SPECIALIST SIGNAL FOUND"
    elif stable_raw or accepted:
        verdict = "B. stable but incremental only"
    else:
        verdict = "C. no specialist signal"

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
        "experiment_contract": {
            "feature_contract": "exact champion HGB/CatBoost 52-column ID-free row-local contract",
            "learner": "source-exact mono63 LightGBM parameters",
            "rounds": N_ROUNDS,
            "seed": 42,
            "blend_weights": list(BLEND_WEIGHTS),
            "minimum_train_rows": MIN_TRAIN_ROWS,
            "large_overall_gain": LARGE_OVERALL_GAIN,
            "test_data_used": False,
            "calibration_added": False,
        },
        "priority_order": [
            [spec.name for spec in stage_specs] for stage_specs in STAGES
        ],
        "evaluated_specialists": [row["specialist"] for row in decisions],
        "stopped_after_stage": stopped_after_stage,
        "stop_reason": stop_reason,
        "specialist_decisions": decisions,
        "accepted_blends": accepted,
        "large_blends": large,
        "multiple_specialists_combined": False,
        "production": {
            "created": False,
            "reason": (
                "Stage A only; production requires a separate full-train parity gate"
                if verdict == "A. LARGE SPECIALIST SIGNAL FOUND"
                else "verdict is not A"
            ),
            "filename_length_hard_gate": 30,
        },
    }
    write_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "evaluated_specialists": summary["evaluated_specialists"],
                "stopped_after_stage": stopped_after_stage,
                "stable_raw": [row["specialist"] for row in stable_raw],
                "large_blends": [
                    (row["specialist"], row["weight"]) for row in large
                ],
                "champion_unchanged": checksum_before == checksum_after,
                "production_created": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
