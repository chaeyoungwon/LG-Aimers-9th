"""Decompose the frozen TrackMan crosswalk signal into minimal feature experts.

The accepted Hungarian HIGH mapping, cutoffs, feature construction, ET25 recipe,
and temporal folds are imported from ``validate_tm_crosswalk`` and never tuned.
No test rows or external identity information are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extratrees_runtime import (
    build_hgb52_features,
    fit_preprocessor,
    transform_features,
)
from scripts.validate_tm_crosswalk import (
    BLEND_WEIGHTS as PRIOR_BLEND_WEIGHTS,
    CHAMPION,
    CHAMPION_SHA256,
    ET25_CACHE_DIR,
    MODEL_RECIPE,
    OOF_CACHE_DIR,
    attach_physical_features,
    brier,
    build_main_physical_lookup,
    build_physical_profiles,
    error_top10_win_rate,
    frame_signature,
    load_champion_meta,
    load_current_champion_oof,
    read_trackman_physical,
    safe_corr,
    sha256_path,
)


DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_TRACKMAN = Path("/Users/wooh/Documents/dev/open/data/trackman_history.csv")
DEFAULT_OUTPUT = ROOT / "artifacts" / "tm_signal_ablation"
CROSSWALK_DIR = ROOT / "artifacts" / "tm_crosswalk"
MAPPING_REFERENCE = CROSSWALK_DIR / "mapping_candidates.csv"
CROSSWALK_SUMMARY_REFERENCE = CROSSWALK_DIR / "summary.json"
BASELINE_BLEND_REFERENCE = CROSSWALK_DIR / "blend_results.csv"
BASELINE_OOF_REFERENCE = CROSSWALK_DIR / "oof_results.csv"
REFERENCE_HASHES = {
    "mapping_candidates.csv": "7506a294ef4443d048d204313d9169e17c1871d279271d81b353487716b31ae2",
    "summary.json": "f76452b72f1f75fc7ce98a8e4248f38c573d388c8d6ef01d292ddb9972a43185",
    "blend_results.csv": "c6e3b466e5f5fa2b939d983d2a5de23dde8c2b98d38523b042e9b11e84527dc8",
    "oof_results.csv": "127106c70bc9e0e50fe7d84a29dc69ea3327a3d0c6ca7eba922c701a5015ef25",
}
BASELINE_LEARNER = "accepted52_plus_tm"
BASELINE_WEIGHT = 0.10
BASELINE_GAINS = {
    "2022": 1.2124564658211323e-05,
    "2023": 2.639287196165574e-05,
    "2024": 2.1054167049033845e-05,
    "pooled": 1.9849831855711653e-05,
}
BASELINE_PARITY_TOLERANCE = 1e-15
FAMILY_BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20)
ADAPTIVE_WEIGHT_MIN = 0.05
ADAPTIVE_WEIGHT_MAX = 0.15
SUBSET_FLOOR = -1e-5
MODEL_CACHE_VERSION = "tm-signal-family-et25-v1"
VALIDATION_SEASONS = (2022, 2023, 2024)

FAMILY_LABELS = {
    "usage_history": "A",
    "velocity": "B",
    "movement": "C",
    "release_mechanics": "D",
    "spin": "E",
    "pitch_separation": "F",
    "all_physical": "G",
    "all_trackman": "H",
}

PASSED_FEATURE_DEFINITIONS = {
    "count_two_strike_n": {
        "definition": "TrackMan <= cutoff rows in the mutually exclusive two_strike bucket (balls < 3, strikes = 2)",
        "physical_mechanics": False,
        "usage_history": True,
        "career_recent": "career/cutoff history",
    },
    "fastball_n": {
        "definition": "TrackMan <= cutoff fastball pitch count",
        "physical_mechanics": False,
        "usage_history": True,
        "career_recent": "career/cutoff history",
    },
    "tm_history_n": {
        "definition": "all TrackMan pitcher rows with season <= cutoff",
        "physical_mechanics": False,
        "usage_history": True,
        "career_recent": "career/cutoff history",
    },
    "count_neutral_n": {
        "definition": "TrackMan <= cutoff rows outside three_ball and two_strike buckets",
        "physical_mechanics": False,
        "usage_history": True,
        "career_recent": "career/cutoff history",
    },
    "offspeed_n": {
        "definition": "TrackMan <= cutoff offspeed pitch count",
        "physical_mechanics": False,
        "usage_history": True,
        "career_recent": "career/cutoff history",
    },
    "fastball_offspeed_velocity_gap": {
        "definition": "career fastball mean rel_speed minus career offspeed mean rel_speed",
        "physical_mechanics": True,
        "usage_history": False,
        "career_recent": "career/cutoff history",
    },
    "count_three_ball_n": {
        "definition": "TrackMan <= cutoff rows with balls = 3 (priority bucket, including full count)",
        "physical_mechanics": False,
        "usage_history": True,
        "career_recent": "career/cutoff history",
    },
}


def write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def verify_reference_hashes() -> dict:
    paths = {
        "mapping_candidates.csv": MAPPING_REFERENCE,
        "summary.json": CROSSWALK_SUMMARY_REFERENCE,
        "blend_results.csv": BASELINE_BLEND_REFERENCE,
        "oof_results.csv": BASELINE_OOF_REFERENCE,
    }
    actual = {name: sha256_path(path) for name, path in paths.items()}
    if actual != REFERENCE_HASHES:
        raise RuntimeError(f"immutable TrackMan reference changed: {actual}")
    return actual


def physical_feature_columns(lookup: pd.DataFrame) -> list[str]:
    metadata = {
        "pitcher_id",
        "pitcher_trackman_id",
        "best_distance",
        "margin",
        "seasons_overlap",
    }
    return [column for column in lookup.columns if column not in metadata]


def build_family_contract(columns: Iterable[str]) -> dict[str, list[str]]:
    columns = list(columns)
    usage = [
        column
        for column in columns
        if column.endswith("_n")
        and column not in {"tm_matched"}
    ]
    separation = [
        column
        for column in columns
        if "velocity_gap" in column or "movement_separation" in column
    ]
    velocity = [
        column
        for column in columns
        if (
            "rel_speed" in column
            or "zone_speed" in column
            or column == "velocity_variability"
        )
        and column not in separation
    ]
    movement = [
        column
        for column in columns
        if (
            "induced_vert_break" in column
            or "horz_break" in column
            or column == "movement_variability"
        )
        and column not in separation
    ]
    release = [
        column
        for column in columns
        if (
            "extension" in column
            or "rel_height" in column
            or "rel_side" in column
            or column == "release_spread"
        )
    ]
    spin = [column for column in columns if "spin_rate" in column]
    physical = sorted(set(velocity + movement + release + spin + separation))
    all_trackman = list(columns)
    contracts = {
        "usage_history": usage,
        "velocity": velocity,
        "movement": movement,
        "release_mechanics": release,
        "spin": spin,
        "pitch_separation": separation,
        "all_physical": physical,
        "all_trackman": all_trackman,
        "usage_plus_fo_gap": sorted(
            set(usage + ["fastball_offspeed_velocity_gap"])
        ),
        "history_count_only": ["tm_history_n"],
        "recent_count_only": ["tm_recent_n"],
        "pitch_type_count_only": ["fastball_n", "breaking_n", "offspeed_n"],
        "count_bucket_count_only": [
            "count_three_ball_n",
            "count_two_strike_n",
            "count_neutral_n",
        ],
    }
    for name, selected in contracts.items():
        missing = sorted(set(selected) - set(columns))
        if missing:
            raise ValueError(f"{name} references absent features: {missing}")
    return contracts


def load_baseline_fold(season: int, validation: pd.DataFrame) -> dict:
    cache = np.load(
        CROSSWALK_DIR / "cache" / f"{BASELINE_LEARNER}_{season}.npz",
        allow_pickle=False,
    )
    matched = cache["matched"].astype(bool)
    prediction = cache["prediction"].astype("float64")
    target = validation["control_success"].to_numpy(dtype="float64")
    champion, et25 = load_current_champion_oof(season, target)
    if len(matched) != len(validation) or len(prediction) != matched.sum():
        raise ValueError(f"baseline cache shape mismatch for {season}")
    return {
        "target": target,
        "champion": champion,
        "et25": et25,
        "matched": matched,
        "prediction": prediction,
    }


def apply_matched_blend(fold: Mapping, weight: float | np.ndarray) -> np.ndarray:
    candidate = np.asarray(fold["champion"], dtype="float64").copy()
    matched = np.asarray(fold["matched"], dtype=bool)
    if np.isscalar(weight):
        current_weight = float(weight)
    else:
        current_weight = np.asarray(weight, dtype="float64")[matched]
    candidate[matched] = (
        (1.0 - current_weight) * np.asarray(fold["champion"])[matched]
        + current_weight * np.asarray(fold["prediction"])
    )
    return candidate


def reconstruct_baseline(folds: Mapping[int, Mapping]) -> list[dict]:
    reference = pd.read_csv(BASELINE_BLEND_REFERENCE)
    reference = reference[
        reference["learner"].eq(BASELINE_LEARNER)
        & np.isclose(reference["weight"], BASELINE_WEIGHT)
    ]
    rows = []
    for season in (*VALIDATION_SEASONS, "pooled"):
        if season == "pooled":
            target = np.concatenate([folds[s]["target"] for s in VALIDATION_SEASONS])
            champion = np.concatenate(
                [folds[s]["champion"] for s in VALIDATION_SEASONS]
            )
            candidate = np.concatenate(
                [apply_matched_blend(folds[s], BASELINE_WEIGHT) for s in VALIDATION_SEASONS]
            )
        else:
            fold = folds[int(season)]
            target = fold["target"]
            champion = fold["champion"]
            candidate = apply_matched_blend(fold, BASELINE_WEIGHT)
        gain = brier(target, champion) - brier(target, candidate)
        matched = reference[
            reference["validation_season"].astype(str).eq(str(season))
        ]
        if len(matched) != 1:
            raise ValueError(f"missing baseline reference {season}")
        expected = float(matched.iloc[0]["brier_gain"])
        rows.append(
            {
                "validation_season": season,
                "reconstructed_gain": gain,
                "reference_gain": expected,
                "absolute_diff": abs(gain - expected),
                "passed": abs(gain - expected) <= BASELINE_PARITY_TOLERANCE,
            }
        )
    if not all(row["passed"] for row in rows):
        raise RuntimeError("frozen TrackMan baseline parity failed")
    return rows


def model_cache_path(output: Path, expert: str, season: int) -> Path:
    return output / "cache" / f"{expert}_{season}.npz"


def fit_family_expert(
    expert: str,
    selected_columns: list[str],
    training: pd.DataFrame,
    validation: pd.DataFrame,
    train_physical: pd.DataFrame,
    validation_physical: pd.DataFrame,
    champion_meta: Mapping,
    output: Path,
    force: bool,
    train_all_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if train_all_rows:
        train_mask = np.ones(len(training), dtype=bool)
        validation_mask = np.ones(len(validation), dtype=bool)
    else:
        train_mask = train_physical["tm_matched"].eq(1.0).to_numpy()
        validation_mask = validation_physical["tm_matched"].eq(1.0).to_numpy()
    cache_path = model_cache_path(output, expert, int(validation["season"].iloc[0]))
    validation_signature = frame_signature(validation, ["row_id"])
    feature_signature = hashlib.sha256(
        json.dumps(selected_columns, separators=(",", ":")).encode()
    ).hexdigest()
    if cache_path.is_file() and not force:
        payload = np.load(cache_path, allow_pickle=False)
        valid = (
            str(payload["cache_version"].item()) == MODEL_CACHE_VERSION
            and str(payload["feature_signature"].item()) == feature_signature
            and str(payload["validation_signature"].item()) == validation_signature
            and bool(payload["train_all_rows"].item()) == train_all_rows
        )
        if valid:
            prediction = payload["prediction"].astype("float64")
            matched = payload["matched"].astype(bool)
            if len(matched) == len(validation) and len(prediction) == matched.sum():
                return (
                    prediction,
                    matched,
                    float(payload["fit_seconds"].item()),
                    float(payload["predict_seconds"].item()),
                )

    train_rows = training.loc[train_mask]
    validation_rows = validation.loc[validation_mask]
    train_base = build_hgb52_features(train_rows, champion_meta["cat_levels"])
    validation_base = build_hgb52_features(
        validation_rows, champion_meta["cat_levels"]
    )
    train_features = pd.concat(
        [
            train_base.reset_index(drop=True),
            train_physical.loc[train_mask, selected_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    validation_features = pd.concat(
        [
            validation_base.reset_index(drop=True),
            validation_physical.loc[
                validation_mask, selected_columns
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
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_version=np.asarray(MODEL_CACHE_VERSION),
        feature_signature=np.asarray(feature_signature),
        validation_signature=np.asarray(validation_signature),
        train_all_rows=np.asarray(train_all_rows),
        prediction=prediction,
        matched=validation_mask,
        fit_seconds=np.asarray(fit_seconds),
        predict_seconds=np.asarray(predict_seconds),
    )
    return prediction, validation_mask, fit_seconds, predict_seconds


def fold_result(expert: str, season: int | str, fold: Mapping) -> dict:
    matched = np.asarray(fold["matched"], dtype=bool)
    target = np.asarray(fold["target"])
    champion = np.asarray(fold["champion"])
    prediction = np.asarray(fold["prediction"])
    routed = champion.copy()
    routed[matched] = prediction
    return {
        "expert": expert,
        "family": FAMILY_LABELS.get(expert, "candidate_or_usage"),
        "validation_season": season,
        "n": len(target),
        "matched_n": int(matched.sum()),
        "standalone_brier_matched": brier(target[matched], prediction),
        "champion_brier_matched": brier(target[matched], champion[matched]),
        "hard_routed_overall_gain": brier(target, champion) - brier(target, routed),
        "corr_champion": safe_corr(prediction, champion[matched]),
        "corr_et25": safe_corr(prediction, np.asarray(fold["et25"])[matched]),
        "champion_top10_error_model_win_rate": error_top10_win_rate(
            target[matched], champion[matched], prediction
        ),
        "fit_seconds": float(fold.get("fit_seconds", 0.0)),
        "predict_seconds": float(fold.get("predict_seconds", 0.0)),
    }


def pooled_fold(folds: Mapping[int, Mapping]) -> dict:
    return {
        key: np.concatenate([np.asarray(folds[s][key]) for s in VALIDATION_SEASONS])
        for key in ("target", "champion", "et25", "matched")
    } | {
        "prediction": np.concatenate(
            [np.asarray(folds[s]["prediction"]) for s in VALIDATION_SEASONS]
        ),
        "fit_seconds": sum(float(folds[s].get("fit_seconds", 0.0)) for s in VALIDATION_SEASONS),
        "predict_seconds": sum(float(folds[s].get("predict_seconds", 0.0)) for s in VALIDATION_SEASONS),
    }


def subset_masks(validation: pd.DataFrame, oof: Mapping[str, np.ndarray]) -> list[tuple[str, str, np.ndarray]]:
    form = oof["pitcher_form_pred"].astype("float64") - oof["champion_base"].astype("float64")
    return [
        ("game_type", "R", validation["game_type"].astype(str).eq("R").to_numpy()),
        ("game_type", "F", validation["game_type"].astype(str).eq("F").to_numpy()),
        ("pitcher_seen", "seen", oof["x_pitcher_seen"].astype(bool)),
        ("pitcher_seen", "unseen", ~oof["x_pitcher_seen"].astype(bool)),
        ("full_count", "full", (validation["balls_before"].eq(3) & validation["strikes_before"].eq(2)).to_numpy()),
        ("full_count", "non_full", ~(validation["balls_before"].eq(3) & validation["strikes_before"].eq(2)).to_numpy()),
        ("pitcher_form", "positive", form >= 0.0),
        ("pitcher_form", "negative", form < 0.0),
    ]


def candidate_safety(
    expert_folds: Mapping[int, Mapping], candidates: Mapping[int, np.ndarray],
    validations: Mapping[int, pd.DataFrame], confidence_bins: Mapping[int, np.ndarray]
) -> tuple[float, float, list[dict]]:
    rows = []
    for season in VALIDATION_SEASONS:
        fold = expert_folds[season]
        target = fold["target"]
        champion = fold["champion"]
        oof = np.load(OOF_CACHE_DIR / f"season_{season}.npz", allow_pickle=False)
        for axis, subset, mask in subset_masks(validations[season], oof):
            if mask.sum() < 100:
                continue
            rows.append(
                {
                    "validation_season": season,
                    "axis": axis,
                    "subset": subset,
                    "n": int(mask.sum()),
                    "gain": brier(target[mask], champion[mask]) - brier(target[mask], candidates[season][mask]),
                }
            )
        for label in ("high-A", "high-B", "high-C"):
            mask = confidence_bins[season] == label
            if mask.sum() < 100:
                continue
            rows.append(
                {
                    "validation_season": season,
                    "axis": "mapping_confidence",
                    "subset": label,
                    "n": int(mask.sum()),
                    "gain": brier(target[mask], champion[mask]) - brier(target[mask], candidates[season][mask]),
                }
            )
    major = [row["gain"] for row in rows if row["axis"] != "mapping_confidence"]
    confidence = [row["gain"] for row in rows if row["axis"] == "mapping_confidence"]
    return min(major), min(confidence), rows


def percentile_weight(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    finite_reference = np.sort(reference[np.isfinite(reference)])
    if len(finite_reference) < 2:
        return np.full(len(values), 0.10)
    ranks = np.searchsorted(finite_reference, values, side="right") - 1
    percentile = np.clip(ranks / (len(finite_reference) - 1), 0.0, 1.0)
    return ADAPTIVE_WEIGHT_MIN + (ADAPTIVE_WEIGHT_MAX - ADAPTIVE_WEIGHT_MIN) * percentile


def confidence_bins(values: np.ndarray, reference: np.ndarray, matched: np.ndarray) -> np.ndarray:
    finite_reference = reference[np.isfinite(reference)]
    q1, q2 = np.quantile(finite_reference, [1 / 3, 2 / 3])
    output = np.full(len(values), "unmatched", dtype=object)
    output[matched & (values <= q1)] = "high-C"
    output[matched & (values > q1) & (values <= q2)] = "high-B"
    output[matched & (values > q2)] = "high-A"
    return output


def _feature_max_diff(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """Compare aligned mixed-type feature frames, treating paired NaNs as equal."""
    if list(left.columns) != list(right.columns) or not left.index.equals(right.index):
        return math.inf
    maximum = 0.0
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]):
            left_values = left[column].to_numpy(dtype="float64")
            right_values = right[column].to_numpy(dtype="float64")
            paired_nan = np.isnan(left_values) & np.isnan(right_values)
            differences = np.abs(left_values - right_values)
            differences[paired_nan] = 0.0
            maximum = max(maximum, float(np.nanmax(differences, initial=0.0)))
        else:
            equal = left[column].astype("string").fillna("<NA>").eq(
                right[column].astype("string").fillna("<NA>")
            )
            if not bool(equal.all()):
                return math.inf
    return maximum


def independence_audit(
    validation: pd.DataFrame,
    lookup: pd.DataFrame,
    selected_columns: list[str],
    champion_meta: Mapping,
) -> dict:
    """Audit row-local feature parity on real validation rows without test data."""
    positions = np.unique(
        np.linspace(0, len(validation) - 1, num=min(257, len(validation)), dtype=int)
    )
    sample = validation.iloc[positions].copy()

    def matrix(rows: pd.DataFrame) -> pd.DataFrame:
        base = build_hgb52_features(rows, champion_meta["cat_levels"])
        physical = attach_physical_features(rows, lookup)[selected_columns]
        return pd.concat([base, physical], axis=1)

    full = matrix(sample)
    single = pd.concat([matrix(sample.iloc[[i]]) for i in range(len(sample))]).loc[
        sample.index
    ]
    shuffled_rows = sample.sample(frac=1.0, random_state=42)
    shuffled = matrix(shuffled_rows).loc[sample.index]
    reversed_matrix = matrix(sample.iloc[::-1]).loc[sample.index]
    subset_rows = sample.iloc[::2]
    subset = matrix(subset_rows)
    flipped = sample.copy()
    flipped["control_success"] = 1 - flipped["control_success"]
    target_flipped = matrix(flipped)
    return {
        "n": len(sample),
        "single_max_abs_diff": _feature_max_diff(full, single),
        "full_max_abs_diff": 0.0,
        "shuffle_max_abs_diff": _feature_max_diff(full, shuffled),
        "subset_max_abs_diff": _feature_max_diff(full.loc[subset.index], subset),
        "reverse_max_abs_diff": _feature_max_diff(full, reversed_matrix),
        "validation_target_flip_max_abs_diff": _feature_max_diff(full, target_flipped),
    }


def build_feature_inventory(
    feature_results: pd.DataFrame,
    validation_physical: Mapping[int, pd.DataFrame],
    validations: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    pivot = feature_results[
        feature_results["feature"].isin(PASSED_FEATURE_DEFINITIONS)
    ].pivot_table(index="feature", columns="validation_season", values="residual_correlation")
    rows = []
    for feature, definition in PASSED_FEATURE_DEFINITIONS.items():
        missing = 0
        total = 0
        finite_pitchers = 0
        matched_pitchers = 0
        for season in VALIDATION_SEASONS:
            physical = validation_physical[season]
            validation = validations[season]
            matched = physical["tm_matched"].eq(1.0).to_numpy()
            values = physical[feature].to_numpy(dtype="float64")
            missing += int((matched & ~np.isfinite(values)).sum())
            total += int(matched.sum())
            pitcher_frame = pd.DataFrame(
                {"pitcher_id": validation["pitcher_id"].to_numpy(), "matched": matched, "finite": np.isfinite(values)}
            )
            current = pitcher_frame[pitcher_frame["matched"]].groupby("pitcher_id")["finite"].any()
            matched_pitchers += len(current)
            finite_pitchers += int(current.sum())
        rows.append(
            {
                "feature_name": feature,
                **definition,
                "missing_rows": missing,
                "matched_rows": total,
                "missingness": missing / total,
                "finite_matched_pitchers": finite_pitchers,
                "matched_pitchers": matched_pitchers,
                "matched_pitcher_coverage": finite_pitchers / matched_pitchers,
                "residual_association_2022": float(pivot.loc[feature, "2022"]),
                "residual_association_2023": float(pivot.loc[feature, "2023"]),
                "residual_association_2024": float(pivot.loc[feature, "2024"]),
            }
        )
    return pd.DataFrame(rows)


def coverage_diagnostics(
    validations: Mapping[int, pd.DataFrame],
    physical: Mapping[int, pd.DataFrame],
    baseline: Mapping[int, Mapping],
    indicator_folds: Mapping[int, Mapping],
) -> pd.DataFrame:
    rows = []
    for season in VALIDATION_SEASONS:
        validation = validations[season]
        fold = baseline[season]
        matched = physical[season]["tm_matched"].eq(1.0).to_numpy()
        for label, mask in (("matched_HIGH", matched), ("unmatched", ~matched)):
            rows.append(
                {
                    "record_type": "coverage",
                    "validation_season": season,
                    "subset": label,
                    "n": int(mask.sum()),
                    "row_coverage": float(mask.mean()),
                    "target_rate": float(fold["target"][mask].mean()),
                    "champion_brier": brier(fold["target"][mask], fold["champion"][mask]),
                    "asof_pitcher_n_mean": float(validation.loc[mask, "asof_pitcher_n"].mean()),
                    "asof_pitcher_pitchmix_n_mean": float(validation.loc[mask, "asof_pitcher_pitchmix_n"].mean()),
                    "indicator_hard_routed_gain": np.nan,
                }
            )
        indicator = indicator_folds[season]
        routed = indicator["prediction"]
        rows.append(
            {
                "record_type": "tm_matched_indicator_model",
                "validation_season": season,
                "subset": "all",
                "n": len(validation),
                "row_coverage": 1.0,
                "target_rate": float(fold["target"].mean()),
                "champion_brier": brier(fold["target"], fold["champion"]),
                "asof_pitcher_n_mean": float(validation["asof_pitcher_n"].mean()),
                "asof_pitcher_pitchmix_n_mean": float(validation["asof_pitcher_pitchmix_n"].mean()),
                "indicator_hard_routed_gain": brier(fold["target"], fold["champion"]) - brier(fold["target"], routed),
            }
        )
    return pd.DataFrame(rows)


def usage_associations(
    training_frames: Mapping[int, pd.DataFrame],
    validation_frames: Mapping[int, pd.DataFrame],
    train_physical: Mapping[int, pd.DataFrame],
    validation_physical: Mapping[int, pd.DataFrame],
    baseline: Mapping[int, Mapping],
    usage_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for season in VALIDATION_SEASONS:
        training = training_frames[season]
        validation = validation_frames[season]
        train_p = train_physical[season]
        val_p = validation_physical[season]
        train_mask = train_p["tm_matched"].eq(1.0).to_numpy()
        val_mask = val_p["tm_matched"].eq(1.0).to_numpy()
        train_controls = np.column_stack(
            [
                np.ones(train_mask.sum()),
                np.log1p(training.loc[train_mask, "asof_pitcher_n"].to_numpy(dtype="float64")),
                np.log1p(training.loc[train_mask, "asof_pitcher_pitchmix_n"].to_numpy(dtype="float64")),
            ]
        )
        validation_controls = np.column_stack(
            [
                np.ones(val_mask.sum()),
                np.log1p(validation.loc[val_mask, "asof_pitcher_n"].to_numpy(dtype="float64")),
                np.log1p(validation.loc[val_mask, "asof_pitcher_pitchmix_n"].to_numpy(dtype="float64")),
            ]
        )
        residual = baseline[season]["target"][val_mask] - baseline[season]["champion"][val_mask]
        for feature in usage_columns:
            train_values = np.log1p(train_p.loc[train_mask, feature].to_numpy(dtype="float64"))
            validation_values = np.log1p(val_p.loc[val_mask, feature].to_numpy(dtype="float64"))
            train_finite = np.isfinite(train_values) & np.isfinite(train_controls).all(axis=1)
            coefficients = np.linalg.lstsq(train_controls[train_finite], train_values[train_finite], rcond=None)[0]
            controlled = validation_values - validation_controls @ coefficients
            main_n = validation_controls[:, 1]
            mix_n = validation_controls[:, 2]
            rows.append(
                {
                    "record_type": "association",
                    "name": feature,
                    "validation_season": season,
                    "n": int(np.isfinite(validation_values).sum()),
                    "corr_log_asof_pitcher_n": safe_corr(validation_values, main_n),
                    "corr_log_asof_pitcher_pitchmix_n": safe_corr(validation_values, mix_n),
                    "raw_residual_association": safe_corr(validation_values, residual),
                    "controlled_residual_association": safe_corr(controlled, residual),
                }
            )
    return pd.DataFrame(rows)


def advantage_quantiles(
    trainings: Mapping[int, pd.DataFrame],
    validations: Mapping[int, pd.DataFrame],
    train_physical: Mapping[int, pd.DataFrame],
    validation_physical: Mapping[int, pd.DataFrame],
    baseline: Mapping[int, Mapping],
) -> pd.DataFrame:
    specifications = {
        "mapping_confidence": ("physical", "tm_confidence"),
        "history_n": ("physical", "tm_history_n"),
        "recent_history_n": ("physical", "tm_recent_n"),
        "pitchmix_history_n": ("main", "asof_pitcher_pitchmix_n"),
        "fastball_offspeed_velocity_gap": ("physical", "fastball_offspeed_velocity_gap"),
    }
    rows = []
    for season in VALIDATION_SEASONS:
        fold = baseline[season]
        matched = fold["matched"]
        advantage = (
            (fold["target"][matched] - fold["champion"][matched]) ** 2
            - (fold["target"][matched] - fold["prediction"]) ** 2
        )
        for name, (source, column) in specifications.items():
            if source == "physical":
                train_values = train_physical[season].loc[
                    train_physical[season]["tm_matched"].eq(1.0), column
                ].to_numpy(dtype="float64")
                values = validation_physical[season].loc[matched, column].to_numpy(dtype="float64")
            else:
                train_match = train_physical[season]["tm_matched"].eq(1.0).to_numpy()
                train_values = trainings[season].loc[train_match, column].to_numpy(dtype="float64")
                values = validations[season].loc[matched, column].to_numpy(dtype="float64")
            finite_train = train_values[np.isfinite(train_values)]
            edges = np.unique(np.quantile(finite_train, [0.25, 0.50, 0.75]))
            bins = np.digitize(values, edges, right=True)
            for index in range(len(edges) + 1):
                mask = (bins == index) & np.isfinite(values)
                if mask.sum() < 100:
                    continue
                rows.append(
                    {
                        "validation_season": season,
                        "feature": name,
                        "quantile": f"Q{index + 1}",
                        "n": int(mask.sum()),
                        "value_mean": float(values[mask].mean()),
                        "mean_advantage": float(advantage[mask].mean()),
                        "expert_win_rate": float((advantage[mask] > 0.0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--trackman", type=Path, default=DEFAULT_TRACKMAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference_hashes = verify_reference_hashes()
    champion_before = sha256_path(CHAMPION)
    if champion_before != CHAMPION_SHA256:
        raise RuntimeError("immutable champion mismatch")
    crosswalk_summary = json.loads(CROSSWALK_SUMMARY_REFERENCE.read_text())
    if not crosswalk_summary["mapping_gate_passed"] or crosswalk_summary["selected_method"] != "hungarian":
        raise RuntimeError("accepted Hungarian crosswalk reference is not frozen")
    mappings = pd.read_csv(MAPPING_REFERENCE)
    full_train = pd.read_csv(args.train)
    trackman = read_trackman_physical(args.trackman)
    champion_meta = load_champion_meta()

    trainings = {}
    validations = {}
    train_physical = {}
    validation_physical = {}
    lookups = {}
    baseline = {}
    family_contract = None
    for season in VALIDATION_SEASONS:
        cutoff = season - 1
        trainings[season] = full_train[full_train["season"].le(cutoff)]
        validations[season] = full_train[full_train["season"].eq(season)]
        profile = build_physical_profiles(trackman, cutoff)
        lookup = build_main_physical_lookup(mappings, profile, cutoff, "hungarian")
        lookups[season] = lookup
        train_physical[season] = attach_physical_features(trainings[season], lookup)
        validation_physical[season] = attach_physical_features(validations[season], lookup)
        baseline[season] = load_baseline_fold(season, validations[season])
        if not np.array_equal(
            baseline[season]["matched"],
            validation_physical[season]["tm_matched"].eq(1.0).to_numpy(),
        ):
            raise RuntimeError(f"baseline mapping parity failed for {season}")
        current_contract = build_family_contract(physical_feature_columns(lookup))
        if family_contract is None:
            family_contract = current_contract
        elif current_contract != family_contract:
            raise RuntimeError("TrackMan family contract changes by cutoff")
    assert family_contract is not None
    baseline_parity = reconstruct_baseline(baseline)
    print("[baseline parity] exact", flush=True)

    prior_features = pd.read_csv(CROSSWALK_DIR / "feature_results.csv")
    inventory = build_feature_inventory(prior_features, validation_physical, validations)
    inventory.to_csv(args.output / "feature_inventory.csv", index=False)

    experts_to_run = [
        "usage_history",
        "velocity",
        "movement",
        "release_mechanics",
        "spin",
        "pitch_separation",
        "all_physical",
        "usage_plus_fo_gap",
    ]
    expert_folds: dict[str, dict[int, dict]] = {"all_trackman": baseline}
    for expert in experts_to_run:
        expert_folds[expert] = {}
        for season in VALIDATION_SEASONS:
            prediction, matched, fit_seconds, predict_seconds = fit_family_expert(
                expert,
                family_contract[expert],
                trainings[season],
                validations[season],
                train_physical[season],
                validation_physical[season],
                champion_meta,
                args.output,
                args.force_models,
            )
            expert_folds[expert][season] = {
                **baseline[season],
                "prediction": prediction,
                "matched": matched,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
            print(f"[{expert} {season}] fit={fit_seconds:.2f}s", flush=True)

    # Usage-only passed in the prior diagnostic, so evaluate only actual count subsets.
    usage_experts = [
        "history_count_only",
        "recent_count_only",
        "pitch_type_count_only",
        "count_bucket_count_only",
    ]
    for expert in usage_experts:
        expert_folds[expert] = {}
        for season in VALIDATION_SEASONS:
            prediction, matched, fit_seconds, predict_seconds = fit_family_expert(
                expert,
                family_contract[expert],
                trainings[season],
                validations[season],
                train_physical[season],
                validation_physical[season],
                champion_meta,
                args.output,
                args.force_models,
            )
            expert_folds[expert][season] = {
                **baseline[season],
                "prediction": prediction,
                "matched": matched,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }

    # Coverage-only audit: raw52 + tm_matched, trained and predicted on every row.
    indicator_folds = {}
    for season in VALIDATION_SEASONS:
        prediction, mask, fit_seconds, predict_seconds = fit_family_expert(
            "tm_matched_indicator",
            ["tm_matched"],
            trainings[season],
            validations[season],
            train_physical[season],
            validation_physical[season],
            champion_meta,
            args.output,
            args.force_models,
            train_all_rows=True,
        )
        indicator_folds[season] = {
            **baseline[season],
            "prediction": prediction,
            "matched": mask,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }

    family_rows = []
    for expert in [*FAMILY_LABELS, "usage_plus_fo_gap"]:
        folds = expert_folds[expert]
        for season in VALIDATION_SEASONS:
            family_rows.append(fold_result(expert, season, folds[season]))
        family_rows.append(fold_result(expert, "pooled", pooled_fold(folds)))
    family_oof = pd.DataFrame(family_rows)
    family_oof.to_csv(args.output / "family_oof.csv", index=False)

    usage_model_rows = []
    for expert in ["usage_history", *usage_experts]:
        folds = expert_folds[expert]
        for season in VALIDATION_SEASONS:
            usage_model_rows.append({"record_type": "model", **fold_result(expert, season, folds[season])})
        usage_model_rows.append({"record_type": "model", **fold_result(expert, "pooled", pooled_fold(folds))})
    association_rows = usage_associations(
        trainings,
        validations,
        train_physical,
        validation_physical,
        baseline,
        family_contract["usage_history"],
    ).to_dict("records")
    usage_ablation = pd.DataFrame([*usage_model_rows, *association_rows])
    usage_ablation.to_csv(args.output / "usage_ablation.csv", index=False)

    coverage = coverage_diagnostics(
        validations, validation_physical, baseline, indicator_folds
    )
    coverage.to_csv(args.output / "coverage_diagnostics.csv", index=False)

    confidence_rows = []
    confidence_bin_masks = {}
    adaptive_weights = {"confidence_linear": {}, "history_linear": {}}
    for season in VALIDATION_SEASONS:
        fold = baseline[season]
        validation_values = validation_physical[season]
        train_values = train_physical[season]
        matched_train = train_values["tm_matched"].eq(1.0).to_numpy()
        matched_validation = fold["matched"]
        scores = validation_values["tm_confidence"].to_numpy(dtype="float64")
        score_reference = train_values.loc[matched_train, "tm_confidence"].to_numpy(dtype="float64")
        bins = confidence_bins(scores, score_reference, matched_validation)
        confidence_bin_masks[season] = bins
        adaptive_weights["confidence_linear"][season] = percentile_weight(scores, score_reference)
        history = np.log1p(validation_values["tm_history_n"].to_numpy(dtype="float64"))
        history_reference = np.log1p(train_values.loc[matched_train, "tm_history_n"].to_numpy(dtype="float64"))
        adaptive_weights["history_linear"][season] = percentile_weight(history, history_reference)
        advantage = (
            (fold["target"][matched_validation] - fold["champion"][matched_validation]) ** 2
            - (fold["target"][matched_validation] - fold["prediction"]) ** 2
        )
        matched_bins = bins[matched_validation]
        for label in ("high-A", "high-B", "high-C"):
            current = matched_bins == label
            confidence_rows.append(
                {
                    "validation_season": season,
                    "confidence_group": label,
                    "n": int(current.sum()),
                    "row_coverage": float(current.sum() / len(fold["target"])),
                    "champion_brier": brier(fold["target"][matched_validation][current], fold["champion"][matched_validation][current]),
                    "expert_brier": brier(fold["target"][matched_validation][current], fold["prediction"][current]),
                    "expert_gain": float(advantage[current].mean()),
                    "expert_win_rate": float((advantage[current] > 0.0).mean()),
                }
            )
    confidence = pd.DataFrame(confidence_rows)
    confidence.to_csv(args.output / "confidence_diagnostics.csv", index=False)

    advantage = advantage_quantiles(
        trainings,
        validations,
        train_physical,
        validation_physical,
        baseline,
    )
    advantage.to_csv(args.output / "advantage_quantiles.csv", index=False)

    independence = {
        str(season): independence_audit(
            validations[season],
            lookups[season],
            family_contract["all_trackman"],
            champion_meta,
        )
        for season in VALIDATION_SEASONS
    }
    if any(
        value != 0.0
        for audit in independence.values()
        for key, value in audit.items()
        if key.endswith("max_abs_diff")
    ):
        raise RuntimeError(f"row independence audit failed: {independence}")

    blend_rows = []
    decisions = []
    candidate_arrays = {}
    all_blend_experts = [*FAMILY_LABELS, "usage_plus_fo_gap", *usage_experts]
    for expert in all_blend_experts:
        folds = expert_folds[expert]
        for weight in FAMILY_BLEND_WEIGHTS:
            gains = {}
            candidates = {}
            for season in VALIDATION_SEASONS:
                candidate = apply_matched_blend(folds[season], weight)
                candidates[season] = candidate
                gain = brier(folds[season]["target"], folds[season]["champion"]) - brier(folds[season]["target"], candidate)
                gains[str(season)] = gain
                blend_rows.append(
                    {
                        "expert": expert,
                        "strategy": "constant",
                        "weight": weight,
                        "validation_season": season,
                        "brier_gain": gain,
                    }
                )
            target = np.concatenate([folds[s]["target"] for s in VALIDATION_SEASONS])
            champion = np.concatenate([folds[s]["champion"] for s in VALIDATION_SEASONS])
            candidate = np.concatenate([candidates[s] for s in VALIDATION_SEASONS])
            gains["pooled"] = brier(target, champion) - brier(target, candidate)
            blend_rows.append(
                {
                    "expert": expert,
                    "strategy": "constant",
                    "weight": weight,
                    "validation_season": "pooled",
                    "brier_gain": gains["pooled"],
                }
            )
            worst_major, worst_confidence, details = candidate_safety(
                folds, candidates, validations, confidence_bin_masks
            )
            passed = (
                all(gains[str(season)] > 0.0 for season in VALIDATION_SEASONS)
                and gains["pooled"] > BASELINE_GAINS["pooled"]
                and worst_major >= SUBSET_FLOOR
                and worst_confidence >= SUBSET_FLOOR
            )
            decision = {
                "expert": expert,
                "strategy": "constant",
                "weight": weight,
                "gains": gains,
                "delta_vs_all_trackman_w010": gains["pooled"] - BASELINE_GAINS["pooled"],
                "worst_major_subset_gain": worst_major,
                "worst_confidence_group_gain": worst_confidence,
                "passed": passed,
                "safety_details": details,
            }
            decisions.append(decision)
            candidate_arrays[(expert, "constant", weight)] = candidates

    # Deterministic adaptive weighting is evaluated only on frozen ALL TRACKMAN predictions.
    for strategy in ("confidence_linear", "history_linear"):
        gains = {}
        candidates = {}
        for season in VALIDATION_SEASONS:
            fold = baseline[season]
            candidate = apply_matched_blend(fold, adaptive_weights[strategy][season])
            candidates[season] = candidate
            gain = brier(fold["target"], fold["champion"]) - brier(fold["target"], candidate)
            gains[str(season)] = gain
            blend_rows.append(
                {
                    "expert": "all_trackman",
                    "strategy": strategy,
                    "weight": "0.05_to_0.15",
                    "validation_season": season,
                    "brier_gain": gain,
                }
            )
        target = np.concatenate([baseline[s]["target"] for s in VALIDATION_SEASONS])
        champion = np.concatenate([baseline[s]["champion"] for s in VALIDATION_SEASONS])
        candidate = np.concatenate([candidates[s] for s in VALIDATION_SEASONS])
        gains["pooled"] = brier(target, champion) - brier(target, candidate)
        blend_rows.append(
            {
                "expert": "all_trackman",
                "strategy": strategy,
                "weight": "0.05_to_0.15",
                "validation_season": "pooled",
                "brier_gain": gains["pooled"],
            }
        )
        worst_major, worst_confidence, details = candidate_safety(
            baseline, candidates, validations, confidence_bin_masks
        )
        passed = (
            all(gains[str(season)] > 0.0 for season in VALIDATION_SEASONS)
            and gains["pooled"] > BASELINE_GAINS["pooled"]
            and worst_major >= SUBSET_FLOOR
            and worst_confidence >= SUBSET_FLOOR
        )
        decisions.append(
            {
                "expert": "all_trackman",
                "strategy": strategy,
                "weight": "0.05_to_0.15",
                "gains": gains,
                "delta_vs_all_trackman_w010": gains["pooled"] - BASELINE_GAINS["pooled"],
                "worst_major_subset_gain": worst_major,
                "worst_confidence_group_gain": worst_confidence,
                "passed": passed,
                "safety_details": details,
            }
        )
        candidate_arrays[("all_trackman", strategy, "0.05_to_0.15")] = candidates

    blends = pd.DataFrame(blend_rows)
    constant_h = blends[
        blends["expert"].eq("all_trackman")
        & blends["strategy"].eq("constant")
    ][["weight", "validation_season", "brier_gain"]].rename(
        columns={"brier_gain": "all_trackman_same_weight_gain"}
    )
    blends = blends.merge(
        constant_h,
        on=["weight", "validation_season"],
        how="left",
    )
    blends["family_removal_delta"] = (
        blends["brier_gain"] - blends["all_trackman_same_weight_gain"]
    )
    blends.to_csv(args.output / "blend_results.csv", index=False)
    passing = [decision for decision in decisions if decision["passed"]]
    selected = max(
        passing,
        key=lambda row: (row["gains"]["pooled"], str(row["expert"])),
        default=None,
    )
    if selected:
        verdict = "A. PURIFIED TRACKMAN SIGNAL READY FOR LB"
    else:
        baseline_stable = all(BASELINE_GAINS[str(season)] > 0 for season in VALIDATION_SEASONS)
        verdict = (
            "B. TRACKMAN SIGNAL REAL BUT BASELINE IS BEST"
            if baseline_stable
            else "C. TRACKMAN GAIN EXPLAINED BY COVERAGE / UNSTABLE"
        )

    champion_after = sha256_path(CHAMPION)
    reference_hashes_after = verify_reference_hashes()
    if champion_after != champion_before:
        raise RuntimeError("champion changed during signal ablation")
    summary = {
        "verdict": verdict,
        "champion": {
            "artifact": str(CHAMPION),
            "sha256_before": champion_before,
            "sha256_after": champion_after,
            "unchanged": champion_after == CHAMPION_SHA256,
        },
        "immutable_crosswalk": {
            "method": "hungarian",
            "confidence": "HIGH",
            "reference_hashes_before": reference_hashes,
            "reference_hashes_after": reference_hashes_after,
            "unchanged": reference_hashes == reference_hashes_after,
        },
        "baseline_parity": baseline_parity,
        "family_contract": family_contract,
        "model_recipe": MODEL_RECIPE,
        "family_blend_weights": FAMILY_BLEND_WEIGHTS,
        "adaptive_weight_range": [ADAPTIVE_WEIGHT_MIN, ADAPTIVE_WEIGHT_MAX],
        "independence_audit": independence,
        "future_season_exclusion": {
            str(season): {
                "trackman_cutoff": season - 1,
                "validation_season": season,
                "passed": season - 1 < season,
            }
            for season in VALIDATION_SEASONS
        },
        "candidate_decisions": decisions,
        "selected_candidate": selected,
        "production": {
            "created": False,
            "reason": "guarded packaging required" if selected else "no purified candidate passed",
            "filename_limit": 30,
        },
        "test_read": False,
        "external_identity_information_used": False,
    }
    write_json(args.output / "summary.json", summary)
    print(verdict, flush=True)
    if selected:
        print(json.dumps(selected, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
