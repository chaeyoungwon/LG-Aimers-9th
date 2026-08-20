"""Inference-only estimator-prefix ablation for the accepted 300-tree ExtraTrees.

Each temporal fold fits the frozen 300-tree recipe once.  The 25/50/100/200
candidates are predictions from ``estimators_[:n]`` of that one fitted forest;
no tree-count-specific model or blend-weight search is performed.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_final_submission import run_submission, safe_extract  # noqa: E402
from scripts.build_extratrees_candidate import (  # noqa: E402
    CHAMPION,
    ENDPOINT_FORMULA_ATOL,
    EXPECTED_CHAMPION_SHA256,
    EXPECTED_CHAMPION_SIZE,
    META_FILE,
    MODEL_FILE,
    OOF_PREDICTION_ATOL,
    ROW_INDEPENDENCE_ATOL,
    build_candidate_zip,
    candidate_row_independence,
    determinism_audit,
    extra_row_independence,
    package_audit,
    read_zip_json,
    run_champion_component,
    run_raw_prediction,
    runtime_audit,
    sha256_path,
    test_delta_audit,
    write_json,
)
from scripts.extratrees_runtime import (  # noqa: E402
    CHAMPION_WEIGHT,
    EXTRA_WEIGHT,
    build_hgb52_features,
    fit_preprocessor,
    predict_extratrees,
    transform_features,
)


DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_DATA_DIR = Path("/Users/wooh/Documents/dev/open/data")
OOF_SOURCE_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
ACCEPTED_OOF_DIR = ROOT / "artifacts" / "extratrees_complement" / "cache"
REFERENCE_PRODUCTION_DIR = ROOT / "artifacts" / "extratrees_production"
REFERENCE_FULL_MODEL = REFERENCE_PRODUCTION_DIR / "cache" / MODEL_FILE
REFERENCE_CANDIDATE = ROOT / "artifacts" / "submit_r10_pitcher_w418194_extra01.zip"
REFERENCE_CANDIDATE_SHA256 = (
    "42658ee044671925cf8c08f1d26c127f92b6ff587bfc038cf59915e3862965b9"
)
OUTPUT_DIR = ROOT / "artifacts" / "extratrees_size_ablation"
CACHE_DIR = OUTPUT_DIR / "cache"

TARGET = "control_success"
VALIDATION_SEASONS = (2022, 2023, 2024)
TREE_COUNTS = (25, 50, 100, 200, 300)
REFERENCE_TREE_COUNT = 300
RANDOM_STATE = 42
MIN_SAMPLES_LEAF = 8
MAX_FEATURES = "sqrt"
SUBSET_LOSS_FLOOR = -1e-5
MIN_SUBSET_ROWS = 5_000
MIN_REFERENCE_GAIN_RETENTION = 0.50
CACHE_VERSION = "extratrees-prefix-ablation-300-v1"
ZIP_TARGET_BYTES = 500_000_000
PEAK_RSS_TARGET_BYTES = 4 * 1024**3
RUNTIME_TARGET_SECONDS = 20.0

OOF_COLUMNS = (
    "tree_count",
    "validation_season",
    "n",
    "champion_brier",
    "blend_brier",
    "brier_gain",
    "reference_300_gain",
    "reference_gain_retention",
    "positive",
)
CONVERGENCE_COLUMNS = (
    "tree_count",
    "validation_season",
    "n",
    "corr_vs_300",
    "mae_vs_300",
    "max_abs_diff_vs_300",
    "corr_vs_champion",
    "champion_error_top10_extra_win_rate",
    "production_prediction_max_abs_diff",
)
SUBSET_COLUMNS = (
    "tree_count",
    "validation_season",
    "axis",
    "subset",
    "n",
    "brier_gain",
    "eligible_for_gate",
    "passes_loss_floor",
)
SIZE_COLUMNS = (
    "tree_count",
    "actual_model_size_bytes",
    "zip_size_estimate_bytes",
    "actual_candidate_zip_size_bytes",
    "selected",
)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype="float64")
    prediction = np.asarray(prediction, dtype="float64")
    return float(np.mean((target - prediction) ** 2))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype="float64")
    right = np.asarray(right, dtype="float64")
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("correlation inputs must be same-length vectors")
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def frozen_300_model(n_jobs: int = -1) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=REFERENCE_TREE_COUNT,
        max_depth=None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=False,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
    )


def prefix_model(model: ExtraTreesClassifier, tree_count: int) -> ExtraTreesClassifier:
    """Return a shallow forest view containing the first ``tree_count`` trees."""
    if tree_count not in TREE_COUNTS:
        raise ValueError(f"tree_count must be one of {TREE_COUNTS}")
    if len(model.estimators_) != REFERENCE_TREE_COUNT:
        raise ValueError("source forest is not the accepted 300-tree reference")
    prefix = copy.copy(model)
    prefix.estimators_ = list(model.estimators_[:tree_count])
    prefix.n_estimators = tree_count
    return prefix


def prefix_predictions(
    model: ExtraTreesClassifier,
    matrix: np.ndarray,
) -> dict[int, np.ndarray]:
    """Predict every candidate from ordered prefixes without mutating the source."""
    source_estimators = tuple(model.estimators_)
    predictions = {}
    for tree_count in TREE_COUNTS:
        prefix = prefix_model(model, tree_count)
        predictions[tree_count] = prefix.predict_proba(matrix)[:, 1].astype(
            "float64"
        )
    if tuple(model.estimators_) != source_estimators or model.n_estimators != 300:
        raise RuntimeError("prefix inference mutated the accepted reference forest")
    return predictions


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save_npz_atomic(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def fold_cache_path(season: int) -> Path:
    return CACHE_DIR / f"season_{season}_prefixes.npz"


def validate_feature_order(features: pd.DataFrame, champion_meta: Mapping) -> None:
    hgb = next(
        member for member in champion_meta["members"] if member["name"] == "hgb"
    )
    if list(features.columns) != hgb["feature_columns"]:
        raise ValueError("prefix ablation changed the accepted 52-column contract")
    forbidden = {"row_id", TARGET, "pitcher_id", "batter_id"}
    if forbidden & set(features.columns):
        raise ValueError("raw ID/target leaked into prefix ablation")


def fit_fold_prefixes(
    train_fold: pd.DataFrame,
    validation: pd.DataFrame,
    champion_meta: Mapping,
    accepted_prediction: np.ndarray,
    n_jobs: int,
) -> tuple[dict[int, np.ndarray], dict[int, float], float]:
    train_features = build_hgb52_features(train_fold, champion_meta["cat_levels"])
    validation_features = build_hgb52_features(
        validation, champion_meta["cat_levels"]
    )
    validate_feature_order(train_features, champion_meta)
    preprocessor = fit_preprocessor(train_features)
    x_train = transform_features(train_features, preprocessor)
    x_validation = transform_features(validation_features, preprocessor)
    del train_features, validation_features
    gc.collect()

    model = frozen_300_model(n_jobs=n_jobs)
    started = time.monotonic()
    model.fit(x_train, train_fold[TARGET].to_numpy(dtype="int8"))
    fit_seconds = time.monotonic() - started
    predictions = prefix_predictions(model, x_validation)
    reference_diff = float(
        np.max(np.abs(predictions[REFERENCE_TREE_COUNT] - accepted_prediction))
    )
    if reference_diff > OOF_PREDICTION_ATOL:
        raise RuntimeError(
            "reconstructed 300-tree OOF prediction differs from accepted cache: "
            f"{reference_diff:.3e}"
        )

    metadata = {
        "feature_columns": preprocessor["feature_columns"],
        "cat_levels": champion_meta["cat_levels"],
        "preprocessor": preprocessor,
    }
    production_differences = {}
    for tree_count in TREE_COUNTS:
        production = predict_extratrees(
            validation,
            "",
            model=prefix_model(model, tree_count),
            metadata=metadata,
        )
        production_differences[tree_count] = float(
            np.max(np.abs(production - predictions[tree_count]))
        )
    del model, x_train, x_validation
    gc.collect()
    return predictions, production_differences, fit_seconds


def load_or_fit_fold(
    season: int,
    train_fold: pd.DataFrame,
    validation: pd.DataFrame,
    champion_meta: Mapping,
    accepted_prediction: np.ndarray,
    n_jobs: int,
    force: bool,
) -> tuple[dict[int, np.ndarray], dict[int, float], float]:
    cache_path = fold_cache_path(season)
    target = validation[TARGET].to_numpy(dtype="float64")
    if cache_path.is_file() and not force:
        payload = load_npz(cache_path)
        expected = {
            "cache_version": CACHE_VERSION,
            "season": season,
            "target_sha256": array_sha256(target),
        }
        actual = {
            "cache_version": str(payload["cache_version"].item()),
            "season": int(payload["season"].item()),
            "target_sha256": str(payload["target_sha256"].item()),
        }
        if actual != expected:
            raise ValueError(f"stale prefix cache {cache_path}: {actual} != {expected}")
        predictions = {
            count: payload[f"prediction_{count}"].astype("float64")
            for count in TREE_COUNTS
        }
        production_differences = {
            count: float(payload[f"production_diff_{count}"].item())
            for count in TREE_COUNTS
        }
        reference_diff = float(
            np.max(
                np.abs(predictions[REFERENCE_TREE_COUNT] - accepted_prediction)
            )
        )
        if reference_diff > OOF_PREDICTION_ATOL:
            raise RuntimeError("cached 300-tree prefix lost accepted OOF parity")
        return predictions, production_differences, float(
            payload["fit_seconds"].item()
        )

    print(
        f"[{season}] fit one frozen 300-tree forest; evaluate ordered prefixes",
        flush=True,
    )
    predictions, production_differences, fit_seconds = fit_fold_prefixes(
        train_fold,
        validation,
        champion_meta,
        accepted_prediction,
        n_jobs,
    )
    payload = {
        "cache_version": np.asarray(CACHE_VERSION),
        "season": np.asarray(season, dtype="int16"),
        "target_sha256": np.asarray(array_sha256(target)),
        "fit_seconds": np.asarray(fit_seconds),
    }
    for count in TREE_COUNTS:
        payload[f"prediction_{count}"] = predictions[count]
        payload[f"production_diff_{count}"] = np.asarray(
            production_differences[count]
        )
    save_npz_atomic(cache_path, **payload)
    return predictions, production_differences, fit_seconds


def common_fold_data(
    validation: pd.DataFrame,
    train_fold: pd.DataFrame,
    oof: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray | float]:
    return {
        "target": validation[TARGET].to_numpy(dtype="float64"),
        "champion": oof["champion_final"].astype("float64"),
        "game_type": validation["game_type"].to_numpy(),
        "pitcher_seen": oof["x_pitcher_seen"].astype(bool),
        "balls_before": validation["balls_before"].to_numpy(),
        "strikes_before": validation["strikes_before"].to_numpy(),
        "pitcher_form_adjustment": (
            oof["pitcher_form_pred"].astype("float64")
            - oof["champion_base"].astype("float64")
        ),
        "history_threshold": float(train_fold["asof_pitcher_n"].median()),
    }


def subset_masks(fold: Mapping) -> list[tuple[str, str, np.ndarray]]:
    game_type = np.asarray(fold["game_type"]).astype(str)
    seen = np.asarray(fold["pitcher_seen"], dtype=bool)
    full = (
        np.asarray(fold["balls_before"], dtype="int64") == 3
    ) & (np.asarray(fold["strikes_before"], dtype="int64") == 2)
    form = np.asarray(fold["pitcher_form_adjustment"], dtype="float64")
    return [
        ("game_type", "R", game_type == "R"),
        ("game_type", "F", game_type == "F"),
        ("pitcher_seen", "seen", seen),
        ("pitcher_seen", "unseen", ~seen),
        ("full_count", "full", full),
        ("full_count", "non_full", ~full),
        ("pitcher_form", "positive", form > 0.0),
        ("pitcher_form", "negative", form < 0.0),
    ]


def top10_win_rate(
    target: np.ndarray,
    extra: np.ndarray,
    champion: np.ndarray,
) -> float:
    champion_error = (target - champion) ** 2
    threshold = float(np.quantile(champion_error, 0.90))
    mask = champion_error >= threshold
    extra_error = (target[mask] - extra[mask]) ** 2
    return float(np.mean(extra_error < champion_error[mask]))


def build_oof_tables(
    folds: Mapping[int, Mapping],
    production_differences: Mapping[int, Mapping[int, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof_rows = []
    convergence_rows = []
    subset_records = []
    for count in TREE_COUNTS:
        for season in (*VALIDATION_SEASONS, "pooled"):
            if season == "pooled":
                target = np.concatenate(
                    [np.asarray(folds[s]["target"]) for s in VALIDATION_SEASONS]
                )
                champion = np.concatenate(
                    [np.asarray(folds[s]["champion"]) for s in VALIDATION_SEASONS]
                )
                extra = np.concatenate(
                    [np.asarray(folds[s]["predictions"][count]) for s in VALIDATION_SEASONS]
                )
                reference = np.concatenate(
                    [np.asarray(folds[s]["predictions"][300]) for s in VALIDATION_SEASONS]
                )
                production_diff = max(
                    production_differences[s][count] for s in VALIDATION_SEASONS
                )
            else:
                fold = folds[int(season)]
                target = np.asarray(fold["target"])
                champion = np.asarray(fold["champion"])
                extra = np.asarray(fold["predictions"][count])
                reference = np.asarray(fold["predictions"][300])
                production_diff = production_differences[int(season)][count]
            candidate = CHAMPION_WEIGHT * champion + EXTRA_WEIGHT * extra
            gain = brier(target, champion) - brier(target, candidate)
            oof_rows.append(
                {
                    "tree_count": count,
                    "validation_season": season,
                    "n": len(target),
                    "champion_brier": brier(target, champion),
                    "blend_brier": brier(target, candidate),
                    "brier_gain": gain,
                    "reference_300_gain": np.nan,
                    "reference_gain_retention": np.nan,
                    "positive": gain > 0.0,
                }
            )
            convergence_rows.append(
                {
                    "tree_count": count,
                    "validation_season": season,
                    "n": len(target),
                    "corr_vs_300": safe_corr(extra, reference),
                    "mae_vs_300": float(np.mean(np.abs(extra - reference))),
                    "max_abs_diff_vs_300": float(np.max(np.abs(extra - reference))),
                    "corr_vs_champion": safe_corr(extra, champion),
                    "champion_error_top10_extra_win_rate": top10_win_rate(
                        target, extra, champion
                    ),
                    "production_prediction_max_abs_diff": production_diff,
                }
            )

        for season in VALIDATION_SEASONS:
            fold = folds[season]
            target = np.asarray(fold["target"])
            champion = np.asarray(fold["champion"])
            extra = np.asarray(fold["predictions"][count])
            candidate = CHAMPION_WEIGHT * champion + EXTRA_WEIGHT * extra
            for axis, subset, mask in subset_masks(fold):
                mask = np.asarray(mask, dtype=bool)
                n = int(mask.sum())
                gain = (
                    brier(target[mask], champion[mask])
                    - brier(target[mask], candidate[mask])
                    if n
                    else np.nan
                )
                eligible = n >= MIN_SUBSET_ROWS
                subset_records.append(
                    {
                        "tree_count": count,
                        "validation_season": season,
                        "axis": axis,
                        "subset": subset,
                        "n": n,
                        "brier_gain": gain,
                        "eligible_for_gate": eligible,
                        "passes_loss_floor": (not eligible) or gain >= SUBSET_LOSS_FLOOR,
                    }
                )

        for axis, subset, _ in subset_masks(folds[VALIDATION_SEASONS[0]]):
            targets = []
            champions = []
            candidates = []
            for season in VALIDATION_SEASONS:
                fold = folds[season]
                mask = next(
                    mask
                    for current_axis, current_subset, mask in subset_masks(fold)
                    if current_axis == axis and current_subset == subset
                )
                target = np.asarray(fold["target"])[mask]
                champion = np.asarray(fold["champion"])[mask]
                extra = np.asarray(fold["predictions"][count])[mask]
                targets.append(target)
                champions.append(champion)
                candidates.append(CHAMPION_WEIGHT * champion + EXTRA_WEIGHT * extra)
            target = np.concatenate(targets)
            champion = np.concatenate(champions)
            candidate = np.concatenate(candidates)
            gain = brier(target, champion) - brier(target, candidate)
            eligible = len(target) >= MIN_SUBSET_ROWS
            subset_records.append(
                {
                    "tree_count": count,
                    "validation_season": "pooled",
                    "axis": axis,
                    "subset": subset,
                    "n": len(target),
                    "brier_gain": gain,
                    "eligible_for_gate": eligible,
                    "passes_loss_floor": (not eligible) or gain >= SUBSET_LOSS_FLOOR,
                }
            )

    oof = pd.DataFrame(oof_rows, columns=OOF_COLUMNS)
    for season in (*VALIDATION_SEASONS, "pooled"):
        reference_gain = float(
            oof[
                (oof["tree_count"] == 300)
                & (oof["validation_season"].astype(str) == str(season))
            ]["brier_gain"].iloc[0]
        )
        mask = oof["validation_season"].astype(str) == str(season)
        oof.loc[mask, "reference_300_gain"] = reference_gain
        oof.loc[mask, "reference_gain_retention"] = (
            oof.loc[mask, "brier_gain"] / reference_gain
        )
    return (
        oof,
        pd.DataFrame(convergence_rows, columns=CONVERGENCE_COLUMNS),
        pd.DataFrame(subset_records, columns=SUBSET_COLUMNS),
    )


def select_smallest_candidate(
    oof: pd.DataFrame,
    subsets: pd.DataFrame,
) -> tuple[int | None, list[dict]]:
    decisions = []
    reference_pooled = float(
        oof[
            (oof["tree_count"] == REFERENCE_TREE_COUNT)
            & oof["validation_season"].astype(str).eq("pooled")
        ]["brier_gain"].iloc[0]
    )
    for count in TREE_COUNTS:
        rows = oof[oof["tree_count"] == count]
        gains = {
            str(row.validation_season): float(row.brier_gain)
            for row in rows.itertuples()
        }
        eligible_subsets = subsets[
            (subsets["tree_count"] == count) & subsets["eligible_for_gate"]
        ]
        conditions = {
            "2022_positive": gains["2022"] > 0.0,
            "2023_positive": gains["2023"] > 0.0,
            "2024_positive": gains["2024"] > 0.0,
            "pooled_positive": gains["pooled"] > 0.0,
            "retains_at_least_half_reference_pooled_gain": (
                gains["pooled"] >= MIN_REFERENCE_GAIN_RETENTION * reference_pooled
            ),
            "major_subsets_safe": bool(
                eligible_subsets["passes_loss_floor"].all()
            ),
        }
        decisions.append(
            {
                "tree_count": count,
                "gains": gains,
                "reference_pooled_gain": reference_pooled,
                "reference_gain_retention": gains["pooled"] / reference_pooled,
                "worst_eligible_subset_gain": float(
                    eligible_subsets["brier_gain"].min()
                ),
                "conditions": conditions,
                "passed": all(conditions.values()),
            }
        )
    passing = [row["tree_count"] for row in decisions if row["passed"]]
    return (min(passing) if passing else None), decisions


def validate_full_model(model: ExtraTreesClassifier) -> None:
    params = model.get_params()
    expected = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "bootstrap": False,
        "n_jobs": -1,
        "random_state": 42,
    }
    for key, value in expected.items():
        if params[key] != value:
            raise ValueError(f"full model parameter mismatch: {key}")
    if len(model.estimators_) != 300:
        raise ValueError("full model does not contain exactly 300 estimators")


def model_size_table(
    full_model: ExtraTreesClassifier,
    selected: int | None,
) -> tuple[pd.DataFrame, Path | None]:
    reference_overhead = REFERENCE_CANDIDATE.stat().st_size - REFERENCE_FULL_MODEL.stat().st_size
    selected_path = CACHE_DIR / (
        f"extra_trees_et{selected}.joblib" if selected is not None else "unused.joblib"
    )
    rows = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extra_prefix_sizes_") as temporary:
        temporary_root = Path(temporary)
        for count in TREE_COUNTS:
            path = selected_path if count == selected else temporary_root / f"et{count}.joblib"
            joblib.dump(
                prefix_model(full_model, count),
                path,
                compress=("zlib", 3),
                protocol=5,
            )
            size = path.stat().st_size
            rows.append(
                {
                    "tree_count": count,
                    "actual_model_size_bytes": size,
                    "zip_size_estimate_bytes": size + reference_overhead,
                    "actual_candidate_zip_size_bytes": np.nan,
                    "selected": count == selected,
                }
            )
    return pd.DataFrame(rows, columns=SIZE_COLUMNS), selected_path if selected else None


def candidate_metadata(reference: Mapping, selected: int) -> dict:
    metadata = copy.deepcopy(dict(reference))
    metadata["candidate"] = f"0.99 champion + 0.01 ExtraTrees first {selected}/300 trees"
    metadata["model_params"]["n_estimators"] = selected
    metadata["selected_tree_count"] = selected
    metadata["prefix_source_tree_count"] = 300
    metadata["prefix_selection"] = "estimators_[:selected_tree_count]"
    metadata["tree_count_ablation_only"] = True
    return metadata


def production_audit(
    selected: int,
    candidate: Path,
    selected_model_path: Path,
    full_model: ExtraTreesClassifier,
    metadata: Mapping,
    test: pd.DataFrame,
    oof: pd.DataFrame,
    convergence: pd.DataFrame,
    champion_sha_before: str,
    reference_candidate_sha_before: str,
) -> dict:
    selected_model = prefix_model(full_model, selected)
    extra_test = predict_extratrees(
        test, "", model=selected_model, metadata=metadata
    )
    extra_independence = extra_row_independence(selected_model, metadata, test)
    reloaded = joblib.load(selected_model_path)
    reloaded_prediction = predict_extratrees(
        test, "", model=reloaded, metadata=metadata
    )
    reload_diff = float(np.max(np.abs(reloaded_prediction - extra_test)))
    del reloaded
    gc.collect()

    build_candidate_zip(CHAMPION, candidate, selected_model_path, metadata)
    with tempfile.TemporaryDirectory(prefix="et_size_champion_") as champion_tmp, tempfile.TemporaryDirectory(
        prefix="et_size_candidate_"
    ) as candidate_tmp:
        champion_root = Path(champion_tmp)
        candidate_root = Path(candidate_tmp)
        safe_extract(CHAMPION, champion_root)
        safe_extract(candidate, candidate_root)
        champion_run = run_submission(champion_root, test)
        champion_prediction = champion_run["output"][TARGET].to_numpy(dtype="float64")
        champion_raw = run_raw_prediction(champion_root, test, "predict")
        candidate_champion = run_champion_component(candidate_root, test)
        champion_diff = float(np.max(np.abs(champion_raw - candidate_champion)))
        row_result = candidate_row_independence(candidate_root, test)
        candidate_output = row_result.pop("full_output")
        candidate_prediction = candidate_output["output"][TARGET].to_numpy(
            dtype="float64"
        )
        delta = test_delta_audit(champion_prediction, extra_test, candidate_prediction)
        determinism = determinism_audit(candidate_root, test)
        runtime = runtime_audit(candidate_root, test)

    package = package_audit(candidate, selected_model_path)
    champion_sha_after = sha256_path(CHAMPION)
    reference_candidate_sha_after = sha256_path(REFERENCE_CANDIDATE)
    oof_rows = oof[
        (oof["tree_count"] == selected)
        & oof["validation_season"].astype(str).isin(
            [str(season) for season in VALIDATION_SEASONS]
        )
    ]
    parity_rows = convergence[
        (convergence["tree_count"] == selected)
        & convergence["validation_season"].astype(str).isin(
            [str(season) for season in VALIDATION_SEASONS]
        )
    ]
    parity = []
    for result in oof_rows.itertuples():
        diff = float(
            parity_rows[
                parity_rows["validation_season"].astype(str)
                == str(result.validation_season)
            ]["production_prediction_max_abs_diff"].iloc[0]
        )
        parity.append(
            {
                "validation_season": result.validation_season,
                "max_abs_diff": diff,
                "brier_gain": result.brier_gain,
                "passed": diff <= OOF_PREDICTION_ATOL and result.brier_gain > 0.0,
            }
        )

    row_independence = {
        "champion_component": {
            "max_abs_diff": champion_diff,
            "passed": champion_diff == 0.0,
        },
        "extra_trees": extra_independence,
        "candidate": row_result["candidate"],
    }
    row_independence["passed"] = bool(
        row_independence["champion_component"]["passed"]
        and extra_independence["passed"]
        and row_result["candidate"]["max_abs_diff"] == 0.0
    )
    integrity = {
        "champion_sha256_before": champion_sha_before,
        "champion_sha256_after": champion_sha_after,
        "champion_immutable": (
            champion_sha_before == champion_sha_after == EXPECTED_CHAMPION_SHA256
        ),
        "reference_300_candidate_sha256_before": reference_candidate_sha_before,
        "reference_300_candidate_sha256_after": reference_candidate_sha_after,
        "reference_300_candidate_immutable": (
            reference_candidate_sha_before
            == reference_candidate_sha_after
            == REFERENCE_CANDIDATE_SHA256
        ),
    }
    technical_conditions = {
        "oof_production_parity": all(row["passed"] for row in parity),
        "serialized_prefix_parity": reload_diff <= OOF_PREDICTION_ATOL,
        "champion_component_exact": champion_diff == 0.0,
        "row_independence": row_independence["passed"],
        "candidate_formula_and_bounds": delta["passed"],
        "determinism": determinism["passed"],
        "package_structure": package["passed"],
        "champion_and_reference_candidate_immutable": all(
            [integrity["champion_immutable"], integrity["reference_300_candidate_immutable"]]
        ),
    }
    deployment_targets = {
        "zip_under_500_mb": candidate.stat().st_size < ZIP_TARGET_BYTES,
        "peak_rss_under_4_gib": runtime["peak_rss_bytes"] < PEAK_RSS_TARGET_BYTES,
        "runtime_under_20_seconds": runtime["wall_seconds"] < RUNTIME_TARGET_SECONDS,
    }
    return {
        "performed": True,
        "tree_count": selected,
        "candidate": {
            "path": str(candidate),
            "sha256": sha256_path(candidate),
            "size_bytes": candidate.stat().st_size,
        },
        "model": {
            "path": str(selected_model_path),
            "size_bytes": selected_model_path.stat().st_size,
            "source_tree_count": 300,
            "selection": f"estimators_[:{selected}]",
            "serialized_reload_max_abs_diff": reload_diff,
        },
        "oof_production_parity": parity,
        "row_independence": row_independence,
        "test_delta": delta,
        "determinism": determinism,
        "runtime": runtime,
        "package": package,
        "integrity": integrity,
        "targets": {
            "zip_bytes": ZIP_TARGET_BYTES,
            "peak_rss_bytes": PEAK_RSS_TARGET_BYTES,
            "runtime_seconds": RUNTIME_TARGET_SECONDS,
        },
        "technical_conditions": technical_conditions,
        "deployment_targets": deployment_targets,
        "technical_passed": all(technical_conditions.values()),
        "deployment_passed": all(deployment_targets.values()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    required = [
        args.train,
        args.data_dir / "test.csv",
        args.data_dir / "sample_submission.csv",
        CHAMPION,
        REFERENCE_CANDIDATE,
        REFERENCE_FULL_MODEL,
    ]
    required.extend(OOF_SOURCE_DIR / f"season_{season}.npz" for season in VALIDATION_SEASONS)
    required.extend(
        ACCEPTED_OOF_DIR / f"season_{season}_trees300.npz"
        for season in VALIDATION_SEASONS
    )
    for path in required:
        if not path.is_file():
            parser.error(f"required file not found: {path}")
    if CHAMPION.stat().st_size != EXPECTED_CHAMPION_SIZE:
        raise SystemExit("champion size mismatch")
    champion_sha_before = sha256_path(CHAMPION)
    reference_candidate_sha_before = sha256_path(REFERENCE_CANDIDATE)
    if champion_sha_before != EXPECTED_CHAMPION_SHA256:
        raise SystemExit("champion checksum mismatch before ablation")
    if reference_candidate_sha_before != REFERENCE_CANDIDATE_SHA256:
        raise SystemExit("300-tree candidate checksum mismatch before ablation")

    args.output.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"champion before: {champion_sha_before}", flush=True)
    print(f"reference 300-tree candidate before: {reference_candidate_sha_before}", flush=True)
    train = pd.read_csv(args.train, encoding="utf-8-sig")
    test = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(
        args.data_dir / "sample_submission.csv", encoding="utf-8-sig"
    )
    if test["row_id"].tolist() != sample["row_id"].tolist():
        raise ValueError("test/sample row order mismatch")
    champion_meta = read_zip_json(CHAMPION, "model/meta.json")

    folds = {}
    production_differences = {}
    fit_seconds = {}
    for season in VALIDATION_SEASONS:
        train_fold = train[train["season"] <= season - 1]
        validation = train[train["season"] == season]
        source = load_npz(OOF_SOURCE_DIR / f"season_{season}.npz")
        accepted = load_npz(
            ACCEPTED_OOF_DIR / f"season_{season}_trees300.npz"
        )["prediction"].astype("float64")
        common = common_fold_data(validation, train_fold, source)
        if not np.array_equal(common["target"], source["y"].astype("float64")):
            raise ValueError(f"train/OOF target order mismatch for {season}")
        predictions, differences, seconds = load_or_fit_fold(
            season,
            train_fold,
            validation,
            champion_meta,
            accepted,
            args.n_jobs,
            args.force,
        )
        common["predictions"] = predictions
        folds[season] = common
        production_differences[season] = differences
        fit_seconds[season] = seconds
        print(
            f"[{season}] 300 parity max={np.max(np.abs(predictions[300] - accepted)):.3e}, "
            f"fit={seconds:.1f}s",
            flush=True,
        )
        del train_fold, validation, source
        gc.collect()

    oof, convergence, subsets = build_oof_tables(folds, production_differences)
    selected, decisions = select_smallest_candidate(oof, subsets)
    oof.to_csv(args.output / "oof_results.csv", index=False)
    convergence.to_csv(args.output / "prediction_convergence.csv", index=False)
    subsets.to_csv(args.output / "subset_results.csv", index=False)
    print(f"smallest passing prefix: {selected}", flush=True)

    full_model = joblib.load(REFERENCE_FULL_MODEL)
    validate_full_model(full_model)
    sizes, selected_model_path = model_size_table(full_model, selected)

    audit = {"performed": False, "reason": "no OOF prefix passed all gates"}
    candidate_path = None
    if selected is not None and selected_model_path is not None:
        candidate_path = ROOT / "artifacts" / f"submit_r10_pitcher_extra01_et{selected}.zip"
        reference_metadata = read_zip_json(
            REFERENCE_CANDIDATE, f"model/{META_FILE}"
        )
        metadata = candidate_metadata(reference_metadata, selected)
        audit = production_audit(
            selected,
            candidate_path,
            selected_model_path,
            full_model,
            metadata,
            test,
            oof,
            convergence,
            champion_sha_before,
            reference_candidate_sha_before,
        )
        sizes.loc[
            sizes["tree_count"] == selected,
            "actual_candidate_zip_size_bytes",
        ] = candidate_path.stat().st_size
    sizes.to_csv(args.output / "size_estimates.csv", index=False)
    write_json(args.output / "production_audit.json", audit)

    if selected is None:
        verdict = "C. complementary signal disappears after compression"
    elif audit["technical_passed"] and audit["deployment_passed"]:
        verdict = "A. COMPRESSED CANDIDATE READY FOR ONE LB SUBMISSION"
    else:
        verdict = "B. signal survives but deployment still too heavy"
    summary = {
        "verdict": verdict,
        "experiment": "accepted 300-tree ExtraTrees ordered-prefix size ablation",
        "tree_counts": list(TREE_COUNTS),
        "reference_tree_count": REFERENCE_TREE_COUNT,
        "blend_weights": {"champion": CHAMPION_WEIGHT, "extra_trees": EXTRA_WEIGHT},
        "separate_tree_count_models_trained": False,
        "fold_models_trained": len(VALIDATION_SEASONS),
        "prefix_rule": "first n estimators in estimators_ order",
        "fit_seconds": fit_seconds,
        "selection_gate": {
            "all_three_folds_positive": True,
            "pooled_positive": True,
            "minimum_reference_pooled_gain_retention": MIN_REFERENCE_GAIN_RETENTION,
            "major_subset_loss_floor": SUBSET_LOSS_FLOOR,
            "minimum_subset_rows": MIN_SUBSET_ROWS,
        },
        "candidate_decisions": decisions,
        "selected_tree_count": selected,
        "candidate": audit.get("candidate"),
        "production_technical_passed": audit.get("technical_passed", False),
        "deployment_targets_passed": audit.get("deployment_passed", False),
        "champion_sha256_before": champion_sha_before,
        "champion_sha256_after": sha256_path(CHAMPION),
        "reference_300_candidate_sha256_before": reference_candidate_sha_before,
        "reference_300_candidate_sha256_after": sha256_path(REFERENCE_CANDIDATE),
        "leaderboard_submission_performed": False,
    }
    write_json(args.output / "summary.json", summary)
    print(f"verdict: {verdict}", flush=True)
    if candidate_path is not None:
        print(f"candidate: {candidate_path}", flush=True)
        print(f"sha256: {sha256_path(candidate_path)}", flush=True)


if __name__ == "__main__":
    main()
