"""Validate a fixed upper-weight grid above the immutable ET25 w=.015 champion."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_final_submission import run_submission, safe_extract  # noqa: E402
from scripts.build_extratrees_candidate import (  # noqa: E402
    CHAMPION as ORIGINAL_CHAMPION,
    ENDPOINT_FORMULA_ATOL,
    EXPECTED_CHAMPION_SHA256 as ORIGINAL_CHAMPION_SHA256,
    META_FILE,
    MODEL_FILE,
    OOF_PREDICTION_ATOL,
    ROW_INDEPENDENCE_ATOL,
    candidate_row_independence,
    determinism_audit,
    extra_row_independence,
    package_audit,
    read_zip_json,
    run_champion_component,
    run_raw_prediction,
    runtime_audit,
    sha256_path,
    write_json,
    zip_info,
)
from scripts.extratrees_runtime import predict_extratrees  # noqa: E402
from scripts.validate_et25_weight import (  # noqa: E402
    brier,
    load_npz,
    subset_masks,
    zip_member_sha256,
)


CURRENT_CHAMPION = ROOT / "artifacts" / "sub_et25_w015.zip"
CURRENT_CHAMPION_SHA256 = (
    "11b7d514dadb47df505461176961cd6e436676bf99ddfa4e10f510fa65516b18"
)
CURRENT_CHAMPION_LB_BSS = 1016.9655337572
ORIGINAL_CHAMPION_LB_BSS = 1016.4442212358
DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_DATA_DIR = Path("/Users/wooh/Documents/dev/open/data")
OOF_SOURCE_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
ET25_CACHE_DIR = ROOT / "artifacts" / "extratrees_size_ablation" / "cache"
ET25_MODEL = ET25_CACHE_DIR / "extra_trees_et25.joblib"
OUTPUT_DIR = ROOT / "artifacts" / "et25_upper_weight"

TARGET = "control_success"
VALIDATION_SEASONS = (2022, 2023, 2024)
WEIGHTS = (0.0150, 0.0175, 0.0200, 0.0225, 0.0250)
REFERENCE_WEIGHT = 0.0150
SUBSET_LOSS_FLOOR = -1e-5
SUBSET_DELTA_FLOOR_VS_REFERENCE = -5e-6
MIN_SUBSET_ROWS = 5_000
MIN_CLEAR_POOLED_DELTA = 5e-7
MAX_SUBMISSION_FILENAME_LENGTH = 30

REFERENCE_ZIP_SIZE_BYTES = 121_125_222
REFERENCE_RUNTIME_SECONDS = 9.605823833029717
REFERENCE_PEAK_RSS_BYTES = 2_452_914_176
ZIP_SIZE_RATIO_LIMIT = 1.02
RUNTIME_TARGET_SECONDS = 15.0
PEAK_RSS_TARGET_BYTES = int(2.5 * 1024**3)

WEIGHT_COLUMNS = (
    "weight",
    "validation_season",
    "n",
    "champion_brier",
    "candidate_brier",
    "brier_gain",
    "worst_major_subset_gain",
)
DELTA_COLUMNS = (
    "weight",
    "validation_season",
    "n",
    "brier_gain",
    "reference_w015_gain",
    "brier_gain_delta_vs_w015",
    "better_than_w015",
)
SUBSET_COLUMNS = (
    "weight",
    "validation_season",
    "axis",
    "subset",
    "n",
    "gain_vs_original_champion",
    "reference_w015_gain",
    "gain_delta_vs_w015",
    "eligible_for_gate",
    "passes_original_loss_floor",
    "passes_delta_floor_vs_w015",
)
SHIFT_COLUMNS = (
    "weight",
    "validation_season",
    "n",
    "mean_prediction",
    "mean_prediction_delta",
    "mean_abs_delta",
    "p50_abs_delta",
    "p95_abs_delta",
    "p99_abs_delta",
    "max_abs_delta",
)


def blend_original_champion(
    champion: np.ndarray,
    extra: np.ndarray,
    weight: float,
) -> np.ndarray:
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}")
    champion = np.asarray(champion, dtype="float64")
    extra = np.asarray(extra, dtype="float64")
    if champion.shape != extra.shape:
        raise ValueError("original champion and ET25 shapes differ")
    return (1.0 - weight) * champion + weight * extra


def build_raw_subset_results(folds: Mapping[int, Mapping]) -> pd.DataFrame:
    rows = []
    for weight in WEIGHTS:
        for season in VALIDATION_SEASONS:
            fold = folds[season]
            target = np.asarray(fold["target"])
            champion = np.asarray(fold["champion"])
            candidate = blend_original_champion(
                champion, np.asarray(fold["extra"]), weight
            )
            for axis, subset, mask in subset_masks(fold):
                mask = np.asarray(mask, dtype=bool)
                n = int(mask.sum())
                gain = (
                    brier(target[mask], champion[mask])
                    - brier(target[mask], candidate[mask])
                    if n
                    else np.nan
                )
                rows.append(
                    {
                        "weight": weight,
                        "validation_season": season,
                        "axis": axis,
                        "subset": subset,
                        "n": n,
                        "gain_vs_original_champion": gain,
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
                extra = np.asarray(fold["extra"])[mask]
                targets.append(target)
                champions.append(champion)
                candidates.append(blend_original_champion(champion, extra, weight))
            target = np.concatenate(targets)
            champion = np.concatenate(champions)
            candidate = np.concatenate(candidates)
            rows.append(
                {
                    "weight": weight,
                    "validation_season": "pooled",
                    "axis": axis,
                    "subset": subset,
                    "n": len(target),
                    "gain_vs_original_champion": (
                        brier(target, champion) - brier(target, candidate)
                    ),
                }
            )
    raw = pd.DataFrame(rows)
    reference = raw[np.isclose(raw["weight"], REFERENCE_WEIGHT)][
        ["validation_season", "axis", "subset", "gain_vs_original_champion"]
    ].rename(columns={"gain_vs_original_champion": "reference_w015_gain"})
    output = raw.merge(
        reference,
        on=["validation_season", "axis", "subset"],
        how="left",
        validate="many_to_one",
    )
    output["gain_delta_vs_w015"] = (
        output["gain_vs_original_champion"] - output["reference_w015_gain"]
    )
    output["eligible_for_gate"] = output["n"] >= MIN_SUBSET_ROWS
    output["passes_original_loss_floor"] = (
        ~output["eligible_for_gate"]
        | (output["gain_vs_original_champion"] >= SUBSET_LOSS_FLOOR)
    )
    output["passes_delta_floor_vs_w015"] = (
        ~output["eligible_for_gate"]
        | (output["gain_delta_vs_w015"] >= SUBSET_DELTA_FLOOR_VS_REFERENCE)
    )
    return output.loc[:, SUBSET_COLUMNS]


def build_weight_and_shift_results(
    folds: Mapping[int, Mapping],
    subsets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weight_rows = []
    shift_rows = []
    for weight in WEIGHTS:
        for season in (*VALIDATION_SEASONS, "pooled"):
            if season == "pooled":
                target = np.concatenate(
                    [np.asarray(folds[s]["target"]) for s in VALIDATION_SEASONS]
                )
                champion = np.concatenate(
                    [np.asarray(folds[s]["champion"]) for s in VALIDATION_SEASONS]
                )
                extra = np.concatenate(
                    [np.asarray(folds[s]["extra"]) for s in VALIDATION_SEASONS]
                )
            else:
                fold = folds[int(season)]
                target = np.asarray(fold["target"])
                champion = np.asarray(fold["champion"])
                extra = np.asarray(fold["extra"])
            candidate = blend_original_champion(champion, extra, weight)
            delta = candidate - champion
            abs_delta = np.abs(delta)
            eligible = subsets[
                np.isclose(subsets["weight"], weight)
                & subsets["validation_season"].astype(str).eq(str(season))
                & subsets["eligible_for_gate"]
            ]
            weight_rows.append(
                {
                    "weight": weight,
                    "validation_season": season,
                    "n": len(target),
                    "champion_brier": brier(target, champion),
                    "candidate_brier": brier(target, candidate),
                    "brier_gain": brier(target, champion) - brier(target, candidate),
                    "worst_major_subset_gain": float(
                        eligible["gain_vs_original_champion"].min()
                    ),
                }
            )
            shift_rows.append(
                {
                    "weight": weight,
                    "validation_season": season,
                    "n": len(target),
                    "mean_prediction": float(np.mean(candidate)),
                    "mean_prediction_delta": float(np.mean(delta)),
                    "mean_abs_delta": float(np.mean(abs_delta)),
                    "p50_abs_delta": float(np.quantile(abs_delta, 0.50)),
                    "p95_abs_delta": float(np.quantile(abs_delta, 0.95)),
                    "p99_abs_delta": float(np.quantile(abs_delta, 0.99)),
                    "max_abs_delta": float(np.max(abs_delta)),
                }
            )
    return (
        pd.DataFrame(weight_rows, columns=WEIGHT_COLUMNS),
        pd.DataFrame(shift_rows, columns=SHIFT_COLUMNS),
    )


def build_delta_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for weight in WEIGHTS:
        for season in (*VALIDATION_SEASONS, "pooled"):
            current = results[
                np.isclose(results["weight"], weight)
                & results["validation_season"].astype(str).eq(str(season))
            ].iloc[0]
            reference = results[
                np.isclose(results["weight"], REFERENCE_WEIGHT)
                & results["validation_season"].astype(str).eq(str(season))
            ].iloc[0]
            delta = float(current["brier_gain"] - reference["brier_gain"])
            rows.append(
                {
                    "weight": weight,
                    "validation_season": season,
                    "n": int(current["n"]),
                    "brier_gain": float(current["brier_gain"]),
                    "reference_w015_gain": float(reference["brier_gain"]),
                    "brier_gain_delta_vs_w015": delta,
                    "better_than_w015": delta > 0.0,
                }
            )
    return pd.DataFrame(rows, columns=DELTA_COLUMNS)


def select_upper_weight(
    results: pd.DataFrame,
    deltas: pd.DataFrame,
    subsets: pd.DataFrame,
) -> tuple[str, float | None, list[dict]]:
    decisions = []
    for weight in WEIGHTS:
        weight_results = results[np.isclose(results["weight"], weight)]
        gains = {
            str(row.validation_season): float(row.brier_gain)
            for row in weight_results.itertuples()
        }
        weight_deltas = deltas[np.isclose(deltas["weight"], weight)]
        delta_map = {
            str(row.validation_season): float(row.brier_gain_delta_vs_w015)
            for row in weight_deltas.itertuples()
        }
        eligible = subsets[
            np.isclose(subsets["weight"], weight) & subsets["eligible_for_gate"]
        ]
        is_upper = weight > REFERENCE_WEIGHT
        conditions = {
            "is_upper_candidate": is_upper,
            "2022_positive_vs_original": gains["2022"] > 0.0,
            "2023_positive_vs_original": gains["2023"] > 0.0,
            "2024_positive_vs_original": gains["2024"] > 0.0,
            "pooled_positive_vs_original": gains["pooled"] > 0.0,
            "pooled_positive_vs_w015": delta_map["pooled"] > 0.0,
            "2023_positive_vs_w015": delta_map["2023"] > 0.0,
            "2024_positive_vs_w015": delta_map["2024"] > 0.0,
            "2022_nonnegative_vs_w015": delta_map["2022"] >= 0.0,
            "pooled_delta_clearly_positive": (
                delta_map["pooled"] >= MIN_CLEAR_POOLED_DELTA
            ),
            "major_subsets_safe_vs_original": bool(
                eligible["passes_original_loss_floor"].all()
            ),
            "no_severe_subset_degradation_vs_w015": bool(
                eligible["passes_delta_floor_vs_w015"].all()
            ),
        }
        decisions.append(
            {
                "weight": weight,
                "gains": gains,
                "delta_vs_w015": delta_map,
                "worst_subset_gain_vs_original": float(
                    eligible["gain_vs_original_champion"].min()
                ),
                "worst_subset_delta_vs_w015": float(
                    eligible["gain_delta_vs_w015"].min()
                ),
                "conditions": conditions,
                "passed": all(conditions.values()),
            }
        )
    passing = sorted(row["weight"] for row in decisions if row["passed"])
    if passing:
        return "A. NEW UPPER WEIGHT READY FOR ONE LB SUBMISSION", passing[0], decisions
    pooled = {
        row["weight"]: row["gains"]["pooled"] for row in decisions
    }
    if pooled[REFERENCE_WEIGHT] >= max(pooled.values()):
        return "B. w=.015 REMAINS BEST", None, decisions
    return "C. SAFE PLATEAU REACHED — STOP WEIGHT AXIS", None, decisions


def candidate_filename(weight: float) -> str:
    if weight not in WEIGHTS or weight <= REFERENCE_WEIGHT:
        raise ValueError("filename requires a preregistered upper weight")
    scaled_10000 = int(round(weight * 10_000))
    if scaled_10000 % 10:
        code = f"{scaled_10000:04d}"
    else:
        code = f"{scaled_10000 // 10:03d}"
    name = f"sub_et25_w{code}.zip"
    if len(name) > MAX_SUBMISSION_FILENAME_LENGTH:
        raise ValueError("submission filename exceeds the 30-character hard gate")
    return name


def format_weight(weight: float) -> str:
    return f"{weight:.4f}".rstrip("0").rstrip(".")


def patch_runtime_weights(
    source: str,
    weight: float,
    source_weight: float = REFERENCE_WEIGHT,
) -> str:
    extra_anchor = f"EXTRA_WEIGHT = {format_weight(source_weight)}"
    champion_anchor = f"CHAMPION_WEIGHT = {format_weight(1.0 - source_weight)}"
    if source.count(extra_anchor) != 1 or source.count(champion_anchor) != 1:
        raise ValueError("source runtime weight anchors are missing or ambiguous")
    return source.replace(
        extra_anchor, f"EXTRA_WEIGHT = {format_weight(weight)}", 1
    ).replace(
        champion_anchor, f"CHAMPION_WEIGHT = {format_weight(1.0 - weight)}", 1
    )


def patch_script_weight_doc(
    source: str,
    weight: float,
    source_weight: float = REFERENCE_WEIGHT,
) -> str:
    anchor = (
        "Frozen endpoint: original champion first, then "
        f"{format_weight(1.0 - source_weight)}/{format_weight(source_weight)} "
        "ET25 blend."
    )
    if source.count(anchor) != 1:
        raise ValueError("source script weight anchor is missing or ambiguous")
    replacement = (
        "Frozen endpoint: original champion first, then "
        f"{format_weight(1.0 - weight)}/{format_weight(weight)} ET25 blend."
    )
    return source.replace(anchor, replacement, 1)


def weight_metadata(
    reference: Mapping,
    weight: float,
    source_champion_sha256: str = CURRENT_CHAMPION_SHA256,
) -> dict:
    metadata = copy.deepcopy(dict(reference))
    metadata["candidate"] = (
        f"{format_weight(1.0 - weight)} original champion + "
        f"{format_weight(weight)} frozen ET25"
    )
    metadata["champion_weight"] = 1.0 - weight
    metadata["extra_weight"] = weight
    metadata["upper_weight_ablation_only"] = True
    metadata["source_champion_sha256"] = source_champion_sha256
    return metadata


def build_weight_candidate(
    source_path: Path,
    candidate: Path,
    weight: float,
    source_weight: float = REFERENCE_WEIGHT,
    source_champion_sha256: str = CURRENT_CHAMPION_SHA256,
) -> None:
    if len(candidate.name) > MAX_SUBMISSION_FILENAME_LENGTH:
        raise ValueError("submission filename length guard failed")
    with zipfile.ZipFile(source_path) as source, zipfile.ZipFile(
        candidate, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as output:
        for original in source.infolist():
            if original.is_dir():
                continue
            name = original.filename
            if name == "extratrees_runtime.py":
                runtime = source.read(name).decode("utf-8")
                output.writestr(
                    zip_info(name),
                    patch_runtime_weights(runtime, weight, source_weight).encode("utf-8"),
                )
            elif name == "script.py":
                script = source.read(name).decode("utf-8")
                output.writestr(
                    zip_info(name),
                    patch_script_weight_doc(script, weight, source_weight).encode("utf-8"),
                )
            elif name == f"model/{META_FILE}":
                metadata = weight_metadata(
                    json.loads(source.read(name)), weight, source_champion_sha256
                )
                output.writestr(
                    zip_info(name),
                    json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
                )
            elif name == f"model/{MODEL_FILE}":
                with source.open(name) as source_model, output.open(
                    zip_info(name, zipfile.ZIP_STORED), "w"
                ) as target_model:
                    shutil.copyfileobj(
                        source_model, target_model, length=16 * 1024 * 1024
                    )
            else:
                output.writestr(zip_info(name), source.read(name))


def endpoint_audit(
    champion: np.ndarray,
    extra: np.ndarray,
    candidate: np.ndarray,
    weight: float,
) -> dict:
    expected = blend_original_champion(champion, extra, weight)
    diff = float(np.max(np.abs(candidate - expected)))
    return {
        "rows": len(candidate),
        "formula_max_abs_diff": diff,
        "nan_count": int(np.isnan(candidate).sum()),
        "inf_count": int(np.isinf(candidate).sum()),
        "within_bounds": bool(np.all((candidate >= 0.0) & (candidate <= 1.0))),
        "passed": bool(
            diff <= ENDPOINT_FORMULA_ATOL
            and np.isfinite(candidate).all()
            and np.all((candidate >= 0.0) & (candidate <= 1.0))
        ),
    }


def production_audit(
    candidate: Path,
    weight: float,
    test: pd.DataFrame,
    results: pd.DataFrame,
    current_sha_before: str,
    original_sha_before: str,
    current_champion: Path = CURRENT_CHAMPION,
    expected_current_sha256: str = CURRENT_CHAMPION_SHA256,
    reference_zip_size_bytes: int = REFERENCE_ZIP_SIZE_BYTES,
    reference_runtime_seconds: float = REFERENCE_RUNTIME_SECONDS,
    reference_peak_rss_bytes: int = REFERENCE_PEAK_RSS_BYTES,
) -> dict:
    metadata = read_zip_json(candidate, f"model/{META_FILE}")
    model = joblib.load(ET25_MODEL)
    extra_test = predict_extratrees(test, "", model=model, metadata=metadata)
    extra_independence = extra_row_independence(model, metadata, test)
    del model
    gc.collect()
    with tempfile.TemporaryDirectory(prefix="upper_base_") as base_tmp, tempfile.TemporaryDirectory(
        prefix="upper_candidate_"
    ) as candidate_tmp:
        base_root = Path(base_tmp)
        candidate_root = Path(candidate_tmp)
        safe_extract(ORIGINAL_CHAMPION, base_root)
        safe_extract(candidate, candidate_root)
        base_run = run_submission(base_root, test)
        base_prediction = base_run["output"][TARGET].to_numpy(dtype="float64")
        base_raw = run_raw_prediction(base_root, test, "predict")
        candidate_base = run_champion_component(candidate_root, test)
        champion_diff = float(np.max(np.abs(base_raw - candidate_base)))
        row_result = candidate_row_independence(candidate_root, test)
        candidate_output = row_result.pop("full_output")
        candidate_prediction = candidate_output["output"][TARGET].to_numpy(
            dtype="float64"
        )
        endpoint = endpoint_audit(base_prediction, extra_test, candidate_prediction, weight)
        determinism = determinism_audit(candidate_root, test)
        runtime = runtime_audit(candidate_root, test)

    current_model_hash = zip_member_sha256(current_champion, f"model/{MODEL_FILE}")
    candidate_model_hash = zip_member_sha256(candidate, f"model/{MODEL_FILE}")
    package = package_audit(candidate, ET25_MODEL)
    parity = []
    for season in VALIDATION_SEASONS:
        source = load_npz(OOF_SOURCE_DIR / f"season_{season}.npz")
        extra = load_npz(ET25_CACHE_DIR / f"season_{season}_prefixes.npz")[
            "prediction_25"
        ].astype("float64")
        experiment = blend_original_champion(
            source["champion_final"].astype("float64"), extra, weight
        )
        packaged_formula = (
            float(metadata["champion_weight"])
            * source["champion_final"].astype("float64")
            + float(metadata["extra_weight"]) * extra
        )
        diff = float(np.max(np.abs(experiment - packaged_formula)))
        gain = float(
            results[
                np.isclose(results["weight"], weight)
                & results["validation_season"].astype(str).eq(str(season))
            ]["brier_gain"].iloc[0]
        )
        parity.append(
            {
                "validation_season": season,
                "max_abs_diff": diff,
                "brier_gain": gain,
                "passed": diff <= OOF_PREDICTION_ATOL and gain > 0.0,
            }
        )
    row_independence = {
        "champion_component": {"max_abs_diff": champion_diff, "passed": champion_diff == 0.0},
        "extra_trees": extra_independence,
        "candidate": row_result["candidate"],
    }
    row_independence["passed"] = bool(
        champion_diff == 0.0
        and extra_independence["max_abs_diff"] <= ROW_INDEPENDENCE_ATOL
        and row_result["candidate"]["max_abs_diff"] == 0.0
    )
    current_sha_after = sha256_path(current_champion)
    original_sha_after = sha256_path(ORIGINAL_CHAMPION)
    integrity = {
        "current_champion_path": str(current_champion),
        "current_champion_sha256_before": current_sha_before,
        "current_champion_sha256_after": current_sha_after,
        "current_champion_immutable": (
            current_sha_before == current_sha_after == expected_current_sha256
        ),
        "original_champion_sha256_before": original_sha_before,
        "original_champion_sha256_after": original_sha_after,
        "original_champion_immutable": (
            original_sha_before == original_sha_after == ORIGINAL_CHAMPION_SHA256
        ),
        "et25_model_member_sha256": current_model_hash,
        "candidate_model_member_sha256": candidate_model_hash,
        "et25_model_byte_identical": current_model_hash == candidate_model_hash,
    }
    filename = {
        "name": candidate.name,
        "length": len(candidate.name),
        "maximum": MAX_SUBMISSION_FILENAME_LENGTH,
        "passed": len(candidate.name) <= MAX_SUBMISSION_FILENAME_LENGTH,
    }
    technical = {
        "original_champion_component_exact": champion_diff == 0.0,
        "et25_model_byte_identical": integrity["et25_model_byte_identical"],
        "oof_formula_parity": all(row["passed"] for row in parity),
        "row_independence": row_independence["passed"],
        "determinism": determinism["passed"],
        "endpoint_finite_and_bounded": endpoint["passed"],
        "isolated_package_structure": package["passed"],
        "immutable_inputs": (
            integrity["current_champion_immutable"]
            and integrity["original_champion_immutable"]
        ),
        "filename_length": filename["passed"],
    }
    resources = {
        "zip_size_near_w015": candidate.stat().st_size <= int(
            reference_zip_size_bytes * ZIP_SIZE_RATIO_LIMIT
        ),
        "runtime_under_15_seconds": runtime["wall_seconds"] < RUNTIME_TARGET_SECONDS,
        "peak_rss_under_2_5_gib": runtime["peak_rss_bytes"] < PEAK_RSS_TARGET_BYTES,
    }
    return {
        "performed": True,
        "candidate": {
            "path": str(candidate),
            "file_name": candidate.name,
            "file_name_length": len(candidate.name),
            "sha256": sha256_path(candidate),
            "size_bytes": candidate.stat().st_size,
            "weight": weight,
            "tree_count": 25,
        },
        "filename": filename,
        "oof_formula_parity": parity,
        "row_independence": row_independence,
        "et25_model_parity": {"byte_identical": integrity["et25_model_byte_identical"], "sha256": candidate_model_hash},
        "endpoint": endpoint,
        "determinism": determinism,
        "runtime": runtime,
        "package": package,
        "integrity": integrity,
        "reference_resources": {
            "zip_size_bytes": reference_zip_size_bytes,
            "runtime_seconds": reference_runtime_seconds,
            "peak_rss_bytes": reference_peak_rss_bytes,
        },
        "technical_conditions": technical,
        "resource_conditions": resources,
        "passed": all(technical.values()) and all(resources.values()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    required = [
        args.train,
        args.data_dir / "test.csv",
        args.data_dir / "sample_submission.csv",
        CURRENT_CHAMPION,
        ORIGINAL_CHAMPION,
        ET25_MODEL,
    ]
    required.extend(OOF_SOURCE_DIR / f"season_{s}.npz" for s in VALIDATION_SEASONS)
    required.extend(ET25_CACHE_DIR / f"season_{s}_prefixes.npz" for s in VALIDATION_SEASONS)
    for path in required:
        if not path.is_file():
            parser.error(f"required file not found: {path}")
    current_sha_before = sha256_path(CURRENT_CHAMPION)
    original_sha_before = sha256_path(ORIGINAL_CHAMPION)
    if current_sha_before != CURRENT_CHAMPION_SHA256:
        raise SystemExit("immutable w=.015 champion checksum mismatch")
    if original_sha_before != ORIGINAL_CHAMPION_SHA256:
        raise SystemExit("original champion checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"w=.015 champion before: {current_sha_before}", flush=True)
    print("LB history is record-only; selection uses temporal OOF and safety", flush=True)

    train = pd.read_csv(args.train, encoding="utf-8-sig")
    folds = {}
    for season in VALIDATION_SEASONS:
        validation = train[train["season"] == season]
        source = load_npz(OOF_SOURCE_DIR / f"season_{season}.npz")
        extra = load_npz(ET25_CACHE_DIR / f"season_{season}_prefixes.npz")
        target = validation[TARGET].to_numpy(dtype="float64")
        if not np.array_equal(target, source["y"].astype("float64")):
            raise ValueError(f"train/OOF target order mismatch for {season}")
        folds[season] = {
            "target": target,
            "champion": source["champion_final"].astype("float64"),
            "extra": extra["prediction_25"].astype("float64"),
            "game_type": validation["game_type"].to_numpy(),
            "pitcher_seen": source["x_pitcher_seen"].astype(bool),
            "balls_before": validation["balls_before"].to_numpy(),
            "strikes_before": validation["strikes_before"].to_numpy(),
            "pitcher_form_adjustment": (
                source["pitcher_form_pred"].astype("float64")
                - source["champion_base"].astype("float64")
            ),
        }

    subsets = build_raw_subset_results(folds)
    results, shifts = build_weight_and_shift_results(folds, subsets)
    deltas = build_delta_results(results)
    verdict, selected_weight, decisions = select_upper_weight(results, deltas, subsets)
    results.to_csv(args.output / "weight_results.csv", index=False)
    deltas.to_csv(args.output / "delta_vs_w015.csv", index=False)
    subsets.to_csv(args.output / "subset_results.csv", index=False)
    shifts.to_csv(args.output / "prediction_shift.csv", index=False)
    print(f"OOF verdict before production audit: {verdict}", flush=True)

    audit = {"performed": False, "reason": "no upper weight passed the frozen gate"}
    candidate = None
    if selected_weight is not None:
        name = candidate_filename(selected_weight)
        existing = sorted((ROOT / "artifacts").glob("sub_et25_w*.zip"))
        allowed = {CURRENT_CHAMPION.name, name}
        unexpected = [path for path in existing if path.name not in allowed]
        if unexpected:
            raise RuntimeError(f"multiple upper-weight candidates would exist: {unexpected}")
        candidate = ROOT / "artifacts" / name
        build_weight_candidate(CURRENT_CHAMPION, candidate, selected_weight)
        test = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")
        sample = pd.read_csv(
            args.data_dir / "sample_submission.csv", encoding="utf-8-sig"
        )
        if test["row_id"].tolist() != sample["row_id"].tolist():
            raise ValueError("test/sample row order mismatch")
        audit = production_audit(
            candidate,
            selected_weight,
            test,
            results,
            current_sha_before,
            original_sha_before,
        )
        if not audit["passed"]:
            raise RuntimeError("selected upper weight failed production audit")

    ordered_pooled = results[
        results["validation_season"].astype(str).eq("pooled")
    ].sort_values("weight")
    response_curve = []
    previous_gain = None
    for row in ordered_pooled.itertuples():
        response_curve.append(
            {
                "weight": row.weight,
                "pooled_gain": row.brier_gain,
                "increment_from_previous_grid_point": (
                    None if previous_gain is None else row.brier_gain - previous_gain
                ),
            }
        )
        previous_gain = row.brier_gain
    summary = {
        "verdict": verdict,
        "current_lb_champion": {
            "lb_bss": CURRENT_CHAMPION_LB_BSS,
            "sha256": CURRENT_CHAMPION_SHA256,
            "selection_use": "record only",
        },
        "tree_count": 25,
        "weights": list(WEIGHTS),
        "reference_weight": REFERENCE_WEIGHT,
        "selection_source": "temporal OOF and fixed subset safety only",
        "selection_policy": {
            "choose_smallest_passing_upper_weight": True,
            "minimum_pooled_delta_vs_w015": MIN_CLEAR_POOLED_DELTA,
            "require_2022_nonnegative_delta_vs_w015": True,
            "subset_loss_floor_vs_original": SUBSET_LOSS_FLOOR,
            "subset_delta_floor_vs_w015": SUBSET_DELTA_FLOOR_VS_REFERENCE,
        },
        "candidate_decisions": decisions,
        "response_curve": response_curve,
        "selected_weight": selected_weight,
        "production_audit": audit,
        "candidate": audit.get("candidate"),
        "current_w015_sha256_before": current_sha_before,
        "current_w015_sha256_after": sha256_path(CURRENT_CHAMPION),
        "original_champion_sha256_before": original_sha_before,
        "original_champion_sha256_after": sha256_path(ORIGINAL_CHAMPION),
        "leaderboard_submission_performed": False,
    }
    write_json(args.output / "summary.json", summary)
    print(f"final verdict: {verdict}", flush=True)
    if candidate is not None:
        print(f"candidate: {candidate.name} (length={len(candidate.name)})", flush=True)
        print(f"sha256: {sha256_path(candidate)}", flush=True)


if __name__ == "__main__":
    main()
