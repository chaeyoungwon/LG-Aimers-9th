"""Narrow, OOF-only weight sensitivity around the immutable ET25 champion.

The ExtraTrees model and original pre-ET champion endpoint are frozen.  Only
five preregistered blend weights are evaluated.  A new package is built only
when the OOF-only clarity gate selects a weight other than 0.010.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
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


CURRENT_CHAMPION = ROOT / "artifacts" / "submit_r10_pitcher_extra01_et25.zip"
CURRENT_CHAMPION_SHA256 = (
    "53fb5dc24a70b693a606ba59e2407c447f4050b71e4391d8e0a8314f4d6a3672"
)
CURRENT_CHAMPION_LB_BSS = 1016.8497637345
ORIGINAL_CHAMPION_LB_BSS = 1016.4442212358
DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_DATA_DIR = Path("/Users/wooh/Documents/dev/open/data")
OOF_SOURCE_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
ET25_OOF_DIR = ROOT / "artifacts" / "extratrees_size_ablation" / "cache"
ET25_MODEL = ET25_OOF_DIR / "extra_trees_et25.joblib"
OUTPUT_DIR = ROOT / "artifacts" / "et25_weight"

TARGET = "control_success"
VALIDATION_SEASONS = (2022, 2023, 2024)
WEIGHTS = (0.005, 0.008, 0.010, 0.012, 0.015)
REFERENCE_WEIGHT = 0.010
SUBSET_LOSS_FLOOR = -1e-5
MIN_SUBSET_ROWS = 5_000

# Preregistered before reading this round's weight results.  The threshold is
# about 14% of the accepted ET25 pooled gain and prevents packaging numerical
# wins that are too small or concentrated in one season.
MIN_CLEAR_POOLED_DELTA = 5e-7
MAX_CLEAR_FOLD_LOSS = 5e-7
MIN_FOLDS_BETTER_THAN_REFERENCE = 2

MAX_SUBMISSION_FILENAME_LENGTH = 30
ZIP_TARGET_BYTES = 500_000_000
PEAK_RSS_TARGET_BYTES = 4 * 1024**3
RUNTIME_TARGET_SECONDS = 20.0

WEIGHT_COLUMNS = (
    "weight",
    "validation_season",
    "n",
    "champion_brier",
    "candidate_brier",
    "brier_gain",
    "worst_major_subset_gain",
    "mean_candidate_prediction",
    "mean_delta_vs_champion",
    "p01_delta_vs_champion",
    "p50_delta_vs_champion",
    "p99_delta_vs_champion",
)
SUBSET_COLUMNS = (
    "weight",
    "validation_season",
    "axis",
    "subset",
    "n",
    "brier_gain",
    "eligible_for_gate",
    "passes_loss_floor",
)
DELTA_COLUMNS = (
    "weight",
    "validation_season",
    "n",
    "brier_gain",
    "reference_w010_gain",
    "brier_gain_delta_vs_w010",
    "better_than_w010",
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype="float64")
    prediction = np.asarray(prediction, dtype="float64")
    return float(np.mean((target - prediction) ** 2))


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
        raise ValueError("champion and ET25 predictions must have identical shape")
    return (1.0 - weight) * champion + weight * extra


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


def build_subset_results(folds: Mapping[int, Mapping]) -> pd.DataFrame:
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
                eligible = n >= MIN_SUBSET_ROWS
                rows.append(
                    {
                        "weight": weight,
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
                champion = np.asarray(fold["champion"])[mask]
                targets.append(np.asarray(fold["target"])[mask])
                champions.append(champion)
                candidates.append(
                    blend_original_champion(
                        champion, np.asarray(fold["extra"])[mask], weight
                    )
                )
            target = np.concatenate(targets)
            champion = np.concatenate(champions)
            candidate = np.concatenate(candidates)
            gain = brier(target, champion) - brier(target, candidate)
            eligible = len(target) >= MIN_SUBSET_ROWS
            rows.append(
                {
                    "weight": weight,
                    "validation_season": "pooled",
                    "axis": axis,
                    "subset": subset,
                    "n": len(target),
                    "brier_gain": gain,
                    "eligible_for_gate": eligible,
                    "passes_loss_floor": (not eligible) or gain >= SUBSET_LOSS_FLOOR,
                }
            )
    return pd.DataFrame(rows, columns=SUBSET_COLUMNS)


def build_weight_results(
    folds: Mapping[int, Mapping],
    subsets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
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
            eligible = subsets[
                np.isclose(subsets["weight"], weight)
                & subsets["validation_season"].astype(str).eq(str(season))
                & subsets["eligible_for_gate"]
            ]
            rows.append(
                {
                    "weight": weight,
                    "validation_season": season,
                    "n": len(target),
                    "champion_brier": brier(target, champion),
                    "candidate_brier": brier(target, candidate),
                    "brier_gain": brier(target, champion) - brier(target, candidate),
                    "worst_major_subset_gain": float(eligible["brier_gain"].min()),
                    "mean_candidate_prediction": float(np.mean(candidate)),
                    "mean_delta_vs_champion": float(np.mean(delta)),
                    "p01_delta_vs_champion": float(np.quantile(delta, 0.01)),
                    "p50_delta_vs_champion": float(np.quantile(delta, 0.50)),
                    "p99_delta_vs_champion": float(np.quantile(delta, 0.99)),
                }
            )
    return pd.DataFrame(rows, columns=WEIGHT_COLUMNS)


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
                    "reference_w010_gain": float(reference["brier_gain"]),
                    "brier_gain_delta_vs_w010": delta,
                    "better_than_w010": delta > 0.0,
                }
            )
    return pd.DataFrame(rows, columns=DELTA_COLUMNS)


def select_weight(
    results: pd.DataFrame,
    subsets: pd.DataFrame,
    deltas: pd.DataFrame,
) -> tuple[str, float | None, list[dict]]:
    decisions = []
    for weight in WEIGHTS:
        weight_results = results[np.isclose(results["weight"], weight)]
        gains = {
            str(row.validation_season): float(row.brier_gain)
            for row in weight_results.itertuples()
        }
        eligible = subsets[
            np.isclose(subsets["weight"], weight) & subsets["eligible_for_gate"]
        ]
        conditions = {
            "2022_positive": gains["2022"] > 0.0,
            "2023_positive": gains["2023"] > 0.0,
            "2024_positive": gains["2024"] > 0.0,
            "pooled_positive": gains["pooled"] > 0.0,
            "major_subsets_safe": bool(eligible["passes_loss_floor"].all()),
        }
        decisions.append(
            {
                "weight": weight,
                "gains": gains,
                "worst_major_subset_gain": float(eligible["brier_gain"].min()),
                "base_conditions": conditions,
                "base_passed": all(conditions.values()),
            }
        )
    passing = [row for row in decisions if row["base_passed"]]
    if not passing:
        return "C. OOF SIGNAL TOO FLAT — STOP WEIGHT AXIS", None, decisions
    best = max(passing, key=lambda row: row["gains"]["pooled"])
    if math.isclose(best["weight"], REFERENCE_WEIGHT, abs_tol=1e-15):
        return "B. w=.010 REMAINS BEST — KEEP CURRENT CHAMPION", None, decisions

    selected_deltas = deltas[
        np.isclose(deltas["weight"], best["weight"])
        & deltas["validation_season"].astype(str).isin(
            [str(season) for season in VALIDATION_SEASONS]
        )
    ]
    pooled_delta = float(
        deltas[
            np.isclose(deltas["weight"], best["weight"])
            & deltas["validation_season"].astype(str).eq("pooled")
        ]["brier_gain_delta_vs_w010"].iloc[0]
    )
    clarity = {
        "pooled_delta_at_least_5e_7": pooled_delta >= MIN_CLEAR_POOLED_DELTA,
        "at_least_two_folds_better": int(
            (selected_deltas["brier_gain_delta_vs_w010"] > 0.0).sum()
        )
        >= MIN_FOLDS_BETTER_THAN_REFERENCE,
        "no_fold_worse_by_more_than_5e_7": bool(
            (
                selected_deltas["brier_gain_delta_vs_w010"]
                >= -MAX_CLEAR_FOLD_LOSS
            ).all()
        ),
    }
    best["delta_vs_w010"] = {
        str(row.validation_season): float(row.brier_gain_delta_vs_w010)
        for row in deltas[np.isclose(deltas["weight"], best["weight"])].itertuples()
    }
    best["clarity_conditions"] = clarity
    best["clearly_better_than_w010"] = all(clarity.values())
    if all(clarity.values()):
        return "A. NEW WEIGHT READY FOR ONE LB SUBMISSION", best["weight"], decisions
    return "C. OOF SIGNAL TOO FLAT — STOP WEIGHT AXIS", None, decisions


def candidate_filename(weight: float) -> str:
    if weight not in WEIGHTS or math.isclose(weight, REFERENCE_WEIGHT, abs_tol=1e-15):
        raise ValueError("candidate filename requires a preregistered new weight")
    weight_code = int(round(weight * 1000))
    name = f"sub_et25_w{weight_code:03d}.zip"
    if len(name) > MAX_SUBMISSION_FILENAME_LENGTH:
        raise ValueError(
            f"submission filename exceeds {MAX_SUBMISSION_FILENAME_LENGTH} chars: {name}"
        )
    return name


def patch_runtime_weights(source: str, weight: float) -> str:
    extra_anchor = "EXTRA_WEIGHT = 0.01"
    champion_anchor = "CHAMPION_WEIGHT = 0.99"
    if source.count(extra_anchor) != 1 or source.count(champion_anchor) != 1:
        raise ValueError("ET25 runtime weight anchors are missing or ambiguous")
    return source.replace(
        extra_anchor, f"EXTRA_WEIGHT = {weight:.3f}", 1
    ).replace(
        champion_anchor, f"CHAMPION_WEIGHT = {1.0 - weight:.3f}", 1
    )


def patch_script_weight_doc(source: str, weight: float) -> str:
    anchor = "Frozen endpoint: original champion first, then 99/1 ExtraTrees blend."
    if source.count(anchor) != 1:
        raise ValueError("ET25 script weight description anchor is missing or ambiguous")
    replacement = (
        "Frozen endpoint: original champion first, then "
        f"{1.0 - weight:.3f}/{weight:.3f} ET25 blend."
    )
    return source.replace(anchor, replacement, 1)


def weight_metadata(reference: Mapping, weight: float) -> dict:
    metadata = copy.deepcopy(dict(reference))
    metadata["candidate"] = (
        f"{1.0 - weight:.3f} original champion + {weight:.3f} frozen ET25"
    )
    metadata["champion_weight"] = 1.0 - weight
    metadata["extra_weight"] = weight
    metadata["weight_ablation_only"] = True
    metadata["original_champion_lb_bss"] = ORIGINAL_CHAMPION_LB_BSS
    metadata["source_et25_champion_sha256"] = CURRENT_CHAMPION_SHA256
    return metadata


def build_weight_candidate(source_path: Path, candidate: Path, weight: float) -> None:
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
                    zip_info(name), patch_runtime_weights(runtime, weight).encode("utf-8")
                )
            elif name == "script.py":
                script = source.read(name).decode("utf-8")
                output.writestr(
                    zip_info(name), patch_script_weight_doc(script, weight).encode("utf-8")
                )
            elif name == f"model/{META_FILE}":
                metadata = weight_metadata(json.loads(source.read(name)), weight)
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


def zip_member_sha256(path: Path, member: str) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def endpoint_delta_audit(
    champion: np.ndarray,
    extra: np.ndarray,
    candidate: np.ndarray,
    weight: float,
) -> dict:
    expected = blend_original_champion(champion, extra, weight)
    formula_diff = float(np.max(np.abs(candidate - expected)))
    delta = candidate - champion
    return {
        "rows": len(delta),
        "weight": weight,
        "candidate_formula_max_abs_diff": formula_diff,
        "mean_delta": float(np.mean(delta)),
        "p01_delta": float(np.quantile(delta, 0.01)),
        "p50_delta": float(np.quantile(delta, 0.50)),
        "p99_delta": float(np.quantile(delta, 0.99)),
        "nan_count": int(np.isnan(candidate).sum()),
        "inf_count": int(np.isinf(candidate).sum()),
        "within_bounds": bool(np.all((candidate >= 0.0) & (candidate <= 1.0))),
        "passed": bool(
            formula_diff <= ENDPOINT_FORMULA_ATOL
            and np.isfinite(candidate).all()
            and np.all((candidate >= 0.0) & (candidate <= 1.0))
            and np.max(np.abs(delta)) <= weight
        ),
    }


def run_production_audit(
    candidate: Path,
    weight: float,
    test: pd.DataFrame,
    results: pd.DataFrame,
    current_sha_before: str,
    original_sha_before: str,
) -> dict:
    metadata = read_zip_json(candidate, f"model/{META_FILE}")
    model = joblib.load(ET25_MODEL)
    extra_test = predict_extratrees(test, "", model=model, metadata=metadata)
    extra_independence = extra_row_independence(model, metadata, test)
    del model
    gc.collect()

    with tempfile.TemporaryDirectory(prefix="et25_weight_base_") as base_tmp, tempfile.TemporaryDirectory(
        prefix="et25_weight_candidate_"
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
        endpoint = endpoint_delta_audit(
            base_prediction, extra_test, candidate_prediction, weight
        )
        determinism = determinism_audit(candidate_root, test)
        runtime = runtime_audit(candidate_root, test)

    current_model_hash = zip_member_sha256(
        CURRENT_CHAMPION, f"model/{MODEL_FILE}"
    )
    candidate_model_hash = zip_member_sha256(candidate, f"model/{MODEL_FILE}")
    current_sha_after = sha256_path(CURRENT_CHAMPION)
    original_sha_after = sha256_path(ORIGINAL_CHAMPION)
    package = package_audit(candidate, ET25_MODEL)
    parity = []
    for season in VALIDATION_SEASONS:
        row = results[
            np.isclose(results["weight"], weight)
            & results["validation_season"].astype(str).eq(str(season))
        ].iloc[0]
        # The ET25 OOF prediction already passed production parity.  This round
        # changes only the scalar metadata/runtime formula, evaluated twice here.
        source = load_npz(OOF_SOURCE_DIR / f"season_{season}.npz")
        prefix = load_npz(ET25_OOF_DIR / f"season_{season}_prefixes.npz")[
            "prediction_25"
        ].astype("float64")
        experiment = blend_original_champion(
            source["champion_final"].astype("float64"), prefix, weight
        )
        production_formula = (
            float(metadata["champion_weight"])
            * source["champion_final"].astype("float64")
            + float(metadata["extra_weight"]) * prefix
        )
        diff = float(np.max(np.abs(experiment - production_formula)))
        parity.append(
            {
                "validation_season": season,
                "max_abs_diff": diff,
                "brier_gain": float(row["brier_gain"]),
                "passed": diff <= OOF_PREDICTION_ATOL and row["brier_gain"] > 0.0,
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
        champion_diff == 0.0
        and extra_independence["max_abs_diff"] <= ROW_INDEPENDENCE_ATOL
        and row_result["candidate"]["max_abs_diff"] == 0.0
    )
    filename = {
        "name": candidate.name,
        "length": len(candidate.name),
        "maximum_length": MAX_SUBMISSION_FILENAME_LENGTH,
        "passed": len(candidate.name) <= MAX_SUBMISSION_FILENAME_LENGTH,
    }
    integrity = {
        "current_et25_sha256_before": current_sha_before,
        "current_et25_sha256_after": current_sha_after,
        "current_et25_immutable": (
            current_sha_before == current_sha_after == CURRENT_CHAMPION_SHA256
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
    technical = {
        "oof_parity": all(row["passed"] for row in parity),
        "champion_component_exact": champion_diff == 0.0,
        "et25_model_byte_identical": integrity["et25_model_byte_identical"],
        "row_independence": row_independence["passed"],
        "endpoint_formula_and_bounds": endpoint["passed"],
        "determinism": determinism["passed"],
        "isolated_package_structure": package["passed"],
        "immutable_inputs": (
            integrity["current_et25_immutable"]
            and integrity["original_champion_immutable"]
        ),
        "filename_length": filename["passed"],
    }
    resources = {
        "zip_under_500_mb": candidate.stat().st_size < ZIP_TARGET_BYTES,
        "peak_rss_under_4_gib": runtime["peak_rss_bytes"] < PEAK_RSS_TARGET_BYTES,
        "runtime_under_20_seconds": runtime["wall_seconds"] < RUNTIME_TARGET_SECONDS,
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
        "oof_parity": parity,
        "champion_component_parity": row_independence["champion_component"],
        "et25_model_parity": {
            "byte_identical": integrity["et25_model_byte_identical"],
            "sha256": candidate_model_hash,
        },
        "row_independence": row_independence,
        "endpoint": endpoint,
        "determinism": determinism,
        "runtime": runtime,
        "package": package,
        "integrity": integrity,
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
    required.extend(ET25_OOF_DIR / f"season_{s}_prefixes.npz" for s in VALIDATION_SEASONS)
    for path in required:
        if not path.is_file():
            parser.error(f"required file not found: {path}")

    current_sha_before = sha256_path(CURRENT_CHAMPION)
    original_sha_before = sha256_path(ORIGINAL_CHAMPION)
    if current_sha_before != CURRENT_CHAMPION_SHA256:
        raise SystemExit("immutable ET25 champion checksum mismatch")
    if original_sha_before != ORIGINAL_CHAMPION_SHA256:
        raise SystemExit("original champion checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"ET25 champion before: {current_sha_before}", flush=True)
    print("selection uses OOF only; LB score is record-only", flush=True)

    train = pd.read_csv(args.train, encoding="utf-8-sig")
    folds = {}
    for season in VALIDATION_SEASONS:
        validation = train[train["season"] == season]
        source = load_npz(OOF_SOURCE_DIR / f"season_{season}.npz")
        prefix = load_npz(ET25_OOF_DIR / f"season_{season}_prefixes.npz")
        target = validation[TARGET].to_numpy(dtype="float64")
        if not np.array_equal(target, source["y"].astype("float64")):
            raise ValueError(f"train/OOF target order mismatch for {season}")
        folds[season] = {
            "target": target,
            "champion": source["champion_final"].astype("float64"),
            "extra": prefix["prediction_25"].astype("float64"),
            "game_type": validation["game_type"].to_numpy(),
            "pitcher_seen": source["x_pitcher_seen"].astype(bool),
            "balls_before": validation["balls_before"].to_numpy(),
            "strikes_before": validation["strikes_before"].to_numpy(),
            "pitcher_form_adjustment": (
                source["pitcher_form_pred"].astype("float64")
                - source["champion_base"].astype("float64")
            ),
        }

    subsets = build_subset_results(folds)
    results = build_weight_results(folds, subsets)
    deltas = build_delta_results(results)
    verdict, selected_weight, decisions = select_weight(results, subsets, deltas)
    results.to_csv(args.output / "weight_results.csv", index=False)
    subsets.to_csv(args.output / "subset_results.csv", index=False)
    deltas.to_csv(args.output / "delta_vs_w010.csv", index=False)
    print(f"OOF verdict before production audit: {verdict}", flush=True)

    audit = {"performed": False, "reason": "OOF did not select a clear new weight"}
    candidate = None
    if selected_weight is not None:
        name = candidate_filename(selected_weight)
        existing = sorted((ROOT / "artifacts").glob("sub_et25_w*.zip"))
        unexpected = [path for path in existing if path.name != name]
        if unexpected:
            raise RuntimeError(f"multiple weight candidates would exist: {unexpected}")
        candidate = ROOT / "artifacts" / name
        build_weight_candidate(CURRENT_CHAMPION, candidate, selected_weight)
        test = pd.read_csv(
            args.data_dir / "test.csv", encoding="utf-8-sig"
        )
        sample = pd.read_csv(
            args.data_dir / "sample_submission.csv", encoding="utf-8-sig"
        )
        if test["row_id"].tolist() != sample["row_id"].tolist():
            raise ValueError("test/sample row order mismatch")
        audit = run_production_audit(
            candidate,
            selected_weight,
            test,
            results,
            current_sha_before,
            original_sha_before,
        )
        if not audit["passed"]:
            raise RuntimeError("selected weight failed production audit")

    summary = {
        "verdict": verdict,
        "new_lb_champion": {
            "lb_bss": CURRENT_CHAMPION_LB_BSS,
            "improvement_vs_original": 0.4055424987,
            "sha256": CURRENT_CHAMPION_SHA256,
            "selection_use": "record only; not used for weight selection",
        },
        "tree_count": 25,
        "weights": list(WEIGHTS),
        "reference_weight": REFERENCE_WEIGHT,
        "selection_source": "temporal OOF only",
        "clarity_gate": {
            "minimum_pooled_gain_delta_vs_w010": MIN_CLEAR_POOLED_DELTA,
            "maximum_fold_loss_vs_w010": MAX_CLEAR_FOLD_LOSS,
            "minimum_folds_better_than_w010": MIN_FOLDS_BETTER_THAN_REFERENCE,
        },
        "candidate_decisions": decisions,
        "selected_weight": selected_weight,
        "production_audit": audit,
        "candidate": audit.get("candidate"),
        "current_et25_sha256_before": current_sha_before,
        "current_et25_sha256_after": sha256_path(CURRENT_CHAMPION),
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
