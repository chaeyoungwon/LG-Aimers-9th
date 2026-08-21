"""Build and audit the frozen HIGH-only TrackMan w=.10 production candidate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_final_submission import (  # noqa: E402
    inference_environment,
    run_submission,
    safe_extract,
    validate_submission,
)
from scripts.extratrees_runtime import (  # noqa: E402
    build_hgb52_features,
    fit_preprocessor,
    transform_features,
)
from scripts.trackman_runtime import (  # noqa: E402
    CHAMPION_WEIGHT,
    LOOKUP_FILE,
    MATCHED_OUTPUT_DECIMALS,
    META_FILE,
    MODEL_FILE,
    TRACKMAN_WEIGHT,
    predict_trackman,
)
from scripts.validate_tm_crosswalk import (  # noqa: E402
    CHAMPION_SHA256,
    MODEL_RECIPE,
    attach_physical_features,
    brier,
    build_distance_table,
    build_main_fingerprints,
    build_main_physical_lookup,
    build_physical_profiles,
    build_tm_fingerprints,
    infer_hand_map,
    load_current_champion_oof,
    read_main,
    read_trackman,
    read_trackman_physical,
)
from scripts.validate_tm_signal import (  # noqa: E402
    BASELINE_GAINS,
    CROSSWALK_DIR,
    MAPPING_REFERENCE,
    REFERENCE_HASHES,
    VALIDATION_SEASONS,
    physical_feature_columns,
    verify_reference_hashes,
)
from scripts.validate_tm_crosswalk import assign_candidates  # noqa: E402


CHAMPION = ROOT / "artifacts" / "sub_et25_w020.zip"
CANDIDATE = ROOT / "artifacts" / "sub_tm10.zip"
OUTPUT = ROOT / "artifacts" / "tm_production"
CACHE = OUTPUT / "cache"
DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_TRACKMAN = Path("/Users/wooh/Documents/dev/open/data/trackman_history.csv")
DEFAULT_DATA_DIR = Path("/Users/wooh/Documents/dev/open/data")
TARGET = "control_success"
EXPECTED_FILENAME = "sub_tm10.zip"
MAX_FILENAME_LENGTH = 30
OOF_PREDICTION_ATOL = 1e-12
OOF_GAIN_ATOL = 1e-15
ROW_INDEPENDENCE_ATOL = 0.0
PACKAGE_GUARD_BYTES = 500 * 1024 * 1024
RSS_GUARD_BYTES = 4 * 1024**3
RUNTIME_GUARD_SECONDS = 20.0
RUNTIME_ROWS = 253_507


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def mapping_signature(mapping: pd.DataFrame) -> str:
    columns = ["main_pitcher_id", "tm_pitcher_id", "confidence"]
    canonical = mapping[columns].sort_values(columns).to_csv(index=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def regenerate_mapping(
    main_fingerprints: pd.DataFrame,
    tm_fingerprints: pd.DataFrame,
    cutoff: int,
    hand_map: Mapping,
) -> pd.DataFrame:
    distances = build_distance_table(
        main_fingerprints, tm_fingerprints, cutoff, hand_map
    )
    return assign_candidates(distances, "hungarian")


def mapping_parity_audit(
    train_path: Path, trackman_path: Path
) -> tuple[dict, dict[int, pd.DataFrame], pd.DataFrame]:
    main = read_main(train_path)
    tm = read_trackman(trackman_path)
    main_fingerprints = build_main_fingerprints(main)
    tm_fingerprints = build_tm_fingerprints(tm)
    hand_map, _ = infer_hand_map(main_fingerprints, tm_fingerprints)
    frozen = pd.read_csv(MAPPING_REFERENCE)
    regenerated = {}
    folds = []
    for season in VALIDATION_SEASONS:
        cutoff = season - 1
        current = regenerate_mapping(
            main_fingerprints, tm_fingerprints, cutoff, hand_map
        )
        reference = frozen[
            frozen["cutoff"].eq(cutoff) & frozen["method"].eq("hungarian")
        ].reset_index(drop=True)
        key_columns = ["main_pitcher_id", "tm_pitcher_id", "confidence"]
        exact = current[key_columns].equals(reference[key_columns])
        regenerated[cutoff] = current
        folds.append(
            {
                "validation_season": season,
                "cutoff": cutoff,
                "high_pitcher_count": int(current["confidence"].eq("HIGH").sum()),
                "mapping_signature": mapping_signature(
                    current[current["confidence"].eq("HIGH")]
                ),
                "reference_signature": mapping_signature(
                    reference[reference["confidence"].eq("HIGH")]
                ),
                "key_classification_exact": exact,
            }
        )
    # The production mapping itself is generated twice to audit Hungarian ties.
    production_1 = regenerate_mapping(
        main_fingerprints, tm_fingerprints, 2024, hand_map
    )
    production_2 = regenerate_mapping(
        main_fingerprints, tm_fingerprints, 2024, hand_map
    )
    production_columns = ["main_pitcher_id", "tm_pitcher_id", "confidence"]
    production_exact = production_1[production_columns].equals(
        production_2[production_columns]
    )
    production_signature_1 = mapping_signature(production_1)
    production_signature_2 = mapping_signature(production_2)
    result = {
        "algorithm": "Hungarian with sorted integer ids and deterministic cost matrix",
        "confidence_used": "HIGH only",
        "folds": folds,
        "production": {
            "cutoff": 2024,
            "runs": 2,
            "high_pitcher_count": int(production_1["confidence"].eq("HIGH").sum()),
            "signature_run_1": production_signature_1,
            "signature_run_2": production_signature_2,
            "key_classification_exact": production_exact,
            "byte_canonical_identical": production_signature_1 == production_signature_2,
        },
        "official_data_only": True,
        "external_identity_information_used": False,
        "passed": bool(
            all(
                row["key_classification_exact"]
                and row["mapping_signature"] == row["reference_signature"]
                for row in folds
            )
            and production_exact
            and production_signature_1 == production_signature_2
        ),
    }
    return result, regenerated, production_1


def frozen_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**MODEL_RECIPE)


def lookup_arrays(lookup: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    ordered = lookup.sort_values("pitcher_id")
    return (
        ordered["pitcher_id"].to_numpy(dtype="int64"),
        ordered[list(columns)].to_numpy(dtype="float64"),
    )


def train_expert(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    lookup: pd.DataFrame,
    champion_meta: Mapping,
) -> tuple[ExtraTreesClassifier, dict, np.ndarray, np.ndarray]:
    columns = physical_feature_columns(lookup)
    train_physical = attach_physical_features(training, lookup)
    validation_physical = attach_physical_features(validation, lookup)
    train_mask = train_physical["tm_matched"].eq(1.0).to_numpy()
    validation_mask = validation_physical["tm_matched"].eq(1.0).to_numpy()
    base_train = build_hgb52_features(
        training.loc[train_mask], champion_meta["cat_levels"]
    )
    base_validation = build_hgb52_features(
        validation.loc[validation_mask], champion_meta["cat_levels"]
    )
    features_train = pd.concat(
        [base_train, train_physical.loc[train_mask, columns]], axis=1
    )
    features_validation = pd.concat(
        [base_validation, validation_physical.loc[validation_mask, columns]], axis=1
    )
    preprocessor = fit_preprocessor(features_train)
    x_train = transform_features(features_train, preprocessor)
    x_validation = transform_features(features_validation, preprocessor)
    model = frozen_model()
    model.fit(x_train, training.loc[train_mask, TARGET].to_numpy(dtype="int8"))
    prediction = (
        model.predict_proba(x_validation)[:, 1].astype("float64")
        if len(x_validation)
        else np.empty(0, dtype="float64")
    )
    metadata = {
        "schema_version": 1,
        "candidate": "HIGH-only ALL TRACKMAN expert, matched-row w=0.10",
        "champion_weight": CHAMPION_WEIGHT,
        "trackman_weight": TRACKMAN_WEIGHT,
        "matched_output_round_decimals": MATCHED_OUTPUT_DECIMALS,
        "feature_columns": list(features_train.columns),
        "trackman_feature_columns": columns,
        "cat_levels": champion_meta["cat_levels"],
        "preprocessor": preprocessor,
        "model_params": frozen_model().get_params(),
        "train_rows_total": len(training),
        "train_rows_matched": int(train_mask.sum()),
        "train_max_season": int(training["season"].max()),
        "test_statistics_used": False,
        "official_data_only": True,
    }
    del x_train, x_validation, features_train, features_validation
    gc.collect()
    return model, metadata, prediction, validation_mask


def read_champion_meta() -> dict:
    with zipfile.ZipFile(CHAMPION) as archive:
        return json.loads(archive.read("model/meta.json"))


def oof_parity_audit(
    train: pd.DataFrame,
    physical_trackman: pd.DataFrame,
    mappings: Mapping[int, pd.DataFrame],
    champion_meta: Mapping,
) -> dict:
    rows = []
    pooled_y = []
    pooled_champion = []
    pooled_candidate = []
    for season in VALIDATION_SEASONS:
        cutoff = season - 1
        training = train[train["season"].le(cutoff)]
        validation = train[train["season"].eq(season)]
        profile = build_physical_profiles(physical_trackman, cutoff)
        lookup = build_main_physical_lookup(
            mappings[cutoff], profile, cutoff, "hungarian"
        )
        model, metadata, prediction, matched = train_expert(
            training, validation, lookup, champion_meta
        )
        reference_payload = np.load(
            CROSSWALK_DIR / "cache" / f"accepted52_plus_tm_{season}.npz",
            allow_pickle=False,
        )
        reference_prediction = reference_payload["prediction"].astype("float64")
        reference_matched = reference_payload["matched"].astype(bool)
        lookup_payload = lookup_arrays(lookup, metadata["trackman_feature_columns"])
        runtime_prediction, runtime_matched = predict_trackman(
            validation,
            "",
            model=model,
            metadata=metadata,
            lookup=lookup_payload,
        )
        target = validation[TARGET].to_numpy(dtype="float64")
        champion, _ = load_current_champion_oof(season, target)
        candidate = champion.copy()
        candidate[matched] = np.round(
            CHAMPION_WEIGHT * champion[matched] + TRACKMAN_WEIGHT * prediction,
            decimals=MATCHED_OUTPUT_DECIMALS,
        )
        gain = brier(target, champion) - brier(target, candidate)
        coverage = float(matched.mean())
        prediction_diff = float(np.max(np.abs(prediction - reference_prediction)))
        runtime_diff = float(np.max(np.abs(prediction - runtime_prediction)))
        row = {
            "validation_season": season,
            "cutoff": cutoff,
            "n": len(validation),
            "high_pitcher_count": len(lookup),
            "high_row_count": int(matched.sum()),
            "high_row_coverage": coverage,
            "mapping_signature": mapping_signature(
                mappings[cutoff][mappings[cutoff]["confidence"].eq("HIGH")]
            ),
            "model_prediction_max_abs_diff": prediction_diff,
            "runtime_prediction_max_abs_diff": runtime_diff,
            "matched_mask_exact": bool(
                np.array_equal(matched, reference_matched)
                and np.array_equal(matched, runtime_matched)
            ),
            "reference_brier_gain": BASELINE_GAINS[str(season)],
            "production_brier_gain": gain,
            "brier_gain_abs_diff": abs(gain - BASELINE_GAINS[str(season)]),
        }
        row["passed"] = bool(
            prediction_diff <= OOF_PREDICTION_ATOL
            and runtime_diff <= OOF_PREDICTION_ATOL
            and row["matched_mask_exact"]
            and row["brier_gain_abs_diff"] <= OOF_GAIN_ATOL
        )
        rows.append(row)
        pooled_y.append(target)
        pooled_champion.append(champion)
        pooled_candidate.append(candidate)
        print(
            f"[OOF {season}] model={prediction_diff:.3e} runtime={runtime_diff:.3e} "
            f"gain={gain:+.12e} coverage={coverage:.4%}",
            flush=True,
        )
        del model, training, validation, profile, lookup
        gc.collect()
    target = np.concatenate(pooled_y)
    champion = np.concatenate(pooled_champion)
    candidate = np.concatenate(pooled_candidate)
    pooled_gain = brier(target, champion) - brier(target, candidate)
    pooled = {
        "validation_season": "pooled",
        "n": len(target),
        "reference_brier_gain": BASELINE_GAINS["pooled"],
        "production_brier_gain": pooled_gain,
        "brier_gain_abs_diff": abs(pooled_gain - BASELINE_GAINS["pooled"]),
        "passed": abs(pooled_gain - BASELINE_GAINS["pooled"]) <= OOF_GAIN_ATOL,
    }
    return {
        "folds": rows,
        "pooled": pooled,
        "passed": bool(all(row["passed"] for row in rows) and pooled["passed"]),
    }


def save_lookup(path: Path, lookup: tuple[np.ndarray, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, pitcher_ids=lookup[0], values=lookup[1])
    os.replace(temporary, path)


def train_full_production(
    train: pd.DataFrame,
    test: pd.DataFrame,
    physical_trackman: pd.DataFrame,
    production_mapping: pd.DataFrame,
    champion_meta: Mapping,
) -> tuple[Path, Path, dict, dict]:
    profile = build_physical_profiles(physical_trackman, 2024)
    lookup_frame = build_main_physical_lookup(
        production_mapping, profile, 2024, "hungarian"
    )
    started = time.monotonic()
    model_1, metadata, prediction_1, matched_1 = train_expert(
        train, test, lookup_frame, champion_meta
    )
    fit_1 = time.monotonic() - started
    lookup = lookup_arrays(lookup_frame, metadata["trackman_feature_columns"])
    runtime_1, runtime_mask = predict_trackman(
        test, "", model=model_1, metadata=metadata, lookup=lookup
    )
    runtime_diff = float(np.max(np.abs(runtime_1 - prediction_1), initial=0.0))
    model_path = CACHE / MODEL_FILE
    lookup_path = CACHE / LOOKUP_FILE
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_1, model_path, compress=("zlib", 3), protocol=5)
    save_lookup(lookup_path, lookup)
    del model_1
    gc.collect()
    loaded = joblib.load(model_path)
    serialized, serialized_mask = predict_trackman(
        test, "", model=loaded, metadata=metadata, lookup=lookup
    )
    serialization_diff = float(
        np.max(np.abs(serialized - prediction_1), initial=0.0)
    )
    del loaded
    gc.collect()
    started = time.monotonic()
    model_2, metadata_2, prediction_2, matched_2 = train_expert(
        train, test, lookup_frame, champion_meta
    )
    fit_2 = time.monotonic() - started
    reproducibility_diff = float(
        np.max(np.abs(prediction_2 - prediction_1), initial=0.0)
    )
    metadata_equal = metadata == metadata_2
    del model_2
    gc.collect()
    report = {
        "training_runs": 2,
        "fit_seconds": [fit_1, fit_2],
        "train_rows_total": len(train),
        "train_rows_matched": metadata["train_rows_matched"],
        "high_pitcher_count": len(lookup_frame),
        "test_high_row_count": int(matched_1.sum()),
        "test_high_row_coverage": float(matched_1.mean()),
        "model_reproducibility_max_abs_diff": reproducibility_diff,
        "serialization_max_abs_diff": serialization_diff,
        "runtime_formula_max_abs_diff": runtime_diff,
        "matched_mask_exact": bool(
            np.array_equal(matched_1, matched_2)
            and np.array_equal(matched_1, runtime_mask)
            and np.array_equal(matched_1, serialized_mask)
        ),
        "metadata_exact": metadata_equal,
        "model_size_bytes": model_path.stat().st_size,
        "lookup_size_bytes": lookup_path.stat().st_size,
    }
    report["passed"] = bool(
        reproducibility_diff <= OOF_PREDICTION_ATOL
        and serialization_diff <= OOF_PREDICTION_ATOL
        and runtime_diff <= OOF_PREDICTION_ATOL
        and report["matched_mask_exact"]
        and metadata_equal
    )
    return model_path, lookup_path, metadata, report


def zip_info(name: str, compression: int = zipfile.ZIP_DEFLATED) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 8, 21, 0, 0, 0))
    info.compress_type = compression
    info.external_attr = 0o644 << 16
    return info


def patch_champion_script(source: str) -> str:
    import_anchor = "from extratrees_runtime import apply_extratrees_blend\n"
    predict_anchor = "def predict(test, model, meta, verbose=True):"
    main_anchor = "\ndef main():\n"
    for anchor in (import_anchor, predict_anchor, main_anchor):
        if source.count(anchor) != 1:
            raise ValueError(f"champion patch anchor count != 1: {anchor!r}")
    patched = source.replace(
        import_anchor,
        import_anchor + "from trackman_runtime import apply_trackman_blend\n",
        1,
    ).replace(
        predict_anchor,
        "def predict_current_champion(test, model, meta, verbose=True):",
        1,
    )
    wrapper = '''

def predict(test, model, meta, verbose=True):
    """Frozen endpoint: current ET25 champion, then HIGH-only TrackMan w=.10."""
    champion = predict_current_champion(test, model, meta, verbose=verbose)
    return apply_trackman_blend(champion, test, "./model")
'''
    return patched.replace(main_anchor, wrapper + main_anchor, 1)


def build_candidate_zip(
    model_path: Path, lookup_path: Path, metadata: Mapping
) -> None:
    if CANDIDATE.name != EXPECTED_FILENAME or len(CANDIDATE.name) > MAX_FILENAME_LENGTH:
        raise ValueError("candidate filename hard gate failed")
    runtime = (ROOT / "scripts" / "trackman_runtime.py").read_bytes()
    temporary = CANDIDATE.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(CHAMPION) as source, zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as output:
        champion_script = source.read("script.py").decode("utf-8")
        patched_script = patch_champion_script(champion_script).encode("utf-8")
        for original in source.infolist():
            if original.is_dir():
                continue
            payload = source.read(original.filename)
            if original.filename == "script.py":
                payload = patched_script
            output.writestr(zip_info(original.filename), payload)
        output.writestr(zip_info("trackman_runtime.py"), runtime)
        output.writestr(
            zip_info(f"model/{META_FILE}"),
            json.dumps(json_safe(metadata), ensure_ascii=False, indent=2).encode(),
        )
        for source_path, member in (
            (model_path, f"model/{MODEL_FILE}"),
            (lookup_path, f"model/{LOOKUP_FILE}"),
        ):
            with source_path.open("rb") as source_file, output.open(
                zip_info(member, zipfile.ZIP_STORED), "w"
            ) as target:
                shutil.copyfileobj(source_file, target, length=16 * 1024 * 1024)
    os.replace(temporary, CANDIDATE)


def run_raw_prediction(extracted: Path, test: pd.DataFrame, predictor: str) -> np.ndarray:
    if predictor not in {"predict", "predict_current_champion"}:
        raise ValueError(predictor)
    data = extracted / "data"
    output = extracted / "output"
    data.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    test.to_csv(data / "test.csv", index=False, encoding="utf-8-sig")
    code = f'''\
import json
import numpy as np
import pandas as pd
import script
test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
with open("./model/meta.json", encoding="utf-8") as handle:
    meta = json.load(handle)
models = script.load_models("./model", meta)
prediction = script.{predictor}(test, models, meta, verbose=False)
np.save("./output/raw.npy", prediction)
'''
    env, _ = inference_environment()
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=extracted,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stdout[-2000:] + process.stderr[-4000:])
    return np.load(output / "raw.npy")


def prediction_map(output: pd.DataFrame) -> dict[str, float]:
    return dict(zip(output["row_id"], output[TARGET].astype(float)))


def production_row_independence(extracted: Path, test: pd.DataFrame) -> dict:
    variants = {
        "single": test.iloc[[0]].copy(),
        "full": test.copy(),
        "shuffle": test.sample(frac=1.0, random_state=42).reset_index(drop=True),
        "subset": test.iloc[[0, 2, 4]].reset_index(drop=True),
        "reverse": test.iloc[::-1].reset_index(drop=True),
    }
    full = run_submission(extracted, variants["full"])
    reference = prediction_map(full["output"])
    details = {}
    maximum = 0.0
    for name, frame in variants.items():
        run = full if name == "full" else run_submission(extracted, frame)
        values = prediction_map(run["output"])
        current = max(
            (abs(value - reference[row_id]) for row_id, value in values.items()),
            default=0.0,
        )
        details[name] = {"rows": len(frame), "max_abs_diff": current}
        maximum = max(maximum, current)
    return {
        "variants": details,
        "max_abs_diff": maximum,
        "atol": ROW_INDEPENDENCE_ATOL,
        "passed": maximum == 0.0,
        "full_output_sha256": full["output_sha256"],
    }


def determinism_audit(extracted: Path, test: pd.DataFrame) -> dict:
    runs = [run_submission(extracted, test, timeout=600) for _ in range(3)]
    hashes = [run["output_sha256"] for run in runs]
    arrays = [run["output"][TARGET].to_numpy(dtype="float64") for run in runs]
    difference = max(
        (float(np.max(np.abs(values - arrays[0]))) for values in arrays[1:]),
        default=0.0,
    )
    return {
        "runs": 3,
        "submission_sha256": hashes,
        "byte_identical": len(set(hashes)) == 1,
        "prediction_max_abs_diff": difference,
        "passed": len(set(hashes)) == 1 and difference == 0.0,
    }


def timed_isolated_run(extracted: Path, test: pd.DataFrame) -> dict:
    data = extracted / "data"
    output = extracted / "output"
    data.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    test.to_csv(data / "test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"row_id": test["row_id"], TARGET: np.full(len(test), 0.5)}
    ).to_csv(data / "sample_submission.csv", index=False, encoding="utf-8-sig")
    (output / "submission.csv").unlink(missing_ok=True)
    env, warnings = inference_environment()
    wrapper = '''
import json
import platform
import resource
import subprocess
import sys
import time
started = time.perf_counter()
process = subprocess.run(
    [sys.executable, "script.py"], capture_output=True, text=True, check=False
)
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
peak_bytes = int(peak if platform.system() == "Darwin" else peak * 1024)
print("__TM_AUDIT__" + json.dumps({
    "returncode": process.returncode,
    "wall_seconds": time.perf_counter() - started,
    "peak_rss_bytes": peak_bytes,
    "stdout_tail": process.stdout[-2000:],
    "stderr_tail": process.stderr[-4000:],
}))
'''
    process = subprocess.run(
        [sys.executable, "-c", wrapper],
        cwd=extracted,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stdout[-2000:] + process.stderr[-4000:])
    marker = next(
        (line for line in process.stdout.splitlines() if line.startswith("__TM_AUDIT__")),
        None,
    )
    if marker is None:
        raise RuntimeError("could not parse isolated runtime report")
    measured = json.loads(marker.removeprefix("__TM_AUDIT__"))
    if measured["returncode"]:
        raise RuntimeError(measured["stdout_tail"] + measured["stderr_tail"])
    submission = pd.read_csv(output / "submission.csv")
    validation = validate_submission(submission, test["row_id"].tolist())
    return {
        "rows": len(test),
        "wall_seconds": measured["wall_seconds"],
        "peak_rss_bytes": measured["peak_rss_bytes"],
        "submission_size_bytes": (output / "submission.csv").stat().st_size,
        "submission_validation": validation,
        "environment_warnings": warnings,
        "passed": validation["passed"],
    }


def package_audit(model_path: Path, lookup_path: Path) -> dict:
    required = {
        "script.py",
        "requirements.txt",
        "trackman_runtime.py",
        f"model/{MODEL_FILE}",
        f"model/{META_FILE}",
        f"model/{LOOKUP_FILE}",
    }
    with zipfile.ZipFile(CHAMPION) as champion, zipfile.ZipFile(CANDIDATE) as archive:
        names = {info.filename for info in archive.infolist() if not info.is_dir()}
        champion_names = {
            info.filename for info in champion.infolist() if not info.is_dir()
        }
        crc_bad = archive.testzip()
        experiment_hits = sorted(
            name
            for name in names
            if name.startswith(("artifacts/", "docs/", "tests/", "experiments/"))
            or "/cache/" in f"/{name}"
            or name.endswith(".csv")
        )
        absolute_hits = []
        for name in names:
            if Path(name).suffix.lower() not in {".py", ".json", ".txt"}:
                continue
            text = archive.read(name).decode("utf-8", errors="replace")
            if any(token in text for token in ("/Users/", "/home/", "../open/")):
                absolute_hits.append(name)
        requirements_equal = (
            archive.read("requirements.txt") == champion.read("requirements.txt")
        )
        preserved = all(
            name == "script.py" or archive.read(name) == champion.read(name)
            for name in champion_names
        )
    result = {
        "filename": CANDIDATE.name,
        "filename_length": len(CANDIDATE.name),
        "zip_size_bytes": CANDIDATE.stat().st_size,
        "champion_zip_size_bytes": CHAMPION.stat().st_size,
        "model_size_bytes": model_path.stat().st_size,
        "lookup_size_bytes": lookup_path.stat().st_size,
        "zip_crc_bad_member": crc_bad,
        "required_members_present": required.issubset(names),
        "requirements_byte_identical": requirements_equal,
        "champion_payloads_preserved_except_script": preserved,
        "experiment_artifact_hits": experiment_hits,
        "absolute_path_hits": absolute_hits,
        "new_members": sorted(names - champion_names),
        "package_guard_bytes": PACKAGE_GUARD_BYTES,
        "within_package_guard": CANDIDATE.stat().st_size < PACKAGE_GUARD_BYTES,
    }
    result["passed"] = bool(
        result["filename_length"] <= MAX_FILENAME_LENGTH
        and crc_bad is None
        and result["required_members_present"]
        and requirements_equal
        and preserved
        and not experiment_hits
        and not absolute_hits
        and result["within_package_guard"]
    )
    return result


def build_audit_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    production_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build audit-only rows that exercise both HIGH and unmatched routing."""
    high_ids = production_mapping.loc[
        production_mapping["confidence"].eq("HIGH"), "main_pitcher_id"
    ]
    matched = (
        train[train["pitcher_id"].isin(high_ids)]
        .groupby("pitcher_id", observed=True, sort=True)
        .tail(1)
    )
    matched = matched[test.columns]
    audit = pd.concat([test, matched], ignore_index=True)
    audit["row_id"] = [f"TM_AUDIT_{index:06d}" for index in range(len(audit))]
    repeats = math.ceil(RUNTIME_ROWS / len(audit))
    runtime = pd.concat([audit] * repeats, ignore_index=True).iloc[:RUNTIME_ROWS].copy()
    runtime["row_id"] = [f"TM_RUNTIME_{index:09d}" for index in range(len(runtime))]
    return audit, runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--trackman", type=Path, default=DEFAULT_TRACKMAN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--reuse-parity",
        action="store_true",
        help="reuse already-passed mapping/OOF JSON and frozen mapping rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.train,
        args.trackman,
        args.data_dir / "test.csv",
        args.data_dir / "sample_submission.csv",
        CHAMPION,
        MAPPING_REFERENCE,
    ):
        if not path.is_file():
            raise SystemExit(f"required file missing: {path}")
    if CANDIDATE.name != EXPECTED_FILENAME or len(CANDIDATE.name) > MAX_FILENAME_LENGTH:
        raise SystemExit("candidate filename hard gate failed")
    args.output.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    champion_before = sha256_path(CHAMPION)
    if champion_before != CHAMPION_SHA256:
        raise SystemExit(f"immutable champion checksum mismatch: {champion_before}")
    reference_before = verify_reference_hashes()
    if reference_before != REFERENCE_HASHES:
        raise SystemExit("immutable TrackMan reference mismatch")

    if args.reuse_parity:
        mapping = json.loads((args.output / "mapping_parity.json").read_text())
        frozen_mapping = pd.read_csv(MAPPING_REFERENCE)
        fold_mappings = {
            cutoff: frozen_mapping[
                frozen_mapping["cutoff"].eq(cutoff)
                & frozen_mapping["method"].eq("hungarian")
            ].reset_index(drop=True)
            for cutoff in (2021, 2022, 2023)
        }
        production_mapping = frozen_mapping[
            frozen_mapping["cutoff"].eq(2024)
            & frozen_mapping["method"].eq("hungarian")
        ].reset_index(drop=True)
        print("[1/5] reuse passed mapping parity", flush=True)
    else:
        print("[1/5] regenerate and audit frozen Hungarian mappings", flush=True)
        mapping, fold_mappings, production_mapping = mapping_parity_audit(
            args.train, args.trackman
        )
        write_json(args.output / "mapping_parity.json", mapping)
    if not mapping["passed"]:
        raise RuntimeError("mapping parity failed before model/package build")

    train = pd.read_csv(args.train, encoding="utf-8-sig")
    physical_trackman = read_trackman_physical(args.trackman)
    champion_meta = read_champion_meta()
    if args.reuse_parity:
        print("[2/5] reuse passed temporal OOF parity", flush=True)
        oof = json.loads((args.output / "oof_parity.json").read_text())
    else:
        print("[2/5] reproduce frozen temporal OOF", flush=True)
        oof = oof_parity_audit(
            train, physical_trackman, fold_mappings, champion_meta
        )
        write_json(args.output / "oof_parity.json", oof)
    if not oof["passed"]:
        raise RuntimeError("OOF parity failed before production package build")

    print("[3/5] train and serialize full-cutoff frozen expert", flush=True)
    test = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(
        args.data_dir / "sample_submission.csv", encoding="utf-8-sig"
    )
    if test["row_id"].tolist() != sample["row_id"].tolist():
        raise RuntimeError("test/sample row order mismatch")
    model_path, lookup_path, metadata, full_training = train_full_production(
        train,
        test,
        physical_trackman,
        production_mapping,
        champion_meta,
    )
    if not full_training["passed"]:
        raise RuntimeError("full production model reproducibility failed")
    write_json(CACHE / META_FILE, metadata)
    build_candidate_zip(model_path, lookup_path, metadata)
    print(
        f"built {CANDIDATE.name} ({CANDIDATE.stat().st_size:,} bytes)",
        flush=True,
    )
    audit_frame, runtime_frame = build_audit_frames(
        train, test, production_mapping
    )
    # Keep the audit parent small before spawning memory-heavy champion inference.
    del train, physical_trackman, production_mapping, fold_mappings, champion_meta
    gc.collect()

    print("[4/5] isolated parity, row-independence and determinism", flush=True)
    with tempfile.TemporaryDirectory(prefix="tm_champion_") as champion_tmp, tempfile.TemporaryDirectory(
        prefix="tm_candidate_"
    ) as candidate_tmp:
        champion_root = Path(champion_tmp)
        candidate_root = Path(candidate_tmp)
        safe_extract(CHAMPION, champion_root)
        safe_extract(CANDIDATE, candidate_root)
        champion_test_prediction = run_raw_prediction(champion_root, test, "predict")
        candidate_test_component = run_raw_prediction(
            candidate_root, test, "predict_current_champion"
        )
        component_raw_diff = float(
            np.max(
                np.abs(candidate_test_component - champion_test_prediction),
                initial=0.0,
            )
        )
        component_diff = float(
            np.max(
                np.abs(
                    np.round(candidate_test_component, MATCHED_OUTPUT_DECIMALS)
                    - np.round(champion_test_prediction, MATCHED_OUTPUT_DECIMALS)
                ),
                initial=0.0,
            )
        )
        champion_prediction = run_raw_prediction(
            candidate_root, audit_frame, "predict_current_champion"
        )
        model = joblib.load(model_path)
        lookup_payload = (
            np.load(lookup_path, allow_pickle=False)["pitcher_ids"],
            np.load(lookup_path, allow_pickle=False)["values"],
        )
        expert, matched = predict_trackman(
            audit_frame,
            "",
            model=model,
            metadata=metadata,
            lookup=lookup_payload,
        )
        del model
        gc.collect()
        expected = champion_prediction.copy()
        unrounded_expected = (
            CHAMPION_WEIGHT * champion_prediction[matched] + TRACKMAN_WEIGHT * expert
        )
        expected[matched] = np.round(
            unrounded_expected, decimals=MATCHED_OUTPUT_DECIMALS
        )
        candidate_prediction = run_raw_prediction(candidate_root, audit_frame, "predict")
        formula_diff = float(
            np.max(np.abs(candidate_prediction - expected), initial=0.0)
        )
        unmatched_diff = float(
            np.max(
                np.abs(candidate_prediction[~matched] - champion_prediction[~matched]),
                initial=0.0,
            )
        )
        row_independence = production_row_independence(candidate_root, audit_frame)
        row_independence.update(
            {
                "champion_component_max_abs_diff": component_diff,
                "champion_component_raw_max_abs_diff": component_raw_diff,
                "champion_component_serialization_decimals": MATCHED_OUTPUT_DECIMALS,
                "formula_max_abs_diff": formula_diff,
                "audit_rows": len(audit_frame),
                "matched_rows": int(matched.sum()),
                "unmatched_rows": int((~matched).sum()),
                "unmatched_max_abs_diff": unmatched_diff,
            }
        )
        row_independence["passed"] = bool(
            row_independence["passed"]
            and component_diff == 0.0
            and formula_diff <= OOF_PREDICTION_ATOL
            and unmatched_diff == 0.0
        )
        write_json(args.output / "row_independence.json", row_independence)
        determinism = determinism_audit(candidate_root, test)
        write_json(args.output / "determinism.json", determinism)
        champion_runtime = timed_isolated_run(champion_root, runtime_frame)
        candidate_runtime = timed_isolated_run(candidate_root, runtime_frame)
        runtime = {
            "champion": champion_runtime,
            "candidate": candidate_runtime,
            "runtime_delta_seconds": (
                candidate_runtime["wall_seconds"] - champion_runtime["wall_seconds"]
            ),
            "peak_rss_delta_bytes": (
                candidate_runtime["peak_rss_bytes"]
                - champion_runtime["peak_rss_bytes"]
            ),
            "internal_targets": {
                "runtime_seconds": RUNTIME_GUARD_SECONDS,
                "peak_rss_bytes": RSS_GUARD_BYTES,
            },
            "passed": bool(
                candidate_runtime["passed"]
                and candidate_runtime["wall_seconds"] < RUNTIME_GUARD_SECONDS
                and candidate_runtime["peak_rss_bytes"] < RSS_GUARD_BYTES
            ),
        }
        write_json(args.output / "runtime.json", runtime)

    print("[5/5] package/rule audit and immutable checks", flush=True)
    package = package_audit(model_path, lookup_path)
    package["candidate_sha256"] = sha256_path(CANDIDATE)
    package["submission_validation"] = candidate_runtime["submission_validation"]
    package["competition_rule_audit"] = {
        "data_sources": ["official train.csv", "official trackman_history.csv"],
        "external_identity_information": False,
        "test_other_rows_used": False,
        "test_distribution_used": False,
        "row_independent_prediction": row_independence["passed"],
        "crosswalk": "official-data-only Hungarian HIGH lookup frozen at cutoff 2024",
    }
    package["passed"] = bool(
        package["passed"] and package["submission_validation"]["passed"]
    )
    write_json(args.output / "package_audit.json", package)

    champion_after = sha256_path(CHAMPION)
    reference_after = verify_reference_hashes()
    immutable = bool(
        champion_after == champion_before == CHAMPION_SHA256
        and reference_after == reference_before == REFERENCE_HASHES
    )
    technical = {
        "mapping_parity": mapping["passed"],
        "oof_production_parity": oof["passed"],
        "full_train_reproducibility": full_training["passed"],
        "champion_component_parity_exact": component_diff == 0.0,
        "unmatched_identity_exact": unmatched_diff == 0.0,
        "row_independence_exact": row_independence["passed"],
        "determinism_byte_identical": determinism["passed"],
        "package_and_rule_audit": package["passed"],
        "immutable_references": immutable,
    }
    risk = {
        "zip_under_500_mib": package["within_package_guard"],
        "runtime_under_20_seconds": candidate_runtime["wall_seconds"] < RUNTIME_GUARD_SECONDS,
        "peak_rss_under_4_gib": candidate_runtime["peak_rss_bytes"] < RSS_GUARD_BYTES,
    }
    if not all(technical.values()):
        verdict = "C. PARITY / RULE / SAFETY FAILURE"
    elif not all(risk.values()):
        verdict = "B. TECHNICALLY VALID BUT DEPLOYMENT RISK"
    else:
        verdict = "A. READY FOR ONE LB SUBMISSION"
    summary = {
        "verdict": verdict,
        "candidate": {
            "file": str(CANDIDATE),
            "filename": CANDIDATE.name,
            "filename_length": len(CANDIDATE.name),
            "sha256": sha256_path(CANDIDATE),
            "size_bytes": CANDIDATE.stat().st_size,
            "formula": "unmatched=champion; HIGH=0.90*champion+0.10*TrackMan expert",
        },
        "champion": {
            "file": str(CHAMPION),
            "sha256_before": champion_before,
            "sha256_after": champion_after,
            "immutable": champion_after == champion_before,
        },
        "mapping": mapping,
        "oof": oof,
        "full_training": full_training,
        "row_independence": row_independence,
        "determinism": determinism,
        "runtime": runtime,
        "package": package,
        "technical_conditions": technical,
        "risk_conditions": risk,
        "test_other_rows_used_for_features": False,
        "leaderboard_submission_performed": False,
    }
    write_json(args.output / "summary.json", summary)
    print(f"candidate sha256: {summary['candidate']['sha256']}", flush=True)
    print(f"champion after: {champion_after}", flush=True)
    print(verdict, flush=True)


if __name__ == "__main__":
    main()
