"""Build and audit the frozen 99% champion + 1% ExtraTrees candidate.

No parameter, weight, calibration, or seed search is performed.  The original
champion is read-only; the candidate is always written to a separate path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
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
    CHAMPION_WEIGHT,
    EXTRA_WEIGHT,
    META_FILE,
    MODEL_FILE,
    build_hgb52_features,
    fit_preprocessor,
    predict_extratrees,
    transform_features,
)


CHAMPION = ROOT / "artifacts" / "submit_r10_pitcher_w418194.zip"
CANDIDATE = ROOT / "artifacts" / "submit_r10_pitcher_w418194_extra01.zip"
EXPECTED_CHAMPION_SHA256 = (
    "ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47"
)
EXPECTED_CHAMPION_SIZE = 3_345_014
DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_DATA_DIR = Path("/Users/wooh/Documents/dev/open/data")
DEFAULT_OOF_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
REFERENCE_DIR = ROOT / "artifacts" / "extratrees_complement"
OUTPUT_DIR = ROOT / "artifacts" / "extratrees_production"
CACHE_DIR = OUTPUT_DIR / "cache"

TARGET = "control_success"
VALIDATION_SEASONS = (2022, 2023, 2024)
N_ESTIMATORS = 300
RANDOM_STATE = 42
MIN_SAMPLES_LEAF = 8
MAX_FEATURES = "sqrt"
BOOTSTRAP = False
N_JOBS = -1
FROZEN_GAINS = {
    "2022": 1.1302609010666043e-06,
    "2023": 7.195466559029029e-06,
    "2024": 2.0101703540886806e-06,
    "pooled": 3.4239161593452305e-06,
}
OOF_PREDICTION_ATOL = 1e-12
OOF_GAIN_ATOL = 5e-12
FULL_TRAIN_PREDICTION_ATOL = 1e-12
ENDPOINT_FORMULA_ATOL = 1e-12
CHAMPION_PARITY_ATOL = 0.0
ROW_INDEPENDENCE_ATOL = 1e-12
PACKAGE_GUARD_BYTES = 1024**3
CHAMPION_RUNTIME_SECONDS = 8.963606916833669
CHAMPION_PEAK_RSS_BYTES = 1_440_284_672
RUNTIME_RATIO_GUARD = 5.0
MEMORY_RATIO_GUARD = 4.0
RUNTIME_ROWS = 253_507


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_zip_json(path: Path, member: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(member))


def frozen_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )


def validate_recipe(champion_meta: Mapping, reference_summary: Mapping) -> list[str]:
    blockers = []
    hgb = [member for member in champion_meta["members"] if member["name"] == "hgb"]
    if len(hgb) != 1:
        blockers.append("champion HGB feature contract is missing or ambiguous")
        return blockers
    feature_columns = hgb[0]["feature_columns"]
    reference_contract = reference_summary["feature_contract"]
    if feature_columns != reference_contract["feature_columns"]:
        blockers.append("OOF/champion feature column order mismatch")
    if len(feature_columns) != 52:
        blockers.append(f"expected 52 features, got {len(feature_columns)}")
    forbidden = {"row_id", TARGET, "pitcher_id", "batter_id"}
    if forbidden & set(feature_columns):
        blockers.append("raw ID/target leaked into ExtraTrees feature contract")
    expected = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": None,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "max_features": MAX_FEATURES,
        "bootstrap": BOOTSTRAP,
        "n_jobs": N_JOBS,
        "random_state": RANDOM_STATE,
    }
    actual = reference_summary["model"]["parameters"]
    for key, value in expected.items():
        if actual.get(key) != value:
            blockers.append(f"frozen model parameter mismatch: {key}")
    if [item["weight"] for item in reference_summary["passing_weights"]] != [
        EXTRA_WEIGHT
    ]:
        blockers.append("OOF passing weight is not exactly 0.01")
    if sklearn.__version__ != "1.8.0":
        blockers.append(f"local sklearn must be 1.8.0, got {sklearn.__version__}")
    return blockers


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    return float(np.mean((y - p) ** 2))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save_prediction_cache(path: Path, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, prediction=prediction)
    os.replace(temporary, path)


def validate_feature_order(features: pd.DataFrame, champion_meta: Mapping) -> None:
    hgb = next(
        member for member in champion_meta["members"] if member["name"] == "hgb"
    )
    if list(features.columns) != hgb["feature_columns"]:
        raise ValueError("production feature order differs from champion HGB contract")


def production_oof_parity(
    train: pd.DataFrame,
    champion_meta: Mapping,
    oof_dir: Path,
    force: bool,
) -> dict:
    rows = []
    pooled_target = []
    pooled_champion = []
    pooled_candidate = []
    blockers = []
    for season in VALIDATION_SEASONS:
        train_fold = train[train["season"] <= season - 1]
        validation = train[train["season"] == season]
        source = load_npz(oof_dir / f"season_{season}.npz")
        reference = load_npz(
            REFERENCE_DIR / "cache" / f"season_{season}_trees{N_ESTIMATORS}.npz"
        )["prediction"].astype("float64")
        target = validation[TARGET].to_numpy(dtype="float64")
        if not np.array_equal(target, source["y"].astype("float64")):
            raise ValueError(f"train/OOF order mismatch for {season}")
        cache_path = CACHE_DIR / f"oof_production_{season}.npz"
        if cache_path.is_file() and not force:
            production = load_npz(cache_path)["prediction"].astype("float64")
            fit_seconds = None
            print(f"[{season}] reuse production OOF parity cache", flush=True)
        else:
            train_features = build_hgb52_features(
                train_fold, champion_meta["cat_levels"]
            )
            validation_features = build_hgb52_features(
                validation, champion_meta["cat_levels"]
            )
            validate_feature_order(train_features, champion_meta)
            preprocessor = fit_preprocessor(train_features)
            x_train = transform_features(train_features, preprocessor)
            x_validation = transform_features(validation_features, preprocessor)
            del train_features, validation_features
            gc.collect()
            model = frozen_model()
            started = time.monotonic()
            model.fit(x_train, train_fold[TARGET].to_numpy(dtype="int8"))
            fit_seconds = time.monotonic() - started
            production = model.predict_proba(x_validation)[:, 1].astype("float64")
            save_prediction_cache(cache_path, production)
            del model, x_train, x_validation
            gc.collect()
        if production.shape != reference.shape:
            raise ValueError(f"production/reference OOF length mismatch for {season}")
        difference = np.abs(production - reference)
        champion = source["champion_final"].astype("float64")
        candidate = CHAMPION_WEIGHT * champion + EXTRA_WEIGHT * production
        gain = brier(target, champion) - brier(target, candidate)
        row = {
            "validation_season": season,
            "n": len(target),
            "max_abs_prediction_diff": float(np.max(difference)),
            "mean_abs_prediction_diff": float(np.mean(difference)),
            "reference_brier_gain": FROZEN_GAINS[str(season)],
            "production_brier_gain": gain,
            "brier_gain_abs_diff": abs(gain - FROZEN_GAINS[str(season)]),
            "fit_seconds": fit_seconds,
        }
        row["passed"] = bool(
            row["max_abs_prediction_diff"] <= OOF_PREDICTION_ATOL
            and row["brier_gain_abs_diff"] <= OOF_GAIN_ATOL
            and gain > 0.0
        )
        if not row["passed"]:
            blockers.append(f"OOF production parity failed for {season}")
        rows.append(row)
        pooled_target.append(target)
        pooled_champion.append(champion)
        pooled_candidate.append(candidate)
        print(
            f"[{season}] OOF parity max={row['max_abs_prediction_diff']:.3e}, "
            f"gain={gain:+.12e}",
            flush=True,
        )
        del train_fold, validation, source
        gc.collect()
    target = np.concatenate(pooled_target)
    champion = np.concatenate(pooled_champion)
    candidate = np.concatenate(pooled_candidate)
    pooled_gain = brier(target, champion) - brier(target, candidate)
    pooled = {
        "validation_season": "pooled",
        "n": len(target),
        "reference_brier_gain": FROZEN_GAINS["pooled"],
        "production_brier_gain": pooled_gain,
        "brier_gain_abs_diff": abs(pooled_gain - FROZEN_GAINS["pooled"]),
        "passed": bool(
            pooled_gain > 0.0
            and abs(pooled_gain - FROZEN_GAINS["pooled"]) <= OOF_GAIN_ATOL
        ),
    }
    if not pooled["passed"]:
        blockers.append("pooled OOF production parity failed")
    return {"folds": rows, "pooled": pooled, "blockers": blockers, "passed": not blockers}


def extra_row_independence(
    model,
    metadata: Mapping,
    test: pd.DataFrame,
) -> dict:
    variants = {
        "single": test.iloc[[0]].copy(),
        "full": test.copy(),
        "shuffled": test.sample(frac=1.0, random_state=42).reset_index(drop=True),
        "subset": test.iloc[[0, 2, 4]].reset_index(drop=True),
        "reverse": test.iloc[::-1].reset_index(drop=True),
    }
    full_prediction = predict_extratrees(
        variants["full"], "", model=model, metadata=metadata
    )
    reference = dict(zip(variants["full"]["row_id"], full_prediction))
    results = {}
    maximum = 0.0
    for name, frame in variants.items():
        prediction = predict_extratrees(frame, "", model=model, metadata=metadata)
        values = dict(zip(frame["row_id"], prediction))
        differences = [
            abs(float(value) - float(reference[row_id]))
            for row_id, value in values.items()
        ]
        current = max(differences, default=0.0)
        maximum = max(maximum, current)
        results[name] = {"rows": len(frame), "max_abs_diff": current}
    return {
        "variants": results,
        "max_abs_diff": maximum,
        "atol": ROW_INDEPENDENCE_ATOL,
        "passed": maximum <= ROW_INDEPENDENCE_ATOL,
    }


def train_full_twice(
    train: pd.DataFrame,
    test: pd.DataFrame,
    champion_meta: Mapping,
    reuse_first_model: bool = False,
) -> tuple[Path, dict, np.ndarray, dict]:
    features = build_hgb52_features(train, champion_meta["cat_levels"])
    test_features = build_hgb52_features(test, champion_meta["cat_levels"])
    validate_feature_order(features, champion_meta)
    preprocessor = fit_preprocessor(features)
    x_train = transform_features(features, preprocessor)
    x_test = transform_features(test_features, preprocessor)
    target = train[TARGET].to_numpy(dtype="int8")
    del features, test_features
    gc.collect()

    model_path = CACHE_DIR / MODEL_FILE
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if reuse_first_model:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"cannot reuse missing full-train model: {model_path}"
            )
        model_1 = joblib.load(model_path)
        fit_1 = None
    else:
        model_1 = frozen_model()
        started = time.monotonic()
        model_1.fit(x_train, target)
        fit_1 = time.monotonic() - started
    prediction_1 = model_1.predict_proba(x_test)[:, 1].astype("float64")
    metadata = {
        "schema_version": 1,
        "candidate": "0.99 champion + 0.01 ExtraTrees",
        "champion_weight": CHAMPION_WEIGHT,
        "extra_weight": EXTRA_WEIGHT,
        "model_file": MODEL_FILE,
        "feature_columns": preprocessor["feature_columns"],
        "cat_levels": champion_meta["cat_levels"],
        "preprocessor": preprocessor,
        "model_class": "sklearn.ensemble.ExtraTreesClassifier",
        "model_params": frozen_model().get_params(),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "train_rows": len(train),
        "train_max_season": int(train["season"].max()),
        "target": TARGET,
        "raw_player_ids_included": False,
        "test_statistics_used": False,
    }
    extra_independence = extra_row_independence(model_1, metadata, test)
    if not reuse_first_model:
        joblib.dump(model_1, model_path, compress=("zlib", 3), protocol=5)
    serialized_size = model_path.stat().st_size
    del model_1
    gc.collect()
    loaded = joblib.load(model_path)
    loaded_prediction = loaded.predict_proba(x_test)[:, 1].astype("float64")
    serialization_diff = float(np.max(np.abs(loaded_prediction - prediction_1)))
    del loaded
    gc.collect()

    model_2 = frozen_model()
    started = time.monotonic()
    model_2.fit(x_train, target)
    fit_2 = time.monotonic() - started
    prediction_2 = model_2.predict_proba(x_test)[:, 1].astype("float64")
    reproducibility_diff = float(np.max(np.abs(prediction_2 - prediction_1)))
    del model_2, x_train, x_test, target
    gc.collect()
    report = {
        "runs": 2,
        "fit_seconds": [fit_1, fit_2],
        "reused_existing_first_fit": reuse_first_model,
        "test_prediction_max_abs_diff": reproducibility_diff,
        "serialized_reload_max_abs_diff": serialization_diff,
        "prediction_atol": FULL_TRAIN_PREDICTION_ATOL,
        "model_file_size_bytes": serialized_size,
        "prediction_identical": reproducibility_diff == 0.0,
        "prediction_effectively_identical": (
            reproducibility_diff <= FULL_TRAIN_PREDICTION_ATOL
        ),
        "serialization_prediction_identical": serialization_diff == 0.0,
        "passed": bool(
            reproducibility_diff <= FULL_TRAIN_PREDICTION_ATOL
            and serialization_diff <= FULL_TRAIN_PREDICTION_ATOL
        ),
    }
    print(
        "full-train parity "
        f"max={reproducibility_diff:.3e}, "
        f"serialized={serialization_diff:.3e}, "
        f"atol={FULL_TRAIN_PREDICTION_ATOL:.1e}",
        flush=True,
    )
    return model_path, metadata, prediction_1, {
        "training": report,
        "extra_row_independence": extra_independence,
    }


def patch_champion_script(source: str) -> str:
    import_anchor = "from coldstart_runtime import apply_coldstart_expert\n"
    predict_anchor = "def predict(test, model, meta, verbose=True):"
    main_anchor = "\ndef main():\n"
    for anchor in (import_anchor, predict_anchor, main_anchor):
        if source.count(anchor) != 1:
            raise ValueError(f"champion script patch anchor count != 1: {anchor!r}")
    patched = source.replace(
        import_anchor,
        import_anchor + "from extratrees_runtime import apply_extratrees_blend\n",
        1,
    )
    patched = patched.replace(
        predict_anchor,
        "def predict_champion(test, model, meta, verbose=True):",
        1,
    )
    wrapper = """

def predict(test, model, meta, verbose=True):
    \"\"\"Frozen endpoint: original champion first, then 99/1 ExtraTrees blend.\"\"\"
    champion = predict_champion(test, model, meta, verbose=verbose)
    return apply_extratrees_blend(champion, test, \"./model\")
"""
    return patched.replace(main_anchor, wrapper + main_anchor, 1)


def zip_info(name: str, compression: int = zipfile.ZIP_DEFLATED) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 8, 20, 0, 0, 0))
    info.compress_type = compression
    info.external_attr = 0o644 << 16
    return info


def build_candidate_zip(
    champion: Path,
    candidate: Path,
    model_path: Path,
    metadata: Mapping,
) -> None:
    runtime_source = (ROOT / "scripts" / "extratrees_runtime.py").read_bytes()
    with zipfile.ZipFile(champion) as source:
        champion_script = source.read("script.py").decode("utf-8")
        patched_script = patch_champion_script(champion_script).encode("utf-8")
        with zipfile.ZipFile(
            candidate, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as output:
            for original in source.infolist():
                if original.is_dir():
                    continue
                payload = source.read(original.filename)
                if original.filename == "script.py":
                    payload = patched_script
                output.writestr(zip_info(original.filename), payload)
            output.writestr(zip_info("extratrees_runtime.py"), runtime_source)
            output.writestr(
                zip_info(f"model/{META_FILE}"),
                json.dumps(
                    json_safe(metadata), ensure_ascii=False, indent=2
                ).encode("utf-8"),
            )
            with model_path.open("rb") as source_model, output.open(
                zip_info(f"model/{MODEL_FILE}", zipfile.ZIP_STORED), "w"
            ) as target_model:
                shutil.copyfileobj(source_model, target_model, length=16 * 1024 * 1024)


def run_raw_prediction(
    extracted: Path,
    test: pd.DataFrame,
    predictor_name: str,
) -> np.ndarray:
    if predictor_name not in {"predict", "predict_champion"}:
        raise ValueError(f"unsupported raw predictor: {predictor_name}")
    data_dir = extracted / "data"
    output_dir = extracted / "output"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    test.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
    code = """
import json
import numpy as np
import pandas as pd
import script
test = pd.read_csv('./data/test.csv', encoding='utf-8-sig')
with open('./model/meta.json', encoding='utf-8') as handle:
    meta = json.load(handle)
models = script.load_models('./model', meta)
prediction = script.predict_champion(test, models, meta, verbose=False)
np.save('./output/raw_prediction.npy', prediction)
""".replace("script.predict_champion", f"script.{predictor_name}")
    env, _ = inference_environment()
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=extracted,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"raw {predictor_name} execution failed:\n"
            + process.stdout[-2000:]
            + process.stderr[-4000:]
        )
    return np.load(output_dir / "raw_prediction.npy")


def run_champion_component(extracted: Path, test: pd.DataFrame) -> np.ndarray:
    return run_raw_prediction(extracted, test, "predict_champion")


def prediction_map(output: pd.DataFrame) -> dict[str, float]:
    return dict(zip(output["row_id"], output[TARGET].astype(float)))


def candidate_row_independence(extracted: Path, test: pd.DataFrame) -> dict:
    variants = {
        "single": test.iloc[[0]].copy(),
        "full": test.copy(),
        "shuffled": test.sample(frac=1.0, random_state=42).reset_index(drop=True),
        "subset": test.iloc[[0, 2, 4]].reset_index(drop=True),
        "reverse": test.iloc[::-1].reset_index(drop=True),
    }
    full = run_submission(extracted, variants["full"])
    reference = prediction_map(full["output"])
    results = {}
    maximum = 0.0
    for name, frame in variants.items():
        run = full if name == "full" else run_submission(extracted, frame)
        values = prediction_map(run["output"])
        differences = [
            abs(value - reference[row_id]) for row_id, value in values.items()
        ]
        current = max(differences, default=0.0)
        maximum = max(maximum, current)
        results[name] = {"rows": len(frame), "max_abs_diff": current}
    return {
        "candidate": {
            "variants": results,
            "max_abs_diff": maximum,
            "atol": ROW_INDEPENDENCE_ATOL,
            "passed": maximum <= ROW_INDEPENDENCE_ATOL,
        },
        "full_output": full,
    }


def determinism_audit(extracted: Path, test: pd.DataFrame) -> dict:
    runs = [run_submission(extracted, test) for _ in range(3)]
    hashes = [run["output_sha256"] for run in runs]
    predictions = [
        run["output"][TARGET].to_numpy(dtype="float64") for run in runs
    ]
    maximum = max(
        float(np.max(np.abs(values - predictions[0])))
        for values in predictions[1:]
    )
    return {
        "runs": 3,
        "submission_sha256": hashes,
        "byte_identical": len(set(hashes)) == 1,
        "prediction_max_abs_diff": maximum,
        "passed": len(set(hashes)) == 1 and maximum == 0.0,
    }


def runtime_audit(extracted: Path, sample: pd.DataFrame) -> dict:
    repeats = math.ceil(RUNTIME_ROWS / len(sample))
    frame = pd.concat([sample] * repeats, ignore_index=True).iloc[:RUNTIME_ROWS].copy()
    frame["row_id"] = [f"EXTRA_RUNTIME_{index:09d}" for index in range(RUNTIME_ROWS)]
    run = run_submission(extracted, frame, timeout=900)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_bytes = int(peak if platform.system() == "Darwin" else peak * 1024)
    runtime_ratio = run["wall_seconds"] / CHAMPION_RUNTIME_SECONDS
    memory_ratio = peak_bytes / CHAMPION_PEAK_RSS_BYTES
    return {
        "rows": RUNTIME_ROWS,
        "wall_seconds": run["wall_seconds"],
        "rows_per_second": RUNTIME_ROWS / run["wall_seconds"],
        "peak_rss_bytes": peak_bytes,
        "output_size_bytes": run["output_size_bytes"],
        "champion_reference_seconds": CHAMPION_RUNTIME_SECONDS,
        "champion_reference_peak_rss_bytes": CHAMPION_PEAK_RSS_BYTES,
        "runtime_ratio": runtime_ratio,
        "memory_ratio": memory_ratio,
        "runtime_guard": RUNTIME_RATIO_GUARD,
        "memory_guard": MEMORY_RATIO_GUARD,
        "official_limits_found_locally": False,
        "passed": bool(
            runtime_ratio <= RUNTIME_RATIO_GUARD
            and memory_ratio <= MEMORY_RATIO_GUARD
        ),
    }


def test_delta_audit(
    champion: np.ndarray,
    extra: np.ndarray,
    candidate: np.ndarray,
) -> dict:
    expected = CHAMPION_WEIGHT * champion + EXTRA_WEIGHT * extra
    formula_difference = float(np.max(np.abs(candidate - expected)))
    delta = candidate - champion
    return {
        "rows": len(delta),
        "mean_delta": float(np.mean(delta)),
        "std_delta": float(np.std(delta)),
        "min_delta": float(np.min(delta)),
        "max_delta": float(np.max(delta)),
        "p01": float(np.quantile(delta, 0.01)),
        "p50": float(np.quantile(delta, 0.50)),
        "p99": float(np.quantile(delta, 0.99)),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "changed_rows": int(np.sum(delta != 0.0)),
        "candidate_formula_max_abs_diff": formula_difference,
        "candidate_formula_atol": ENDPOINT_FORMULA_ATOL,
        "nan_count": int(np.isnan(candidate).sum()),
        "inf_count": int(np.isinf(candidate).sum()),
        "within_bounds": bool(np.all((candidate >= 0.0) & (candidate <= 1.0))),
        "passed": bool(
            formula_difference <= ENDPOINT_FORMULA_ATOL
            and np.isfinite(candidate).all()
            and np.all((candidate >= 0.0) & (candidate <= 1.0))
            and np.max(np.abs(delta)) <= EXTRA_WEIGHT
        ),
    }


def package_audit(candidate: Path, model_path: Path) -> dict:
    with zipfile.ZipFile(CHAMPION) as champion_archive, zipfile.ZipFile(candidate) as archive:
        bad_crc_member = archive.testzip()
        champion_names = sorted(
            info.filename for info in champion_archive.infolist() if not info.is_dir()
        )
        names = sorted(info.filename for info in archive.infolist() if not info.is_dir())
        required_members = {
            "script.py",
            "requirements.txt",
            f"model/{MODEL_FILE}",
            f"model/{META_FILE}",
        }
        experiment_artifact_hits = [
            name
            for name in names
            if name.startswith(("artifacts/", "docs/", "experiments/", "tests/"))
            or "/cache/" in f"/{name}"
            or name.endswith(".csv")
            or (name.endswith(".npz") and not name.startswith("model/"))
        ]
        requirements_same = (
            archive.read("requirements.txt")
            == champion_archive.read("requirements.txt")
        )
        original_payload_equal = all(
            name == "script.py"
            or archive.read(name) == champion_archive.read(name)
            for name in champion_names
        )
        text_hits = []
        for name in names:
            if Path(name).suffix.lower() not in {".py", ".json", ".txt"}:
                continue
            text = archive.read(name).decode("utf-8", errors="replace")
            for pattern in ("/Users/wooh/", "/Users/", "/home/", "../open/"):
                if pattern in text:
                    text_hits.append({"file": name, "pattern": pattern})
    candidate_size = candidate.stat().st_size
    expected_additions = {
        "extratrees_runtime.py",
        f"model/{MODEL_FILE}",
        f"model/{META_FILE}",
    }
    actual_additions = set(names) - set(champion_names)
    model_load_ok = False
    try:
        model = joblib.load(model_path)
        model_load_ok = isinstance(model, ExtraTreesClassifier)
        del model
        gc.collect()
    except Exception:
        model_load_ok = False
    return {
        "champion_zip_size_bytes": CHAMPION.stat().st_size,
        "candidate_zip_size_bytes": candidate_size,
        "zip_size_delta_bytes": candidate_size - CHAMPION.stat().st_size,
        "model_serialized_size_bytes": model_path.stat().st_size,
        "package_guard_bytes": PACKAGE_GUARD_BYTES,
        "within_package_guard": candidate_size <= PACKAGE_GUARD_BYTES,
        "official_package_limit_found_locally": False,
        "champion_file_count": len(champion_names),
        "candidate_file_count": len(names),
        "candidate_files": names,
        "zip_crc_bad_member": bad_crc_member,
        "zip_crc_passed": bad_crc_member is None,
        "required_members": sorted(required_members),
        "required_members_present": required_members.issubset(names),
        "experiment_artifact_hits": experiment_artifact_hits,
        "production_payload_only": not experiment_artifact_hits,
        "expected_additions_exact": actual_additions == expected_additions,
        "original_payloads_equal_except_script": original_payload_equal,
        "requirements_byte_identical": requirements_same,
        "requirements_change": "none" if requirements_same else "changed",
        "new_dependency": "none; joblib and scikit-learn already pinned",
        "serialization": "joblib compressed pickle, protocol 5",
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "model_load_ok": model_load_ok,
        "absolute_path_hits": text_hits,
        "passed": bool(
            bad_crc_member is None
            and required_members.issubset(names)
            and not experiment_artifact_hits
            and actual_additions == expected_additions
            and original_payload_equal
            and requirements_same
            and model_load_ok
            and not text_hits
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--force-oof", action="store_true")
    parser.add_argument(
        "--reuse-full-model",
        action="store_true",
        help=(
            "reuse the serialized first full-train fit and train one fresh repeat; "
            "intended only to resume an interrupted parity run"
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="reuse the existing candidate/model and rerun inference audits only",
    )
    args = parser.parse_args(argv)
    for required in (
        args.train,
        args.data_dir / "test.csv",
        args.data_dir / "sample_submission.csv",
        CHAMPION,
        REFERENCE_DIR / "summary.json",
    ):
        if not required.is_file():
            parser.error(f"required file not found: {required}")
    if args.candidate.resolve() == CHAMPION.resolve():
        parser.error("candidate path must never overwrite the champion")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    checksum_before = sha256_path(CHAMPION)
    if checksum_before != EXPECTED_CHAMPION_SHA256:
        raise SystemExit(f"champion checksum mismatch before build: {checksum_before}")
    if CHAMPION.stat().st_size != EXPECTED_CHAMPION_SIZE:
        raise SystemExit("champion size mismatch before build")
    print(f"champion before: {checksum_before}", flush=True)

    reference_summary = json.loads(
        (REFERENCE_DIR / "summary.json").read_text(encoding="utf-8")
    )
    champion_meta = read_zip_json(CHAMPION, "model/meta.json")
    recipe_blockers = validate_recipe(champion_meta, reference_summary)
    if recipe_blockers:
        raise SystemExit(f"frozen recipe mismatch: {recipe_blockers}")
    train = pd.read_csv(args.train, encoding="utf-8-sig")
    test = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(
        args.data_dir / "sample_submission.csv", encoding="utf-8-sig"
    )
    if test["row_id"].tolist() != sample["row_id"].tolist():
        raise ValueError("local test/sample row order mismatch")

    model_path = CACHE_DIR / MODEL_FILE
    checkpoint_path = args.output_dir / "cache" / "full_train_reproducibility.json"
    if args.audit_only:
        for required in (
            args.candidate,
            model_path,
            checkpoint_path,
            args.output_dir / "oof_parity.json",
        ):
            if not required.is_file():
                parser.error(f"audit-only prerequisite not found: {required}")
        print("reuse existing OOF/full-train/candidate artifacts", flush=True)
        oof = json.loads(
            (args.output_dir / "oof_parity.json").read_text(encoding="utf-8")
        )
        full_training = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        metadata = read_zip_json(args.candidate, f"model/{META_FILE}")
        model = joblib.load(model_path)
        extra_test = predict_extratrees(
            test, "", model=model, metadata=metadata
        )
        full_training["extra_row_independence"] = extra_row_independence(
            model, metadata, test
        )
        del model
        gc.collect()
        write_json(checkpoint_path, full_training)
    else:
        print("production-equivalent temporal OOF parity", flush=True)
        oof = production_oof_parity(
            train, champion_meta, args.oof_dir, args.force_oof
        )
        write_json(args.output_dir / "oof_parity.json", oof)

        print("full-train ExtraTrees reproducibility (two fits)", flush=True)
        model_path, metadata, extra_test, full_training = train_full_twice(
            train, test, champion_meta, reuse_first_model=args.reuse_full_model
        )
        write_json(checkpoint_path, full_training)
        if not full_training["training"]["passed"]:
            raise RuntimeError("full-train ExtraTrees reproducibility failed")
        build_candidate_zip(CHAMPION, args.candidate, model_path, metadata)
        print(
            f"candidate built: {args.candidate} "
            f"({args.candidate.stat().st_size:,} bytes)",
            flush=True,
        )

    with tempfile.TemporaryDirectory(prefix="extra_champion_") as champion_tmp, tempfile.TemporaryDirectory(
        prefix="extra_candidate_"
    ) as candidate_tmp:
        champion_root = Path(champion_tmp)
        candidate_root = Path(candidate_tmp)
        safe_extract(CHAMPION, champion_root)
        safe_extract(args.candidate, candidate_root)
        champion_run = run_submission(champion_root, test)
        champion_prediction = champion_run["output"][TARGET].to_numpy(dtype="float64")
        champion_component_reference = run_raw_prediction(
            champion_root, test, "predict"
        )
        component_prediction = run_champion_component(candidate_root, test)
        champion_parity_diff = float(
            np.max(
                np.abs(component_prediction - champion_component_reference)
            )
        )
        champion_parity = {
            "rows": len(test),
            "max_abs_diff": champion_parity_diff,
            "atol": CHAMPION_PARITY_ATOL,
            "passed": champion_parity_diff <= CHAMPION_PARITY_ATOL,
        }
        row_result = candidate_row_independence(candidate_root, test)
        candidate_output = row_result.pop("full_output")
        row_independence = {
            "champion_component_parity": champion_parity,
            "extra_trees": full_training["extra_row_independence"],
            "candidate": row_result["candidate"],
        }
        row_independence["passed"] = all(
            item["passed"]
            for item in row_independence.values()
            if isinstance(item, dict) and "passed" in item
        )
        write_json(args.output_dir / "row_independence.json", row_independence)
        candidate_prediction = candidate_output["output"][TARGET].to_numpy(
            dtype="float64"
        )
        test_delta = test_delta_audit(
            champion_prediction, extra_test, candidate_prediction
        )
        write_json(args.output_dir / "test_delta.json", test_delta)
        determinism = determinism_audit(candidate_root, test)
        write_json(args.output_dir / "determinism.json", determinism)
        runtime = runtime_audit(candidate_root, test)
        write_json(args.output_dir / "runtime.json", runtime)

    package = package_audit(args.candidate, model_path)
    package["candidate_sha256"] = sha256_path(args.candidate)
    package["champion_sha256"] = checksum_before
    write_json(args.output_dir / "package_audit.json", package)
    checksum_after = sha256_path(CHAMPION)
    champion_immutable = checksum_after == checksum_before == EXPECTED_CHAMPION_SHA256

    technical_conditions = {
        "oof_production_parity": oof["passed"],
        "all_three_fold_gains_positive": all(
            row["production_brier_gain"] > 0.0 for row in oof["folds"]
        ),
        "champion_base_parity_exact": champion_parity["passed"],
        "row_independence": row_independence["passed"],
        "isolated_run_and_output_bounds": test_delta["passed"],
        "deterministic_inference": determinism["passed"],
        "full_train_reproducibility": full_training["training"]["passed"],
        "champion_checksum_immutable": champion_immutable,
        "package_structure_and_dependencies": package["passed"],
    }
    risk_conditions = {
        "runtime_memory_realistic": runtime["passed"],
        "package_size_within_engineering_guard": package[
            "within_package_guard"
        ],
    }
    if not all(technical_conditions.values()):
        verdict = "C. PARITY OR SAFETY FAILURE — DO NOT SUBMIT"
    elif not all(risk_conditions.values()):
        verdict = "B. TECHNICALLY VALID BUT GAIN TOO SMALL / RISK TOO HIGH"
    else:
        verdict = "A. READY FOR ONE LB SUBMISSION"
    summary = {
        "verdict": verdict,
        "candidate": {
            "path": str(args.candidate),
            "sha256": sha256_path(args.candidate),
            "size_bytes": args.candidate.stat().st_size,
            "formula": "0.99 * frozen champion + 0.01 * frozen ExtraTrees",
        },
        "champion": {
            "path": str(CHAMPION),
            "sha256_before": checksum_before,
            "sha256_after": checksum_after,
            "size_bytes": CHAMPION.stat().st_size,
            "immutable": champion_immutable,
        },
        "recipe": {
            "features": metadata["feature_columns"],
            "feature_count": len(metadata["feature_columns"]),
            "preprocessing": metadata["preprocessor"],
            "model_params": metadata["model_params"],
            "sklearn_version": metadata["sklearn_version"],
            "joblib_version": metadata["joblib_version"],
            "weights": {
                "champion": CHAMPION_WEIGHT,
                "extra_trees": EXTRA_WEIGHT,
            },
        },
        "full_train_reproducibility": full_training["training"],
        "technical_conditions": technical_conditions,
        "risk_conditions": risk_conditions,
        "official_runtime_memory_package_limits_found_locally": False,
        "artifacts": [
            "oof_parity.json",
            "test_delta.json",
            "row_independence.json",
            "determinism.json",
            "runtime.json",
            "package_audit.json",
            "summary.json",
        ],
        "leaderboard_submission_performed": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"champion after: {checksum_after}", flush=True)
    print(f"candidate sha256: {summary['candidate']['sha256']}", flush=True)
    print(f"verdict: {verdict}", flush=True)


if __name__ == "__main__":
    main()
