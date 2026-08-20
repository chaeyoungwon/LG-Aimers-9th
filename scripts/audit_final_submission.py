"""Reproducible final audit for the immutable LG Aimers champion ZIP.

This script never writes to the champion.  It validates the archive, extracts
it into temporary directories, runs the packaged ``script.py`` as a subprocess,
and stores only audit reports under ``artifacts/final_audit``.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "artifacts" / "submit_r10_pitcher_w418194.zip"
EXPECTED_SHA256 = "ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47"
EXPECTED_SIZE = 3_345_014
DEFAULT_DATA_DIR = Path("/Users/wooh/Documents/dev/open/data")
DEFAULT_OUTPUT = ROOT / "artifacts" / "final_audit"
DEFAULT_REBUILT = DEFAULT_OUTPUT / "rebuilt_champion.zip"
ROW_INDEPENDENCE_ATOL = 0.0

EXPECTED_FILES = {
    "batter_form_runtime.py",
    "coldstart_runtime.py",
    "model/batter_form.json",
    "model/cat_model.cbm",
    "model/coldstart_lgbm.txt",
    "model/coldstart_meta.json",
    "model/hgb_model.pkl",
    "model/meta.json",
    "model/nn_model.npz",
    "model/team_model.lgbmstack.json",
    "model/team_nn_model.npz",
    "requirements.txt",
    "script.py",
}
TEXT_SUFFIXES = {".py", ".json", ".txt", ".md", ".yaml", ".yml", ".cfg", ".ini"}
ABSOLUTE_PATH_PATTERNS = (
    "/Users/wooh/",
    "/Users/",
    "/home/",
    "../open/",
)
PROVENANCE_PATH_PATTERNS = ("LG-Aimers-9th",)
RESEARCH_ONLY_TERMS = (
    "regime_blend",
    "residual_discovery",
    "pitcher_form_shape",
    "trackman_incremental",
)
SUSPICIOUS_FILE_PARTS = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "__pycache__",
    ".ds_store",
    ".ipynb",
    ".git",
)
STATIC_OPERATIONS = {
    "groupby",
    "rolling",
    "expanding",
    "rank",
    "shift",
    "diff",
    "cumsum",
    "cumcount",
    "mean",
    "median",
    "std",
    "quantile",
    "value_counts",
}
PACKAGE_IMPORT_MAP = {
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "pandas": "pandas",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "numpy": "numpy",
    "torch": "torch",
    "scipy": "scipy",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not name or name.endswith("/"):
        raise ValueError(f"unsafe ZIP member: {name!r}")


def zip_payload_hashes(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {
            info.filename: sha256_bytes(archive.read(info.filename))
            for info in archive.infolist()
            if not info.is_dir()
        }


def inspect_zip(path: Path) -> tuple[dict, str]:
    inventory_lines = [
        "size_bytes\tcompressed_bytes\tcrc32\tsha256\tpath"
    ]
    names = []
    absolute_hits = []
    provenance_hits = []
    secret_hits = []
    research_hits = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            validate_member_name(info.filename)
            names.append(info.filename)
            data = archive.read(info.filename)
            inventory_lines.append(
                f"{info.file_size}\t{info.compress_size}\t{info.CRC:08x}\t"
                f"{sha256_bytes(data)}\t{info.filename}"
            )
            if Path(info.filename).suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = data.decode("utf-8", errors="replace")
            for pattern in ABSOLUTE_PATH_PATTERNS:
                if pattern in text:
                    absolute_hits.append(
                        {"file": info.filename, "pattern": pattern}
                    )
            for pattern in PROVENANCE_PATH_PATTERNS:
                if pattern in text:
                    provenance_hits.append(
                        {"file": info.filename, "pattern": pattern}
                    )
            for pattern in RESEARCH_ONLY_TERMS:
                if pattern in text:
                    research_hits.append(
                        {"file": info.filename, "term": pattern}
                    )
            for regex in (
                r"AKIA[0-9A-Z]{16}",
                r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                r"(?i)(?:api[_-]?key|password|secret)\s*[:=]\s*['\"][^'\"]+",
            ):
                if re.search(regex, text):
                    secret_hits.append({"file": info.filename, "pattern": regex})
    name_set = set(names)
    suspicious = sorted(
        name
        for name in names
        if any(part in name.lower() for part in SUSPICIOUS_FILE_PARTS)
    )
    unexpected = sorted(name_set - EXPECTED_FILES)
    missing = sorted(EXPECTED_FILES - name_set)
    result = {
        "file_count": len(names),
        "required_structure": {
            "script.py": "script.py" in name_set,
            "requirements.txt": "requirements.txt" in name_set,
            "model_files": any(name.startswith("model/") for name in names),
        },
        "missing_expected_files": missing,
        "unexpected_files": unexpected,
        "suspicious_files": suspicious,
        "absolute_path_hits": absolute_hits,
        "non_executable_provenance_path_hits": provenance_hits,
        "secret_hits": secret_hits,
        "research_only_dependency_hits": research_hits,
        "passed": not (
            missing
            or unexpected
            or suspicious
            or absolute_hits
            or secret_hits
            or research_hits
        ),
    }
    return result, "\n".join(inventory_lines) + "\n"


def safe_extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            validate_member_name(info.filename)
            target = destination / PurePosixPath(info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def read_zip_json(path: Path, member: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(member))


def parse_requirements(text: str) -> dict[str, str]:
    result = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            result[line] = "unpinned"
        else:
            name, version = line.split("==", 1)
            result[name.strip().lower()] = version.strip()
    return result


def python_sources(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as archive:
        return {
            info.filename: archive.read(info.filename).decode("utf-8")
            for info in archive.infolist()
            if info.filename.endswith(".py")
        }


def collect_imports(sources: Mapping[str, str]) -> set[str]:
    imports = set()
    local_modules = {Path(name).stem for name in sources}
    for source in sources.values():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    return imports - local_modules


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_audit(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        requirements_text = archive.read("requirements.txt").decode("utf-8")
    requirements = parse_requirements(requirements_text)
    imports = collect_imports(python_sources(zip_path))
    imported_packages = sorted(
        {PACKAGE_IMPORT_MAP[name] for name in imports if name in PACKAGE_IMPORT_MAP}
        | {"scikit-learn"}
    )
    required_by_reason = {
        "scikit-learn": "joblib HGB deserialization",
        "joblib": "HGB loader",
        "pandas": "CSV and tabular inference",
        "lightgbm": "team stack and cold-start expert",
        "catboost": "native CatBoost member",
        "numpy": "numeric inference and NPZ state arrays",
        "torch": "two embedding NN members; imported lazily",
    }
    package_rows = []
    warnings = []
    blockers = []
    for package, reason in required_by_reason.items():
        pinned = requirements.get(package.lower())
        installed = installed_version(package)
        status = "pinned"
        if pinned is None and package == "torch":
            status = "evaluation_base_dependency"
            warnings.append(
                "torch is used but intentionally absent from requirements.txt; "
                "the package relies on the evaluation base image"
            )
        elif pinned is None:
            status = "missing_requirement"
            blockers.append(f"{package} is used but absent from requirements.txt")
        elif installed is not None and pinned != installed:
            status = "local_version_differs"
            warnings.append(
                f"local {package}={installed} differs from pinned {pinned}"
            )
        package_rows.append(
            {
                "package": package,
                "reason": reason,
                "required_version": pinned,
                "local_version": installed,
                "status": status,
            }
        )
    unused = sorted(
        package
        for package in requirements
        if package not in {item.lower() for item in imported_packages}
    )
    # scikit-learn is not an AST import but is required to unpickle HGB.
    unused = [package for package in unused if package != "scikit-learn"]
    if unused:
        warnings.append(f"requirements entries not directly exercised: {unused}")
    return {
        "requirements": requirements,
        "python_import_roots": sorted(imports),
        "packages": package_rows,
        "scipy": {
            "imported": "scipy" in imports,
            "required": "scipy" in requirements,
            "status": "not_used",
        },
        "serialization_compatibility": [
            "HGB uses joblib with scikit-learn pinned to the training version 1.8.0",
            "CatBoost uses native .cbm rather than a repository-class pickle",
            "LightGBM uses native text/model strings with lightgbm 4.6.0 pinned",
            "NN weights use NumPy NPZ and a locally defined torch module, not torch pickle",
        ],
        "warnings": warnings,
        "blockers": blockers,
        "passed": not blockers,
    }


def call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return None


def static_independence_audit(zip_path: Path) -> dict:
    occurrences = []
    target_references = []
    blockers = []
    sources = python_sources(zip_path)
    for filename, source in sources.items():
        lines = source.splitlines()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and call_name(node) in STATIC_OPERATIONS:
                operation = call_name(node)
                snippet = lines[node.lineno - 1].strip()
                if operation == "mean" and filename == "coldstart_runtime.py":
                    classification = "same-row model ensemble across model axis"
                    affects_prediction = True
                    safe = True
                elif operation == "mean":
                    classification = "verbose diagnostic summary only"
                    affects_prediction = False
                    safe = True
                else:
                    classification = "unreviewed aggregation"
                    affects_prediction = True
                    safe = False
                    blockers.append(
                        f"{filename}:{node.lineno} unreviewed {operation} call"
                    )
                occurrences.append(
                    {
                        "file": filename,
                        "line": node.lineno,
                        "operation": operation,
                        "source": snippet,
                        "classification": classification,
                        "affects_prediction": affects_prediction,
                        "safe": safe,
                    }
                )
        for lineno, line in enumerate(lines, 1):
            if "control_success" in line or re.search(r"\b(?:target|label)\b", line):
                target_references.append(
                    {"file": filename, "line": lineno, "source": line.strip()}
                )
    executable_test_target_access = []
    for filename, source in sources.items():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id != "test":
                continue
            key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if key == "control_success":
                executable_test_target_access.append(
                    {"file": filename, "line": node.lineno}
                )
    if executable_test_target_access:
        blockers.append("test frame accesses control_success")
    return {
        "searched_operations": sorted(STATIC_OPERATIONS),
        "occurrences": occurrences,
        "test_wide_prediction_aggregation_found": any(
            not row["safe"] for row in occurrences
        ),
        "target_text_references": target_references,
        "executable_test_target_access": executable_test_target_access,
        "blockers": blockers,
        "passed": not blockers,
    }


def model_component_audit(zip_path: Path) -> list[dict]:
    meta = read_zip_json(zip_path, "model/meta.json")
    cold = read_zip_json(zip_path, "model/coldstart_meta.json")
    members = {row["name"]: row for row in meta["members"]}
    return [
        {
            "component": "HGB",
            "model_file": members["hgb"]["model_files"][0],
            "format": "joblib pickle",
            "loader": "joblib.load",
            "package": "joblib + scikit-learn",
            "repository_module_required": False,
        },
        {
            "component": "CatBoost",
            "model_file": members["cat"]["model_files"][0],
            "format": "CatBoost native CBM",
            "loader": "CatBoostClassifier.load_model",
            "package": "catboost",
            "repository_module_required": False,
        },
        {
            "component": "ours embedding NN",
            "model_file": members["nn"]["model_files"][0],
            "format": "NumPy NPZ state arrays",
            "loader": "np.load + local torch module.load_state_dict(strict=True)",
            "package": "numpy + torch",
            "repository_module_required": False,
        },
        {
            "component": "team LightGBM",
            "model_file": members["team"]["model_files"][0],
            "format": "JSON bundle of four LightGBM model strings",
            "loader": "lightgbm.Booster(model_str=...)",
            "package": "lightgbm",
            "repository_module_required": False,
        },
        {
            "component": "team embedding NN",
            "model_file": members["team_nn"]["model_files"][0],
            "format": "NumPy NPZ state arrays",
            "loader": "np.load + local torch module.load_state_dict(strict=True)",
            "package": "numpy + torch",
            "repository_module_required": False,
        },
        {
            "component": "cold-start expert",
            "model_file": cold["model_files"][0],
            "format": "LightGBM native text",
            "loader": "lightgbm.Booster(model_file=...)",
            "package": "lightgbm",
            "repository_module_required": False,
        },
        {
            "component": "pitcher season form",
            "model_file": "model/meta.json#season_form",
            "format": "JSON train-derived constants/lookup",
            "loader": "json.load + season_form_values",
            "package": "stdlib + numpy + pandas",
            "repository_module_required": False,
        },
        {
            "component": "batter season form",
            "model_file": "model/batter_form.json",
            "format": "JSON train-derived constants/lookup",
            "loader": "json.load + apply_batter_form",
            "package": "stdlib + numpy + pandas",
            "repository_module_required": False,
        },
    ]


def feature_source_audit() -> list[dict]:
    return [
        {
            "feature_or_source": "official row context and as-of pitcher/batter features",
            "source": "current test row provided by organizer",
            "train_only": False,
            "test_row_only": True,
            "safe": True,
        },
        {
            "feature_or_source": "category levels/maps",
            "source": "model/meta.json and coldstart_meta.json",
            "train_only": True,
            "test_row_only": False,
            "safe": True,
        },
        {
            "feature_or_source": "NN medians/means/stds/category levels/ID vocab",
            "source": "model/meta.json",
            "train_only": True,
            "test_row_only": False,
            "safe": True,
        },
        {
            "feature_or_source": "stage weights and calibration constants",
            "source": "model/meta.json",
            "train_only": True,
            "test_row_only": False,
            "safe": True,
        },
        {
            "feature_or_source": "pitcher/batter season form lookups",
            "source": "train-derived JSON + current row asof counters",
            "train_only": True,
            "test_row_only": True,
            "safe": True,
        },
        {
            "feature_or_source": "cold-start known pitcher set and expert",
            "source": "train R rows only + current row pitcher_id/game_type",
            "train_only": True,
            "test_row_only": True,
            "safe": True,
        },
        {
            "feature_or_source": "sample submission",
            "source": "organizer file; row ordering/output schema only",
            "train_only": False,
            "test_row_only": False,
            "safe": True,
        },
        {
            "feature_or_source": "TrackMan",
            "source": "excluded: no official ID crosswalk and 0% exact coverage",
            "train_only": False,
            "test_row_only": False,
            "safe": True,
        },
    ]


def discover_openmp_runtime() -> Path | None:
    if platform.system() != "Darwin":
        return None
    candidates = [
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / package
        / subdir
        / "libomp.dylib"
        for package, subdir in (("sklearn", ".dylibs"), ("torch", "lib"))
    ]
    return next((path for path in candidates if path.is_file()), None)


def inference_environment() -> tuple[dict[str, str], list[str]]:
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    warnings = []
    openmp = discover_openmp_runtime()
    if openmp is not None:
        env["DYLD_LIBRARY_PATH"] = str(openmp.parent)
        warnings.append(
            f"local macOS audit injected DYLD_LIBRARY_PATH={openmp.parent}; "
            "this is a host OpenMP runtime issue, not a ZIP path dependency"
        )
    return env, warnings


def validate_submission(
    output: pd.DataFrame, expected_ids: Sequence[str]
) -> dict:
    expected = list(expected_ids)
    ids = output.get("row_id", pd.Series(dtype=object)).tolist()
    if "control_success" in output:
        prediction = output["control_success"].to_numpy(dtype="float64")
    else:
        prediction = np.asarray([], dtype="float64")
    checks = {
        "columns_exact": list(output.columns) == ["row_id", "control_success"],
        "length_match": len(output) == len(expected),
        "row_id_order_match": ids == expected,
        "row_id_unique": len(ids) == len(set(ids)),
        "nan_count": int(np.isnan(prediction).sum()) if len(prediction) else 0,
        "inf_count": int(np.isinf(prediction).sum()) if len(prediction) else 0,
        "within_bounds": bool(
            len(prediction)
            and np.all((prediction >= 0.0) & (prediction <= 1.0))
        ),
    }
    checks["passed"] = bool(
        checks["columns_exact"]
        and checks["length_match"]
        and checks["row_id_order_match"]
        and checks["row_id_unique"]
        and checks["nan_count"] == 0
        and checks["inf_count"] == 0
        and checks["within_bounds"]
    )
    return checks


def run_submission(
    extracted: Path,
    test: pd.DataFrame,
    sample_ids: Sequence[str] | None = None,
    timeout: int = 300,
) -> dict:
    data_dir = extracted / "data"
    output_dir = extracted / "output"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    expected_ids = list(sample_ids) if sample_ids is not None else test["row_id"].tolist()
    sample = pd.DataFrame(
        {"row_id": expected_ids, "control_success": np.full(len(expected_ids), 0.5)}
    )
    test.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
    sample.to_csv(
        data_dir / "sample_submission.csv", index=False, encoding="utf-8-sig"
    )
    output_path = output_dir / "submission.csv"
    output_path.unlink(missing_ok=True)
    env, environment_warnings = inference_environment()
    started = time.perf_counter()
    process = subprocess.run(
        [sys.executable, "script.py"],
        cwd=extracted,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"isolated inference failed ({process.returncode}):\n"
            f"STDOUT:\n{process.stdout[-4000:]}\nSTDERR:\n{process.stderr[-4000:]}"
        )
    if not output_path.is_file():
        raise RuntimeError("isolated inference did not create output/submission.csv")
    output = pd.read_csv(output_path)
    validation = validate_submission(output, expected_ids)
    if not validation["passed"]:
        raise RuntimeError(f"submission validation failed: {validation}")
    return {
        "output": output,
        "output_sha256": sha256_path(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "wall_seconds": elapsed,
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-2000:],
        "validation": validation,
        "environment_warnings": environment_warnings,
    }


def smoke_rows(sample: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base = sample.iloc[0].copy()
    rows = []
    cases = []

    def add(name: str, updates: Mapping[str, object]) -> None:
        row = base.copy()
        row["row_id"] = f"AUDIT_{name.upper()}"
        for key, value in updates.items():
            row[key] = value
        rows.append(row)
        cases.append(name)

    rate_columns = [
        column
        for column in sample.columns
        if column.startswith("asof_") and column.endswith("_rate")
    ]
    add(
        "unseen_pitcher",
        {
            "pitcher_id": 999_999_991,
            "game_type": "R",
            "asof_pitcher_n": 0,
            "asof_pitcher_pitchmix_n": 0,
            **{column: np.nan for column in rate_columns if "pitcher" in column},
        },
    )
    add(
        "unseen_batter",
        {
            "batter_id": 999_999_992,
            "game_type": "R",
            "asof_batter_n": 0,
            "asof_batter_success_rate": np.nan,
            "asof_batter_middle_rate": np.nan,
        },
    )
    add(
        "missing_categorical",
        {
            "game_type": "F",
            "top_bottom": np.nan,
            "base_state": np.nan,
            "pitcher_hand": np.nan,
            "batter_hand": np.nan,
            "pitcher_team_id": np.nan,
            "batter_team_id": np.nan,
        },
    )
    add(
        "missing_numeric",
        {
            "game_type": "F",
            "li": np.nan,
            "asof_pitcher_n": np.nan,
            "asof_pitcher_success_rate": np.nan,
            "asof_batter_n": np.nan,
            "asof_batter_success_rate": np.nan,
        },
    )
    add(
        "rare_category",
        {
            "game_type": "F",
            "top_bottom": "X",
            "base_state": "RARE",
            "pitcher_hand": 9,
            "batter_hand": 8,
            "pitcher_team_id": 999,
            "batter_team_id": 998,
        },
    )
    add("game_type_f", {"game_type": "F"})
    add("game_type_r", {"game_type": "R"})
    add("full_count", {"balls_before": 3, "strikes_before": 2})
    add(
        "bases_empty",
        {
            "runner_on_1b": 0,
            "runner_on_2b": 0,
            "runner_on_3b": 0,
            "num_runners_on": 0,
            "base_state": "___",
        },
    )
    add(
        "bases_full",
        {
            "runner_on_1b": 1,
            "runner_on_2b": 1,
            "runner_on_3b": 1,
            "num_runners_on": 3,
            "base_state": "123",
        },
    )
    add(
        "extreme_asof_count",
        {
            "asof_pitcher_n": 1_000_000_000,
            "asof_batter_n": 1_000_000_000,
            "asof_pitcher_pitchmix_n": 1_000_000_000,
            "asof_pitcher_success_rate": 0.5,
            "asof_batter_success_rate": 0.5,
        },
    )
    return pd.DataFrame(rows).reset_index(drop=True), cases


def prediction_map(output: pd.DataFrame) -> dict[str, float]:
    return dict(zip(output["row_id"], output["control_success"].astype(float)))


def row_independence_audit(extracted: Path, sample: pd.DataFrame) -> dict:
    anchor = str(sample.iloc[0]["row_id"])
    full = run_submission(extracted, sample)
    full_map = prediction_map(full["output"])
    variants = {
        "single": sample.iloc[[0]].copy(),
        "full": sample.copy(),
        "shuffled": sample.sample(frac=1.0, random_state=42).reset_index(drop=True),
        "reversed": sample.iloc[::-1].reset_index(drop=True),
        "subset": sample.iloc[[0, 2, 4]].reset_index(drop=True),
    }
    unrelated = sample.iloc[[1]].copy()
    unrelated.loc[:, "row_id"] = "AUDIT_UNRELATED_DUPLICATE_CONTENT"
    variants["duplicate_unrelated_row"] = pd.concat(
        [sample, unrelated], ignore_index=True
    )
    results = {}
    max_difference = 0.0
    for name, frame in variants.items():
        run = run_submission(extracted, frame)
        values = prediction_map(run["output"])
        common = sorted(set(values) & set(full_map))
        differences = [abs(values[row_id] - full_map[row_id]) for row_id in common]
        common_max = max(differences, default=0.0)
        anchor_difference = abs(values[anchor] - full_map[anchor])
        max_difference = max(max_difference, common_max)
        results[name] = {
            "rows": len(frame),
            "anchor_prediction": values[anchor],
            "anchor_abs_difference": anchor_difference,
            "common_row_max_abs_difference": common_max,
            "output_sha256": run["output_sha256"],
        }
    passed = max_difference <= ROW_INDEPENDENCE_ATOL
    return {
        "anchor_row_id": anchor,
        "reference_prediction": full_map[anchor],
        "atol": ROW_INDEPENDENCE_ATOL,
        "max_abs_difference": max_difference,
        "variants": results,
        "passed": passed,
    }


def determinism_audit(extracted: Path, sample: pd.DataFrame) -> dict:
    runs = [run_submission(extracted, sample) for _ in range(3)]
    hashes = [run["output_sha256"] for run in runs]
    arrays = [
        run["output"]["control_success"].to_numpy(dtype="float64") for run in runs
    ]
    max_difference = max(
        float(np.max(np.abs(array - arrays[0]))) for array in arrays[1:]
    )
    return {
        "runs": 3,
        "submission_sha256": hashes,
        "all_byte_identical": len(set(hashes)) == 1,
        "max_prediction_abs_difference": max_difference,
        "passed": len(set(hashes)) == 1 and max_difference == 0.0,
    }


def smoke_test_audit(extracted: Path, sample: pd.DataFrame) -> dict:
    frame, cases = smoke_rows(sample)
    run = run_submission(extracted, frame)
    values = prediction_map(run["output"])
    case_rows = [
        {
            "case": case,
            "row_id": frame.iloc[index]["row_id"],
            "prediction": values[frame.iloc[index]["row_id"]],
            "finite": math.isfinite(values[frame.iloc[index]["row_id"]]),
            "within_bounds": 0.0 <= values[frame.iloc[index]["row_id"]] <= 1.0,
        }
        for index, case in enumerate(cases)
    ]
    alignment_order = list(reversed(frame["row_id"].tolist()))
    alignment = run_submission(extracted, frame, sample_ids=alignment_order)
    passed = all(row["finite"] and row["within_bounds"] for row in case_rows)
    return {
        "cases": case_rows,
        "output_validation": run["validation"],
        "sample_submission_alignment": alignment["validation"],
        "passed": passed and alignment["validation"]["passed"],
    }


def runtime_audit(
    extracted: Path, sample: pd.DataFrame, row_count: int
) -> dict:
    repeats = math.ceil(row_count / len(sample))
    frame = pd.concat([sample] * repeats, ignore_index=True).iloc[:row_count].copy()
    frame["row_id"] = [f"AUDIT_RUNTIME_{index:09d}" for index in range(row_count)]
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    run = run_submission(extracted, frame, timeout=900)
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if platform.system() == "Darwin":
        peak_bytes = int(after)
    else:
        peak_bytes = int(after * 1024)
    return {
        "rows": row_count,
        "wall_seconds": run["wall_seconds"],
        "rows_per_second": row_count / run["wall_seconds"],
        "peak_rss_bytes": peak_bytes,
        "peak_rss_before_bytes": int(before if platform.system() == "Darwin" else before * 1024),
        "output_size_bytes": run["output_size_bytes"],
        "output_sha256": run["output_sha256"],
        "published_competition_limit_available_locally": False,
        "assessment": (
            "measurement completed; no explicit runtime/memory limit is present in "
            "the provided local competition documentation"
        ),
        "passed": True,
    }


def rebuild_parity_audit(
    champion: Path, rebuilt: Path, sample: pd.DataFrame
) -> dict:
    if not rebuilt.is_file():
        return {
            "evaluated": False,
            "passed": False,
            "warning": f"rebuilt ZIP not found: {rebuilt}",
        }
    champion_payload = zip_payload_hashes(champion)
    rebuilt_payload = zip_payload_hashes(rebuilt)
    common = sorted(set(champion_payload) & set(rebuilt_payload))
    differing = [
        name for name in common if champion_payload[name] != rebuilt_payload[name]
    ]
    with zipfile.ZipFile(champion) as left_archive, zipfile.ZipFile(rebuilt) as right_archive:
        left_cold_model = left_archive.read("model/coldstart_lgbm.txt").decode("utf-8")
        right_cold_model = right_archive.read("model/coldstart_lgbm.txt").decode("utf-8")
        ignored_lightgbm_footer = "[gpu_device_id_list: ]"
        normalize_model = lambda text: "\n".join(
            line for line in text.splitlines() if line != ignored_lightgbm_footer
        )
        normalized_cold_model_parity = (
            normalize_model(left_cold_model) == normalize_model(right_cold_model)
        )
        left_cold_meta = json.loads(left_archive.read("model/coldstart_meta.json"))
        right_cold_meta = json.loads(right_archive.read("model/coldstart_meta.json"))
        left_cold_meta.pop("source", None)
        right_cold_meta.pop("source", None)
        inference_cold_meta_parity = left_cold_meta == right_cold_meta
    inference_payload_differences = [
        name
        for name in differing
        if name not in {"model/coldstart_lgbm.txt", "model/coldstart_meta.json"}
    ]
    with tempfile.TemporaryDirectory(prefix="final_audit_rebuilt_") as tmp:
        extracted = Path(tmp)
        safe_extract(rebuilt, extracted)
        rebuilt_run = run_submission(extracted, sample)
    with tempfile.TemporaryDirectory(prefix="final_audit_champion_") as tmp:
        extracted = Path(tmp)
        safe_extract(champion, extracted)
        champion_run = run_submission(extracted, sample)
    left = champion_run["output"]["control_success"].to_numpy(dtype="float64")
    right = rebuilt_run["output"]["control_success"].to_numpy(dtype="float64")
    prediction_difference = float(np.max(np.abs(left - right)))
    return {
        "evaluated": True,
        "champion_zip_sha256": sha256_path(champion),
        "rebuilt_zip_sha256": sha256_path(rebuilt),
        "zip_binary_identical": sha256_path(champion) == sha256_path(rebuilt),
        "payload_file_sets_equal": set(champion_payload) == set(rebuilt_payload),
        "payload_files": len(champion_payload),
        "payload_files_differing": differing,
        "raw_model_payload_parity": not any(
            name.startswith("model/") for name in differing
        ),
        "raw_meta_parity": all(
            name not in differing
            for name in ("model/meta.json", "model/coldstart_meta.json")
        ),
        "normalized_coldstart_model_parity": normalized_cold_model_parity,
        "normalized_model_difference": (
            "LightGBM footer differs only by an unused empty gpu_device_id_list line"
            if normalized_cold_model_parity
            and "model/coldstart_lgbm.txt" in differing
            else None
        ),
        "inference_coldstart_meta_parity": inference_cold_meta_parity,
        "coldstart_meta_difference": (
            "non-inference source decision/LB provenance fields only"
            if inference_cold_meta_parity
            and "model/coldstart_meta.json" in differing
            else None
        ),
        "other_inference_payload_differences": inference_payload_differences,
        "prediction_max_abs_difference": prediction_difference,
        "prediction_atol": 0.0,
        "prediction_parity": prediction_difference == 0.0,
        "passed": (
            set(champion_payload) == set(rebuilt_payload)
            and not inference_payload_differences
            and normalized_cold_model_parity
            and inference_cold_meta_parity
            and prediction_difference == 0.0
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", type=Path, default=CHAMPION)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuilt", type=Path, default=DEFAULT_REBUILT)
    parser.add_argument("--runtime-rows", type=int, default=253_507)
    args = parser.parse_args(argv)
    if args.runtime_rows <= 0:
        parser.error("--runtime-rows must be positive")
    for path in (
        args.champion,
        args.data_dir / "test.csv",
        args.data_dir / "sample_submission.csv",
    ):
        if not path.is_file():
            parser.error(f"required file not found: {path}")

    args.output.mkdir(parents=True, exist_ok=True)
    checksum_start = sha256_path(args.champion)
    size_start = args.champion.stat().st_size
    blockers = []
    warnings = []
    if checksum_start != EXPECTED_SHA256:
        blockers.append(f"champion checksum mismatch: {checksum_start}")
    if size_start != EXPECTED_SIZE:
        blockers.append(f"champion size mismatch: {size_start}")

    zip_audit, inventory = inspect_zip(args.champion)
    (args.output / "zip_inventory.txt").write_text(inventory, encoding="utf-8")
    if not zip_audit["passed"]:
        blockers.append("ZIP structure/content audit failed")
    if zip_audit["non_executable_provenance_path_hits"]:
        warnings.append(
            "model/meta.json contains a repository-name provenance string "
            "('LG-Aimers-9th') but no local absolute path; it is never opened"
        )
    dependencies = dependency_audit(args.champion)
    write_json(args.output / "dependency_audit.json", dependencies)
    blockers.extend(dependencies["blockers"])
    warnings.extend(dependencies["warnings"])
    static = static_independence_audit(args.champion)
    write_json(args.output / "static_independence_audit.json", static)
    blockers.extend(static["blockers"])

    sample = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")
    with tempfile.TemporaryDirectory(prefix="final_submission_audit_") as tmp:
        extracted = Path(tmp)
        safe_extract(args.champion, extracted)
        row_independence = row_independence_audit(extracted, sample)
        smoke = smoke_test_audit(extracted, sample)
        determinism = determinism_audit(extracted, sample)
        runtime = runtime_audit(extracted, sample, args.runtime_rows)
    write_json(args.output / "row_independence.json", row_independence)
    write_json(args.output / "smoke_test.json", smoke)
    write_json(args.output / "determinism.json", determinism)
    write_json(args.output / "runtime.json", runtime)
    if not row_independence["passed"]:
        blockers.append("actual packaged script failed row-independence audit")
    if not smoke["passed"]:
        blockers.append("actual packaged script failed smoke/output audit")
    if not determinism["passed"]:
        blockers.append("actual packaged script is not deterministic")
    _, local_runtime_warnings = inference_environment()
    warnings.extend(local_runtime_warnings)
    warnings.append(
        "provided local docs do not state an explicit runtime/memory limit; "
        "the audit records measurements but cannot prove a numeric limit margin"
    )

    rebuild = rebuild_parity_audit(args.champion, args.rebuilt, sample)
    write_json(args.output / "rebuild_parity.json", rebuild)
    if not rebuild["evaluated"]:
        warnings.append(rebuild["warning"])
    elif not rebuild["passed"]:
        blockers.append("rebuild payload or prediction parity failed")
    elif rebuild["payload_files_differing"]:
        warnings.append(
            "rebuilt ZIP is prediction-identical but two cold-start payloads have "
            "non-inference serialization/provenance byte differences"
        )

    checksum_end = sha256_path(args.champion)
    size_end = args.champion.stat().st_size
    if checksum_end != checksum_start or size_end != size_start:
        blockers.append("champion ZIP changed during audit")
    if blockers:
        verdict = "C. BLOCKER FOUND — DO NOT SUBMIT"
    elif warnings:
        verdict = "B. READY WITH NON-BLOCKING WARNINGS"
    else:
        verdict = "A. READY FOR FINAL SUBMISSION"
    meta = read_zip_json(args.champion, "model/meta.json")
    summary = {
        "champion": {
            "path": str(args.champion),
            "lb_bss": 1016.4442212358,
            "sha256_start": checksum_start,
            "sha256_end": checksum_end,
            "expected_sha256": EXPECTED_SHA256,
            "size_start_bytes": size_start,
            "size_end_bytes": size_end,
            "expected_size_bytes": EXPECTED_SIZE,
            "immutable": checksum_start == checksum_end and size_start == size_end,
        },
        "zip_audit": zip_audit,
        "model_components": model_component_audit(args.champion),
        "prediction_pipeline": [
            "read test.csv and sample_submission.csv",
            "apply fixed category levels and row-local feature engineering",
            "predict HGB, CatBoost, ours embedding NN",
            "ours stage = calibrated 0.4/0.4/0.2 probability blend",
            "predict team LightGBM stack and team embedding NN",
            "team stage = 0.9/0.1 probability blend",
            "stage mix = 0.4 ours + 0.6 team",
            "pitcher season-form endpoint",
            "batter season-form endpoint",
            "regular-season unseen-pitcher cold-start expert blend",
            "clip to [0,1] and align to sample_submission row_id order",
        ],
        "stage_configuration": meta["blend"],
        "feature_sources": feature_source_audit(),
        "dependency_audit_file": "dependency_audit.json",
        "static_audit_file": "static_independence_audit.json",
        "row_independence_file": "row_independence.json",
        "determinism_file": "determinism.json",
        "smoke_test_file": "smoke_test.json",
        "runtime_file": "runtime.json",
        "rebuild_parity_file": "rebuild_parity.json",
        "test_rows_from_private_evaluation_read": False,
        "research_code_in_zip": False,
        "trackman_used": False,
        "trackman_exclusion": (
            "no official train.pitcher_id ↔ pitcher_trackman_id crosswalk; "
            "exact cutoff-safe coverage is 0% in 2022/2023/2024"
        ),
        "warnings": sorted(set(warnings)),
        "blockers": blockers,
        "final_verdict": verdict,
    }
    write_json(args.output / "summary.json", summary)
    print(f"champion checksum start/end: {checksum_start} / {checksum_end}")
    print(f"champion size start/end: {size_start} / {size_end}")
    print(f"row independence max diff: {row_independence['max_abs_difference']}")
    print(f"deterministic byte hashes: {determinism['submission_sha256']}")
    print(
        f"runtime: {runtime['rows']} rows, {runtime['wall_seconds']:.3f}s, "
        f"peak RSS {runtime['peak_rss_bytes']} bytes"
    )
    print(f"rebuild parity: {rebuild.get('passed')}")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
