"""Evaluate architecture diversity on the frozen accepted HGB-52 contract.

Stage A is one preregistered XGBoost ``hist`` recipe.  It uses the exact same
ID-free 52 row-local features and cutoff-train preprocessing as the accepted
ExtraTrees member.  No feature engineering, calibration, seed averaging, or
hyperparameter search is performed.  Later learners are conditional stage
gates and are not run by the XGBoost worker.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import importlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import rtdl_revisiting_models as rtdl
import tabm
import torch
from pandas.api.types import is_object_dtype


def _import_xgboost():
    try:
        return importlib.import_module("xgboost")
    except Exception as error:
        if "libomp.dylib" not in str(error):
            raise
        prefix = Path(sys.prefix)
        candidates = list(
            prefix.glob("lib/python*/site-packages/torch/lib/libomp.dylib")
        )
        candidates += list(
            prefix.glob("lib/python*/site-packages/sklearn/.dylibs/libomp.dylib")
        )
        if not candidates:
            raise
        ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)
        return importlib.import_module("xgboost")


xgboost = _import_xgboost()
XGBClassifier = xgboost.XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime_blend_oof import load_champion_meta  # noqa: E402
from scripts.extratrees_runtime import (  # noqa: E402
    build_hgb52_features,
    fit_preprocessor,
    transform_features,
)
from scripts.validate_control_profile import (  # noqa: E402
    array_sha256,
    brier,
    load_npz,
    pooled_array,
    safe_corr,
    save_npz_atomic,
    sha256_path,
    subset_masks,
    temporal_split,
    write_json,
)


DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
OUTPUT_DIR = ROOT / "artifacts" / "model_diversity"
OOF_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
RAW_ET25_DIR = ROOT / "artifacts" / "extratrees_size_ablation" / "cache"
WEIGHT_REFERENCE = ROOT / "artifacts" / "et25_upper_weight" / "weight_results.csv"
CHAMPION = ROOT / "artifacts" / "sub_et25_w020.zip"
CHAMPION_SHA256 = (
    "316bf6dab11fb78f06cc74089ea38bdb3b957355f4510db5fcc8a485dc26cecf"
)
CHAMPION_LB_BSS = 1017.0233029621

TARGET = "control_success"
VALIDATION_SEASONS = (2022, 2023, 2024)
BLEND_WEIGHTS = (0.005, 0.010, 0.020, 0.050)
CURRENT_ET25_WEIGHT = 0.020
XGB_CACHE_VERSION = f"model-diversity-xgb-hist-v1-xgb{xgboost.__version__}"
TABM_CACHE_VERSION = f"model-diversity-tabm-small-v1-tabm{tabm.__version__}"
FT_CACHE_VERSION = f"model-diversity-ft-small-v1-rtdl{rtdl.__version__}"

XGB_RECIPE = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 5,
    "reg_alpha": 0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "tree_method": "hist",
}
TABM_RECIPE = {
    "arch_type": "tabm",
    "k": 8,
    "n_blocks": 2,
    "d_block": 128,
    "dropout": 0.1,
    "learning_rate": 0.002,
    "weight_decay": 0.0003,
    "batch_size": 4096,
    "eval_batch_size": 16384,
    "epochs": 8,
    "random_state": 42,
}
FT_RECIPE = {
    "n_blocks": 2,
    "batch_size": 2048,
    "eval_batch_size": 4096,
    "epochs": 6,
    "learning_rate": 0.0001,
    "weight_decay": 0.00001,
    "random_state": 42,
}

# Fixed before viewing Stage-A predictions.
CLEAR_STANDALONE_GAIN = 1e-5
CATASTROPHIC_STANDALONE_FLOOR = -5e-4
MAX_DIVERSITY_CORRELATION = 0.90
MIN_CHAMPION_TOP10_WIN_RATE = 0.60
MAX_TOP10_WIN_RATE_SEASON_GAP = 0.15
WORST_FOLD_BLEND_FLOOR = -1e-5
MAJOR_SUBSET_FLOOR = -1e-5
MIN_SUBSET_ROWS = 5_000
LARGE_SIGNAL_GAIN = 2e-5

RESULT_COLUMNS = (
    "learner",
    "validation_season",
    "n",
    "standalone_brier",
    "current_champion_brier",
    "standalone_gain_vs_champion",
    "corr_model_champion",
    "corr_model_et25",
    "champion_top10_error_model_win_rate",
    "model_top10_error_champion_win_rate",
    "prediction_mean",
    "prediction_std",
    "fit_seconds",
    "predict_seconds",
)
COMPLEMENT_COLUMNS = (
    *RESULT_COLUMNS[:10],
    "path_a_passed",
    "path_b_passed",
    "complementarity_gate_passed",
)
BLEND_COLUMNS = (
    "learner",
    "weight",
    "validation_season",
    "n",
    "current_champion_brier",
    "candidate_brier",
    "brier_gain",
    "worst_major_subset_gain",
    "production_gate_passed",
)


@dataclass(frozen=True)
class TabMEncoding:
    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_medians: Mapping[str, float]
    numeric_means: Mapping[str, float]
    numeric_scales: Mapping[str, float]
    categorical_levels: Mapping[str, tuple[str, ...]]


def make_xgb(n_jobs: int = -1) -> XGBClassifier:
    return XGBClassifier(**XGB_RECIPE, n_jobs=n_jobs)


def cache_path(output: Path, season: int) -> Path:
    return output / "cache" / f"xgb_hist_{season}.npz"


def tabm_cache_path(output: Path, season: int) -> Path:
    return output / "cache" / f"tabm_{season}.npz"


def ft_cache_path(output: Path, season: int) -> Path:
    return output / "cache" / f"ft_transformer_{season}.npz"


def validate_raw_contract(features: pd.DataFrame, champion_meta: Mapping) -> None:
    hgb = next(member for member in champion_meta["members"] if member["name"] == "hgb")
    if list(features.columns) != list(hgb["feature_columns"]):
        raise ValueError("XGBoost input differs from accepted HGB-52 contract")
    if len(features.columns) != 52:
        raise ValueError(f"expected 52 raw accepted features, got {len(features.columns)}")
    forbidden = {"row_id", TARGET, "pitcher_id", "batter_id"}
    if forbidden & set(features.columns):
        raise ValueError("ID/target leaked into XGBoost features")


def fit_tabm_encoding(features: pd.DataFrame) -> TabMEncoding:
    numeric_columns = []
    categorical_columns = []
    medians = {}
    means = {}
    scales = {}
    levels = {}
    for column in features.columns:
        series = features[column]
        if isinstance(series.dtype, pd.CategoricalDtype) or is_object_dtype(series.dtype):
            categorical_columns.append(column)
            observed = series.astype("object").dropna().astype(str).unique().tolist()
            levels[column] = tuple(sorted(observed))
        else:
            numeric_columns.append(column)
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            filled = np.where(np.isfinite(values), values, median)
            mean = float(np.mean(filled))
            scale = float(np.std(filled))
            medians[column] = median
            means[column] = mean
            scales[column] = scale if scale > 1e-12 else 1.0
    return TabMEncoding(
        feature_columns=tuple(features.columns),
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        numeric_medians=medians,
        numeric_means=means,
        numeric_scales=scales,
        categorical_levels=levels,
    )


def transform_tabm(
    features: pd.DataFrame, encoding: TabMEncoding
) -> tuple[np.ndarray, np.ndarray]:
    if tuple(features.columns) != encoding.feature_columns:
        raise ValueError("TabM feature order differs from fitted HGB-52 contract")
    numeric = np.empty((len(features), len(encoding.numeric_columns)), dtype="float32")
    for index, column in enumerate(encoding.numeric_columns):
        values = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype="float64")
        values = np.where(np.isfinite(values), values, encoding.numeric_medians[column])
        numeric[:, index] = (
            (values - encoding.numeric_means[column]) / encoding.numeric_scales[column]
        ).astype("float32")
    categorical = np.empty(
        (len(features), len(encoding.categorical_columns)), dtype="int64"
    )
    for index, column in enumerate(encoding.categorical_columns):
        mapping = {
            level: position
            for position, level in enumerate(encoding.categorical_levels[column])
        }
        objects = features[column].astype("object")
        strings = objects.where(objects.notna(), "__UNKNOWN__").astype(str)
        categorical[:, index] = (
            strings.map(mapping)
            .fillna(len(mapping))
            .to_numpy(dtype="int64")
        )
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite TabM numeric input remains")
    return numeric, categorical


def tabm_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_tabm_model(encoding: TabMEncoding):
    cardinalities = [
        len(encoding.categorical_levels[column]) + 1
        for column in encoding.categorical_columns
    ]
    return tabm.TabM.make(
        n_num_features=len(encoding.numeric_columns),
        cat_cardinalities=cardinalities,
        d_out=1,
        arch_type=TABM_RECIPE["arch_type"],
        k=TABM_RECIPE["k"],
        n_blocks=TABM_RECIPE["n_blocks"],
        d_block=TABM_RECIPE["d_block"],
        dropout=TABM_RECIPE["dropout"],
    )


def _tabm_logits(model, numeric: np.ndarray, categorical: np.ndarray, indices, device):
    x_num = torch.as_tensor(numeric[indices], dtype=torch.float32, device=device)
    x_cat = torch.as_tensor(categorical[indices], dtype=torch.long, device=device)
    return model(x_num=x_num, x_cat=x_cat).squeeze(-1)


def fit_tabm_fold(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    champion_meta: Mapping,
) -> tuple[np.ndarray, float, float, list[str], list[float], str]:
    train_features = build_hgb52_features(training, champion_meta["cat_levels"])
    validation_features = build_hgb52_features(validation, champion_meta["cat_levels"])
    validate_raw_contract(train_features, champion_meta)
    validate_raw_contract(validation_features, champion_meta)
    encoding = fit_tabm_encoding(train_features)
    x_train_num, x_train_cat = transform_tabm(train_features, encoding)
    x_validation_num, x_validation_cat = transform_tabm(validation_features, encoding)
    y_train = training[TARGET].to_numpy(dtype="float32")
    feature_columns = list(train_features.columns)
    del train_features, validation_features
    gc.collect()

    seed = TABM_RECIPE["random_state"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = tabm_device()
    model = make_tabm_model(encoding).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TABM_RECIPE["learning_rate"],
        weight_decay=TABM_RECIPE["weight_decay"],
    )
    rng = np.random.default_rng(seed)
    batch_size = TABM_RECIPE["batch_size"]
    epoch_losses = []
    started = time.monotonic()
    for epoch in range(TABM_RECIPE["epochs"]):
        model.train()
        loss_sum = 0.0
        seen = 0
        permutation = rng.permutation(len(y_train))
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            logits = _tabm_logits(
                model, x_train_num, x_train_cat, indices, device
            )
            target = torch.as_tensor(
                y_train[indices], dtype=torch.float32, device=device
            ).unsqueeze(1)
            # Official TabM contract: optimize every head independently.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target.expand_as(logits)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        epoch_loss = loss_sum / seen
        epoch_losses.append(epoch_loss)
        print(f"  TabM epoch {epoch + 1}/{TABM_RECIPE['epochs']} loss={epoch_loss:.6f}", flush=True)
    fit_seconds = time.monotonic() - started

    model.eval()
    predictions = []
    started = time.monotonic()
    eval_batch_size = TABM_RECIPE["eval_batch_size"]
    with torch.inference_mode():
        for start in range(0, len(x_validation_num), eval_batch_size):
            indices = slice(start, min(start + eval_batch_size, len(x_validation_num)))
            logits = _tabm_logits(
                model, x_validation_num, x_validation_cat, indices, device
            )
            # Official classification inference: average probabilities, not logits.
            predictions.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
    predict_seconds = time.monotonic() - started
    prediction = np.concatenate(predictions).astype("float64")
    del model, optimizer, x_train_num, x_train_cat, x_validation_num, x_validation_cat
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("TabM produced invalid probability")
    return prediction, fit_seconds, predict_seconds, feature_columns, epoch_losses, device


def make_ft_model(encoding: TabMEncoding):
    cardinalities = [
        len(encoding.categorical_levels[column]) + 1
        for column in encoding.categorical_columns
    ]
    defaults = rtdl.FTTransformer.get_default_kwargs(
        n_blocks=FT_RECIPE["n_blocks"]
    )
    return rtdl.FTTransformer(
        n_cont_features=len(encoding.numeric_columns),
        cat_cardinalities=cardinalities,
        d_out=1,
        **defaults,
    )


def _ft_logits(model, numeric: np.ndarray, categorical: np.ndarray, indices, device):
    x_cont = torch.as_tensor(numeric[indices], dtype=torch.float32, device=device)
    x_cat = torch.as_tensor(categorical[indices], dtype=torch.long, device=device)
    return model(x_cont, x_cat).squeeze(-1)


def fit_ft_fold(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    champion_meta: Mapping,
) -> tuple[np.ndarray, float, float, list[str], list[float], str, int]:
    train_features = build_hgb52_features(training, champion_meta["cat_levels"])
    validation_features = build_hgb52_features(validation, champion_meta["cat_levels"])
    validate_raw_contract(train_features, champion_meta)
    validate_raw_contract(validation_features, champion_meta)
    encoding = fit_tabm_encoding(train_features)
    x_train_num, x_train_cat = transform_tabm(train_features, encoding)
    x_validation_num, x_validation_cat = transform_tabm(validation_features, encoding)
    y_train = training[TARGET].to_numpy(dtype="float32")
    feature_columns = list(train_features.columns)
    del train_features, validation_features
    gc.collect()

    seed = FT_RECIPE["random_state"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = tabm_device()
    model = make_ft_model(encoding).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = model.make_default_optimizer()
    rng = np.random.default_rng(seed)
    batch_size = FT_RECIPE["batch_size"]
    epoch_losses = []
    started = time.monotonic()
    for epoch in range(FT_RECIPE["epochs"]):
        model.train()
        loss_sum = 0.0
        seen = 0
        permutation = rng.permutation(len(y_train))
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            logits = _ft_logits(model, x_train_num, x_train_cat, indices, device)
            target = torch.as_tensor(
                y_train[indices], dtype=torch.float32, device=device
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        epoch_loss = loss_sum / seen
        epoch_losses.append(epoch_loss)
        print(
            f"  FT epoch {epoch + 1}/{FT_RECIPE['epochs']} loss={epoch_loss:.6f}",
            flush=True,
        )
    fit_seconds = time.monotonic() - started

    model.eval()
    predictions = []
    started = time.monotonic()
    eval_batch_size = FT_RECIPE["eval_batch_size"]
    with torch.inference_mode():
        for start in range(0, len(x_validation_num), eval_batch_size):
            indices = slice(start, min(start + eval_batch_size, len(x_validation_num)))
            logits = _ft_logits(
                model, x_validation_num, x_validation_cat, indices, device
            )
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    predict_seconds = time.monotonic() - started
    prediction = np.concatenate(predictions).astype("float64")
    del model, optimizer, x_train_num, x_train_cat, x_validation_num, x_validation_cat
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("FT-Transformer produced invalid probability")
    return (
        prediction,
        fit_seconds,
        predict_seconds,
        feature_columns,
        epoch_losses,
        device,
        parameter_count,
    )


def fit_xgb_fold(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    champion_meta: Mapping,
    n_jobs: int,
) -> tuple[np.ndarray, float, float, list[str]]:
    train_features = build_hgb52_features(training, champion_meta["cat_levels"])
    validation_features = build_hgb52_features(
        validation, champion_meta["cat_levels"]
    )
    validate_raw_contract(train_features, champion_meta)
    validate_raw_contract(validation_features, champion_meta)
    preprocessor = fit_preprocessor(train_features)
    x_train = transform_features(train_features, preprocessor)
    x_validation = transform_features(validation_features, preprocessor)
    y_train = training[TARGET].to_numpy(dtype="int8")
    feature_columns = list(train_features.columns)
    del train_features, validation_features
    gc.collect()

    model = make_xgb(n_jobs=n_jobs)
    started = time.monotonic()
    model.fit(x_train, y_train, verbose=False)
    fit_seconds = time.monotonic() - started
    started = time.monotonic()
    prediction = model.predict_proba(x_validation)[:, 1].astype("float64")
    predict_seconds = time.monotonic() - started
    del model, x_train, x_validation, y_train
    gc.collect()
    if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("XGBoost produced invalid probability")
    return prediction, fit_seconds, predict_seconds, feature_columns


def run_xgb_worker(args: argparse.Namespace) -> None:
    if sha256_path(CHAMPION) != CHAMPION_SHA256:
        raise SystemExit("immutable champion checksum mismatch")
    frame = pd.read_csv(args.train)
    training, validation = temporal_split(frame, args.worker_xgb_season)
    champion_meta = load_champion_meta(CHAMPION)
    print(
        f"[xgb hist {args.worker_xgb_season}] {len(training):,} -> "
        f"{len(validation):,}, 600 trees",
        flush=True,
    )
    prediction, fit_seconds, predict_seconds, columns = fit_xgb_fold(
        training, validation, champion_meta, args.n_jobs
    )
    target = validation[TARGET].to_numpy(dtype="float64")
    save_npz_atomic(
        cache_path(args.output, args.worker_xgb_season),
        cache_version=np.asarray(XGB_CACHE_VERSION),
        season=np.asarray(args.worker_xgb_season, dtype="int16"),
        target_sha256=np.asarray(array_sha256(target)),
        prediction=prediction,
        fit_seconds=np.asarray(fit_seconds),
        predict_seconds=np.asarray(predict_seconds),
        feature_columns=np.asarray(columns),
    )
    print(
        f"[xgb hist {args.worker_xgb_season}] fit={fit_seconds:.1f}s "
        f"predict={predict_seconds:.1f}s",
        flush=True,
    )


def run_tabm_worker(args: argparse.Namespace) -> None:
    if sha256_path(CHAMPION) != CHAMPION_SHA256:
        raise SystemExit("immutable champion checksum mismatch")
    frame = pd.read_csv(args.train)
    training, validation = temporal_split(frame, args.worker_tabm_season)
    champion_meta = load_champion_meta(CHAMPION)
    print(
        f"[TabM {args.worker_tabm_season}] {len(training):,} -> "
        f"{len(validation):,}, device={tabm_device()}",
        flush=True,
    )
    prediction, fit_seconds, predict_seconds, columns, losses, device = fit_tabm_fold(
        training, validation, champion_meta
    )
    target = validation[TARGET].to_numpy(dtype="float64")
    save_npz_atomic(
        tabm_cache_path(args.output, args.worker_tabm_season),
        cache_version=np.asarray(TABM_CACHE_VERSION),
        season=np.asarray(args.worker_tabm_season, dtype="int16"),
        target_sha256=np.asarray(array_sha256(target)),
        prediction=prediction,
        fit_seconds=np.asarray(fit_seconds),
        predict_seconds=np.asarray(predict_seconds),
        feature_columns=np.asarray(columns),
        epoch_losses=np.asarray(losses, dtype="float64"),
        device=np.asarray(device),
    )
    print(
        f"[TabM {args.worker_tabm_season}] fit={fit_seconds:.1f}s "
        f"predict={predict_seconds:.1f}s",
        flush=True,
    )


def run_ft_worker(args: argparse.Namespace) -> None:
    if sha256_path(CHAMPION) != CHAMPION_SHA256:
        raise SystemExit("immutable champion checksum mismatch")
    frame = pd.read_csv(args.train)
    training, validation = temporal_split(frame, args.worker_ft_season)
    champion_meta = load_champion_meta(CHAMPION)
    print(
        f"[FT-Transformer {args.worker_ft_season}] {len(training):,} -> "
        f"{len(validation):,}, device={tabm_device()}",
        flush=True,
    )
    (
        prediction,
        fit_seconds,
        predict_seconds,
        columns,
        losses,
        device,
        parameter_count,
    ) = fit_ft_fold(training, validation, champion_meta)
    target = validation[TARGET].to_numpy(dtype="float64")
    save_npz_atomic(
        ft_cache_path(args.output, args.worker_ft_season),
        cache_version=np.asarray(FT_CACHE_VERSION),
        season=np.asarray(args.worker_ft_season, dtype="int16"),
        target_sha256=np.asarray(array_sha256(target)),
        prediction=prediction,
        fit_seconds=np.asarray(fit_seconds),
        predict_seconds=np.asarray(predict_seconds),
        feature_columns=np.asarray(columns),
        epoch_losses=np.asarray(losses, dtype="float64"),
        device=np.asarray(device),
        parameter_count=np.asarray(parameter_count, dtype="int64"),
    )
    print(
        f"[FT-Transformer {args.worker_ft_season}] fit={fit_seconds:.1f}s "
        f"predict={predict_seconds:.1f}s",
        flush=True,
    )


def validate_cache(path: Path, season: int, target: np.ndarray) -> None:
    payload = load_npz(path)
    expected = (XGB_CACHE_VERSION, season, array_sha256(target))
    actual = (
        str(payload["cache_version"].item()),
        int(payload["season"].item()),
        str(payload["target_sha256"].item()),
    )
    if actual != expected:
        raise ValueError(f"stale XGBoost cache {path}: {actual} != {expected}")


def ensure_xgb_caches(args: argparse.Namespace) -> None:
    # Validate target hashes before trusting any cache. Parent reads train once later;
    # here existence is enough and the strict validation happens on load.
    for season in VALIDATION_SEASONS:
        path = cache_path(args.output, season)
        if path.is_file() and not args.force:
            print(f"[xgb hist {season}] cached", flush=True)
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--train",
            str(args.train),
            "--output",
            str(args.output),
            "--n-jobs",
            str(args.n_jobs),
            "--worker-xgb-season",
            str(season),
        ]
        subprocess.run(command, check=True)


def ensure_tabm_caches(args: argparse.Namespace) -> None:
    for season in VALIDATION_SEASONS:
        path = tabm_cache_path(args.output, season)
        if path.is_file() and not args.force:
            print(f"[TabM {season}] cached", flush=True)
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--train",
            str(args.train),
            "--output",
            str(args.output),
            "--worker-tabm-season",
            str(season),
        ]
        subprocess.run(command, check=True)


def ensure_ft_caches(args: argparse.Namespace) -> None:
    for season in VALIDATION_SEASONS:
        path = ft_cache_path(args.output, season)
        if path.is_file() and not args.force:
            print(f"[FT-Transformer {season}] cached", flush=True)
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--train",
            str(args.train),
            "--output",
            str(args.output),
            "--worker-ft-season",
            str(season),
        ]
        subprocess.run(command, check=True)


def load_fold_context(frame: pd.DataFrame, season: int) -> dict[str, np.ndarray]:
    validation = frame[frame["season"] == season]
    target = validation[TARGET].to_numpy(dtype="float64")
    oof = load_npz(OOF_DIR / f"season_{season}.npz")
    raw_et = load_npz(RAW_ET25_DIR / f"season_{season}_prefixes.npz")
    if not np.array_equal(target, oof["y"].astype("float64")):
        raise ValueError(f"OOF target/order mismatch for {season}")
    if str(raw_et["target_sha256"].item()) != array_sha256(target):
        raise ValueError(f"ET25 target checksum mismatch for {season}")
    original = oof["champion_final"].astype("float64")
    et25 = raw_et["prediction_25"].astype("float64")
    current = (1.0 - CURRENT_ET25_WEIGHT) * original + CURRENT_ET25_WEIGHT * et25
    return {
        "target": target,
        "original_champion": original,
        "current_champion": current,
        "et25": et25,
        "game_type": validation["game_type"].astype(str).to_numpy(),
        "pitcher_seen": oof["x_pitcher_seen"].astype(bool),
        "balls_before": validation["balls_before"].to_numpy(dtype="int64"),
        "strikes_before": validation["strikes_before"].to_numpy(dtype="int64"),
        "form_adjustment": (
            oof["pitcher_form_pred"].astype("float64")
            - oof["champion_base"].astype("float64")
        ),
    }


def load_xgb_prediction(
    output: Path, season: int, target: np.ndarray
) -> tuple[np.ndarray, float, float, tuple[str, ...]]:
    path = cache_path(output, season)
    validate_cache(path, season, target)
    payload = load_npz(path)
    prediction = payload["prediction"].astype("float64")
    if prediction.shape != target.shape:
        raise ValueError(f"XGBoost prediction length mismatch for {season}")
    return (
        prediction,
        float(payload["fit_seconds"].item()),
        float(payload["predict_seconds"].item()),
        tuple(str(value) for value in payload["feature_columns"]),
    )


def load_tabm_prediction(
    output: Path, season: int, target: np.ndarray
) -> tuple[np.ndarray, float, float, tuple[str, ...], list[float], str]:
    path = tabm_cache_path(output, season)
    payload = load_npz(path)
    expected = (TABM_CACHE_VERSION, season, array_sha256(target))
    actual = (
        str(payload["cache_version"].item()),
        int(payload["season"].item()),
        str(payload["target_sha256"].item()),
    )
    if actual != expected:
        raise ValueError(f"stale TabM cache {path}: {actual} != {expected}")
    prediction = payload["prediction"].astype("float64")
    if prediction.shape != target.shape:
        raise ValueError(f"TabM prediction length mismatch for {season}")
    return (
        prediction,
        float(payload["fit_seconds"].item()),
        float(payload["predict_seconds"].item()),
        tuple(str(value) for value in payload["feature_columns"]),
        payload["epoch_losses"].astype("float64").tolist(),
        str(payload["device"].item()),
    )


def load_ft_prediction(
    output: Path, season: int, target: np.ndarray
) -> tuple[np.ndarray, float, float, tuple[str, ...], list[float], str, int]:
    path = ft_cache_path(output, season)
    payload = load_npz(path)
    expected = (FT_CACHE_VERSION, season, array_sha256(target))
    actual = (
        str(payload["cache_version"].item()),
        int(payload["season"].item()),
        str(payload["target_sha256"].item()),
    )
    if actual != expected:
        raise ValueError(f"stale FT cache {path}: {actual} != {expected}")
    prediction = payload["prediction"].astype("float64")
    if prediction.shape != target.shape:
        raise ValueError(f"FT prediction length mismatch for {season}")
    return (
        prediction,
        float(payload["fit_seconds"].item()),
        float(payload["predict_seconds"].item()),
        tuple(str(value) for value in payload["feature_columns"]),
        payload["epoch_losses"].astype("float64").tolist(),
        str(payload["device"].item()),
        int(payload["parameter_count"].item()),
    )


def error_top10_win_rate(
    target: np.ndarray,
    reference: np.ndarray,
    challenger: np.ndarray,
) -> float:
    reference_error = (target - reference) ** 2
    threshold = float(np.quantile(reference_error, 0.90))
    mask = reference_error >= threshold
    challenger_error = (target[mask] - challenger[mask]) ** 2
    return float(np.mean(challenger_error < reference_error[mask]))


def diagnostic_row(
    learner: str,
    season: int | str,
    target: np.ndarray,
    model: np.ndarray,
    champion: np.ndarray,
    et25: np.ndarray,
    fit_seconds: float,
    predict_seconds: float,
) -> dict:
    return {
        "learner": learner,
        "validation_season": season,
        "n": len(target),
        "standalone_brier": brier(target, model),
        "current_champion_brier": brier(target, champion),
        "standalone_gain_vs_champion": brier(target, champion) - brier(target, model),
        "corr_model_champion": safe_corr(model, champion),
        "corr_model_et25": safe_corr(model, et25),
        "champion_top10_error_model_win_rate": error_top10_win_rate(
            target, champion, model
        ),
        "model_top10_error_champion_win_rate": error_top10_win_rate(
            target, model, champion
        ),
        "prediction_mean": float(np.mean(model)),
        "prediction_std": float(np.std(model)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def complementarity_gate(rows: list[dict]) -> dict:
    lookup = {str(row["validation_season"]): row for row in rows}
    latest_safe = all(
        lookup[str(season)]["standalone_gain_vs_champion"]
        >= CATASTROPHIC_STANDALONE_FLOOR
        for season in (2023, 2024)
    )
    path_a = (
        max(
            lookup[str(season)]["standalone_gain_vs_champion"]
            for season in VALIDATION_SEASONS
        )
        >= CLEAR_STANDALONE_GAIN
        and latest_safe
    )
    latest_win_rates = [
        lookup[str(season)]["champion_top10_error_model_win_rate"]
        for season in (2023, 2024)
    ]
    path_b = (
        lookup["pooled"]["corr_model_champion"] <= MAX_DIVERSITY_CORRELATION
        and all(rate >= MIN_CHAMPION_TOP10_WIN_RATE for rate in latest_win_rates)
        and abs(latest_win_rates[0] - latest_win_rates[1])
        <= MAX_TOP10_WIN_RATE_SEASON_GAP
        and latest_safe
    )
    reasons = []
    if not path_a:
        reasons.append("path A failed: no clear standalone fold win or latest fold unsafe")
    if not path_b:
        reasons.append("path B failed: diversity/top10/latest-direction gate")
    return {
        "path_a_passed": path_a,
        "path_b_passed": path_b,
        "passed": path_a or path_b,
        "latest_safe": latest_safe,
        "latest_top10_win_rate_gap": abs(latest_win_rates[0] - latest_win_rates[1]),
        "reasons": reasons,
    }


def worst_subset_gain(
    folds: Mapping[int, Mapping], season: int | str, weight: float
) -> tuple[float, dict]:
    records = []
    template = subset_masks(folds[VALIDATION_SEASONS[0]])
    for axis, subset, _ in template:
        targets = []
        champions = []
        candidates = []
        seasons = VALIDATION_SEASONS if season == "pooled" else (int(season),)
        for current_season in seasons:
            fold = folds[current_season]
            mask = next(
                current_mask
                for current_axis, current_subset, current_mask in subset_masks(fold)
                if current_axis == axis and current_subset == subset
            )
            target = np.asarray(fold["target"])[mask]
            champion = np.asarray(fold["current_champion"])[mask]
            model = np.asarray(fold["model"])[mask]
            targets.append(target)
            champions.append(champion)
            candidates.append((1.0 - weight) * champion + weight * model)
        target = np.concatenate(targets)
        champion = np.concatenate(champions)
        candidate = np.concatenate(candidates)
        if len(target) >= MIN_SUBSET_ROWS:
            records.append(
                {
                    "axis": axis,
                    "subset": subset,
                    "n": len(target),
                    "gain": brier(target, champion) - brier(target, candidate),
                }
            )
    worst = min(records, key=lambda row: row["gain"])
    return float(worst["gain"]), worst


def build_blends(
    learner: str, folds: Mapping[int, Mapping]
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    decisions = []
    for weight in BLEND_WEIGHTS:
        current_rows = []
        worst_details = []
        for season in (*VALIDATION_SEASONS, "pooled"):
            if season == "pooled":
                target = pooled_array(folds, "target")
                champion = pooled_array(folds, "current_champion")
                model = pooled_array(folds, "model")
            else:
                fold = folds[int(season)]
                target = np.asarray(fold["target"])
                champion = np.asarray(fold["current_champion"])
                model = np.asarray(fold["model"])
            candidate = (1.0 - weight) * champion + weight * model
            worst_gain, worst = worst_subset_gain(folds, season, weight)
            worst_details.append({"validation_season": season, **worst})
            row = {
                "learner": learner,
                "weight": weight,
                "validation_season": season,
                "n": len(target),
                "current_champion_brier": brier(target, champion),
                "candidate_brier": brier(target, candidate),
                "brier_gain": brier(target, champion) - brier(target, candidate),
                "worst_major_subset_gain": worst_gain,
            }
            rows.append(row)
            current_rows.append(row)
        lookup = {str(row["validation_season"]): row for row in current_rows}
        passed = (
            lookup["2023"]["brier_gain"] > 0.0
            and lookup["2024"]["brier_gain"] > 0.0
            and lookup["pooled"]["brier_gain"] > 0.0
            and min(lookup[str(season)]["brier_gain"] for season in VALIDATION_SEASONS)
            >= WORST_FOLD_BLEND_FLOOR
            and min(row["worst_major_subset_gain"] for row in current_rows)
            >= MAJOR_SUBSET_FLOOR
        )
        for row in current_rows:
            row["production_gate_passed"] = passed
        decisions.append(
            {
                "learner": learner,
                "weight": weight,
                "gains": {key: lookup[key]["brier_gain"] for key in lookup},
                "worst_major_subset_gain": min(
                    row["worst_major_subset_gain"] for row in current_rows
                ),
                "worst_major_subsets": worst_details,
                "passed": passed,
            }
        )
    return pd.DataFrame(rows, columns=BLEND_COLUMNS), decisions


def current_champion_parity(folds: Mapping[int, Mapping]) -> list[dict]:
    reference = pd.read_csv(WEIGHT_REFERENCE)
    reference = reference[np.isclose(reference["weight"], CURRENT_ET25_WEIGHT)]
    rows = []
    for season in (*VALIDATION_SEASONS, "pooled"):
        if season == "pooled":
            target = pooled_array(folds, "target")
            original = pooled_array(folds, "original_champion")
            current = pooled_array(folds, "current_champion")
        else:
            fold = folds[int(season)]
            target = np.asarray(fold["target"])
            original = np.asarray(fold["original_champion"])
            current = np.asarray(fold["current_champion"])
        reconstructed = brier(target, original) - brier(target, current)
        matched = reference[reference["validation_season"].astype(str).eq(str(season))]
        if len(matched) != 1:
            raise ValueError(f"missing frozen current-champion reference for {season}")
        expected = float(matched.iloc[0]["brier_gain"])
        rows.append(
            {
                "validation_season": season,
                "reconstructed_gain": reconstructed,
                "reference_gain": expected,
                "absolute_diff": abs(reconstructed - expected),
                "passed": abs(reconstructed - expected) <= 1e-15,
            }
        )
    if not all(row["passed"] for row in rows):
        raise RuntimeError("current champion OOF parity failed")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-xgb-season", type=int, choices=VALIDATION_SEASONS)
    parser.add_argument("--worker-tabm-season", type=int, choices=VALIDATION_SEASONS)
    parser.add_argument("--worker-ft-season", type=int, choices=VALIDATION_SEASONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker_xgb_season:
        run_xgb_worker(args)
        return
    if args.worker_tabm_season:
        run_tabm_worker(args)
        return
    if args.worker_ft_season:
        run_ft_worker(args)
        return

    champion_sha_before = sha256_path(CHAMPION)
    if champion_sha_before != CHAMPION_SHA256:
        raise SystemExit("immutable champion checksum mismatch before experiment")
    ensure_xgb_caches(args)

    frame = pd.read_csv(args.train)
    folds = {season: load_fold_context(frame, season) for season in VALIDATION_SEASONS}
    feature_contract = None
    result_rows = []
    for season in VALIDATION_SEASONS:
        fold = folds[season]
        model, fit_seconds, predict_seconds, columns = load_xgb_prediction(
            args.output, season, np.asarray(fold["target"])
        )
        if feature_contract is None:
            feature_contract = columns
        elif columns != feature_contract:
            raise ValueError("XGBoost feature contract varies by fold")
        fold["model"] = model
        result_rows.append(
            diagnostic_row(
                "xgboost_hist",
                season,
                np.asarray(fold["target"]),
                model,
                np.asarray(fold["current_champion"]),
                np.asarray(fold["et25"]),
                fit_seconds,
                predict_seconds,
            )
        )
    result_rows.append(
        diagnostic_row(
            "xgboost_hist",
            "pooled",
            pooled_array(folds, "target"),
            pooled_array(folds, "model"),
            pooled_array(folds, "current_champion"),
            pooled_array(folds, "et25"),
            sum(row["fit_seconds"] for row in result_rows),
            sum(row["predict_seconds"] for row in result_rows),
        )
    )
    results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    results.to_csv(args.output / "xgb_results.csv", index=False)

    gate = complementarity_gate(result_rows)
    complement = results.loc[:, RESULT_COLUMNS[:10]].copy()
    complement["path_a_passed"] = gate["path_a_passed"]
    complement["path_b_passed"] = gate["path_b_passed"]
    complement["complementarity_gate_passed"] = gate["passed"]

    tabm_rows = []
    tabm_gate = None
    tabm_training = []
    ft_rows = []
    ft_gate = None
    ft_training = []
    active_learner = None
    active_folds = None

    if gate["passed"]:
        active_learner = "xgboost_hist"
        active_folds = folds
    else:
        ensure_tabm_caches(args)
        tabm_folds = {season: {**folds[season]} for season in VALIDATION_SEASONS}
        tabm_contract = None
        for season in VALIDATION_SEASONS:
            fold = tabm_folds[season]
            model, fit_seconds, predict_seconds, columns, losses, device = load_tabm_prediction(
                args.output, season, np.asarray(fold["target"])
            )
            if tabm_contract is None:
                tabm_contract = columns
            elif columns != tabm_contract:
                raise ValueError("TabM feature contract varies by fold")
            if columns != feature_contract:
                raise ValueError("TabM and XGBoost do not share the exact HGB-52 contract")
            fold["model"] = model
            tabm_rows.append(
                diagnostic_row(
                    "tabm",
                    season,
                    np.asarray(fold["target"]),
                    model,
                    np.asarray(fold["current_champion"]),
                    np.asarray(fold["et25"]),
                    fit_seconds,
                    predict_seconds,
                )
            )
            tabm_training.append(
                {
                    "validation_season": season,
                    "device": device,
                    "epoch_losses": losses,
                }
            )
        tabm_rows.append(
            diagnostic_row(
                "tabm",
                "pooled",
                pooled_array(tabm_folds, "target"),
                pooled_array(tabm_folds, "model"),
                pooled_array(tabm_folds, "current_champion"),
                pooled_array(tabm_folds, "et25"),
                sum(row["fit_seconds"] for row in tabm_rows),
                sum(row["predict_seconds"] for row in tabm_rows),
            )
        )
        tabm_results = pd.DataFrame(tabm_rows, columns=RESULT_COLUMNS)
        tabm_results.to_csv(args.output / "tabm_results.csv", index=False)
        tabm_gate = complementarity_gate(tabm_rows)
        tabm_complement = tabm_results.loc[:, RESULT_COLUMNS[:10]].copy()
        tabm_complement["path_a_passed"] = tabm_gate["path_a_passed"]
        tabm_complement["path_b_passed"] = tabm_gate["path_b_passed"]
        tabm_complement["complementarity_gate_passed"] = tabm_gate["passed"]
        complement = pd.concat([complement, tabm_complement], ignore_index=True)
        if tabm_gate["passed"]:
            active_learner = "tabm"
            active_folds = tabm_folds

    if tabm_gate is not None and not tabm_gate["passed"]:
        ensure_ft_caches(args)
        ft_folds = {season: {**folds[season]} for season in VALIDATION_SEASONS}
        ft_contract = None
        for season in VALIDATION_SEASONS:
            fold = ft_folds[season]
            (
                model,
                fit_seconds,
                predict_seconds,
                columns,
                losses,
                device,
                parameter_count,
            ) = load_ft_prediction(args.output, season, np.asarray(fold["target"]))
            if ft_contract is None:
                ft_contract = columns
            elif columns != ft_contract:
                raise ValueError("FT-Transformer feature contract varies by fold")
            if columns != feature_contract:
                raise ValueError(
                    "FT-Transformer and XGBoost do not share the exact HGB-52 contract"
                )
            fold["model"] = model
            ft_rows.append(
                diagnostic_row(
                    "ft_transformer",
                    season,
                    np.asarray(fold["target"]),
                    model,
                    np.asarray(fold["current_champion"]),
                    np.asarray(fold["et25"]),
                    fit_seconds,
                    predict_seconds,
                )
            )
            ft_training.append(
                {
                    "validation_season": season,
                    "device": device,
                    "parameter_count": parameter_count,
                    "epoch_losses": losses,
                }
            )
        ft_rows.append(
            diagnostic_row(
                "ft_transformer",
                "pooled",
                pooled_array(ft_folds, "target"),
                pooled_array(ft_folds, "model"),
                pooled_array(ft_folds, "current_champion"),
                pooled_array(ft_folds, "et25"),
                sum(row["fit_seconds"] for row in ft_rows),
                sum(row["predict_seconds"] for row in ft_rows),
            )
        )
        ft_results = pd.DataFrame(ft_rows, columns=RESULT_COLUMNS)
        ft_results.to_csv(args.output / "ft_transformer_results.csv", index=False)
        ft_gate = complementarity_gate(ft_rows)
        ft_complement = ft_results.loc[:, RESULT_COLUMNS[:10]].copy()
        ft_complement["path_a_passed"] = ft_gate["path_a_passed"]
        ft_complement["path_b_passed"] = ft_gate["path_b_passed"]
        ft_complement["complementarity_gate_passed"] = ft_gate["passed"]
        complement = pd.concat([complement, ft_complement], ignore_index=True)
        if ft_gate["passed"]:
            active_learner = "ft_transformer"
            active_folds = ft_folds

    complement.to_csv(args.output / "complementarity.csv", index=False)
    if active_learner is not None and active_folds is not None:
        blends, blend_decisions = build_blends(active_learner, active_folds)
    else:
        blends = pd.DataFrame(columns=BLEND_COLUMNS)
        blend_decisions = []
    blends.to_csv(args.output / "blend_results.csv", index=False)

    passing = [decision for decision in blend_decisions if decision["passed"]]
    selected = max(
        passing,
        key=lambda decision: (decision["gains"]["pooled"], -decision["weight"]),
        default=None,
    )
    if selected and selected["gains"]["pooled"] >= LARGE_SIGNAL_GAIN:
        verdict = "A. LARGE COMPLEMENTARY MODEL SIGNAL"
    elif selected:
        verdict = "B. stable incremental signal"
    else:
        verdict = "C. no useful model diversity signal"

    sha_after = sha256_path(CHAMPION)
    if sha_after != champion_sha_before:
        raise RuntimeError("immutable champion changed during model-diversity experiment")
    parity = current_champion_parity(folds)
    summary = {
        "verdict": verdict,
        "champion": {
            "artifact": str(CHAMPION),
            "lb_bss": CHAMPION_LB_BSS,
            "sha256_before": champion_sha_before,
            "sha256_after": sha_after,
            "unchanged": sha_after == champion_sha_before == CHAMPION_SHA256,
            "oof_parity": parity,
        },
        "feature_contract": {
            "name": "accepted ID-free raw HGB-52",
            "columns": list(feature_contract or ()),
            "feature_count": len(feature_contract or ()),
            "new_features_added": False,
            "preprocessing": {
                "xgboost": "cutoff-train-only ordinal categories and numeric medians",
                "tabm_and_ft_transformer": (
                    "cutoff-train-only category vocabulary, numeric medians, and z-score"
                ),
                "validation_statistics_used": False,
            },
        },
        "xgboost": {
            "version": xgboost.__version__,
            "recipe": {**XGB_RECIPE, "n_jobs": args.n_jobs},
            "early_stopping": False,
            "hyperparameter_search": False,
            "results": result_rows,
            "complementarity_gate": gate,
            "blend_decisions": blend_decisions,
        },
        "stage_policy": {
            "tabm_runs_only_if_xgb_complementarity_fails": True,
            "tabm_required": not gate["passed"],
            "tabm_evaluated": bool(tabm_rows),
            "ft_transformer_required": bool(tabm_gate is not None and not tabm_gate["passed"]),
            "ft_transformer_evaluated": bool(ft_rows),
        },
        "tabm": {
            "version": tabm.__version__,
            "recipe": TABM_RECIPE,
            "official_training_contract": "independent head loss; probability mean at inference",
            "hyperparameter_search": False,
            "results": tabm_rows,
            "training": tabm_training,
            "complementarity_gate": tabm_gate,
            "blend_decisions": blend_decisions if active_learner == "tabm" else [],
        },
        "ft_transformer": {
            "version": rtdl.__version__,
            "architecture_audit": {
                "existing_ours_nn": "player-embedding + two-layer ReLU MLP",
                "candidate": "per-feature token embeddings + multi-head self-attention + CLS token",
                "effectively_identical": False,
            },
            "recipe": FT_RECIPE,
            "official_default_backbone": rtdl.FTTransformer.get_default_kwargs(
                n_blocks=FT_RECIPE["n_blocks"]
            ),
            "hyperparameter_search": False,
            "results": ft_rows,
            "training": ft_training,
            "complementarity_gate": ft_gate,
            "blend_decisions": blend_decisions
            if active_learner == "ft_transformer"
            else [],
        },
        "selected_candidate": selected,
        "production": {
            "created": False,
            "reason": "no evaluated learner blend passed"
            if selected is None
            else "OOF passed; guarded full-train package step remains",
            "filename_length_gate": 30,
        },
        "data": {
            "train": str(args.train),
            "test_read": False,
            "validation_seasons": VALIDATION_SEASONS,
        },
    }
    write_json(args.output / "summary.json", summary)
    print("XGBoost gate", json.dumps(gate, indent=2), flush=True)
    if tabm_gate is not None:
        print("TabM gate", json.dumps(tabm_gate, indent=2), flush=True)
    if ft_gate is not None:
        print("FT-Transformer gate", json.dumps(ft_gate, indent=2), flush=True)
    print(verdict, flush=True)


if __name__ == "__main__":
    main()
