"""Validate row-local pitcher control-profile representations with temporal OOF.

Each candidate is the frozen ID-free HGB-52 feature contract plus exactly one
domain family.  The learner is a single 25-tree ExtraTreesClassifier with the
accepted ET recipe.  Family gain is measured against the accepted raw HGB-52
ET25 prediction; complementarity and blends are measured against the current
immutable champion (98% original champion + 2% accepted ET25).

Only current-row official columns and official ``asof_*`` columns are used.
No test file is read and no validation label is used in feature construction.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime_blend_oof import load_champion_meta  # noqa: E402
from scripts.extratrees_runtime import (  # noqa: E402
    build_hgb52_features,
    fit_preprocessor,
    transform_features,
)


DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
OUTPUT_DIR = ROOT / "artifacts" / "control_profile"
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
FAMILIES = (
    "long_term_control",
    "multi_timescale_form",
    "deterioration_type",
    "recent_consistency",
    "pitcher_batter_relative",
    "pitch_mix",
    "control_state_leverage",
)
FAMILY_LABELS = {
    "long_term_control": "A",
    "multi_timescale_form": "B",
    "deterioration_type": "C",
    "recent_consistency": "D",
    "pitcher_batter_relative": "E",
    "pitch_mix": "F",
    "control_state_leverage": "G",
}
BLEND_WEIGHTS = (0.005, 0.010, 0.020, 0.050)

N_ESTIMATORS = 25
RANDOM_STATE = 42
MIN_SAMPLES_LEAF = 8
MAX_FEATURES = "sqrt"
CURRENT_RAW_ET_WEIGHT = 0.020
STRONG_POOLED_GAIN = 1e-5
WEAK_WORST_FOLD_FLOOR = -2e-5
MAJOR_SUBSET_FLOOR = -1e-5
MIN_SUBSET_ROWS = 5_000
MAX_CHAMPION_CORRELATION = 0.98
MAX_RESIDUAL_CORRELATION = 0.995
MIN_TOP10_WIN_RATE = 0.50
CACHE_VERSION = "control-profile-et25-v1"

DETERIORATION_LEVELS = (
    "STABLE",
    "SUCCESS_DOWN_MIDDLE_DOWN",
    "SUCCESS_DOWN_MIDDLE_FLAT",
    "SUCCESS_DOWN_MIDDLE_UP",
    "SUCCESS_FLAT_MIDDLE_DOWN",
    "SUCCESS_FLAT_MIDDLE_FLAT",
    "SUCCESS_FLAT_MIDDLE_UP",
    "SUCCESS_UP_MIDDLE_DOWN",
    "SUCCESS_UP_MIDDLE_FLAT",
    "SUCCESS_UP_MIDDLE_UP",
    "UNKNOWN",
)
LEVERAGE_LEVELS = ("NORMAL", "HIGH", "UNKNOWN")
DETERIORATION_FULL_LEVELS = tuple(
    f"{label}|FULL={flag}"
    for label in DETERIORATION_LEVELS
    for flag in ("False", "True")
)
DETERIORATION_LEVERAGE_LEVELS = tuple(
    f"{label}|{leverage}"
    for label in DETERIORATION_LEVELS
    for leverage in LEVERAGE_LEVELS
)

FAMILY_OOF_COLUMNS = (
    "family",
    "family_label",
    "validation_season",
    "n",
    "raw_et25_brier",
    "domain_et25_brier",
    "feature_gain_vs_raw_et25",
    "current_champion_brier",
    "domain_gain_vs_current_champion",
    "prediction_mean",
    "prediction_std",
    "fit_seconds",
    "grade",
)
COMPLEMENT_COLUMNS = (
    "family",
    "grade",
    "validation_season",
    "n",
    "champion_correlation",
    "residual_correlation",
    "champion_top10_error_domain_win_rate",
    "gate_passed",
)
BLEND_COLUMNS = (
    "family",
    "grade",
    "weight",
    "validation_season",
    "n",
    "current_champion_brier",
    "candidate_brier",
    "brier_gain",
    "worst_major_subset_gain",
    "family_complementarity_gate_passed",
    "production_gate_passed",
)


@dataclass(frozen=True)
class ControlProfileState:
    high_leverage_threshold: float


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
        return {name: archive[name] for name in archive.files}


def save_npz_atomic(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def write_json(path: Path, payload: Mapping) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), ensure_ascii=False, indent=2) + "\n",
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
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def temporal_split(
    frame: pd.DataFrame, validation_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = frame[frame["season"] < validation_season]
    validation = frame[frame["season"] == validation_season]
    if training.empty or validation.empty:
        raise ValueError(f"empty temporal split for {validation_season}")
    if int(training["season"].max()) >= validation_season:
        raise ValueError("validation/future season leaked into training")
    return training, validation


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _finite_quantile(values: np.ndarray, probability: float, default: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if len(finite) else float(default)


def fit_control_profile_state(training: pd.DataFrame) -> ControlProfileState:
    """Fit the only learned feature threshold from cutoff-training rows."""
    return ControlProfileState(
        high_leverage_threshold=_finite_quantile(
            _numeric(training, "li"), 0.75, 1.0
        )
    )


def _difference(frame: pd.DataFrame, left: str, right: str) -> np.ndarray:
    return _numeric(frame, left) - _numeric(frame, right)


def _recent_arrays(frame: pd.DataFrame, prefix: str) -> tuple[np.ndarray, ...]:
    return (
        _numeric(frame, f"asof_pitcher_prev1_game_{prefix}_rate"),
        _numeric(frame, f"asof_pitcher_prev3_game_{prefix}_rate"),
        _numeric(frame, f"asof_pitcher_prev5_game_{prefix}_rate"),
        _numeric(frame, f"asof_pitcher_{prefix}_rate"),
    )


def _deterioration_labels(frame: pd.DataFrame) -> np.ndarray:
    s1, s3, s5, sc = _recent_arrays(frame, "success")
    m1, m3, m5, mc = _recent_arrays(frame, "middle")
    del s1, s5, m1, m5
    success_dev = s3 - sc
    middle_dev = m3 - mc
    valid = np.isfinite(success_dev) & np.isfinite(middle_dev)
    success_direction = np.where(
        success_dev > 0.0, "SUCCESS_UP", np.where(success_dev < 0.0, "SUCCESS_DOWN", "SUCCESS_FLAT")
    )
    middle_direction = np.where(
        middle_dev > 0.0, "MIDDLE_UP", np.where(middle_dev < 0.0, "MIDDLE_DOWN", "MIDDLE_FLAT")
    )
    labels = np.char.add(np.char.add(success_direction, "_"), middle_direction)
    labels[(success_dev == 0.0) & (middle_dev == 0.0) & valid] = "STABLE"
    labels[~valid] = "UNKNOWN"
    return labels


def _recent_consistency_values(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    s1, s3, s5, _ = _recent_arrays(frame, "success")
    m1, m3, m5, _ = _recent_arrays(frame, "middle")
    success_valid = np.isfinite(s1) & np.isfinite(s3) & np.isfinite(s5)
    middle_valid = np.isfinite(m1) & np.isfinite(m3) & np.isfinite(m5)
    success_range = np.where(
        success_valid,
        np.maximum.reduce([s1, s3, s5]) - np.minimum.reduce([s1, s3, s5]),
        np.nan,
    )
    middle_range = np.where(
        middle_valid,
        np.maximum.reduce([m1, m3, m5]) - np.minimum.reduce([m1, m3, m5]),
        np.nan,
    )
    return {
        "ctrl_success_monotonic_improving": (
            success_valid & (s1 >= s3) & (s3 >= s5) & (s1 > s5)
        ).astype("int8"),
        "ctrl_success_monotonic_declining": (
            success_valid & (s1 <= s3) & (s3 <= s5) & (s1 < s5)
        ).astype("int8"),
        # A lower recent middle rate is treated as the favorable direction.
        "ctrl_middle_monotonic_improving": (
            middle_valid & (m1 <= m3) & (m3 <= m5) & (m1 < m5)
        ).astype("int8"),
        "ctrl_middle_monotonic_declining": (
            middle_valid & (m1 >= m3) & (m3 >= m5) & (m1 > m5)
        ).astype("int8"),
        "ctrl_success_recent_range": success_range,
        "ctrl_middle_recent_range": middle_range,
        "ctrl_recent_instability": (success_range + middle_range) / 2.0,
        "ctrl_recent_consistency_available": (success_valid & middle_valid).astype("int8"),
    }


def _pitch_mix_values(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    fastball = _numeric(frame, "asof_pitcher_fastball_rate")
    breaking = _numeric(frame, "asof_pitcher_breaking_rate")
    offspeed = _numeric(frame, "asof_pitcher_offspeed_rate")
    raw = np.column_stack([fastball, breaking, offspeed])
    valid = np.isfinite(raw).all(axis=1)
    clipped = np.clip(np.where(np.isfinite(raw), raw, 0.0), 0.0, None)
    total = clipped.sum(axis=1)
    shares = np.divide(
        clipped,
        total[:, None],
        out=np.zeros_like(clipped),
        where=total[:, None] > 0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(shares > 0.0, shares * np.log(shares), 0.0)
    entropy = -entropy_terms.sum(axis=1)
    entropy[~valid | (total <= 0.0)] = np.nan
    concentration = np.sum(shares**2, axis=1)
    concentration[~valid | (total <= 0.0)] = np.nan

    pitcher_n = _numeric(frame, "asof_pitcher_n")
    pitchmix_n = _numeric(frame, "asof_pitcher_pitchmix_n")
    coverage = np.divide(
        pitchmix_n,
        np.maximum(pitcher_n, 1.0),
        out=np.full(len(frame), np.nan),
        where=np.isfinite(pitchmix_n) & np.isfinite(pitcher_n),
    )
    success = _numeric(frame, "asof_pitcher_success_rate")
    success_dev3 = _difference(
        frame,
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_success_rate",
    )
    two_strike = (_numeric(frame, "strikes_before") == 2.0).astype("int8")
    return {
        "ctrl_primary_pitch_share": np.where(valid, np.max(raw, axis=1), np.nan),
        "ctrl_pitchmix_entropy": entropy,
        "ctrl_fastball_minus_breaking": fastball - breaking,
        "ctrl_fastball_minus_offspeed": fastball - offspeed,
        "ctrl_pitchmix_concentration": concentration,
        "ctrl_pitchmix_coverage": coverage,
        "ctrl_fastball_x_success": fastball * success,
        "ctrl_breaking_x_success": breaking * success,
        "ctrl_concentration_x_recent_form": concentration * success_dev3,
        "ctrl_concentration_x_two_strike": concentration * two_strike,
        "ctrl_concentration_x_log_pitchmix_n": concentration
        * np.log1p(np.clip(pitchmix_n, 0.0, None)),
        "ctrl_pitchmix_available": valid.astype("int8"),
    }


def build_family_features(
    frame: pd.DataFrame,
    family: str,
    state: ControlProfileState,
) -> pd.DataFrame:
    """Return only the requested family; output is row-local and index-stable."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    output = pd.DataFrame(index=frame.index)

    if family == "long_term_control":
        # control_margin and strike_ball_balance are exact champion duplicates.
        output["ctrl_danger_failure_balance"] = _difference(
            frame, "asof_pitcher_middle_rate", "asof_pitcher_ball_rate"
        )
        output["ctrl_reverse_adjusted_control"] = _difference(
            frame, "asof_pitcher_success_rate", "asof_pitcher_reverse_rate"
        )

    elif family == "multi_timescale_form":
        s1, s3, s5, sc = _recent_arrays(frame, "success")
        m1, m3, m5, mc = _recent_arrays(frame, "middle")
        # success_dev_5 and success_1v5 are exact champion duplicates.
        output["ctrl_success_dev_1"] = s1 - sc
        output["ctrl_success_dev_3"] = s3 - sc
        output["ctrl_success_1v3"] = s1 - s3
        output["ctrl_success_3v5"] = s3 - s5
        output["ctrl_middle_dev_1"] = m1 - mc
        output["ctrl_middle_dev_3"] = m3 - mc
        output["ctrl_middle_dev_5"] = m5 - mc
        output["ctrl_middle_1v3"] = m1 - m3
        output["ctrl_middle_1v5"] = m1 - m5
        output["ctrl_middle_3v5"] = m3 - m5
        output["ctrl_success_recent_up"] = (s3 > sc).astype("int8")
        output["ctrl_success_recent_down"] = (s3 < sc).astype("int8")
        output["ctrl_middle_recent_up"] = (m3 > mc).astype("int8")
        output["ctrl_middle_recent_down"] = (m3 < mc).astype("int8")
        output["ctrl_recent_form_available"] = (
            np.isfinite(s3) & np.isfinite(sc) & np.isfinite(m3) & np.isfinite(mc)
        ).astype("int8")
        log_pitcher_n = np.log1p(
            np.clip(_numeric(frame, "asof_pitcher_n"), 0.0, None)
        )
        output["ctrl_success_dev3_x_log_pitcher_n"] = (s3 - sc) * log_pitcher_n
        output["ctrl_middle_dev3_x_log_pitcher_n"] = (m3 - mc) * log_pitcher_n

    elif family == "deterioration_type":
        labels = _deterioration_labels(frame)
        full_count = (
            (_numeric(frame, "balls_before") == 3.0)
            & (_numeric(frame, "strikes_before") == 2.0)
        )
        output["ctrl_deterioration_type"] = pd.Categorical(
            labels, categories=DETERIORATION_LEVELS
        )
        output["ctrl_deterioration_x_full_count"] = pd.Categorical(
            np.char.add(np.char.add(labels, "|FULL="), full_count.astype(str)),
            categories=DETERIORATION_FULL_LEVELS,
        )

    elif family == "recent_consistency":
        values = _recent_consistency_values(frame)
        for name, values_array in values.items():
            output[name] = values_array
        three_ball = (_numeric(frame, "balls_before") == 3.0).astype("int8")
        output["ctrl_instability_x_three_ball"] = (
            output["ctrl_recent_instability"].to_numpy(dtype="float64") * three_ball
        )

    elif family == "pitcher_batter_relative":
        output["ctrl_pitcher_batter_success_gap"] = _difference(
            frame, "asof_pitcher_success_rate", "asof_batter_success_rate"
        )
        output["ctrl_pitcher_batter_middle_gap"] = _difference(
            frame, "asof_pitcher_middle_rate", "asof_batter_middle_rate"
        )
        output["ctrl_batter_profile_available"] = (
            np.isfinite(_numeric(frame, "asof_batter_success_rate"))
            & np.isfinite(_numeric(frame, "asof_batter_middle_rate"))
            & (_numeric(frame, "asof_batter_n") > 0.0)
        ).astype("int8")

    elif family == "pitch_mix":
        for name, values_array in _pitch_mix_values(frame).items():
            output[name] = values_array

    elif family == "control_state_leverage":
        li = _numeric(frame, "li")
        log_li = np.log1p(np.clip(li, 0.0, None))
        score = _numeric(frame, "score_diff_pitcher_team")
        inning = _numeric(frame, "inning")
        close = np.isfinite(score) & (np.abs(score) <= 1.0)
        late_close = close & np.isfinite(inning) & (inning >= 7.0)
        high = np.isfinite(li) & (li >= state.high_leverage_threshold)
        leverage_label = np.where(np.isfinite(li), np.where(high, "HIGH", "NORMAL"), "UNKNOWN")
        sdev3 = _difference(
            frame,
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_success_rate",
        )
        mdev3 = _difference(
            frame,
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_middle_rate",
        )
        consistency = _recent_consistency_values(frame)
        deterioration = _deterioration_labels(frame)
        output["ctrl_log_li"] = log_li
        output["ctrl_close_game"] = close.astype("int8")
        output["ctrl_late_close"] = late_close.astype("int8")
        output["ctrl_high_leverage"] = pd.Categorical(
            leverage_label, categories=LEVERAGE_LEVELS
        )
        output["ctrl_success_dev3_x_log_li"] = sdev3 * log_li
        output["ctrl_middle_dev3_x_log_li"] = mdev3 * log_li
        output["ctrl_instability_x_log_li"] = (
            consistency["ctrl_recent_instability"] * log_li
        )
        output["ctrl_deterioration_x_leverage"] = pd.Categorical(
            np.char.add(np.char.add(deterioration, "|"), leverage_label),
            categories=DETERIORATION_LEVERAGE_LEVELS,
        )

    if output.empty or not output.index.equals(frame.index):
        raise RuntimeError(f"invalid row-local output for family {family}")
    return output


def add_family_to_base(
    frame: pd.DataFrame,
    family: str,
    state: ControlProfileState,
    champion_meta: Mapping,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    base = build_hgb52_features(frame, champion_meta["cat_levels"])
    hgb = next(member for member in champion_meta["members"] if member["name"] == "hgb")
    if list(base.columns) != list(hgb["feature_columns"]):
        raise ValueError("raw HGB-52 feature contract drifted")
    domain = build_family_features(frame, family, state)
    overlap = set(base.columns) & set(domain.columns)
    if overlap:
        raise ValueError(f"domain feature duplicates base column names: {sorted(overlap)}")
    for column in domain.columns:
        base[column] = domain[column]
    forbidden = {"row_id", TARGET, "pitcher_id", "batter_id"}
    if forbidden & set(base.columns):
        raise ValueError("ID/target leaked into control-profile model")
    return base, tuple(domain.columns)


def make_model(n_jobs: int = -1) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=False,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
    )


def cache_path(output: Path, family: str, season: int) -> Path:
    return output / "cache" / f"{family}_{season}.npz"


def fit_family_fold(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    family: str,
    champion_meta: Mapping,
    n_jobs: int,
) -> tuple[np.ndarray, float, tuple[str, ...]]:
    state = fit_control_profile_state(training)
    train_features, domain_columns = add_family_to_base(
        training, family, state, champion_meta
    )
    validation_features, validation_columns = add_family_to_base(
        validation, family, state, champion_meta
    )
    if domain_columns != validation_columns:
        raise ValueError("training/validation domain column mismatch")
    preprocessor = fit_preprocessor(train_features)
    x_train = transform_features(train_features, preprocessor)
    x_validation = transform_features(validation_features, preprocessor)
    y_train = training[TARGET].to_numpy(dtype="int8")
    del train_features, validation_features
    gc.collect()
    model = make_model(n_jobs=n_jobs)
    started = time.monotonic()
    model.fit(x_train, y_train)
    fit_seconds = time.monotonic() - started
    prediction = model.predict_proba(x_validation)[:, 1].astype("float64")
    del model, x_train, x_validation, y_train
    gc.collect()
    if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("control-profile ExtraTrees produced invalid probability")
    return prediction, fit_seconds, domain_columns


def run_worker(args: argparse.Namespace) -> None:
    if sha256_path(CHAMPION) != CHAMPION_SHA256:
        raise SystemExit("immutable champion checksum mismatch")
    frame = pd.read_csv(args.train)
    champion_meta = load_champion_meta(CHAMPION)
    for season in VALIDATION_SEASONS:
        path = cache_path(args.output, args.worker_family, season)
        if path.is_file() and not args.force:
            print(f"[{args.worker_family} {season}] cached", flush=True)
            continue
        training, validation = temporal_split(frame, season)
        print(
            f"[{args.worker_family} {season}] fit ET25: "
            f"{len(training):,} -> {len(validation):,}",
            flush=True,
        )
        prediction, fit_seconds, columns = fit_family_fold(
            training, validation, args.worker_family, champion_meta, args.n_jobs
        )
        target = validation[TARGET].to_numpy(dtype="float64")
        save_npz_atomic(
            path,
            cache_version=np.asarray(CACHE_VERSION),
            family=np.asarray(args.worker_family),
            season=np.asarray(season, dtype="int16"),
            target_sha256=np.asarray(array_sha256(target)),
            prediction=prediction,
            fit_seconds=np.asarray(fit_seconds),
            domain_columns=np.asarray(columns),
        )
        print(f"[{args.worker_family} {season}] {fit_seconds:.1f}s", flush=True)


def ensure_caches(args: argparse.Namespace) -> None:
    for family in FAMILIES:
        paths = [cache_path(args.output, family, season) for season in VALIDATION_SEASONS]
        if all(path.is_file() for path in paths) and not args.force:
            print(f"[{family}] all folds cached", flush=True)
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
            "--worker-family",
            family,
        ]
        if args.force:
            command.append("--force")
        subprocess.run(command, check=True)


def load_raw_fold(frame: pd.DataFrame, season: int) -> dict[str, np.ndarray]:
    validation = frame[frame["season"] == season]
    oof = load_npz(OOF_DIR / f"season_{season}.npz")
    raw_et = load_npz(RAW_ET25_DIR / f"season_{season}_prefixes.npz")
    target = validation[TARGET].to_numpy(dtype="float64")
    if not np.array_equal(target, oof["y"].astype("float64")):
        raise ValueError(f"OOF target/order mismatch for {season}")
    if str(raw_et["target_sha256"].item()) != array_sha256(target):
        raise ValueError(f"raw ET25 target checksum mismatch for {season}")
    original = oof["champion_final"].astype("float64")
    accepted_et25 = raw_et["prediction_25"].astype("float64")
    current = (1.0 - CURRENT_RAW_ET_WEIGHT) * original + CURRENT_RAW_ET_WEIGHT * accepted_et25
    return {
        "target": target,
        "raw_et25": accepted_et25,
        "original_champion": original,
        "current_champion": current,
        "game_type": validation["game_type"].astype(str).to_numpy(),
        "pitcher_seen": oof["x_pitcher_seen"].astype(bool),
        "balls_before": validation["balls_before"].to_numpy(dtype="int64"),
        "strikes_before": validation["strikes_before"].to_numpy(dtype="int64"),
        "form_adjustment": (
            oof["pitcher_form_pred"].astype("float64")
            - oof["champion_base"].astype("float64")
        ),
    }


def load_domain_prediction(
    output: Path, family: str, season: int, target: np.ndarray
) -> tuple[np.ndarray, float, tuple[str, ...]]:
    path = cache_path(output, family, season)
    payload = load_npz(path)
    expected = (CACHE_VERSION, family, season, array_sha256(target))
    actual = (
        str(payload["cache_version"].item()),
        str(payload["family"].item()),
        int(payload["season"].item()),
        str(payload["target_sha256"].item()),
    )
    if actual != expected:
        raise ValueError(f"stale control-profile cache {path}: {actual} != {expected}")
    prediction = payload["prediction"].astype("float64")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction length mismatch at {path}")
    return (
        prediction,
        float(payload["fit_seconds"].item()),
        tuple(str(value) for value in payload["domain_columns"]),
    )


def family_grade(rows: Sequence[Mapping]) -> dict:
    gains = {
        str(row["validation_season"]): float(row["feature_gain_vs_raw_et25"])
        for row in rows
    }
    fold_gains = [gains[str(season)] for season in VALIDATION_SEASONS]
    strong = (
        gains["2023"] > 0.0
        and gains["2024"] > 0.0
        and gains["pooled"] > STRONG_POOLED_GAIN
    )
    weak = (
        not strong
        and gains["pooled"] > 0.0
        and sum(gain > 0.0 for gain in fold_gains) >= 2
        and min(fold_gains) > WEAK_WORST_FOLD_FLOOR
    )
    return {
        "grade": "STRONG" if strong else ("WEAK_STABLE" if weak else "UNSTABLE"),
        "gains": gains,
        "positive_fold_count": sum(gain > 0.0 for gain in fold_gains),
        "worst_fold_gain": min(fold_gains),
    }


def pooled_array(folds: Mapping[int, Mapping], key: str) -> np.ndarray:
    return np.concatenate([np.asarray(folds[season][key]) for season in VALIDATION_SEASONS])


def top10_domain_win_rate(
    target: np.ndarray, domain: np.ndarray, champion: np.ndarray
) -> float:
    champion_error = (target - champion) ** 2
    threshold = float(np.quantile(champion_error, 0.90))
    mask = champion_error >= threshold
    return float(np.mean((target[mask] - domain[mask]) ** 2 < champion_error[mask]))


def complementarity_rows(
    family: str, grade: str, folds: Mapping[int, Mapping]
) -> tuple[list[dict], bool, list[str]]:
    records = []
    for season in (*VALIDATION_SEASONS, "pooled"):
        if season == "pooled":
            target = pooled_array(folds, "target")
            champion = pooled_array(folds, "current_champion")
            domain = pooled_array(folds, "domain")
        else:
            fold = folds[int(season)]
            target = np.asarray(fold["target"])
            champion = np.asarray(fold["current_champion"])
            domain = np.asarray(fold["domain"])
        records.append(
            {
                "family": family,
                "grade": grade,
                "validation_season": season,
                "n": len(target),
                "champion_correlation": safe_corr(domain, champion),
                "residual_correlation": safe_corr(target - domain, target - champion),
                "champion_top10_error_domain_win_rate": top10_domain_win_rate(
                    target, domain, champion
                ),
            }
        )
    by_season = {str(row["validation_season"]): row for row in records}
    reasons = []
    if by_season["pooled"]["champion_correlation"] >= MAX_CHAMPION_CORRELATION:
        reasons.append("pooled champion correlation is too high")
    if by_season["pooled"]["residual_correlation"] >= MAX_RESIDUAL_CORRELATION:
        reasons.append("pooled residual correlation is too high")
    for season in (2023, 2024):
        if by_season[str(season)]["champion_top10_error_domain_win_rate"] < MIN_TOP10_WIN_RATE:
            reasons.append(f"{season} champion-top10 error win rate is below gate")
    passed = not reasons
    for row in records:
        row["gate_passed"] = passed
    return records, passed, reasons


def subset_masks(fold: Mapping) -> list[tuple[str, str, np.ndarray]]:
    game_type = np.asarray(fold["game_type"]).astype(str)
    seen = np.asarray(fold["pitcher_seen"], dtype=bool)
    full = (
        np.asarray(fold["balls_before"], dtype="int64") == 3
    ) & (np.asarray(fold["strikes_before"], dtype="int64") == 2)
    form = np.asarray(fold["form_adjustment"], dtype="float64")
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


def worst_subset_gain(
    folds: Mapping[int, Mapping], season: int | str, weight: float
) -> tuple[float, dict]:
    candidates = []
    template = subset_masks(folds[VALIDATION_SEASONS[0]])
    for axis, subset, _ in template:
        targets = []
        champions = []
        predictions = []
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
            domain = np.asarray(fold["domain"])[mask]
            targets.append(target)
            champions.append(champion)
            predictions.append((1.0 - weight) * champion + weight * domain)
        target = np.concatenate(targets)
        champion = np.concatenate(champions)
        prediction = np.concatenate(predictions)
        if len(target) >= MIN_SUBSET_ROWS:
            candidates.append(
                {
                    "axis": axis,
                    "subset": subset,
                    "n": len(target),
                    "gain": brier(target, champion) - brier(target, prediction),
                }
            )
    worst = min(candidates, key=lambda row: row["gain"])
    return float(worst["gain"]), worst


def build_blend_rows(
    family: str,
    grade: str,
    folds: Mapping[int, Mapping],
    complementarity_passed: bool,
) -> tuple[list[dict], list[dict]]:
    rows = []
    decisions = []
    for weight in BLEND_WEIGHTS:
        weight_rows = []
        worst_details = []
        for season in (*VALIDATION_SEASONS, "pooled"):
            if season == "pooled":
                target = pooled_array(folds, "target")
                champion = pooled_array(folds, "current_champion")
                domain = pooled_array(folds, "domain")
            else:
                fold = folds[int(season)]
                target = np.asarray(fold["target"])
                champion = np.asarray(fold["current_champion"])
                domain = np.asarray(fold["domain"])
            candidate = (1.0 - weight) * champion + weight * domain
            worst_gain, worst_detail = worst_subset_gain(folds, season, weight)
            worst_details.append({"validation_season": season, **worst_detail})
            row = {
                "family": family,
                "grade": grade,
                "weight": weight,
                "validation_season": season,
                "n": len(target),
                "current_champion_brier": brier(target, champion),
                "candidate_brier": brier(target, candidate),
                "brier_gain": brier(target, champion) - brier(target, candidate),
                "worst_major_subset_gain": worst_gain,
                "family_complementarity_gate_passed": complementarity_passed,
            }
            rows.append(row)
            weight_rows.append(row)
        lookup = {str(row["validation_season"]): row for row in weight_rows}
        passed = (
            lookup["2023"]["brier_gain"] > 0.0
            and lookup["2024"]["brier_gain"] > 0.0
            and lookup["pooled"]["brier_gain"] > 0.0
            and min(row["worst_major_subset_gain"] for row in weight_rows)
            >= MAJOR_SUBSET_FLOOR
        )
        for row in weight_rows:
            row["production_gate_passed"] = passed
        decisions.append(
            {
                "family": family,
                "grade": grade,
                "weight": weight,
                "gains": {key: lookup[key]["brier_gain"] for key in lookup},
                "worst_major_subset_gain": min(
                    row["worst_major_subset_gain"] for row in weight_rows
                ),
                "worst_major_subsets": worst_details,
                "passed": passed,
            }
        )
    return rows, decisions


def feature_audit(champion_meta: Mapping) -> pd.DataFrame:
    team = next(member for member in champion_meta["members"] if member["name"] == "team")
    hgb = next(member for member in champion_meta["members"] if member["name"] == "hgb")
    existing = set(hgb["feature_columns"]) | set(team["feature_columns"])
    specs = [
        ("A", "control_margin", "success-middle", "success_minus_middle"),
        ("A", "strike_ball_balance", "strike-ball", "strike_minus_ball"),
        ("A", "danger_failure_balance", "middle-ball", None),
        ("A", "reverse_adjusted_control", "success-reverse", None),
        ("B", "success_dev_1", "prev1_success-career_success", None),
        ("B", "success_dev_3", "prev3_success-career_success", None),
        ("B", "success_dev_5", "prev5_success-career_success", "prev_vs_career"),
        ("B", "success_1v3", "prev1_success-prev3_success", None),
        ("B", "success_1v5", "prev1_success-prev5_success", "pitcher_rate_gap_1_5"),
        ("B", "success_3v5", "prev3_success-prev5_success", None),
        ("B", "middle_dev_1/3/5", "recent middle-career middle", None),
        ("B", "middle_1v3/1v5/3v5", "recent-window middle gaps", None),
        ("B", "form_direction", "sign(prev3-career)", None),
        ("B", "log_pitcher_n", "log1p(pitcher_n)", "log1p_asof_pitcher_n"),
        ("B", "recent_form_x_reliability", "dev3*log1p(pitcher_n)", None),
        ("C", "control_deterioration_type", "sign(success_dev3) x sign(middle_dev3)", None),
        ("D", "recent_monotonicity", "ordered prev1/prev3/prev5", None),
        ("D", "recent_range/instability", "max(recent)-min(recent)", None),
        ("E", "pitcher_batter_success_gap", "pitcher_success-batter_success", None),
        ("E", "pitcher_batter_middle_gap", "pitcher_middle-batter_middle", None),
        ("F", "log_pitchmix_n", "log1p(pitchmix_n)", "log1p_asof_pitcher_pitchmix_n"),
        ("F", "primary_pitch_share", "max(fastball,breaking,offspeed)", None),
        ("F", "pitchmix_entropy/concentration", "normalized three-type mix", None),
        ("F", "pitchmix_coverage", "pitchmix_n/max(pitcher_n,1)", None),
        ("F", "pitchmix_control_interactions", "limited interpretable products", None),
        ("G", "log_li", "log1p(max(li,0))", None),
        ("G", "close_game/late_close", "abs(score)<=1; inning>=7", None),
        ("G", "high_leverage", "cutoff-train LI q75", None),
        ("G", "control_state_x_leverage", "dev/instability/type x leverage", None),
    ]
    rows = []
    for label, candidate, formula, duplicate in specs:
        is_duplicate = duplicate is not None and duplicate in existing
        rows.append(
            {
                "family_label": label,
                "candidate": candidate,
                "formula": formula,
                "status": "EXACT_DUPLICATE_NOT_ADDED" if is_duplicate else "NEW_EVALUATED",
                "existing_feature": duplicate if is_duplicate else "",
                "current_row_only": True,
                "validation_target_used": False,
            }
        )
    return pd.DataFrame(rows)


def leakage_audit(frame: pd.DataFrame) -> dict:
    training, validation = temporal_split(frame, 2024)
    sample_validation = validation.iloc[:1_000].copy()
    state = fit_control_profile_state(training)
    changed = sample_validation.copy()
    changed[TARGET] = 1 - changed[TARGET]
    max_changed = 0
    row_local = True
    for family in FAMILIES:
        before = build_family_features(sample_validation, family, state)
        after = build_family_features(changed, family, state)
        row_local &= before.astype(str).equals(after.astype(str))
        shuffled = sample_validation.sample(frac=1.0, random_state=7)
        shuffled_features = build_family_features(shuffled, family, state).loc[before.index]
        row_local &= before.astype(str).equals(shuffled_features.astype(str))
        single = build_family_features(sample_validation.iloc[[0]], family, state)
        row_local &= before.iloc[[0]].astype(str).equals(single.astype(str))
        max_changed += int((before.astype(str) != after.astype(str)).to_numpy().sum())
    return {
        "validation_target_flip_changed_cells": max_changed,
        "single_full_shuffle_row_local": bool(row_local),
        "state_fit_max_season": int(training["season"].max()),
        "validation_season": 2024,
        "future_or_validation_rows_in_state_fit": False,
        "test_data_read": False,
        "passed": max_changed == 0 and row_local,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-family", choices=FAMILIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker_family:
        run_worker(args)
        return

    champion_sha_before = sha256_path(CHAMPION)
    if champion_sha_before != CHAMPION_SHA256:
        raise SystemExit("immutable champion checksum mismatch before experiment")

    champion_meta = load_champion_meta(CHAMPION)
    audit = feature_audit(champion_meta)
    audit.to_csv(args.output / "feature_audit.csv", index=False)
    ensure_caches(args)

    frame = pd.read_csv(args.train)
    leakage = leakage_audit(frame)
    folds = {season: load_raw_fold(frame, season) for season in VALIDATION_SEASONS}
    reference_weights = pd.read_csv(WEIGHT_REFERENCE)
    reference_w020 = reference_weights[np.isclose(reference_weights["weight"], 0.020)]
    champion_oof_parity = []
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
        reconstructed_gain = brier(target, original) - brier(target, current)
        matched = reference_w020[
            reference_w020["validation_season"].astype(str).eq(str(season))
        ]
        if len(matched) != 1:
            raise ValueError(f"missing unique frozen w=.020 OOF reference for {season}")
        reference_gain = float(matched.iloc[0]["brier_gain"])
        champion_oof_parity.append(
            {
                "validation_season": season,
                "reconstructed_gain": reconstructed_gain,
                "reference_gain": reference_gain,
                "absolute_diff": abs(reconstructed_gain - reference_gain),
                "passed": abs(reconstructed_gain - reference_gain) <= 1e-15,
            }
        )
    if not all(row["passed"] for row in champion_oof_parity):
        raise RuntimeError("current champion OOF reconstruction lost frozen parity")
    family_rows = []
    decisions = {}
    domain_columns = {}
    family_folds = {}

    for family in FAMILIES:
        current_folds = {}
        fold_rows = []
        column_contract = None
        for season in VALIDATION_SEASONS:
            base_fold = folds[season]
            domain, fit_seconds, columns = load_domain_prediction(
                args.output, family, season, np.asarray(base_fold["target"])
            )
            if column_contract is None:
                column_contract = columns
            elif columns != column_contract:
                raise ValueError(f"domain columns vary by fold for {family}")
            current_folds[season] = {**base_fold, "domain": domain}
            row = {
                "family": family,
                "family_label": FAMILY_LABELS[family],
                "validation_season": season,
                "n": len(domain),
                "raw_et25_brier": brier(base_fold["target"], base_fold["raw_et25"]),
                "domain_et25_brier": brier(base_fold["target"], domain),
                "feature_gain_vs_raw_et25": brier(base_fold["target"], base_fold["raw_et25"])
                - brier(base_fold["target"], domain),
                "current_champion_brier": brier(base_fold["target"], base_fold["current_champion"]),
                "domain_gain_vs_current_champion": brier(base_fold["target"], base_fold["current_champion"])
                - brier(base_fold["target"], domain),
                "prediction_mean": float(np.mean(domain)),
                "prediction_std": float(np.std(domain)),
                "fit_seconds": fit_seconds,
            }
            fold_rows.append(row)
        target = pooled_array(current_folds, "target")
        raw_et25 = pooled_array(current_folds, "raw_et25")
        current = pooled_array(current_folds, "current_champion")
        domain = pooled_array(current_folds, "domain")
        pooled = {
            "family": family,
            "family_label": FAMILY_LABELS[family],
            "validation_season": "pooled",
            "n": len(domain),
            "raw_et25_brier": brier(target, raw_et25),
            "domain_et25_brier": brier(target, domain),
            "feature_gain_vs_raw_et25": brier(target, raw_et25) - brier(target, domain),
            "current_champion_brier": brier(target, current),
            "domain_gain_vs_current_champion": brier(target, current) - brier(target, domain),
            "prediction_mean": float(np.mean(domain)),
            "prediction_std": float(np.std(domain)),
            "fit_seconds": sum(row["fit_seconds"] for row in fold_rows),
        }
        decision = family_grade([*fold_rows, pooled])
        for row in (*fold_rows, pooled):
            row["grade"] = decision["grade"]
        family_rows.extend([*fold_rows, pooled])
        decisions[family] = decision
        domain_columns[family] = list(column_contract or ())
        family_folds[family] = current_folds
        print(
            f"{family}: {decision['grade']} gains="
            f"{json.dumps(decision['gains'], sort_keys=True)}",
            flush=True,
        )

    family_oof = pd.DataFrame(family_rows, columns=FAMILY_OOF_COLUMNS)
    family_oof.to_csv(args.output / "family_oof.csv", index=False)

    complement_records = []
    complement_decisions = {}
    blend_records = []
    blend_decisions = []
    for family in FAMILIES:
        grade = decisions[family]["grade"]
        if grade not in {"STRONG", "WEAK_STABLE"}:
            continue
        records, passed, reasons = complementarity_rows(
            family, grade, family_folds[family]
        )
        complement_records.extend(records)
        complement_decisions[family] = {"passed": passed, "reasons": reasons}
        if passed:
            rows, current_decisions = build_blend_rows(
                family, grade, family_folds[family], passed
            )
            blend_records.extend(rows)
            blend_decisions.extend(current_decisions)

    pd.DataFrame(complement_records, columns=COMPLEMENT_COLUMNS).to_csv(
        args.output / "complementarity.csv", index=False
    )
    pd.DataFrame(blend_records, columns=BLEND_COLUMNS).to_csv(
        args.output / "blend_results.csv", index=False
    )

    passing = [decision for decision in blend_decisions if decision["passed"]]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda decision: (
                decision["gains"]["pooled"],
                -decision["weight"],
            ),
        )

    if selected is not None:
        verdict = "A. CONTROL-PROFILE BLEND PASSED OOF — PRODUCTION BUILD REQUIRED"
    elif complement_records:
        verdict = "B. weak/stable representation signal, not production-ready"
    else:
        verdict = "C. no stable control-profile representation signal"

    champion_sha_after = sha256_path(CHAMPION)
    if champion_sha_after != champion_sha_before:
        raise RuntimeError("immutable champion changed during experiment")
    if not leakage["passed"]:
        raise RuntimeError("control-profile leakage audit failed")

    summary = {
        "verdict": verdict,
        "champion": {
            "artifact": str(CHAMPION),
            "lb_bss": CHAMPION_LB_BSS,
            "sha256_before": champion_sha_before,
            "sha256_after": champion_sha_after,
            "unchanged": champion_sha_before == champion_sha_after == CHAMPION_SHA256,
        },
        "data": {
            "train": str(args.train),
            "test_read": False,
            "folds": {str(season): f"season < {season} -> {season}" for season in VALIDATION_SEASONS},
        },
        "comparison_contract": {
            "family_grade_reference": "accepted raw HGB-52 ET25",
            "complementarity_reference": "current immutable w=.020 champion",
            "current_champion_oof_formula": "0.98 * original champion + 0.02 * accepted raw ET25",
            "current_champion_oof_parity": champion_oof_parity,
        },
        "model": {
            "class": "sklearn.ensemble.ExtraTreesClassifier",
            "n_estimators": N_ESTIMATORS,
            "max_depth": None,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "bootstrap": False,
            "random_state": RANDOM_STATE,
            "parameter_search": False,
        },
        "grade_policy": {
            "strong": "2023 > 0; 2024 > 0; pooled > 1e-5",
            "weak_stable": "pooled > 0; at least 2/3 positive; worst fold > -2e-5",
        },
        "family_decisions": [
            {"family": family, **decisions[family], "domain_columns": domain_columns[family]}
            for family in FAMILIES
        ],
        "complementarity_policy": {
            "max_pooled_champion_correlation": MAX_CHAMPION_CORRELATION,
            "max_pooled_residual_correlation": MAX_RESIDUAL_CORRELATION,
            "min_2023_2024_top10_win_rate": MIN_TOP10_WIN_RATE,
        },
        "complementarity_decisions": complement_decisions,
        "blend_weights": BLEND_WEIGHTS,
        "blend_decisions": blend_decisions,
        "selected_candidate": selected,
        "production": {
            "created": False,
            "reason": "no family reached STRONG/WEAK_STABLE, so complementarity/blend/production were gated off"
            if selected is None and not complement_records
            else "no blend passed the production gate"
            if selected is None
            else "OOF passed; production-equivalent build is a separate guarded step",
            "filename_length_gate": 30,
        },
        "feature_audit": {
            "exact_duplicates_not_added": audit[
                audit["status"] == "EXACT_DUPLICATE_NOT_ADDED"
            ]["candidate"].tolist(),
            "new_candidates_evaluated": audit[
                audit["status"] == "NEW_EVALUATED"
            ]["candidate"].tolist(),
        },
        "leakage_audit": leakage,
    }
    write_json(args.output / "summary.json", summary)
    print(verdict, flush=True)
    if selected is not None:
        print(json.dumps(json_safe(selected), indent=2), flush=True)


if __name__ == "__main__":
    main()
