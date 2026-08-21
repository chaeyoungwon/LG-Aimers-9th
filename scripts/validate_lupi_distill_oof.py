"""Temporal OOF feasibility audit for TrackMan LUPI distillation.

This research-only runner never reads test.csv and never creates a submission.
It reuses the existing reciprocal pitch anchors, intersects them with the
already-built cutoff-specific Hungarian HIGH crosswalk, and uses current-pitch
TrackMan only inside cross-fitted teachers. Students and validation inference
receive the frozen legal ID-free 52-feature contract only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime_blend_oof import load_champion_meta  # noqa: E402
from scripts.extratrees_runtime import build_hgb52_features  # noqa: E402


DEFAULT_TRAIN = Path("/Users/wooh/Documents/dev/open/data/train.csv")
DEFAULT_TRACKMAN = Path("/Users/wooh/Documents/dev/open/data/trackman_history.csv")
DEFAULT_ANCHORS = ROOT / "artifacts" / "lupi_linkage_audit" / "reciprocal_pitch_anchors.csv"
DEFAULT_LINKAGE_SUMMARY = ROOT / "artifacts" / "lupi_linkage_audit" / "linkage_summary.json"
DEFAULT_CROSSWALK = ROOT / "artifacts" / "tm_crosswalk" / "mapping_candidates.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "lupi_distill_oof"
OOF_DIR = ROOT / "artifacts" / "regime_blend_oof" / "cache"
ET25_DIR = ROOT / "artifacts" / "extratrees_size_ablation" / "cache"

TARGET = "control_success"
VALIDATION_SEASONS = (2022, 2023, 2024)
INNER_FOLDS = 3
INNER_RANDOM_STATE = 20260821
ALPHAS = (0.0, 0.1, 0.25, 0.5)
BLEND_WEIGHTS = (0.01, 0.025, 0.05, 0.10)
CURRENT_ET25_WEIGHT = 0.0225
CURRENT_CHAMPION_WEIGHT = 0.9775
MIN_SUBSET_ROWS = 1_000
WORST_SEASON_FLOOR = -1e-5
TRANSFER_EPSILON = 1e-8

PRIVILEGED_NUMERIC = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
PRIVILEGED_CATEGORICAL = ("pitch_type_group",)
PRIVILEGED_COLUMNS = (*PRIVILEGED_NUMERIC, *PRIVILEGED_CATEGORICAL)
FORBIDDEN_MODEL_COLUMNS = {
    "row_id",
    TARGET,
    "pitcher_id",
    "batter_id",
    "trackman_id",
    "pitcher_trackman_id",
    "batter_trackman_id",
}

TEACHER_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.08,
    "max_iter": 60,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 300,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "categorical_features": "from_dtype",
    "early_stopping": False,
    "random_state": 42,
}
STUDENT_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.08,
    "max_iter": 60,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 1_000,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "categorical_features": "from_dtype",
    "early_stopping": False,
    "random_state": 42,
}


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype="float64")
    p = np.asarray(prediction, dtype="float64")
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("Brier inputs must be same-length vectors")
    return float(np.mean((y - p) ** 2))


def clipped_prediction(model, features: pd.DataFrame) -> np.ndarray:
    prediction = np.asarray(model.predict(features), dtype="float64")
    return np.clip(prediction, 0.0, 1.0)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype="float64")
    y = np.asarray(right, dtype="float64")
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def validate_legal_contract(features: pd.DataFrame, champion_meta: Mapping) -> None:
    hgb = [member for member in champion_meta["members"] if member["name"] == "hgb"]
    if len(hgb) != 1:
        raise ValueError("champion HGB contract is missing or ambiguous")
    if list(features.columns) != list(hgb[0]["feature_columns"]):
        raise ValueError("student does not reuse the accepted 52-feature contract")
    if len(features.columns) != 52:
        raise ValueError(f"expected 52 legal features, got {len(features.columns)}")
    forbidden = sorted(FORBIDDEN_MODEL_COLUMNS & set(features.columns))
    if forbidden:
        raise ValueError(f"forbidden model columns in legal contract: {forbidden}")


def load_and_validate_linkage(
    train: pd.DataFrame,
    trackman_path: Path,
    anchors_path: Path,
    summary_path: Path,
) -> tuple[pd.DataFrame, dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    anchors = pd.read_csv(anchors_path)
    if len(anchors) != int(summary["reciprocal_unique_anchors"]):
        raise ValueError("reciprocal anchor count differs from linkage summary")
    if not anchors["row_id"].is_unique or not anchors["candidate_trackman_id"].is_unique:
        raise ValueError("reciprocal anchor keys are not one-to-one")
    if not np.equal(anchors["candidate_trackman_id"] % 1, 0).all():
        raise ValueError("TrackMan anchor IDs are not integral")
    anchors["candidate_trackman_id"] = anchors["candidate_trackman_id"].astype("int64")

    train_keys = train[["row_id", "season", "pitcher_id", "asof_pitcher_n"]]
    linked = anchors.merge(
        train_keys,
        on="row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_anchor", "_train"),
    )
    if linked["season_train"].isna().any():
        raise ValueError("anchor references a missing train row")
    if not np.array_equal(linked["season_anchor"], linked["season_train"]):
        raise ValueError("anchor/train season mismatch")
    if not np.array_equal(linked["pitcher_id_anchor"], linked["pitcher_id_train"]):
        raise ValueError("anchor/train pitcher mismatch")
    if not np.array_equal(linked["asof_pitcher_n_anchor"], linked["asof_pitcher_n_train"]):
        raise ValueError("anchor/train chronological rank mismatch")
    linked = linked.rename(
        columns={
            "season_train": "season",
            "pitcher_id_train": "pitcher_id",
            "asof_pitcher_n_train": "asof_pitcher_n",
        }
    ).drop(columns=["season_anchor", "pitcher_id_anchor", "asof_pitcher_n_anchor"])

    usecols = [
        "trackman_id",
        "season",
        "pitcher_trackman_id",
        *PRIVILEGED_COLUMNS,
    ]
    trackman = pd.read_csv(trackman_path, usecols=usecols, encoding="utf-8-sig")
    if not trackman["trackman_id"].is_unique:
        raise ValueError("official TrackMan ID is not unique")
    linked = linked.merge(
        trackman,
        left_on="candidate_trackman_id",
        right_on="trackman_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_tm"),
    )
    if linked["trackman_id"].isna().any():
        raise ValueError("anchor references a missing TrackMan row")
    if not np.array_equal(linked["season"], linked["season_tm"]):
        raise ValueError("anchor/TrackMan season mismatch")
    linked = linked.drop(columns=["season_tm"])

    actual_seasons = {
        str(int(season)): int(count)
        for season, count in linked.groupby("season").size().items()
    }
    if actual_seasons != {str(k): int(v) for k, v in summary["anchors_by_season"].items()}:
        raise ValueError("anchor season counts differ from linkage summary")
    all8 = linked[list(PRIVILEGED_NUMERIC)].notna().all(axis=1).mean()
    if abs(float(all8) - float(summary["physical_all8_nonnull_rate"])) > 1e-15:
        raise ValueError("privileged physical coverage differs from linkage summary")

    chronological = []
    for _, group in linked.sort_values(["pitcher_id", "asof_pitcher_n"]).groupby(
        "pitcher_id", sort=False
    ):
        main_gap = np.diff(group["asof_pitcher_n"].to_numpy(dtype="int64"))
        tm_gap = np.diff(group["tm_rank"].to_numpy(dtype="int64"))
        chronological.append(np.column_stack([main_gap, tm_gap]))
    gaps = np.concatenate(chronological)
    chronological_audit = {
        "pair_count": len(gaps),
        "tm_rank_strictly_increasing_rate": float(np.mean(gaps[:, 1] > 0)),
        "exact_adjacent_rank_gap_rate": float(np.mean(gaps[:, 0] == gaps[:, 1])),
        "tm_gap_ge_main_gap_rate": float(np.mean(gaps[:, 1] >= gaps[:, 0])),
    }
    expected = {
        "pair_count": int(summary["chronological_pair_count"]),
        "tm_rank_strictly_increasing_rate": float(summary["tm_rank_strictly_increasing_rate"]),
        "exact_adjacent_rank_gap_rate": float(summary["exact_adjacent_rank_gap_rate"]),
        "tm_gap_ge_main_gap_rate": float(summary["tm_gap_ge_main_gap_rate"]),
    }
    for key, value in expected.items():
        if chronological_audit[key] != value:
            raise ValueError(f"chronological linkage parity failed for {key}")
    return linked, {"summary": summary, "chronological": chronological_audit}


def cutoff_safe_anchors(
    linked: pd.DataFrame,
    crosswalk: pd.DataFrame,
    cutoff: int,
) -> tuple[pd.DataFrame, dict]:
    mapping = crosswalk[
        crosswalk["cutoff"].eq(cutoff)
        & crosswalk["method"].eq("hungarian")
        & crosswalk["accepted"].astype(bool)
        & crosswalk["confidence"].eq("HIGH")
    ][["main_pitcher_id", "tm_pitcher_id"]].drop_duplicates()
    if not mapping["main_pitcher_id"].is_unique or not mapping["tm_pitcher_id"].is_unique:
        raise ValueError(f"cutoff {cutoff} HIGH Hungarian mapping is not one-to-one")
    raw = linked[linked["season"].le(cutoff)].copy()
    safe = raw.merge(
        mapping,
        left_on=["pitcher_id", "pitcher_trackman_id"],
        right_on=["main_pitcher_id", "tm_pitcher_id"],
        how="inner",
        validate="many_to_one",
    ).drop(columns=["main_pitcher_id", "tm_pitcher_id"])
    if safe["season"].max() > cutoff:
        raise AssertionError("future TrackMan season entered cutoff-safe linkage")
    return safe, {
        "cutoff": cutoff,
        "raw_full_history_anchor_rows_at_or_before_cutoff": len(raw),
        "cutoff_safe_anchor_rows": len(safe),
        "retention": len(safe) / len(raw),
        "cutoff_high_mapping_count": len(mapping),
        "cutoff_safe_pitcher_count": safe["pitcher_id"].nunique(),
        "anchor_seasons": sorted(safe["season"].unique().astype(int).tolist()),
    }


def teacher_feature_frame(
    legal: pd.DataFrame,
    privileged: pd.DataFrame,
    pitch_categories: Sequence[str],
) -> pd.DataFrame:
    output = legal.copy()
    for column in PRIVILEGED_NUMERIC:
        output[f"tm_current_{column}"] = pd.to_numeric(
            privileged[column], errors="coerce"
        ).astype("float32")
    output["tm_current_pitch_type_group"] = pd.Categorical(
        privileged["pitch_type_group"].astype("object"),
        categories=list(pitch_categories),
    )
    forbidden = FORBIDDEN_MODEL_COLUMNS & set(output.columns)
    if forbidden:
        raise ValueError(f"ID leaked into teacher features: {sorted(forbidden)}")
    return output


def cross_fitted_teacher(
    legal_features: pd.DataFrame,
    privileged: pd.DataFrame,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict], float]:
    if len(legal_features) != len(privileged) or len(target) != len(privileged):
        raise ValueError("teacher inputs differ in length")
    pitch_categories = sorted(
        privileged["pitch_type_group"].dropna().astype(str).unique().tolist()
    )
    if not pitch_categories:
        raise ValueError("no pitch type group in cutoff-safe linked rows")
    splitter = KFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=INNER_RANDOM_STATE,
    )
    privileged_oof = np.full(len(target), np.nan, dtype="float64")
    legal_oof = np.full(len(target), np.nan, dtype="float64")
    fold_rows = []
    fit_seconds = 0.0
    row_numbers = np.arange(len(target))
    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(row_numbers), start=1):
        if np.intersect1d(fit_idx, holdout_idx).size:
            raise AssertionError("inner teacher fit/holdout overlap")
        legal_model = HistGradientBoostingRegressor(**TEACHER_PARAMS)
        started = time.monotonic()
        legal_model.fit(legal_features.iloc[fit_idx], target[fit_idx])
        legal_oof[holdout_idx] = clipped_prediction(
            legal_model, legal_features.iloc[holdout_idx]
        )
        del legal_model

        fit_privileged = teacher_feature_frame(
            legal_features.iloc[fit_idx].reset_index(drop=True),
            privileged.iloc[fit_idx].reset_index(drop=True),
            pitch_categories,
        )
        holdout_privileged = teacher_feature_frame(
            legal_features.iloc[holdout_idx].reset_index(drop=True),
            privileged.iloc[holdout_idx].reset_index(drop=True),
            pitch_categories,
        )
        teacher = HistGradientBoostingRegressor(**TEACHER_PARAMS)
        teacher.fit(fit_privileged, target[fit_idx])
        privileged_oof[holdout_idx] = clipped_prediction(
            teacher, holdout_privileged
        )
        fit_seconds += time.monotonic() - started
        fold_rows.append(
            {
                "inner_fold": fold,
                "fit_rows": len(fit_idx),
                "holdout_rows": len(holdout_idx),
                "overlap_rows": 0,
                "fit_target_rate": float(np.mean(target[fit_idx])),
                "holdout_target_rate": float(np.mean(target[holdout_idx])),
            }
        )
        del teacher, fit_privileged, holdout_privileged
        gc.collect()
    if not np.isfinite(privileged_oof).all() or not np.isfinite(legal_oof).all():
        raise RuntimeError("teacher cross-fit left non-finite OOF predictions")
    return privileged_oof, legal_oof, fold_rows, fit_seconds


def current_champion_oof(season: int, target: np.ndarray) -> tuple[np.ndarray, dict]:
    source = load_npz(OOF_DIR / f"season_{season}.npz")
    extra = load_npz(ET25_DIR / f"season_{season}_prefixes.npz")[
        "prediction_25"
    ].astype("float64")
    if not np.array_equal(target, source["y"].astype("float64")):
        raise ValueError(f"validation/OOF target order mismatch for {season}")
    champion = (
        CURRENT_CHAMPION_WEIGHT * source["champion_final"].astype("float64")
        + CURRENT_ET25_WEIGHT * extra
    )
    return champion, source


def make_student() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**STUDENT_PARAMS)


def distillation_target(
    real_target: np.ndarray,
    linked_positions: np.ndarray,
    teacher_probability: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if alpha not in ALPHAS:
        raise ValueError(f"alpha must be one of {ALPHAS}")
    target = np.asarray(real_target, dtype="float64").copy()
    positions = np.asarray(linked_positions, dtype="int64")
    teacher = np.asarray(teacher_probability, dtype="float64")
    if len(positions) != len(teacher):
        raise ValueError("linked positions and teacher probability lengths differ")
    if len(np.unique(positions)) != len(positions):
        raise ValueError("linked positions are not unique")
    if np.any(positions < 0) or np.any(positions >= len(target)):
        raise ValueError("linked position is outside the student target")
    if not np.isfinite(teacher).all() or np.any((teacher < 0.0) | (teacher > 1.0)):
        raise ValueError("teacher probability is invalid")
    if alpha > 0.0:
        target[positions] = (
            (1.0 - alpha) * target[positions] + alpha * teacher
        )
    return target


def residual_covariance(
    target: np.ndarray, left: np.ndarray, right: np.ndarray
) -> float:
    return float(np.cov(target - left, target - right, ddof=0)[0, 1])


def subset_masks(
    train_fold: pd.DataFrame,
    validation: pd.DataFrame,
    source: Mapping[str, np.ndarray],
    linkage_pitchers: set[int],
) -> list[tuple[str, str, np.ndarray]]:
    seen_batters = set(train_fold["batter_id"].astype(int).unique().tolist())
    batter_seen = validation["batter_id"].astype(int).isin(seen_batters).to_numpy()
    linked_pitcher = validation["pitcher_id"].astype(int).isin(linkage_pitchers).to_numpy()
    masks = [
        ("pitcher_seen", "seen", source["x_pitcher_seen"].astype(bool)),
        ("pitcher_seen", "unseen", ~source["x_pitcher_seen"].astype(bool)),
        ("batter_seen", "seen", batter_seen),
        ("batter_seen", "unseen", ~batter_seen),
        ("linkage_covered_pitcher", "covered", linked_pitcher),
        ("linkage_covered_pitcher", "non_covered", ~linked_pitcher),
        ("cold_start_gate", "active", source["coldstart_mask"].astype(bool)),
        ("cold_start_gate", "inactive", ~source["coldstart_mask"].astype(bool)),
    ]
    for axis in ("pitcher_hand", "batter_hand", "game_type"):
        values = validation[axis].astype(str)
        for value in sorted(values.unique().tolist()):
            masks.append((axis, value, values.eq(value).to_numpy()))
    return masks


def select_alpha(alpha_results: pd.DataFrame) -> float:
    pooled = alpha_results[
        alpha_results["validation_season"].astype(str).eq("pooled")
        & alpha_results["alpha"].gt(0.0)
    ].sort_values(["gain_vs_student_baseline", "alpha"], ascending=[False, True])
    if pooled.empty:
        raise ValueError("no positive-alpha pooled result")
    return float(pooled.iloc[0]["alpha"])


def verdict_from_results(
    temporal: pd.DataFrame,
    pooled: Mapping,
    teacher_pooled_gain: float,
    noncovered_pooled_gain: float | None,
) -> tuple[str, list[str]]:
    gains = temporal["gain_vs_student_baseline"].to_numpy(dtype="float64")
    blend_gains = temporal["best_blend_gain_vs_champion"].to_numpy(dtype="float64")
    improved = int(np.sum(gains > 0.0))
    worsened = int(np.sum(gains < 0.0))
    reasons = []
    kill = False
    if worsened >= 2:
        kill = True
        reasons.append("student baseline보다 악화된 temporal season이 2개 이상")
    if float(pooled["gain_vs_student_baseline"]) > 0.0 and improved < 2:
        kill = True
        reasons.append("pooled만 개선되고 season 방향성이 부족")
    if improved == 1 and float(temporal.loc[temporal["validation_season"].eq(2024), "gain_vs_student_baseline"].iloc[0]) > 0.0:
        kill = True
        reasons.append("2024만 개선")
    if teacher_pooled_gain > 0.0 and float(pooled["gain_vs_student_baseline"]) <= TRANSFER_EPSILON:
        kill = True
        reasons.append("privileged teacher gain이 student로 전달되지 않음")
    if float(pooled["best_blend_gain_vs_champion"]) <= 0.0:
        kill = True
        reasons.append("고정 coarse blend grid에서 champion gain 없음")
    if noncovered_pooled_gain is not None and noncovered_pooled_gain <= 0.0 and float(pooled["gain_vs_student_baseline"]) <= 0.0:
        kill = True
        reasons.append("linkage non-covered pitcher와 전체 validation에서 signal 소실")
    if kill:
        return "KILL", reasons

    go = (
        improved >= 2
        and float(pooled["gain_vs_student_baseline"]) > 0.0
        and float(np.min(gains)) >= WORST_SEASON_FLOOR
        and int(np.sum(blend_gains > 0.0)) >= 2
        and float(pooled["best_blend_gain_vs_champion"]) > 0.0
        and (noncovered_pooled_gain is None or noncovered_pooled_gain > 0.0)
    )
    if go:
        return "GO", [
            "3개 season 중 최소 2개 standalone 개선",
            "pooled standalone 및 champion coarse blend 개선",
            "worst season과 non-covered pitcher 안정성 gate 통과",
        ]
    return "CONDITIONAL GO", [
        "명시적 KILL 조건은 피했지만 GO의 temporal/subset 안정성 조건을 모두 만족하지 못함"
    ]


def run(args: argparse.Namespace) -> dict:
    required = (
        args.train,
        args.trackman,
        args.anchors,
        args.linkage_summary,
        args.crosswalk,
        *(OOF_DIR / f"season_{season}.npz" for season in VALIDATION_SEASONS),
        *(ET25_DIR / f"season_{season}_prefixes.npz" for season in VALIDATION_SEASONS),
    )
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if "test" in str(args.train).lower() or "test" in str(args.trackman).lower():
        raise ValueError("LUPI OOF inputs must not be test data")

    started = time.monotonic()
    train = pd.read_csv(args.train, encoding="utf-8-sig").reset_index(drop=True)
    if train["row_id"].duplicated().any():
        raise ValueError("train row_id is not unique")
    linked, linkage_audit = load_and_validate_linkage(
        train, args.trackman, args.anchors, args.linkage_summary
    )
    crosswalk = pd.read_csv(args.crosswalk)
    champion_meta = load_champion_meta()
    legal_all = build_hgb52_features(train, champion_meta["cat_levels"])
    validate_legal_contract(legal_all, champion_meta)
    row_to_position = pd.Series(train.index.to_numpy(), index=train["row_id"]).to_dict()

    folds: dict[int, dict] = {}
    teacher_rows = []
    leakage_folds = []
    for season in VALIDATION_SEASONS:
        cutoff = season - 1
        train_positions = np.flatnonzero(train["season"].le(cutoff).to_numpy())
        validation_positions = np.flatnonzero(train["season"].eq(season).to_numpy())
        train_fold = train.iloc[train_positions]
        validation = train.iloc[validation_positions]
        if train_fold["season"].max() > cutoff or validation["season"].nunique() != 1:
            raise AssertionError("outer temporal split violated")

        safe_links, cutoff_audit = cutoff_safe_anchors(linked, crosswalk, cutoff)
        linked_positions_global = np.asarray(
            [row_to_position[row_id] for row_id in safe_links["row_id"]],
            dtype="int64",
        )
        if not np.isin(linked_positions_global, train_positions).all():
            raise AssertionError("validation/future anchor entered teacher training")
        local_position = {global_pos: local for local, global_pos in enumerate(train_positions)}
        linked_positions_local = np.asarray(
            [local_position[position] for position in linked_positions_global],
            dtype="int64",
        )
        linked_legal = legal_all.iloc[linked_positions_global].reset_index(drop=True)
        privileged = safe_links[list(PRIVILEGED_COLUMNS)].reset_index(drop=True)
        linked_target = train.iloc[linked_positions_global][TARGET].to_numpy(dtype="float64")
        teacher_oof, legal_teacher_oof, inner_rows, teacher_seconds = cross_fitted_teacher(
            linked_legal, privileged, linked_target
        )
        teacher_rows.append(
            {
                "validation_season": season,
                "cutoff": cutoff,
                "linked_rows": len(linked_target),
                "legal_teacher_brier": brier(linked_target, legal_teacher_oof),
                "privileged_teacher_brier": brier(linked_target, teacher_oof),
                "privileged_gain_vs_legal_teacher": (
                    brier(linked_target, legal_teacher_oof)
                    - brier(linked_target, teacher_oof)
                ),
                "fit_seconds": teacher_seconds,
            }
        )

        x_train = legal_all.iloc[train_positions]
        x_validation = legal_all.iloc[validation_positions]
        y_train = train_fold[TARGET].to_numpy(dtype="float64")
        y_validation = validation[TARGET].to_numpy(dtype="float64")
        champion, source = current_champion_oof(season, y_validation)
        predictions = {}
        student_fit_seconds = {}
        for alpha in ALPHAS:
            target = distillation_target(
                y_train,
                linked_positions_local,
                teacher_oof,
                alpha,
            )
            model = make_student()
            fit_started = time.monotonic()
            model.fit(x_train, target)
            student_fit_seconds[alpha] = time.monotonic() - fit_started
            predictions[alpha] = clipped_prediction(model, x_validation)
            del model, target
            gc.collect()
        baseline = predictions[0.0]
        folds[season] = {
            "validation": validation.reset_index(drop=True),
            "train_fold": train_fold,
            "target": y_validation,
            "champion": champion,
            "source": source,
            "baseline": baseline,
            "predictions": predictions,
            "student_fit_seconds": student_fit_seconds,
            "linkage_pitchers": set(safe_links["pitcher_id"].astype(int).unique()),
        }
        leakage_folds.append(
            {
                "validation_season": season,
                "outer_cutoff": cutoff,
                "outer_training_max_season": int(train_fold["season"].max()),
                "validation_trackman_rows_used": 0,
                "teacher_linkage_max_season": int(safe_links["season"].max()),
                "same_row_teacher_in_sample_predictions": 0,
                "inner_cross_fit": inner_rows,
                **cutoff_audit,
            }
        )
        print(
            f"season={season} linked={len(linked_target):,} "
            f"teacher_gain={teacher_rows[-1]['privileged_gain_vs_legal_teacher']:.8g}",
            flush=True,
        )

    alpha_rows = []
    blend_rows = []
    for alpha in ALPHAS:
        for season in (*VALIDATION_SEASONS, "pooled"):
            selected_folds = VALIDATION_SEASONS if season == "pooled" else (int(season),)
            y = np.concatenate([folds[s]["target"] for s in selected_folds])
            baseline = np.concatenate([folds[s]["baseline"] for s in selected_folds])
            prediction = np.concatenate(
                [folds[s]["predictions"][alpha] for s in selected_folds]
            )
            champion = np.concatenate([folds[s]["champion"] for s in selected_folds])
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "validation_season": season,
                    "n": len(y),
                    "student_baseline_brier": brier(y, baseline),
                    "distilled_brier": brier(y, prediction),
                    "gain_vs_student_baseline": brier(y, baseline) - brier(y, prediction),
                    "champion_brier": brier(y, champion),
                    "corr_distilled_champion": safe_corr(prediction, champion),
                    "residual_covariance": residual_covariance(y, prediction, champion),
                    "mean_abs_disagreement": float(np.mean(np.abs(prediction - champion))),
                    "rms_disagreement": float(np.sqrt(np.mean((prediction - champion) ** 2))),
                }
            )
            if alpha == 0.0:
                continue
            for weight in BLEND_WEIGHTS:
                blended = (1.0 - weight) * champion + weight * prediction
                blend_rows.append(
                    {
                        "alpha": alpha,
                        "validation_season": season,
                        "weight": weight,
                        "n": len(y),
                        "champion_brier": brier(y, champion),
                        "blend_brier": brier(y, blended),
                        "gain_vs_champion": brier(y, champion) - brier(y, blended),
                    }
                )
    alpha_results = pd.DataFrame(alpha_rows)
    blend_results = pd.DataFrame(blend_rows)
    selected_alpha = select_alpha(alpha_results)

    temporal_rows = []
    for season in (*VALIDATION_SEASONS, "pooled"):
        alpha_row = alpha_results[
            np.isclose(alpha_results["alpha"], selected_alpha)
            & alpha_results["validation_season"].astype(str).eq(str(season))
        ].iloc[0]
        blends = blend_results[
            np.isclose(blend_results["alpha"], selected_alpha)
            & blend_results["validation_season"].astype(str).eq(str(season))
        ].sort_values(["blend_brier", "weight"])
        best = blends.iloc[0]
        temporal_rows.append(
            {
                **alpha_row.to_dict(),
                "best_blend_brier": float(best["blend_brier"]),
                "best_blend_weight": float(best["weight"]),
                "best_blend_gain_vs_champion": float(best["gain_vs_champion"]),
            }
        )
    temporal = pd.DataFrame(temporal_rows)

    subset_rows = []
    for season in VALIDATION_SEASONS:
        fold = folds[season]
        prediction = fold["predictions"][selected_alpha]
        baseline = fold["baseline"]
        champion = fold["champion"]
        best_weight = float(
            temporal.loc[
                temporal["validation_season"].astype(str).eq(str(season)),
                "best_blend_weight",
            ].iloc[0]
        )
        blended = (1.0 - best_weight) * champion + best_weight * prediction
        for axis, subset, mask in subset_masks(
            fold["train_fold"],
            fold["validation"],
            fold["source"],
            fold["linkage_pitchers"],
        ):
            n = int(mask.sum())
            if n == 0:
                continue
            y = fold["target"][mask]
            subset_rows.append(
                {
                    "validation_season": season,
                    "axis": axis,
                    "subset": subset,
                    "n": n,
                    "eligible_for_stability": n >= MIN_SUBSET_ROWS,
                    "student_baseline_brier": brier(y, baseline[mask]),
                    "distilled_brier": brier(y, prediction[mask]),
                    "gain_vs_student_baseline": brier(y, baseline[mask]) - brier(y, prediction[mask]),
                    "champion_brier": brier(y, champion[mask]),
                    "best_blend_brier": brier(y, blended[mask]),
                    "blend_gain_vs_champion": brier(y, champion[mask]) - brier(y, blended[mask]),
                }
            )
    subset_results = pd.DataFrame(subset_rows)

    # Pooled subset is built only from already-generated validation predictions.
    pooled_best_weight = float(
        temporal.loc[
            temporal["validation_season"].astype(str).eq("pooled"),
            "best_blend_weight",
        ].iloc[0]
    )
    for axis, subset in subset_results[["axis", "subset"]].drop_duplicates().itertuples(index=False):
        values = []
        for season in VALIDATION_SEASONS:
            fold = folds[season]
            masks = {
                (a, s): m
                for a, s, m in subset_masks(
                    fold["train_fold"], fold["validation"], fold["source"], fold["linkage_pitchers"]
                )
            }
            if (axis, subset) not in masks:
                continue
            mask = masks[(axis, subset)]
            prediction = fold["predictions"][selected_alpha]
            blended = (
                (1.0 - pooled_best_weight) * fold["champion"]
                + pooled_best_weight * prediction
            )
            values.append(
                (
                    fold["target"][mask],
                    fold["baseline"][mask],
                    prediction[mask],
                    fold["champion"][mask],
                    blended[mask],
                )
            )
        if not values:
            continue
        y, baseline, prediction, champion, blended = (
            np.concatenate([item[index] for item in values]) for index in range(5)
        )
        subset_rows.append(
            {
                "validation_season": "pooled",
                "axis": axis,
                "subset": subset,
                "n": len(y),
                "eligible_for_stability": len(y) >= MIN_SUBSET_ROWS,
                "student_baseline_brier": brier(y, baseline),
                "distilled_brier": brier(y, prediction),
                "gain_vs_student_baseline": brier(y, baseline) - brier(y, prediction),
                "champion_brier": brier(y, champion),
                "best_blend_brier": brier(y, blended),
                "blend_gain_vs_champion": brier(y, champion) - brier(y, blended),
            }
        )
    subset_results = pd.DataFrame(subset_rows)

    teacher_results = pd.DataFrame(teacher_rows)
    teacher_pooled_gain = float(
        np.average(
            teacher_results["privileged_gain_vs_legal_teacher"],
            weights=teacher_results["linked_rows"],
        )
    )
    noncovered = subset_results[
        subset_results["validation_season"].astype(str).eq("pooled")
        & subset_results["axis"].eq("linkage_covered_pitcher")
        & subset_results["subset"].eq("non_covered")
    ]
    noncovered_gain = (
        float(noncovered.iloc[0]["gain_vs_student_baseline"])
        if len(noncovered) == 1
        else None
    )
    temporal_only = temporal[
        temporal["validation_season"].astype(str).isin(["2022", "2023", "2024"])
    ].copy()
    pooled_row = temporal[temporal["validation_season"].astype(str).eq("pooled")].iloc[0]
    verdict, verdict_reasons = verdict_from_results(
        temporal_only, pooled_row, teacher_pooled_gain, noncovered_gain
    )

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reproducibility_rows = []
    for season in VALIDATION_SEASONS:
        fold = folds[season]
        cache_path = cache_dir / f"season_{season}.npz"
        previous = None
        if cache_path.is_file():
            try:
                previous = load_npz(cache_path)
            except ValueError as exception:
                if "Object arrays cannot be loaded" not in str(exception):
                    raise
        current_arrays = {
            "target": fold["target"].astype("float64"),
            "champion": fold["champion"].astype("float64"),
            "student_alpha_0": fold["baseline"].astype("float64"),
            "student_alpha_010": fold["predictions"][0.1].astype("float64"),
            "student_alpha_025": fold["predictions"][0.25].astype("float64"),
            "student_alpha_050": fold["predictions"][0.5].astype("float64"),
        }
        if previous is not None:
            differences = {
                name: float(np.max(np.abs(previous[name] - values)))
                for name, values in current_arrays.items()
            }
            row_id_exact = np.array_equal(
                previous["row_id"],
                fold["validation"]["row_id"].astype(str).to_numpy(),
            )
            reproducibility_rows.append(
                {
                    "validation_season": season,
                    "row_id_exact": row_id_exact,
                    "max_abs_diff_by_array": differences,
                    "max_abs_diff": max(differences.values()),
                    "passed": row_id_exact and max(differences.values()) == 0.0,
                }
            )
        with cache_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                cache_version=np.asarray("lupi-distill-oof-v1"),
                validation_season=np.asarray(season, dtype="int64"),
                row_id=np.asarray(
                    fold["validation"]["row_id"].astype(str).tolist(),
                    dtype="U32",
                ),
                **current_arrays,
            )
    alpha_results.to_csv(args.output / "alpha_results.csv", index=False)
    blend_results.to_csv(args.output / "blend_results.csv", index=False)
    temporal.to_csv(args.output / "temporal_results.csv", index=False)
    subset_results.to_csv(args.output / "subset_results.csv", index=False)
    teacher_results.to_csv(args.output / "teacher_results.csv", index=False)
    leakage_audit = {
        "test_csv_read": False,
        "full_trackman_read_for_existing_linkage_parity_audit": True,
        "validation_trackman_used_for_teacher_or_student": False,
        "future_season_excluded": True,
        "teacher_same_row_in_sample_used": False,
        "full_history_anchor_used_directly": False,
        "cutoff_safe_anchor_rule": (
            "existing reciprocal anchor intersect existing cutoff-specific "
            "Hungarian HIGH main_pitcher_id↔tm_pitcher_id mapping"
        ),
        "outer_folds": leakage_folds,
    }
    write_json(args.output / "leakage_audit.json", leakage_audit)

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    summary = {
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "experiment_scope": "temporal OOF feasibility only; no production candidate or submission",
        "data_contract": {
            "teacher_legal_features": list(legal_all.columns),
            "teacher_privileged_features": [
                *(f"tm_current_{column}" for column in PRIVILEGED_NUMERIC),
                "tm_current_pitch_type_group",
            ],
            "student_features": list(legal_all.columns),
            "student_feature_count": len(legal_all.columns),
            "teacher_id_columns_used": [],
            "student_id_columns_used": [],
            "test_dependency": False,
            "teacher_model": "HistGradientBoostingRegressor",
            "student_model": "HistGradientBoostingRegressor",
            "teacher_params": TEACHER_PARAMS,
            "student_params": STUDENT_PARAMS,
            "alpha_grid": ALPHAS,
            "champion_blend_grid": BLEND_WEIGHTS,
        },
        "linkage": {
            **linkage_audit,
            "cutoff_safe_outer_folds": [
                {
                    key: row[key]
                    for key in (
                        "validation_season",
                        "outer_cutoff",
                        "raw_full_history_anchor_rows_at_or_before_cutoff",
                        "cutoff_safe_anchor_rows",
                        "retention",
                        "cutoff_safe_pitcher_count",
                    )
                }
                for row in leakage_folds
            ],
        },
        "leakage_audit": leakage_audit,
        "selected_alpha": selected_alpha,
        "alpha_selection_rule": "maximum pooled standalone gain vs alpha=0 baseline; tie -> smaller alpha",
        "teacher_results": teacher_rows,
        "teacher_pooled_privileged_gain_vs_legal": teacher_pooled_gain,
        "temporal_results": temporal.to_dict(orient="records"),
        "subset_stability": {
            "eligible_min_rows": MIN_SUBSET_ROWS,
            "worst_eligible_gain_vs_student_baseline": float(
                subset_results.loc[
                    subset_results["eligible_for_stability"],
                    "gain_vs_student_baseline",
                ].min()
            ),
            "pooled_noncovered_pitcher_gain_vs_student_baseline": noncovered_gain,
        },
        "complementarity": {
            "pooled_corr_distilled_champion": float(pooled_row["corr_distilled_champion"]),
            "pooled_residual_covariance": float(pooled_row["residual_covariance"]),
            "pooled_mean_abs_disagreement": float(pooled_row["mean_abs_disagreement"]),
            "pooled_rms_disagreement": float(pooled_row["rms_disagreement"]),
            "pooled_best_blend_weight": float(pooled_row["best_blend_weight"]),
            "pooled_best_blend_gain_vs_champion": float(pooled_row["best_blend_gain_vs_champion"]),
        },
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_bytes,
        },
        "reproducibility_against_previous_cache": {
            "performed": len(reproducibility_rows) == len(VALIDATION_SEASONS),
            "folds": reproducibility_rows,
            "passed": (
                len(reproducibility_rows) == len(VALIDATION_SEASONS)
                and all(row["passed"] for row in reproducibility_rows)
            ),
        },
        "input_sha256": {
            "train": sha256_path(args.train),
            "trackman": sha256_path(args.trackman),
            "reciprocal_anchors": sha256_path(args.anchors),
            "linkage_summary": sha256_path(args.linkage_summary),
            "cutoff_crosswalk": sha256_path(args.crosswalk),
        },
        "production_changes": [],
        "submission_zip_created": False,
        "artifacts": [
            "summary.json",
            "leakage_audit.json",
            "teacher_results.csv",
            "alpha_results.csv",
            "blend_results.csv",
            "temporal_results.csv",
            "subset_results.csv",
            "cache/season_2022.npz",
            "cache/season_2023.npz",
            "cache/season_2024.npz",
        ],
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(json_safe({
        "verdict": verdict,
        "selected_alpha": selected_alpha,
        "temporal_results": temporal.to_dict(orient="records"),
        "teacher_pooled_privileged_gain_vs_legal": teacher_pooled_gain,
        "complementarity": summary["complementarity"],
        "resources": summary["resources"],
    }), ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--trackman", type=Path, default=DEFAULT_TRACKMAN)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--linkage-summary", type=Path, default=DEFAULT_LINKAGE_SUMMARY)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
