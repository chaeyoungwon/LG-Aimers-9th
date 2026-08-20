"""Audit and validate cutoff-safe incremental TrackMan feasibility.

The official files expose ``train.pitcher_id`` and
``trackman_history.pitcher_trackman_id`` but no crosswalk.  This validator only
accepts exact official-key matches: it never infers player identity from names,
game rows, or fuzzy profile similarity.  If coverage is insufficient, the
association/model/blend stages stop instead of fabricating a linkage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHAMPION_ZIP = ROOT / "artifacts" / "submit_r10_pitcher_w418194.zip"
EXPECTED_CHAMPION_SHA256 = (
    "ce10c428c9b6bd88ebf713fcc05b91ce22fc555573f3d11f8deb283637484c47"
)
VALIDATION_SEASONS = (2022, 2023, 2024)
PROFILE_VARIANTS = ("career_prior", "recent_weighted")
RECENT_WEIGHTS = {0: 1.0, 1: 0.5}
OLDER_WEIGHT = 0.25
MIN_EXPLORATORY_ROW_COVERAGE = 0.20
OFFSET_RIDGE_LAMBDA = 10.0
OFFSET_CAP = 0.005

TM_KEY = "pitcher_trackman_id"
TRAIN_KEY = "pitcher_id"
TM_NUMERIC_COLUMNS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
TM_STABILITY_COLUMNS = (
    "rel_speed",
    "induced_vert_break",
    "horz_break",
    "rel_height",
    "rel_side",
)
TM_REQUIRED_COLUMNS = {
    "trackman_id",
    "season",
    "game_date",
    "game_month",
    "game_dayofweek",
    "trackman_game_id",
    "pitch_no",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitch_of_pa",
    "pitcher_trackman_id",
    "batter_trackman_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team",
    "batter_team",
    "tagged_pitch_type",
    "auto_pitch_type",
    "pitch_type_group",
    *TM_NUMERIC_COLUMNS,
}


def profile_feature_groups() -> dict[str, str]:
    groups = {f"{column}_mean": "quality" for column in TM_NUMERIC_COLUMNS}
    groups.update(
        {f"{column}_std": "stability" for column in TM_STABILITY_COLUMNS}
    )
    groups.update(
        {
            "fastball_share": "pitchmix",
            "breaking_share": "pitchmix",
            "offspeed_share": "pitchmix",
            "other_share": "pitchmix",
            "pitch_type_entropy": "pitchmix",
            "tm_available": "evidence",
            "tm_pitch_count": "evidence",
            "tm_seasons": "evidence",
            "tm_last_season": "evidence",
            "tm_recency_gap": "evidence",
        }
    )
    return groups


PROFILE_FEATURE_GROUPS = profile_feature_groups()
PROFILE_FEATURES = tuple(PROFILE_FEATURE_GROUPS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype="float64")
    pred = np.asarray(prediction, dtype="float64")
    return float(np.mean((y - pred) ** 2))


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
    return value


def _member_tm_columns(member: Mapping[str, object]) -> list[str]:
    columns = member.get("feature_columns", []) or []
    return sorted(column for column in columns if str(column).startswith("tm_"))


def audit_champion_members(champion_zip: Path) -> list[dict]:
    """Read the immutable champion meta; do not infer use from dormant code."""
    with zipfile.ZipFile(champion_zip) as archive:
        meta = json.loads(archive.read("model/meta.json"))
        coldstart = json.loads(archive.read("model/coldstart_meta.json"))
        runtime = archive.read("script.py").decode("utf-8")
    members = {member["name"]: member for member in meta["members"]}
    inventory = (
        ("HGB", members["hgb"]),
        ("CatBoost", members["cat"]),
        ("ours NN", members["nn"]),
        ("team LightGBM", members["team"]),
        ("team NN", members["team_nn"]),
        (
            "cold-start expert",
            {
                "groups": [],
                "feature_columns": coldstart.get("feature_cols", []),
            },
        ),
    )
    rows = []
    for label, member in inventory:
        groups = list(member.get("groups", []) or [])
        tm_columns = _member_tm_columns(member)
        rows.append(
            {
                "member": label,
                "tm_feature_used": bool("tm" in groups or tm_columns),
                "exact_columns": tm_columns,
                "declared_groups": groups,
            }
        )
    if any(row["tm_feature_used"] for row in rows):
        raise ValueError("champion already contains an active TrackMan member")
    for row in rows:
        row["runtime_has_dormant_tm_support"] = (
            'if "tm" in groups' in runtime or "apply_lookup" in runtime
        )
    return rows


def scan_trackman(path: Path, chunk_size: int = 250_000) -> dict:
    header = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    missing = sorted(TM_REQUIRED_COLUMNS - set(header))
    if missing:
        raise ValueError(f"TrackMan schema missing required columns: {missing}")
    usecols = ["season", TM_KEY, "pitch_type_group", *TM_NUMERIC_COLUMNS]
    rows = 0
    season_counts: Counter = Counter()
    pitch_group_counts: Counter = Counter()
    numeric_missing: Counter = Counter()
    pitchers_by_season: dict[int, set[int]] = defaultdict(set)
    for chunk in pd.read_csv(
        path, usecols=usecols, chunksize=chunk_size, encoding="utf-8-sig"
    ):
        rows += len(chunk)
        season_counts.update(
            {int(key): int(value) for key, value in chunk["season"].value_counts().items()}
        )
        pitch_group_counts.update(
            {
                str(key): int(value)
                for key, value in chunk["pitch_type_group"].value_counts(
                    dropna=False
                ).items()
            }
        )
        for column in TM_NUMERIC_COLUMNS:
            numeric_missing[column] += int(chunk[column].isna().sum())
        for season, values in chunk.groupby("season")[TM_KEY]:
            finite = pd.to_numeric(values, errors="coerce").dropna().astype("int64")
            pitchers_by_season[int(season)].update(map(int, finite.unique()))
    return {
        "columns": header,
        "row_count": rows,
        "season_counts": dict(sorted(season_counts.items())),
        "pitchers_by_season": pitchers_by_season,
        "pitcher_counts": {
            season: len(values) for season, values in sorted(pitchers_by_season.items())
        },
        "pitch_type_group_counts": dict(sorted(pitch_group_counts.items())),
        "numeric_missing_rate": {
            column: numeric_missing[column] / rows for column in TM_NUMERIC_COLUMNS
        },
    }


def cutoff_pitcher_ids(scan: Mapping[str, object], cutoff: int) -> set[int]:
    result: set[int] = set()
    for season, values in scan["pitchers_by_season"].items():
        if int(season) <= cutoff:
            result.update(values)
    return result


def aggregate_trackman_profiles(
    trackman: pd.DataFrame, cutoff: int, variant: str
) -> pd.DataFrame:
    """Build a pitcher profile using TrackMan rows no later than ``cutoff``."""
    if variant not in PROFILE_VARIANTS:
        raise ValueError(f"unknown profile variant: {variant}")
    required = {"season", TM_KEY, "pitch_type_group", *TM_NUMERIC_COLUMNS}
    missing = sorted(required - set(trackman.columns))
    if missing:
        raise ValueError(f"profile input missing columns: {missing}")
    frame = trackman.loc[trackman["season"].le(cutoff)].copy()
    if frame.empty:
        return pd.DataFrame(columns=[TM_KEY, *PROFILE_FEATURES])
    if int(frame["season"].max()) > cutoff:
        raise AssertionError("future TrackMan season entered profile")
    gap = cutoff - frame["season"].to_numpy(dtype="int64")
    if variant == "career_prior":
        weights = np.ones(len(frame), dtype="float64")
    else:
        weights = np.where(
            gap == 0,
            RECENT_WEIGHTS[0],
            np.where(gap == 1, RECENT_WEIGHTS[1], OLDER_WEIGHT),
        ).astype("float64")
    keys = frame[TM_KEY].astype("int64")
    result = pd.DataFrame(index=pd.Index(sorted(keys.unique()), name=TM_KEY))
    grouped = frame.groupby(TM_KEY, sort=True)
    result["tm_pitch_count"] = grouped.size().astype("float64")
    result["tm_seasons"] = grouped["season"].nunique().astype("float64")
    result["tm_last_season"] = grouped["season"].max().astype("float64")
    result["tm_recency_gap"] = cutoff - result["tm_last_season"]
    result["tm_available"] = 1.0

    key_values = keys.to_numpy()
    for column in TM_NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype="float64"
        )
        valid = np.isfinite(values)
        stat = pd.DataFrame(
            {
                TM_KEY: key_values,
                "w": np.where(valid, weights, 0.0),
                "wx": np.where(valid, weights * values, 0.0),
                "wx2": np.where(valid, weights * values * values, 0.0),
            }
        ).groupby(TM_KEY, sort=True).sum()
        mean = stat["wx"] / stat["w"].replace(0.0, np.nan)
        result[f"{column}_mean"] = mean
        if column in TM_STABILITY_COLUMNS:
            variance = stat["wx2"] / stat["w"].replace(0.0, np.nan) - mean**2
            result[f"{column}_std"] = np.sqrt(variance.clip(lower=0.0))

    mix = pd.DataFrame(
        {
            TM_KEY: key_values,
            "pitch_type_group": frame["pitch_type_group"].astype(str).to_numpy(),
            "w": weights,
        }
    )
    mix = mix.groupby([TM_KEY, "pitch_type_group"], sort=True)["w"].sum().unstack(
        fill_value=0.0
    )
    total_weight = mix.sum(axis=1).replace(0.0, np.nan)
    share_columns = []
    for group in ("fastball", "breaking", "offspeed", "other"):
        share = mix[group] / total_weight if group in mix else 0.0
        result[f"{group}_share"] = share
        share_columns.append(f"{group}_share")
    shares = result[share_columns].to_numpy(dtype="float64")
    log_shares = np.zeros_like(shares)
    positive = shares > 0.0
    log_shares[positive] = np.log(shares[positive])
    result["pitch_type_entropy"] = -(shares * log_shares).sum(axis=1)
    return result.reset_index()[[TM_KEY, *PROFILE_FEATURES]]


def attach_trackman_profiles(
    rows: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Row-local exact lookup; unknown pitchers retain explicit missingness."""
    if TRAIN_KEY not in rows:
        raise ValueError(f"rows missing {TRAIN_KEY}")
    if profiles.empty:
        indexed = pd.DataFrame(columns=PROFILE_FEATURES)
        indexed.index.name = TM_KEY
    else:
        if profiles[TM_KEY].duplicated().any():
            raise ValueError("TrackMan profile key must be unique")
        indexed = profiles.set_index(TM_KEY)[list(PROFILE_FEATURES)]
    keys = pd.to_numeric(rows[TRAIN_KEY], errors="coerce").fillna(-1).astype("int64")
    attached = indexed.reindex(keys.to_numpy()).reset_index(drop=True)
    attached.index = rows.index
    attached["tm_available"] = attached["tm_available"].fillna(0.0)
    for column in ("tm_pitch_count", "tm_seasons"):
        attached[column] = attached[column].fillna(0.0)
    return attached


def exact_lookup_row_independence(
    rows: pd.DataFrame, profiles: pd.DataFrame
) -> bool:
    full = attach_trackman_profiles(rows, profiles)
    n = len(rows)
    indices = np.asarray(sorted(set([0, n // 3, n // 2, n - 1])))
    columns = list(PROFILE_FEATURES)
    single = pd.concat(
        [attach_trackman_profiles(rows.iloc[[index]], profiles) for index in indices]
    )
    expected = full.iloc[indices]
    if not np.array_equal(
        single[columns].to_numpy(), expected[columns].to_numpy(), equal_nan=True
    ):
        return False
    subset = attach_trackman_profiles(rows.iloc[indices], profiles)
    if not np.array_equal(
        subset[columns].to_numpy(), expected[columns].to_numpy(), equal_nan=True
    ):
        return False
    reverse = attach_trackman_profiles(rows.iloc[::-1], profiles).iloc[::-1]
    if not np.array_equal(
        reverse[columns].to_numpy(), full[columns].to_numpy(), equal_nan=True
    ):
        return False
    order = np.random.default_rng(42).permutation(n)
    shuffled = attach_trackman_profiles(rows.iloc[order], profiles)
    restored = shuffled.iloc[np.argsort(order)]
    return bool(
        np.array_equal(
            restored[columns].to_numpy(),
            full[columns].to_numpy(),
            equal_nan=True,
        )
    )


def prepare_offset_design(
    attached: pd.DataFrame, feature_columns: Sequence[str]
) -> np.ndarray:
    missing = sorted(set(feature_columns) - set(attached.columns))
    if missing:
        raise ValueError(f"missing offset features: {missing}")
    values = attached[list(feature_columns)].to_numpy(dtype="float64")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def fit_ridge_offset(
    design: np.ndarray, residual: np.ndarray, ridge_lambda: float
) -> np.ndarray:
    matrix = np.asarray(design, dtype="float64")
    target = np.asarray(residual, dtype="float64")
    if matrix.ndim != 2 or len(matrix) != len(target) or len(target) == 0:
        raise ValueError("design/residual shapes are invalid")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("design/residual contains NaN or infinity")
    if not np.isfinite(ridge_lambda) or ridge_lambda <= 0.0:
        raise ValueError("ridge_lambda must be finite and positive")
    gram = matrix.T @ matrix / len(matrix)
    linear = matrix.T @ target / len(matrix)
    return np.linalg.solve(
        gram + ridge_lambda * np.eye(matrix.shape[1]), linear
    )


def nested_offset_beta(
    validation_season: int,
    designs: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    champions: Mapping[int, np.ndarray],
) -> np.ndarray:
    fit_seasons = tuple(
        season for season in sorted(designs) if season < validation_season
    )
    width = next(iter(designs.values())).shape[1]
    if not fit_seasons:
        return np.zeros(width, dtype="float64")
    design = np.concatenate([designs[season] for season in fit_seasons])
    residual = np.concatenate(
        [targets[season] - champions[season] for season in fit_seasons]
    )
    return fit_ridge_offset(design, residual, OFFSET_RIDGE_LAMBDA)


def apply_tm_correction(
    champion: np.ndarray, design: np.ndarray, beta: np.ndarray, cap: float
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(champion, dtype="float64")
    matrix = np.asarray(design, dtype="float64")
    coefficients = np.asarray(beta, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[0] != len(prediction):
        raise ValueError("champion/design shapes are invalid")
    if matrix.shape[1] != len(coefficients):
        raise ValueError("design/beta shapes are invalid")
    if not (
        np.isfinite(prediction).all()
        and np.isfinite(matrix).all()
        and np.isfinite(coefficients).all()
    ):
        raise ValueError("correction input contains NaN or infinity")
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("cap must be finite and positive")
    delta = np.clip(matrix @ coefficients, -cap, cap)
    candidate = np.clip(prediction + delta, 0.0, 1.0)
    if not np.isfinite(candidate).all() or np.any((candidate < 0.0) | (candidate > 1.0)):
        raise AssertionError("candidate prediction is invalid")
    return candidate, delta


def load_validation_payloads(train_path: Path, oof_dir: Path) -> dict[int, dict]:
    columns = ["season", TRAIN_KEY, "game_type", "control_success"]
    train = pd.read_csv(train_path, usecols=columns, encoding="utf-8-sig")
    train = train.loc[train["season"].isin(VALIDATION_SEASONS)].copy()
    payloads = {}
    for season in VALIDATION_SEASONS:
        rows = train.loc[train["season"].eq(season)].reset_index(drop=True)
        cache_path = oof_dir / "cache" / f"season_{season}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"OOF cache missing: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            required = {
                "y",
                "champion_final",
                "ours_stage",
                "team_stage",
                "x_game_type",
                "x_pitcher_seen",
            }
            missing = sorted(required - set(cache.files))
            if missing:
                raise ValueError(f"OOF cache {season} missing {missing}")
            values = {key: cache[key] for key in required}
        y = rows["control_success"].to_numpy(dtype="float64")
        if not np.array_equal(y, np.asarray(values["y"], dtype="float64")):
            raise ValueError(f"train/OOF label order mismatch for {season}")
        game_type = rows["game_type"].astype(str).to_numpy()
        if not np.array_equal(game_type, np.asarray(values["x_game_type"]).astype(str)):
            raise ValueError(f"train/OOF game_type order mismatch for {season}")
        rows["pitcher_seen"] = np.asarray(values["x_pitcher_seen"], dtype=bool)
        payloads[season] = {
            "rows": rows,
            "y": y,
            "champion": np.asarray(values["champion_final"], dtype="float64"),
            "ours_stage": np.asarray(values["ours_stage"], dtype="float64"),
            "team_stage": np.asarray(values["team_stage"], dtype="float64"),
        }
    return payloads


def coverage_table(
    payloads: Mapping[int, Mapping[str, object]], scan: Mapping[str, object]
) -> pd.DataFrame:
    output = []
    for season in VALIDATION_SEASONS:
        cutoff = season - 1
        valid_ids = cutoff_pitcher_ids(scan, cutoff)
        rows = payloads[season]["rows"]
        matched = rows[TRAIN_KEY].isin(valid_ids).to_numpy()
        masks = [("all", "all", np.ones(len(rows), dtype=bool))]
        masks.extend(
            (
                "pitcher_seen",
                value,
                rows["pitcher_seen"].to_numpy(dtype=bool)
                if value == "seen"
                else ~rows["pitcher_seen"].to_numpy(dtype=bool),
            )
            for value in ("seen", "unseen")
        )
        masks.extend(
            (
                "game_type",
                value,
                rows["game_type"].astype(str).eq(value).to_numpy(),
            )
            for value in ("R", "F")
        )
        for subset_type, subset_value, mask in masks:
            subset_rows = rows.loc[mask]
            subset_matched = matched[mask]
            n_rows = int(mask.sum())
            n_pitchers = int(subset_rows[TRAIN_KEY].nunique())
            matched_pitchers = int(
                subset_rows.loc[subset_matched, TRAIN_KEY].nunique()
            )
            unmatched_pitchers = int(
                subset_rows.loc[~subset_matched, TRAIN_KEY].nunique()
            )
            output.append(
                {
                    "validation_season": season,
                    "tm_cutoff": cutoff,
                    "subset_type": subset_type,
                    "subset_value": subset_value,
                    "validation_rows": n_rows,
                    "matched_rows": int(subset_matched.sum()),
                    "unmatched_rows": int(n_rows - subset_matched.sum()),
                    "validation_pitchers": n_pitchers,
                    "matched_pitchers": matched_pitchers,
                    "unmatched_pitchers": unmatched_pitchers,
                    "row_coverage": float(subset_matched.mean()) if n_rows else 0.0,
                    "pitcher_coverage": (
                        matched_pitchers / n_pitchers if n_pitchers else 0.0
                    ),
                    "join_status": "exact_official_id_no_match"
                    if not subset_matched.any()
                    else "exact_official_id_match",
                }
            )
    return pd.DataFrame(output)


def unavailable_result_tables(
    payloads: Mapping[int, Mapping[str, object]]
) -> dict[str, pd.DataFrame]:
    association_rows = []
    stability_rows = []
    for profile in PROFILE_VARIANTS:
        for feature, feature_group in PROFILE_FEATURE_GROUPS.items():
            for season in VALIDATION_SEASONS:
                association_rows.append(
                    {
                        "profile": profile,
                        "feature": feature,
                        "feature_group": feature_group,
                        "validation_season": season,
                        "tm_cutoff": season - 1,
                        "n_matched": 0,
                        "q1_mean_feature": np.nan,
                        "q1_mean_signed_residual": np.nan,
                        "q1_mean_squared_error": np.nan,
                        "q4_mean_feature": np.nan,
                        "q4_mean_signed_residual": np.nan,
                        "q4_mean_squared_error": np.nan,
                        "residual_gradient": np.nan,
                        "error_gradient": np.nan,
                        "residual_direction": "not_evaluated",
                        "error_direction": "not_evaluated",
                        "status": "not_evaluated_zero_exact_key_coverage",
                    }
                )
            stability_rows.append(
                {
                    "profile": profile,
                    "feature": feature,
                    "feature_group": feature_group,
                    "direction_2022": "not_evaluated",
                    "direction_2023": "not_evaluated",
                    "direction_2024": "not_evaluated",
                    "stable": False,
                    "status": "not_evaluated_zero_exact_key_coverage",
                }
            )

    member_rows = []
    complementarity_rows = []
    for profile in PROFILE_VARIANTS:
        for season in VALIDATION_SEASONS:
            payload = payloads[season]
            member_rows.append(
                {
                    "profile": profile,
                    "validation_season": season,
                    "tm_cutoff": season - 1,
                    "n": len(payload["y"]),
                    "champion_brier": brier(payload["y"], payload["champion"]),
                    "tm_member_brier": np.nan,
                    "corr_tm_champion": np.nan,
                    "corr_tm_ours_stage": np.nan,
                    "corr_tm_team_stage": np.nan,
                    "status": "not_trained_zero_exact_key_coverage",
                }
            )
            complementarity_rows.append(
                {
                    "profile": profile,
                    "validation_season": season,
                    "tm_wins_rate": np.nan,
                    "champion_wins_rate": np.nan,
                    "tie_rate": np.nan,
                    "champion_high_error_tm_wins_rate": np.nan,
                    "status": "not_evaluated_member_not_trained",
                }
            )
    blend_rows = [
        {
            "candidate": f"{profile}_blend",
            "weight": np.nan,
            "brier_gain_2023": np.nan,
            "brier_gain_2024": np.nan,
            "pooled_brier_gain": np.nan,
            "verdict": "reject",
            "status": "not_evaluated_no_independent_member",
        }
        for profile in PROFILE_VARIANTS
    ]
    ablation_rows = [
        {
            "profile": profile,
            "removed_group": group,
            "brier_delta_2023": np.nan,
            "brier_delta_2024": np.nan,
            "pooled_brier_delta": np.nan,
            "status": "not_evaluated_no_passing_trackman_model",
        }
        for profile in PROFILE_VARIANTS
        for group in ("quality", "stability", "pitchmix", "evidence")
    ]
    return {
        "residual_association": pd.DataFrame(association_rows),
        "feature_stability": pd.DataFrame(stability_rows),
        "member_results": pd.DataFrame(member_rows),
        "complementarity": pd.DataFrame(complementarity_rows),
        "blend_results": pd.DataFrame(blend_rows),
        "ablation": pd.DataFrame(ablation_rows),
    }


def runtime_self_checks() -> dict[str, bool]:
    synthetic = pd.DataFrame(
        {
            "season": [2020, 2021, 2022, 2022],
            TM_KEY: [10, 10, 10, 20],
            "pitch_type_group": ["fastball", "breaking", "offspeed", "fastball"],
            **{
                column: np.asarray([1.0, 2.0, 1000.0, 4.0])
                for column in TM_NUMERIC_COLUMNS
            },
        }
    )
    profile = aggregate_trackman_profiles(synthetic, 2021, "career_prior")
    future_excluded = bool(
        len(profile) == 1
        and int(profile.loc[0, TM_KEY]) == 10
        and profile.loc[0, "tm_pitch_count"] == 2.0
        and profile.loc[0, "rel_speed_mean"] == 1.5
    )
    rows = pd.DataFrame({TRAIN_KEY: [10, 999, 10, 999, 10]})
    attached = attach_trackman_profiles(rows, profile)
    unseen_fallback = bool(
        attached.loc[1, "tm_available"] == 0.0
        and attached.loc[1, "tm_pitch_count"] == 0.0
        and np.isnan(attached.loc[1, "rel_speed_mean"])
    )
    row_independent = exact_lookup_row_independence(rows, profile)
    champion = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0])
    design = np.arange(10, dtype="float64").reshape(5, 2) / 10.0
    candidate, delta = apply_tm_correction(
        champion, design, np.zeros(2), OFFSET_CAP
    )
    identity = bool(
        np.array_equal(candidate, champion)
        and np.array_equal(delta, np.zeros_like(delta))
    )
    bounded, _ = apply_tm_correction(
        champion, design, np.asarray([100.0, -100.0]), OFFSET_CAP
    )
    prediction_bounds = bool(
        np.isfinite(bounded).all() and np.all((bounded >= 0.0) & (bounded <= 1.0))
    )
    return {
        "cutoff_future_season_exclusion": future_excluded,
        "unseen_pitcher_fallback": unseen_fallback,
        "row_independence": row_independent,
        "champion_identity_at_zero_correction": identity,
        "prediction_finite_and_bounded": prediction_bounds,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("/Users/wooh/Documents/dev/open/data/train.csv"),
    )
    parser.add_argument(
        "--trackman",
        type=Path,
        default=Path("/Users/wooh/Documents/dev/open/data/trackman_history.csv"),
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=ROOT / "artifacts" / "regime_blend_oof"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "trackman_incremental"
    )
    args = parser.parse_args(argv)

    for path in (args.train, args.trackman, CHAMPION_ZIP):
        if not path.is_file():
            parser.error(f"required file not found: {path}")
    checksum_before = sha256(CHAMPION_ZIP)
    if checksum_before != EXPECTED_CHAMPION_SHA256:
        raise SystemExit(f"champion checksum mismatch: {checksum_before}")

    member_audit = audit_champion_members(CHAMPION_ZIP)
    scan = scan_trackman(args.trackman)
    payloads = load_validation_payloads(args.train, args.oof_dir)
    coverage = coverage_table(payloads, scan)
    overall = coverage.loc[coverage["subset_type"].eq("all")]
    coverage_sufficient = bool(
        len(overall) == len(VALIDATION_SEASONS)
        and overall["row_coverage"].ge(MIN_EXPLORATORY_ROW_COVERAGE).all()
    )
    if coverage_sufficient:
        raise SystemExit(
            "exact-key coverage unexpectedly passed; this validator intentionally "
            "requires a separately reviewed modeling extension before fitting"
        )
    tables = unavailable_result_tables(payloads)
    self_checks = runtime_self_checks()
    if not all(self_checks.values()):
        raise SystemExit(f"runtime self-check failed: {self_checks}")

    args.output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output / "coverage.csv", index=False)
    for name, frame in tables.items():
        frame.to_csv(args.output / f"{name}.csv", index=False, na_rep="")

    checksum_after = sha256(CHAMPION_ZIP)
    if checksum_after != checksum_before:
        raise SystemExit("champion checksum changed during TrackMan audit")
    direct_overlap = {
        str(season): int(
            overall.loc[
                overall["validation_season"].eq(season), "matched_pitchers"
            ].iloc[0]
        )
        for season in VALIDATION_SEASONS
    }
    summary = {
        "research_question": (
            "does official TrackMan history add stable signal after champion prediction"
        ),
        "champion_lb_bss": 1016.4442212358,
        "champion_member_audit": member_audit,
        "champion_uses_trackman": False,
        "trackman_schema": {
            "columns": scan["columns"],
            "row_count": scan["row_count"],
            "season_counts": scan["season_counts"],
            "pitcher_counts": scan["pitcher_counts"],
            "pitch_type_group_counts": scan["pitch_type_group_counts"],
            "numeric_missing_rate": scan["numeric_missing_rate"],
            "train_join_key": TRAIN_KEY,
            "trackman_join_key": TM_KEY,
            "official_crosswalk_present": False,
        },
        "profile_definitions": {
            "career_prior": "all TrackMan seasons <= validation season - 1",
            "recent_weighted": {
                "last_season": 1.0,
                "two_seasons_ago": 0.5,
                "older": 0.25,
            },
            "features": PROFILE_FEATURE_GROUPS,
        },
        "coverage_gate": {
            "minimum_exploratory_row_coverage": MIN_EXPLORATORY_ROW_COVERAGE,
            "passed": coverage_sufficient,
            "exact_matched_pitchers_by_validation_season": direct_overlap,
            "reason": (
                "train.pitcher_id and trackman_history.pitcher_trackman_id have "
                "disjoint namespaces; no official crosswalk is provided"
            ),
        },
        "stages": {
            "residual_association": "not evaluated: zero exact-key coverage",
            "temporal_feature_stability": "not evaluated: zero exact-key coverage",
            "trackman_member": "not trained: zero exact-key coverage",
            "incremental_offset": "not trained: zero exact-key coverage",
            "complementarity": "not evaluated: no TrackMan member",
            "simple_blend": "not evaluated: no independent TrackMan member",
            "ablation": "not evaluated: no passing TrackMan model",
        },
        "identity_inference_policy": (
            "exact official IDs only; no names, external data, game-row linkage, "
            "or fuzzy player-profile matching"
        ),
        "self_checks": self_checks,
        "validation_labels_used_for_trackman_fit": False,
        "test_rows_read": False,
        "production_modified": False,
        "champion_sha256_before": checksum_before,
        "champion_sha256_after": checksum_after,
        "final_verdict": "C. No stable incremental TrackMan signal",
    }
    (args.output / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("champion TrackMan usage audit:")
    for row in member_audit:
        print(
            f"  {row['member']}: used={row['tm_feature_used']} "
            f"columns={row['exact_columns']}"
        )
    print("exact-key TrackMan coverage:")
    for row in overall.itertuples():
        print(
            f"  validation={row.validation_season} cutoff={row.tm_cutoff} "
            f"rows={row.matched_rows}/{row.validation_rows} "
            f"pitchers={row.matched_pitchers}/{row.validation_pitchers}"
        )
    print(f"saved audit artifacts to {args.output}")
    print(f"verdict: {summary['final_verdict']}")
    print(f"champion checksum unchanged: {checksum_before == checksum_after}")


if __name__ == "__main__":
    main()
