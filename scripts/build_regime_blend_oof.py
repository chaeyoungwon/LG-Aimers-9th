"""Reconstruct leakage-safe temporal OOF predictions for the R10 champion.

Cutoff c is trained on season <= c and predicts season c + 1.  Team Platt
calibration for target season s is fitted on the previous forward prediction
(models <= s-2 predicting s-1), never on target-season labels.

The immutable leaderboard champion ZIP is read only for recipe metadata.  This
script never reads test.csv and never modifies submission/ or existing ZIPs.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import importlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


def _import_lightgbm():
    """Load a wheel-local libomp on macOS before importing LightGBM if needed."""
    try:
        return importlib.import_module("lightgbm")
    except OSError as error:
        if "libomp.dylib" not in str(error):
            raise
        prefix = Path(sys.prefix)
        candidates = list(prefix.glob("lib/python*/site-packages/torch/lib/libomp.dylib"))
        candidates += list(prefix.glob("lib/python*/site-packages/sklearn/.dylibs/libomp.dylib"))
        if not candidates:
            raise
        ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)
        return importlib.import_module("lightgbm")


lgb = _import_lightgbm()


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nn_embed import (  # noqa: E402
    IdVocab,
    TabularPrep,
    predict_embed_nn,
    train_embed_nn,
)
from src.train_base import (  # noqa: E402
    CAT_COLS as TEAM_CAT_COLS,
    MODEL_CONFIGS,
    PARAMS as TEAM_PARAMS,
    add_features as team_add_features,
    apply_category_maps,
    build_category_maps,
)
from src.train_nn_candidate import build_nn_features as build_team_nn_features  # noqa: E402


ID_COL = "row_id"
TARGET_COL = "control_success"
RECIPE_VERSION = "r10-temporal-oof-v1"
CHAMPION_ZIP = ROOT / "artifacts" / "submit_r10_pitcher_w418194.zip"
TARGET_SEASONS = (2020, 2021, 2022, 2023, 2024)
REPORT_SEASONS = (2021, 2022, 2023, 2024)
FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024))
FEATURE_COLUMNS = (
    "game_type",
    "balls_before",
    "strikes_before",
    "pitcher_hand",
    "batter_hand",
    "base_state",
    "inning",
    "top_bottom",
    "asof_pitcher_n",
    "pitcher_seen",
)
MEMBERS = ("hgb", "cat", "nn", "team", "team_nn")
BLEND_MEMBERS = ("ours_stage", "team_stage")
ANCHOR_WEIGHTS = np.asarray([0.4, 0.6], dtype="float64")
TEAM_GROUP_WEIGHTS = np.asarray(
    [0.6090563539773146, 0.3909436460226854], dtype="float64"
)
TEAM_STAGE_NN_WEIGHT = 0.10
OURS_WEIGHTS = np.asarray([0.4, 0.4, 0.2], dtype="float64")
COLDSTART_WEIGHT = 0.4181944103545871
PITCHER_FORM_M = 75.0
PITCHER_FORM_ALPHA = 0.28
BATTER_FORM_M = 50.0
BATTER_FORM_ALPHA = 0.11250648296393913
EPS = 1e-6


def log(message: str) -> None:
    print(message, flush=True)


def load_champion_meta(path: Path = CHAMPION_ZIP) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("model/meta.json"))


def _member_spec(meta: Mapping[str, object], name: str) -> dict:
    matches = [member for member in meta["members"] if member["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"champion meta member {name!r}: expected one, got {len(matches)}")
    return matches[0]


def apply_logit_shift(prediction: np.ndarray, bias: float, slope: float = 1.0) -> np.ndarray:
    p = np.clip(np.asarray(prediction, dtype="float64"), EPS, 1.0 - EPS)
    logits = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-(float(slope) * logits + float(bias))))


def fit_platt(prediction: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Unregularized two-parameter Platt fit by deterministic Newton updates."""
    p = np.clip(np.asarray(prediction, dtype="float64"), EPS, 1.0 - EPS)
    y = np.asarray(target, dtype="float64")
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("Platt prediction/target must be same-length vectors")
    x = np.column_stack([np.log(p / (1.0 - p)), np.ones(len(p))])
    coef = np.asarray([1.0, 0.0], dtype="float64")
    for _ in range(100):
        z = np.clip(x @ coef, -35.0, 35.0)
        fitted = 1.0 / (1.0 + np.exp(-z))
        variance = np.maximum(fitted * (1.0 - fitted), 1e-12)
        gradient = x.T @ (fitted - y)
        hessian = x.T @ (variance[:, None] * x)
        step = np.linalg.solve(hessian, gradient)
        coef -= step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return float(coef[0]), float(coef[1])


def build_ours_features(frame: pd.DataFrame, cat_levels: Mapping[str, Sequence]) -> pd.DataFrame:
    """Recovered R10 HGB/Cat/ours-NN row-local 52-column feature frame."""
    exclude = {ID_COL, TARGET_COL, "pitcher_id", "batter_id"}
    out = frame[[column for column in frame.columns if column not in exclude]].copy()
    for column, levels in cat_levels.items():
        out[column] = pd.Categorical(out[column], categories=levels)
    count = frame["balls_before"] * 3 + frame["strikes_before"]
    out["count_state"] = pd.Categorical(count, categories=list(range(12)))
    out["same_hand"] = (frame["pitcher_hand"] == frame["batter_hand"]).astype("int8")
    combo = frame["pitcher_hand"].astype(str) + "-" + frame["batter_hand"].astype(str)
    out["hand_combo"] = pd.Categorical(
        combo, categories=["1-1", "1-2", "2-1", "2-2"]
    )
    out["strike_minus_ball"] = (
        frame["asof_pitcher_strike_rate"] - frame["asof_pitcher_ball_rate"]
    ).astype("float32")
    out["success_minus_middle"] = (
        frame["asof_pitcher_success_rate"] - frame["asof_pitcher_middle_rate"]
    ).astype("float32")
    out["prev_vs_career"] = (
        frame["asof_pitcher_prev5_game_success_rate"]
        - frame["asof_pitcher_success_rate"]
    ).astype("float32")
    second = np.fmax(
        frame["asof_pitcher_breaking_rate"], frame["asof_pitcher_offspeed_rate"]
    )
    out["fastball_dominance"] = (
        frame["asof_pitcher_fastball_rate"] - second
    ).astype("float32")
    return out


def catboost_frame(frame: pd.DataFrame, categorical: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in categorical:
        values = out[column].astype("object")
        out[column] = values.where(values.notna(), "__NA__").astype(str)
    return out


def train_hgb(
    train_features: pd.DataFrame,
    target: np.ndarray,
    prediction_frames: Sequence[pd.DataFrame],
) -> Sequence[np.ndarray]:
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.1,
        max_iter=50,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=3000,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features="from_dtype",
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=42,
        class_weight=None,
    )
    model.fit(train_features, target)
    return [model.predict_proba(frame)[:, 1] for frame in prediction_frames]


def train_catboost(
    train_features: pd.DataFrame,
    target: np.ndarray,
    prediction_frames: Sequence[pd.DataFrame],
    categorical: Sequence[str],
    threads: int,
) -> Sequence[np.ndarray]:
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=20.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        random_strength=1.0,
        bootstrap_type="MVS",
        subsample=0.8,
        border_count=254,
        one_hot_max_size=2,
        allow_writing_files=False,
        thread_count=threads,
        verbose=False,
    )
    train_ready = catboost_frame(train_features, categorical)
    model.fit(train_ready, target, cat_features=list(categorical), verbose=False)
    del train_ready
    return [
        model.predict_proba(catboost_frame(frame, categorical))[:, 1]
        for frame in prediction_frames
    ]


def train_nn(
    train_features: pd.DataFrame,
    train_frame: pd.DataFrame,
    target: np.ndarray,
    prediction_features: Sequence[pd.DataFrame],
    prediction_frames: Sequence[pd.DataFrame],
    threads: int,
    verbose: bool,
) -> Sequence[np.ndarray]:
    import torch

    torch.set_num_threads(threads)
    prep = TabularPrep().fit(train_features)
    pitcher_vocab = IdVocab.fit(train_frame["pitcher_id"])
    batter_vocab = IdVocab.fit(train_frame["batter_id"])
    x_train = prep.transform(train_features)
    pitcher = pitcher_vocab.transform(train_frame["pitcher_id"])
    batter = batter_vocab.transform(train_frame["batter_id"])
    model, _ = train_embed_nn(
        x_train,
        pitcher,
        batter,
        np.asarray(target, dtype="float32"),
        n_pitcher=pitcher_vocab.size,
        n_batter=batter_vocab.size,
        emb_dim=8,
        id_dropout=0.08,
        hidden=(128, 64),
        dropout=0.1,
        lr=1e-3,
        batch_size=4096,
        max_epochs=30,
        patience=3,
        val_frac=0.1,
        seed=42,
        device="cpu",
        verbose=verbose,
    )
    del x_train, pitcher, batter
    predictions = []
    for features, rows in zip(prediction_features, prediction_frames):
        predictions.append(
            predict_embed_nn(
                model,
                prep.transform(features),
                pitcher_vocab.transform(rows["pitcher_id"]),
                batter_vocab.transform(rows["batter_id"]),
                batch_size=16384,
                device="cpu",
            )
        )
    return predictions


def train_team_lgbm(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    cutoff: int,
    threads: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    train_features = team_add_features(train_frame)
    target_features = team_add_features(target_frame)
    feature_columns = [
        column for column in train_features.columns if column not in (ID_COL, TARGET_COL)
    ]
    categorical = [column for column in TEAM_CAT_COLS if column in feature_columns]
    category_maps = build_category_maps(train_features, categorical)
    train_encoded = apply_category_maps(train_features, categorical, category_maps)
    target_encoded = apply_category_maps(target_features, categorical, category_maps)
    predictions: Dict[str, np.ndarray] = {}
    for config in MODEL_CONFIGS:
        params = dict(
            TEAM_PARAMS,
            seed=config["seed"],
            num_leaves=config["num_leaves"],
            min_data_in_leaf=config["min_data_in_leaf"],
            objective=config["objective"],
            metric="binary_logloss",
            num_threads=threads,
        )
        sample_weight = None
        if config["season_decay"] is not None:
            sample_weight = config["season_decay"] ** (
                (cutoff + 1) - train_encoded["season"].to_numpy()
            )
        dataset = lgb.Dataset(
            train_encoded[feature_columns],
            label=train_encoded[TARGET_COL],
            weight=sample_weight,
            categorical_feature=categorical,
            free_raw_data=False,
        )
        booster = lgb.train(params, dataset, num_boost_round=config["rounds"])
        predictions[config["name"]] = booster.predict(target_encoded[feature_columns])

    positive = {
        "asof_pitcher_success_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_batter_success_rate",
    }
    negative = {
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_middle_rate",
    }
    mono_params = dict(
        TEAM_PARAMS,
        seed=42,
        num_leaves=63,
        min_data_in_leaf=1200,
        num_threads=threads,
        monotone_constraints=[
            1 if column in positive else -1 if column in negative else 0
            for column in feature_columns
        ],
        monotone_constraints_method="advanced",
    )
    dataset = lgb.Dataset(
        train_encoded[feature_columns],
        label=train_encoded[TARGET_COL],
        categorical_feature=categorical,
        free_raw_data=False,
    )
    booster = lgb.train(mono_params, dataset, num_boost_round=220)
    predictions["mono63"] = booster.predict(target_encoded[feature_columns])

    first = np.average(
        [predictions["eng31"], predictions["leaf63"], predictions["decay85"]],
        axis=0,
        weights=[0.370886, 0.470361, 0.158753],
    )
    second = np.average(
        [predictions["leaf63"], predictions["decay85"], predictions["mono63"]],
        axis=0,
        weights=[0.263430, 0.045001, 0.691570],
    )
    return first, second, predictions


def train_coldstart_expert(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    threads: int,
) -> Tuple[np.ndarray, np.ndarray]:
    regular = train_frame[train_frame["game_type"].eq("R")]
    train_features = team_add_features(regular)
    target_features = team_add_features(target_frame)
    drop = {ID_COL, TARGET_COL, "pitcher_id", "batter_id"}
    feature_columns = [column for column in train_features.columns if column not in drop]
    categorical = [column for column in TEAM_CAT_COLS if column in feature_columns]
    category_maps = build_category_maps(train_features, categorical)
    train_encoded = apply_category_maps(train_features, categorical, category_maps)
    target_encoded = apply_category_maps(target_features, categorical, category_maps)
    params = dict(
        TEAM_PARAMS,
        seed=42,
        num_leaves=31,
        min_data_in_leaf=800,
        num_threads=threads,
    )
    dataset = lgb.Dataset(
        train_encoded[feature_columns],
        label=train_encoded[TARGET_COL],
        categorical_feature=categorical,
        free_raw_data=False,
    )
    model = lgb.train(params, dataset, num_boost_round=236)
    prediction = model.predict(target_encoded[feature_columns])
    known = regular["pitcher_id"].dropna().to_numpy(dtype="int64")
    pitcher = target_frame["pitcher_id"].to_numpy(dtype="int64")
    cold = target_frame["game_type"].eq("R").to_numpy() & ~np.isin(pitcher, known)
    return prediction, cold


def build_player_table(
    frame: pd.DataFrame,
    id_column: str,
    n_column: str,
    rate_column: str,
) -> dict:
    n = frame[n_column].to_numpy(dtype="float64")
    rate = frame[rate_column].to_numpy(dtype="float64")
    successes = np.rint(n * np.where(np.isfinite(rate), rate, 0.0))
    player = frame[id_column].to_numpy(dtype="int64")
    target = frame[TARGET_COL].to_numpy(dtype="float64")
    order = np.lexsort((n, player))
    ordered_player = player[order]
    last = np.r_[ordered_player[1:] != ordered_player[:-1], True]
    index = order[last]
    return {
        "ids": ordered_player[last],
        "n0": n[index] + 1.0,
        "s0": successes[index] + target[index],
    }


def player_form_values(
    frame: pd.DataFrame,
    table: Mapping[str, np.ndarray],
    id_column: str,
    n_column: str,
    rate_column: str,
    m: float,
    mu: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(table["ids"], dtype="int64")
    player = frame[id_column].to_numpy(dtype="int64")
    position = np.searchsorted(ids, player)
    clipped = np.clip(position, 0, max(len(ids) - 1, 0))
    seen = (position < len(ids)) & (ids[clipped] == player)
    n = frame[n_column].to_numpy(dtype="float64")
    rate = frame[rate_column].to_numpy(dtype="float64")
    successes = np.rint(n * np.where(np.isfinite(rate), rate, 0.0))
    n0 = np.where(seen, np.asarray(table["n0"])[clipped], 0.0)
    s0 = np.where(seen, np.asarray(table["s0"])[clipped], 0.0)
    p0 = np.divide(s0, np.where(n0 > 0, n0, 1.0))
    season_n = n - n0
    form = (successes - s0 + float(m) * p0) / (season_n + float(m)) - p0
    active = (
        seen
        & np.isfinite(season_n)
        & (season_n > 0)
        & np.isfinite(form)
        & frame["game_type"].eq("R").to_numpy()
    )
    return np.where(active, form - float(mu), 0.0), seen


def form_offsets(
    all_train: pd.DataFrame,
    cutoff: int,
    target_frame: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    train = all_train[all_train["season"] <= cutoff]
    previous = all_train[all_train["season"] <= cutoff - 1]
    reference = all_train[all_train["season"] == cutoff]

    pitcher_table = build_player_table(
        train,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    )
    previous_pitcher_table = build_player_table(
        previous,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    )
    reference_form, reference_seen = player_form_values(
        reference,
        previous_pitcher_table,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        PITCHER_FORM_M,
    )
    reference_seen_r = reference_seen & reference["game_type"].eq("R").to_numpy()
    pitcher_mu = float(reference_form[reference_seen_r].mean())
    pitcher_form, _ = player_form_values(
        target_frame,
        pitcher_table,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        PITCHER_FORM_M,
        pitcher_mu,
    )

    batter_table = build_player_table(
        train,
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
    )
    batter_form, _ = player_form_values(
        target_frame,
        batter_table,
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
        BATTER_FORM_M,
    )
    return (
        PITCHER_FORM_ALPHA * pitcher_form,
        BATTER_FORM_ALPHA * batter_form,
        {"pitcher_mu": pitcher_mu},
    )


def forecast_next_rate(train_frame: pd.DataFrame, target_season: int) -> float:
    rates = train_frame.groupby("season", sort=True)[TARGET_COL].mean()
    if len(rates) < 2:
        raise ValueError("at least two training seasons are required for c0 forecast")
    slope, intercept = np.polyfit(rates.index.to_numpy(), rates.to_numpy(), 1)
    return float(np.clip(slope * target_season + intercept, EPS, 1.0 - EPS))


def save_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    version = str(payload.get("recipe_version", ""))
    if version != RECIPE_VERSION:
        raise ValueError(f"cache recipe mismatch at {path}: {version!r}")
    return payload


def summarize_prediction(target: np.ndarray, prediction: np.ndarray) -> dict:
    p = np.asarray(prediction, dtype="float64")
    y = np.asarray(target, dtype="float64")
    finite = np.isfinite(p)
    return {
        "brier": None if not finite.all() else float(np.mean((p - y) ** 2)),
        "min": None if not finite.any() else float(np.nanmin(p)),
        "max": None if not finite.any() else float(np.nanmax(p)),
        "mean": None if not finite.any() else float(np.nanmean(p)),
        "nan_count": int((~finite).sum()),
    }


def build_target_season(
    all_train: pd.DataFrame,
    target_season: int,
    meta: dict,
    output_dir: Path,
    threads: int,
    force: bool,
    verbose_nn: bool,
) -> Tuple[dict, dict]:
    cache_path = output_dir / "cache" / f"season_{target_season}.npz"
    metadata_path = output_dir / "cache" / f"season_{target_season}.json"
    if cache_path.exists() and metadata_path.exists() and not force:
        log(f"[season {target_season}] reuse {cache_path}")
        return load_npz(cache_path), json.loads(metadata_path.read_text(encoding="utf-8"))

    cutoff = target_season - 1
    train_frame = all_train[all_train["season"] <= cutoff]
    target_frame = all_train[all_train["season"] == target_season]
    if train_frame.empty or target_frame.empty:
        raise ValueError(f"empty cutoff/target: <= {cutoff} -> {target_season}")
    y_target = target_frame[TARGET_COL].to_numpy(dtype="float64")
    started = time.monotonic()
    log(
        f"[season {target_season}] train<={cutoff} rows={len(train_frame):,}; "
        f"target rows={len(target_frame):,}"
    )

    log(f"[season {target_season}] train team LightGBM stack")
    team_group_1, team_group_2, team_raw_members = train_team_lgbm(
        train_frame, target_frame, cutoff, threads
    )
    payload: Dict[str, np.ndarray] = {
        "recipe_version": np.asarray(RECIPE_VERSION),
        "season": np.asarray(target_season, dtype="int16"),
        "y": y_target,
        "team_group_1_raw": team_group_1,
        "team_group_2_raw": team_group_2,
    }
    for name, prediction in team_raw_members.items():
        payload[f"team_raw_{name}"] = prediction

    season_metadata = {
        "recipe_version": RECIPE_VERSION,
        "target_season": target_season,
        "cutoff": cutoff,
        "train_rows": len(train_frame),
        "target_rows": len(target_frame),
        "reproducibility": {
            "team": "source-exact",
            "team_nn": "source-exact recipe",
            "hgb": "recipe-reconstructed",
            "cat": "recipe-reconstructed",
            "nn": "recipe-reconstructed",
        },
    }

    if target_season == 2020:
        save_npz_atomic(cache_path, payload)
        metadata_path.write_text(
            json.dumps(season_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload, season_metadata

    previous = load_npz(output_dir / "cache" / f"season_{target_season - 1}.npz")
    previous_target = previous["y"].astype("float64")
    calibrations = []
    calibrated_groups = []
    for group_name in ("team_group_1_raw", "team_group_2_raw"):
        scale, bias = fit_platt(previous[group_name], previous_target)
        calibrations.append({"scale": scale, "bias": bias})
        calibrated_groups.append(apply_logit_shift(payload[group_name], bias, scale))
    team_prediction = np.average(
        calibrated_groups, axis=0, weights=TEAM_GROUP_WEIGHTS
    )
    payload["team"] = team_prediction
    season_metadata["team_calibration_from_season"] = target_season - 1
    season_metadata["team_calibration"] = calibrations

    cat_levels = meta["cat_levels"]
    ours_spec = _member_spec(meta, "hgb")
    cat_spec = _member_spec(meta, "cat")
    train_ours = build_ours_features(train_frame, cat_levels)
    reference_frame = all_train[all_train["season"] == cutoff]
    reference_ours = build_ours_features(reference_frame, cat_levels)
    target_ours = build_ours_features(target_frame, cat_levels)
    if list(train_ours.columns) != ours_spec["feature_columns"]:
        raise ValueError("recovered ours feature order differs from champion meta")

    log(f"[season {target_season}] train recovered HGB")
    hgb_ref, hgb_target = train_hgb(
        train_ours,
        train_frame[TARGET_COL].to_numpy(dtype="int8"),
        (reference_ours, target_ours),
    )
    payload["hgb"] = hgb_target
    gc.collect()

    log(f"[season {target_season}] train recovered CatBoost")
    cat_ref, cat_target = train_catboost(
        train_ours,
        train_frame[TARGET_COL].to_numpy(dtype="int8"),
        (reference_ours, target_ours),
        cat_spec["cat_features"],
        threads,
    )
    payload["cat"] = cat_target
    gc.collect()

    log(f"[season {target_season}] train recovered ours NN")
    nn_ref, nn_target = train_nn(
        train_ours,
        train_frame,
        train_frame[TARGET_COL].to_numpy(dtype="float32"),
        (reference_ours, target_ours),
        (reference_frame, target_frame),
        threads,
        verbose_nn,
    )
    payload["nn"] = nn_target
    ours_ref_raw = (
        OURS_WEIGHTS[0] * hgb_ref
        + OURS_WEIGHTS[1] * cat_ref
        + OURS_WEIGHTS[2] * nn_ref
    )
    ours_target_raw = (
        OURS_WEIGHTS[0] * hgb_target
        + OURS_WEIGHTS[1] * cat_target
        + OURS_WEIGHTS[2] * nn_target
    )
    forecast = forecast_next_rate(train_frame, target_season)
    c0 = math.log(forecast / (1.0 - forecast)) - math.log(
        float(ours_ref_raw.mean()) / (1.0 - float(ours_ref_raw.mean()))
    )
    payload["ours_stage"] = apply_logit_shift(ours_target_raw, c0)
    season_metadata["ours_calibration"] = {
        "forecast_rate": forecast,
        "reference_season": cutoff,
        "reference_raw_mean": float(ours_ref_raw.mean()),
        "bias": c0,
    }
    del train_ours, reference_ours, target_ours
    gc.collect()

    log(f"[season {target_season}] train source-exact team NN recipe")
    train_team_nn = build_team_nn_features(train_frame)
    target_team_nn = build_team_nn_features(target_frame)
    (team_nn_target,) = train_nn(
        train_team_nn,
        train_frame,
        train_frame[TARGET_COL].to_numpy(dtype="float32"),
        (target_team_nn,),
        (target_frame,),
        threads,
        verbose_nn,
    )
    payload["team_nn"] = team_nn_target
    payload["team_stage"] = (
        (1.0 - TEAM_STAGE_NN_WEIGHT) * team_prediction
        + TEAM_STAGE_NN_WEIGHT * team_nn_target
    )
    payload["champion_base"] = (
        ANCHOR_WEIGHTS[0] * payload["ours_stage"]
        + ANCHOR_WEIGHTS[1] * payload["team_stage"]
    )
    del train_team_nn, target_team_nn
    gc.collect()

    log(f"[season {target_season}] build train-only pitcher/batter form")
    pitcher_offset, batter_offset, form_meta = form_offsets(
        all_train, cutoff, target_frame
    )
    payload["pitcher_form_pred"] = np.clip(
        payload["champion_base"] + pitcher_offset, 0.0, 1.0
    )
    payload["batter_form_pred"] = np.clip(
        payload["pitcher_form_pred"] + batter_offset, 0.0, 1.0
    )
    payload["form_offset"] = payload["batter_form_pred"] - payload["champion_base"]
    season_metadata["form"] = form_meta

    log(f"[season {target_season}] train cutoff-safe cold-start expert")
    cold_prediction, cold_mask = train_coldstart_expert(
        train_frame, target_frame, threads
    )
    payload["coldstart_expert_pred"] = cold_prediction
    payload["coldstart_mask"] = cold_mask
    final_prediction = payload["batter_form_pred"].copy()
    final_prediction[cold_mask] = (
        (1.0 - COLDSTART_WEIGHT) * final_prediction[cold_mask]
        + COLDSTART_WEIGHT * cold_prediction[cold_mask]
    )
    payload["champion_final"] = np.clip(final_prediction, 0.0, 1.0)

    known_r_pitchers = set(
        train_frame.loc[train_frame["game_type"].eq("R"), "pitcher_id"]
        .dropna()
        .astype("int64")
    )
    for column in FEATURE_COLUMNS:
        if column == "pitcher_seen":
            values = target_frame["pitcher_id"].astype("int64").isin(known_r_pitchers).to_numpy()
        else:
            values = target_frame[column].to_numpy()
        if values.dtype.kind in "OUS":
            values = values.astype("U32")
        payload[f"x_{column}"] = values

    for name in (*MEMBERS, *BLEND_MEMBERS, "champion_base", "batter_form_pred", "champion_final"):
        values = np.asarray(payload[name], dtype="float64")
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"invalid probability in {target_season}/{name}")
    exact_base = (
        ANCHOR_WEIGHTS[0] * payload["ours_stage"]
        + ANCHOR_WEIGHTS[1] * payload["team_stage"]
    )
    if not np.array_equal(exact_base, payload["champion_base"]):
        raise ValueError("champion base arithmetic parity failed")

    season_metadata["elapsed_seconds"] = time.monotonic() - started
    save_npz_atomic(cache_path, payload)
    metadata_path.write_text(
        json.dumps(season_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(
        f"[season {target_season}] saved {cache_path}; "
        f"elapsed={season_metadata['elapsed_seconds']:.1f}s"
    )
    return payload, season_metadata


def write_fold_artifacts(output_dir: Path, seasons: Mapping[int, dict]) -> dict:
    fold_specs = []
    for inner_season, validation_season in FOLDS:
        inner = seasons[inner_season]
        validation = seasons[validation_season]
        payload: Dict[str, np.ndarray] = {
            "y_inner": inner["y"],
            "y_val": validation["y"],
        }
        for name in (*MEMBERS, *BLEND_MEMBERS):
            payload[f"p_inner_{name}"] = inner[name]
            payload[f"p_val_{name}"] = validation[name]
        payload.update(
            {
                "p_inner_champion_base": inner["champion_base"],
                "p_val_champion_base": validation["champion_base"],
                "p_inner_champion_form": inner["batter_form_pred"],
                "p_val_champion_form": validation["batter_form_pred"],
                "form_offset_inner": inner["form_offset"],
                "form_offset_val": validation["form_offset"],
                "p_inner_pitcher_form": inner["pitcher_form_pred"],
                "p_val_pitcher_form": validation["pitcher_form_pred"],
                "p_inner_batter_form": inner["batter_form_pred"],
                "p_val_batter_form": validation["batter_form_pred"],
                "p_inner_coldstart_expert": inner["coldstart_expert_pred"],
                "p_val_coldstart_expert": validation["coldstart_expert_pred"],
                "p_inner_champion_final": inner["champion_final"],
                "p_val_champion_final": validation["champion_final"],
                "coldstart_mask_inner": inner["coldstart_mask"],
                "coldstart_mask_val": validation["coldstart_mask"],
            }
        )
        for feature in FEATURE_COLUMNS:
            payload[f"x_inner_{feature}"] = inner[f"x_{feature}"]
            payload[f"x_val_{feature}"] = validation[f"x_{feature}"]
        filename = f"fold_{inner_season}_{validation_season}.npz"
        save_npz_atomic(output_dir / filename, payload)
        fold_specs.append(
            {
                "inner_season": inner_season,
                "validation_season": validation_season,
                "path": filename,
            }
        )
    manifest = {
        "schema_version": 1,
        "recipe_version": RECIPE_VERSION,
        "members": list(MEMBERS),
        "blend_members": list(BLEND_MEMBERS),
        "anchor_weights": ANCHOR_WEIGHTS.tolist(),
        "regimes": [
            "game_type",
            "hand_combo",
            "count_state",
            "game_type_count_bucket",
        ],
        "feature_columns": list(FEATURE_COLUMNS),
        "folds": fold_specs,
        "provenance": {
            "train": "official train.csv only",
            "test_rows_used": False,
            "champion_reference": CHAMPION_ZIP.name,
            "champion_reference_immutable": True,
            "source_exact": ["team", "team_nn"],
            "recipe_reconstructed": ["hgb", "cat", "nn"],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_parity_report(
    all_train: pd.DataFrame,
    seasons: Mapping[int, dict],
    season_metadata: Mapping[int, dict],
) -> dict:
    fold_rows = []
    parity_pass = True
    for season in REPORT_SEASONS:
        payload = seasons[season]
        y = payload["y"]
        predictions = {
            name: summarize_prediction(y, payload[name])
            for name in (
                *MEMBERS,
                *BLEND_MEMBERS,
                "champion_base",
                "pitcher_form_pred",
                "batter_form_pred",
                "champion_final",
            )
        }
        invalid = any(item["nan_count"] for item in predictions.values())
        parity_pass = parity_pass and not invalid
        fold_rows.append(
            {
                "validation_season": season,
                "rows": len(y),
                "positive_rate": float(np.mean(y)),
                "predictions": predictions,
                "status": "pass" if not invalid else "fail",
            }
        )

    raw_2024 = seasons[2024]
    self_fit = [
        fit_platt(raw_2024["team_group_1_raw"], raw_2024["y"]),
        fit_platt(raw_2024["team_group_2_raw"], raw_2024["y"]),
    ]
    expected = [(1.07022, -0.04495), (1.021134, -0.038516)]
    deltas = [
        {"scale": got[0] - want[0], "bias": got[1] - want[1]}
        for got, want in zip(self_fit, expected)
    ]
    max_calibration_delta = max(
        abs(value) for item in deltas for value in item.values()
    )
    # The historical constants were rounded to six decimals.
    team_recipe_parity = max_calibration_delta <= 5e-4
    parity_pass = parity_pass and team_recipe_parity
    return {
        "recipe_version": RECIPE_VERSION,
        "train_rows": len(all_train),
        "seasons": fold_rows,
        "team_2024_self_fit_calibration": {
            "reconstructed": [
                {"scale": pair[0], "bias": pair[1]} for pair in self_fit
            ],
            "historical": [
                {"scale": pair[0], "bias": pair[1]} for pair in expected
            ],
            "delta": deltas,
            "max_abs_delta": max_calibration_delta,
            "status": "pass" if team_recipe_parity else "fail",
            "note": "diagnostic only; 2024 labels are not used by 2024 OOF prediction",
        },
        "season_metadata": {str(key): value for key, value in season_metadata.items()},
        "parity_pass": parity_pass,
        "historical_fold_reference": "missing; no old OOF/cache was available",
    }


def parity_markdown(report: dict) -> str:
    lines = [
        "# Reconstructed temporal OOF parity",
        "",
        "Positive Brier deltas are not reported here; this file validates the OOF inputs before blend research.",
        "",
        "| season | rows | positive rate | hgb | cat | nn | team | team_nn | ours_stage | team_stage | champion_base | champion_final | status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in report["seasons"]:
        p = row["predictions"]
        lines.append(
            f"| {row['validation_season']} | {row['rows']:,} | {row['positive_rate']:.6f} "
            f"| {p['hgb']['brier']:.9f} | {p['cat']['brier']:.9f} | {p['nn']['brier']:.9f} "
            f"| {p['team']['brier']:.9f} | {p['team_nn']['brier']:.9f} "
            f"| {p['ours_stage']['brier']:.9f} | {p['team_stage']['brier']:.9f} "
            f"| {p['champion_base']['brier']:.9f} | {p['champion_final']['brier']:.9f} "
            f"| {row['status']} |"
        )
    calibration = report["team_2024_self_fit_calibration"]
    lines.extend(
        [
            "",
            "## Team recipe parity",
            "",
            f"2024 self-fit Platt max absolute coefficient delta: `{calibration['max_abs_delta']:.3e}` ({calibration['status']}).",
            "This is a diagnostic fit only; the actual 2024 OOF prediction uses calibration learned from 2023.",
            "",
            f"Overall parity gate: **{'PASS' if report['parity_pass'] else 'FAIL'}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT.parent / "open" / "data" / "train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "regime_blend_oof",
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet-nn", action="store_true")
    args = parser.parse_args(argv)
    if not args.train.is_file():
        parser.error(f"official train.csv not found: {args.train}")
    if not CHAMPION_ZIP.is_file():
        parser.error(f"immutable champion reference not found: {CHAMPION_ZIP}")
    if args.threads < 1:
        parser.error("--threads must be positive")

    log(f"load official train: {args.train}")
    train = pd.read_csv(args.train, encoding="utf-8-sig", low_memory=False)
    required_seasons = set(range(2019, 2025))
    seasons = set(int(value) for value in train["season"].unique())
    if not required_seasons.issubset(seasons):
        raise ValueError(f"train seasons missing: {sorted(required_seasons - seasons)}")
    meta = load_champion_meta()
    args.output.mkdir(parents=True, exist_ok=True)

    reconstructed = {}
    metadata = {}
    for target_season in TARGET_SEASONS:
        reconstructed[target_season], metadata[target_season] = build_target_season(
            train,
            target_season,
            meta,
            args.output,
            args.threads,
            args.force,
            not args.quiet_nn,
        )
    write_fold_artifacts(args.output, reconstructed)
    report = build_parity_report(train, reconstructed, metadata)
    (args.output / "oof_parity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "oof_parity.md").write_text(
        parity_markdown(report), encoding="utf-8"
    )
    log(parity_markdown(report))
    if not report["parity_pass"]:
        raise SystemExit(
            "OOF parity gate failed; do not run regime blend until the mismatch is resolved"
        )
    log(f"OOF reconstruction complete: {args.output}")


if __name__ == "__main__":
    main()
