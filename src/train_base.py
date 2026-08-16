# train.py — 로컬 학습 전용 스크립트 (제출 대상 아님)
#
# data/train.csv를 읽어 LightGBM 이진분류 모델 5개(시드 배깅)를 학습하고,
# submit/model/ 아래에 (1) 모델 파일들 (2) 전처리 메타(범주형 인코딩 맵)를 저장한다.
# submit/script.py는 이 산출물만 읽어서 추론한다 — 학습 로직과 절대 분리되어야 함
# (평가 서버는 오프라인이라 학습을 다시 돌릴 수 없다).
#
# 검증 전략: train.csv는 season 2019~2024, 실제 평가는 season 2025(미래 시즌)이므로
# 무작위 K-Fold 대신 "2019~2023 학습 / 2024 검증(시간 기반 홀드아웃)"으로 일반화 성능을 추정한다.
#
# 하이퍼파라미터/피처 선택은 explore*.py 실험으로 결정했다 (자세한 내용은 대화 기록 참고):
#   - pitcher_id / batter_id를 범주형으로 넣으면 과적합으로 성능이 크게 떨어짐(BSS 268대) →
#     수치형 그대로 사용(BSS 600~630대로 개선).
#   - pitcher_team_id / batter_team_id는 카디널리티가 낮아(13개) 범주형 유지가 유리.
#   - 단일 시드보다 5-시드 배깅 평균이 더 안정적.

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

# 범주형으로 취급할 컬럼 (카디널리티 낮은 것만 — pitcher_id/batter_id는 수치형으로 둔다)
CAT_COLS = [
    "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]

MODEL_CONFIGS = [
    # Weights are constrained on the 2024 out-of-time holdout. The decay model
    # still uses every historical row, only with 0.85 ** year_age sample weights.
    {"name": "eng31", "seed": 1, "num_leaves": 31, "min_data_in_leaf": 800,
     "rounds": 236, "weight": 0.370886, "season_decay": None, "objective": "binary"},
    {"name": "leaf63", "seed": 3, "num_leaves": 63, "min_data_in_leaf": 1200,
     "rounds": 171, "weight": 0.470361, "season_decay": None, "objective": "binary"},
    {"name": "decay85", "seed": 4, "num_leaves": 63, "min_data_in_leaf": 1200,
     "rounds": 228, "weight": 0.158753, "season_decay": 0.85, "objective": "binary"},
]

DATA_DIR = "data"
MODEL_DIR = "submission/model"
HOLDOUT_SEASON = 2024  # 이 시즌을 검증용으로 떼어 시간기반 홀드아웃 구성

PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=800,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=1.0,
    lambda_l2=3.0,
    verbose=-1,
)


def load_train():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    return df


def get_feature_cols(df):
    return [c for c in df.columns if c not in (ID_COL, TARGET_COL)]


def build_category_maps(df, cat_cols):
    """train 데이터만 사용해 값(문자열화) -> 정수코드 매핑 생성."""
    maps = {}
    for c in cat_cols:
        cats = sorted(df[c].dropna().astype(str).unique().tolist())
        maps[c] = {v: i for i, v in enumerate(cats)}
    return maps


def apply_category_maps(df, cat_cols, maps):
    """값 -> 정수코드로 치환. 매핑에 없는(미등장) 값은 NaN(결측)으로 처리해
    LightGBM이 학습된 missing-value 분기로 라우팅하도록 한다."""
    df = df.copy()
    for c in cat_cols:
        codes = df[c].astype(str).map(maps[c])
        df[c] = codes.astype("float32")  # NaN 보존 위해 float
    return df


def add_features(df):
    """Only row-local transformations; never use statistics from other test rows."""
    df = df.copy()
    df["count_state"] = (df["balls_before"] * 3 + df["strikes_before"]).astype("int8")
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype("int8")
    df["late_inning"] = (df["inning"] >= 7).astype("int8")
    df["two_strikes"] = (df["strikes_before"] == 2).astype("int8")
    df["three_balls"] = (df["balls_before"] == 3).astype("int8")
    df["pitcher_rate_gap_1_5"] = (df["asof_pitcher_prev1_game_success_rate"]
                                   - df["asof_pitcher_prev5_game_success_rate"])
    df["pitcher_rate_gap_career_5"] = (df["asof_pitcher_prev5_game_success_rate"]
                                        - df["asof_pitcher_success_rate"])
    for c in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
        df[f"log1p_{c}"] = np.log1p(df[c].clip(lower=0))
    return df


def brier_skill_score(y_true, p):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    brier = float(np.mean((p - y_true) ** 2))
    r = float(y_true.mean())
    baseline = r * (1 - r)
    bss = max(0.0, 100000 * (1 - brier / baseline)) if baseline > 0 else 0.0
    return bss, brier


def main():
    print("Load train.csv ...")
    df = add_features(load_train())
    print(f" shape={df.shape}")

    feature_cols = get_feature_cols(df)
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    print(f" features={len(feature_cols)}  categorical={len(cat_cols)}  (ID 2개는 수치형 유지)")

    maps = build_category_maps(df, cat_cols)
    df_enc = apply_category_maps(df, cat_cols, maps)

    # ---- 시간 기반 홀드아웃으로 최종 설정 검증 (2019~2023 학습 / 2024 검증) ----
    train_mask = df_enc["season"] <= (HOLDOUT_SEASON - 1)
    valid_mask = df_enc["season"] == HOLDOUT_SEASON
    X_tr, y_tr = df_enc.loc[train_mask, feature_cols], df_enc.loc[train_mask, TARGET_COL]
    X_va, y_va = df_enc.loc[valid_mask, feature_cols], df_enc.loc[valid_mask, TARGET_COL]
    print(f" holdout check: train={len(X_tr)}  valid(season={HOLDOUT_SEASON})={len(X_va)}")

    holdout_preds = []
    weights = []
    for config in MODEL_CONFIGS:
        params = dict(PARAMS, seed=config["seed"], num_leaves=config["num_leaves"],
                      min_data_in_leaf=config["min_data_in_leaf"], objective=config["objective"],
                      metric="l2" if config["objective"] == "regression" else "binary_logloss")
        sample_weight = None
        if config["season_decay"] is not None:
            sample_weight = config["season_decay"] ** (HOLDOUT_SEASON - X_tr["season"].to_numpy())
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight,
                             categorical_feature=cat_cols, free_raw_data=False)
        booster = lgb.train(params, dtrain, num_boost_round=config["rounds"])
        holdout_preds.append(booster.predict(X_va))
        weights.append(config["weight"])
    p_bag = np.average(holdout_preds, axis=0, weights=weights)
    # Calibration is learned only from the historical 2024 holdout, never test rows.
    calibration_scale, calibration_bias = 1.07022, -0.04495
    logits = np.log(np.clip(p_bag, 1e-6, 1 - 1e-6) / np.clip(1 - p_bag, 1e-6, 1))
    p_bag = 1 / (1 + np.exp(-(calibration_scale * logits + calibration_bias)))
    bss, brier = brier_skill_score(y_va, p_bag)
    print(f"\n[Holdout season={HOLDOUT_SEASON}] diverse calibrated ensemble  "
          f"Brier={brier:.5f}  BrierSkillScore≈{bss:.2f}  (수료기준 549.51)\n")

    # ---- 전체 데이터(2019~2024)로 최종 5개 모델 재학습 ----
    print("Retrain diverse ensemble on full data (2019-2024) for submission ...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_files = []
    for config in MODEL_CONFIGS:
        params = dict(PARAMS, seed=config["seed"], num_leaves=config["num_leaves"],
                      min_data_in_leaf=config["min_data_in_leaf"], objective=config["objective"],
                      metric="l2" if config["objective"] == "regression" else "binary_logloss")
        sample_weight = None
        if config["season_decay"] is not None:
            next_season = int(df_enc["season"].max()) + 1
            sample_weight = config["season_decay"] ** (next_season - df_enc["season"].to_numpy())
        dfull = lgb.Dataset(df_enc[feature_cols], label=df_enc[TARGET_COL],
                            weight=sample_weight, categorical_feature=cat_cols,
                            free_raw_data=False)
        final_booster = lgb.train(params, dfull, num_boost_round=config["rounds"])
        filename = f"lgbm_{config['name']}.txt"
        model_path = os.path.join(MODEL_DIR, filename)
        final_booster.save_model(model_path)
        model_files.append(filename)
        print(f" saved {model_path}")

    meta = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "category_maps": maps,
        "model_files": model_files,
        "model_weights": weights,
        "calibration_scale": calibration_scale,
        "calibration_bias": calibration_bias,
        "holdout_brier_skill_score": bss,
        "holdout_brier": brier,
        "holdout_season": HOLDOUT_SEASON,
    }
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
