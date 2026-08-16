# script.py — 평가 서버에서 자동 실행되는 추론 스크립트
#
# ./data/test.csv 를 읽어 ./model/ 의 LightGBM 모델로 control_success(제구 성공) 확률을
# 예측하고 ./output/submission.csv 로 저장한다.
#
# ⚠️ 여기 쓰인 전처리(범주형 인코딩)는 반드시 학습 시점(train.py)과 동일해야 한다.
#    model/meta.json 에 학습 때 만든 카테고리 매핑을 그대로 저장해두고 여기서 재사용한다.

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


def load_model_bundle(model_dir):
    """학습 산출물(모델들 + 전처리 메타) 로드. 시드가 다른 LightGBM 모델 여러 개를
    배깅(평균)하는 앙상블 구조 — meta.json의 seeds 목록에 맞춰 전부 로드한다."""
    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    filenames = meta.get("all_model_files", meta["model_files"])
    boosters = [lgb.Booster(model_file=os.path.join(model_dir, filename))
                for filename in filenames]
    return boosters, meta


# =======================
# 학습 때 사용한 전처리 (train.py 와 반드시 동일해야 함)
# =======================

def apply_category_maps(df, cat_cols, maps):
    """값 -> 정수코드로 치환. 학습 때 없던(미등장) 값은 NaN(결측)으로 처리해
    LightGBM이 학습된 missing-value 분기로 라우팅하도록 한다."""
    df = df.copy()
    for c in cat_cols:
        codes = df[c].astype(str).map(maps[c])
        df[c] = codes.astype("float32")
    return df


def build_features(df, meta):
    """모델 입력 생성 — 학습 때와 동일한 순서/인코딩으로 feature_cols만 추출."""
    df = add_features(df)
    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]
    maps = meta["category_maps"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"test 데이터에 없는 피처 컬럼: {missing}")

    df_enc = apply_category_maps(df, cat_cols, maps)
    return df_enc[feature_cols]


def add_features(df):
    """Training-identical row-local features (no aggregation across test rows)."""
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


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    if len(ids) != len(set(ids)):
        raise ValueError("test.csv row_id가 중복됨")
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        raise ValueError(f"예측이 없는 sample_submission row_id: {n_missing}건")
    if len(pred_map) != len(sub):
        raise ValueError("test.csv와 sample_submission.csv의 row_id 집합이 다름")
    sub[TARGET_COL] = values
    if not np.isfinite(sub[TARGET_COL]).all() or not sub[TARGET_COL].between(0, 1).all():
        raise ValueError("예측값에 NaN/Inf 또는 [0, 1] 범위 밖 값이 있음")
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    OUT_DIR = os.path.join(BASE_DIR, "output")
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    boosters, meta = load_model_bundle(MODEL_DIR)
    print(f" OK. n_features={len(meta['feature_cols'])}  n_models={len(boosters)}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, meta)
    print(f" features={X.shape[1]}")

    # ---- 예측 (제구 성공 확률, 시드별 모델 평균 = 배깅) ----
    print("Inference model...")
    if len(X):
        model_preds = {name: booster.predict(X)
                       for name, booster in zip(meta.get("all_model_files", meta["model_files"]), boosters)}
        if "ensemble_groups" in meta:
            group_preds = []
            for group in meta["ensemble_groups"]:
                raw = np.average([model_preds[name] for name in group["model_files"]], axis=0,
                                 weights=group["model_weights"])
                logits = np.log(np.clip(raw, 1e-6, 1-1e-6) / np.clip(1-raw, 1e-6, 1))
                group_preds.append(1 / (1 + np.exp(-(group["scale"] * logits + group["bias"]))))
            preds = np.average(group_preds, axis=0, weights=meta["group_weights"])
        else:
            raw = np.average(list(model_preds.values()), axis=0, weights=meta["model_weights"])
            logits = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
            preds = 1 / (1 + np.exp(-(meta["calibration_scale"] * logits
                                      + meta["calibration_bias"])))
    else:
        preds = []
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"✅ Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
