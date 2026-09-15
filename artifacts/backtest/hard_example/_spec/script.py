"""평가 서버 실행 스크립트. baseline 규약 준수:
- data/test.csv, data/sample_submission.csv 를 utf-8-sig로 읽음
- sample_submission의 row_id 순서 기준으로 예측 병합
- output/submission.csv 저장

meta.json의 logit_shift(드리프트 보정 절편)와 league_priors(수축용 리그
사전확률)를 예측에 적용한다. 두 상수 모두 train_final.py가 **학습 데이터만으로**
미리 계산해 저장한 값이며, 여기서는 읽어서 행 단위로 적용만 한다 — test 행들의
통계는 일절 계산하지 않는다. 제출물은 독립 실행이므로 src를 임포트하지 않고
numpy/pandas로 인라인 구현한다. src/features.py·src/calibrate.py와의 출력
동등성은 리포지토리의 tests/test_submit_parity.py, tests/test_calibrate.py가 고정한다.

**seed 앙상블 (R3b E트랙)**: meta에 "seeds"가 있으면 `model_0.pkl` ~
`model_{n-1}.pkl`을 전부 읽어 predict_proba를 평균한 뒤 보정을 얹는다
(평균 -> 보정 순서. 검증 실험과 같은 순서여야 로컬 점수와 제출 예측이 갈라지지
않는다). seeds가 없으면 예전처럼 `model.pkl` 하나만 읽는다 — 이미 배포된 단일
모델 산출물이 그대로 동작한다. 여러 모델을 평균해도 **각 모델이 같은 행을 보고
낸 값을 합칠 뿐**이라 행 단위 독립성은 유지된다.

**멤버 블렌드 / meta 스키마 v2 (R4 H5)**: meta에 "members"가 있으면 멤버마다
피처 구성도 보정도 다르다. 멤버 하나는 seed 앙상블(모델 여러 개)일 수 있고,
수축·모드 곱 같은 추가 피처 그룹을 가질 수 있으며, 보정은 스칼라 절편이거나
game_type별 절편이다. 연산 순서는

    멤버별 (모델 predict_proba 평균 -> 그 멤버의 보정) -> 멤버 균등 산술평균

이다. logit이 비선형이라 "평균 후 한 번 보정"과 값이 다르므로, 검증 실험과 같은
순서를 지켜야 로컬 점수와 제출 예측이 갈라지지 않는다. members가 없는 meta는
예전 경로 그대로 동작한다(하위호환). 멤버를 몇 개 평균하든 **각 멤버가 같은 행을
보고 낸 값을 합칠 뿐**이라 행 단위 독립성은 유지된다.

**이종 3종 가중 앙상블 / meta 스키마 v3 (R5 A트랙)**: 두 가지가 늘었다.

  1. `meta["blend"]["weights"]` — 멤버 균등 평균 대신 **가중평균**. 없으면 균등
     (v2 동작 그대로).
  2. `meta["blend"]["calibration"]` — 가중평균 **뒤에 한 번** 얹는 블렌드 보정.
     v2는 멤버마다 따로 보정했지만 v3의 세 멤버(HGB/LightGBM/CatBoost)는 같은
     피처셋·같은 타깃을 보는 동종 구성이라 절편을 나눌 근거가 없고, 보정 지점이
     하나면 검증(exp17)과 제출의 재현 대상도 하나다. 멤버 보정이 함께 있으면
     순서는 멤버 보정 -> 가중평균 -> 블렌드 보정이다.

**CatBoost는 pickle이 아니라 native `.cbm`으로 실린다.** 리포지토리의 래퍼
(`src.models.CatBoostCategoricalClassifier`)를 joblib로 굳히면 pickle에 그 모듈
경로가 박히는데, 이 스크립트는 의도적으로 `src`를 임포트하지 않으므로 평가
서버에서 로드가 죽는다. 대신 `.cbm`을 직접 읽고 래퍼가 하던 전처리(범주 결측 ->
고정 토큰, 열 순서 검사)를 아래 `CatBoostNativePredictor`가 인라인으로 재현한다.
두 구현의 동등성은 tests/test_submit_ens3.py가 실제 학습된 모델로 고정한다.
`load_models`는 **파일 확장자**로 로더를 고른다 — zip 안 model/은 평평하므로
이름이 곧 유일한 단서다.

**선수 임베딩 NN 멤버 (R5-D 최종 조합).** 마지막 멤버로 신경망이 하나 붙는다.
CatBoost와 같은 이유로 여기도 pickle을 쓸 수 없고(모듈 경로), 게다가 torch는
**버전 방향**이 불리하다 — 로컬 torch가 평가 서버(2.7.1)보다 새 버전이라 torch
파일의 전방 호환이 보장되지 않는다. 그래서 셋을 전부 이식했다.

  * `nn_prep_transform` — `src.nn_embed.TabularPrep.transform`의 인라인 재현.
    중앙값/평균/표준편차/범주 레벨은 전부 meta에 박힌 **학습 데이터 유래 상수**다.
  * `nn_vocab_transform` — `IdVocab.transform`의 인라인 재현. 학습에 없던 id와
    결측은 UNK(0번)로 떨어진다 — 2025 신인이 그 경로를 탄다.
  * `EmbedNNPredictor` — 아키텍처를 이 파일에서 **다시 정의**하고 가중치는
    numpy `.npz`(state_dict의 배열들)만 읽어 `load_state_dict(strict=True)`로
    넣는다. torch 직렬화를 쓰지 않으므로 버전 차이의 영향을 받지 않고, 이름·모양이
    어긋나면 조용히 틀리는 대신 로드에서 즉시 터진다.

NN도 **행 단위 독립**이다: BatchNorm이 없고(정규화는 fit 시점 상수로 하는 표준화
뿐이다), eval 모드라 드롭아웃이 꺼지며, forward는 행마다 독립적인 아핀+ReLU
연산이다. 추론 배치 크기를 바꿔도 값이 변하지 않는 것을 tests/test_submit_nn.py가
고정한다. 세 이식 모두 `src` 원본과 atol=0으로 같은 값을 내는지도 같은 파일이
실제 학습된 모델로 검사한다.

**룩업 피처 / meta 스키마 v4 (R6 C1).** 멤버 넷 중 셋이 **테이블에서 읽는 피처**를
쓴다. 지금까지 meta에 실린 상수는 스칼라 몇 개(수축 M, 시즌 사전확률, 보정 절편)
뿐이었는데 여기서는 표가 통째로 실린다.

  platoon : (pitcher_id, batter_hand, season) -> rate / diff / n   (src.platoon)
  tm      : (pitcher_id, season)              -> 물리량 11개        (src.tm_match)

둘 다 **학습 데이터(≤2024)에서만** 만들어진 as-of 집계다. 각 행의 값은 그 행의
season보다 과거 시즌 재료로만 계산돼 있고(테이블을 만들 때 이미 한 칸 밀렸다),
이 스크립트는 **읽기만** 한다 — 평가 데이터가 재료로 들어갈 경로가 없다. 표는
`meta["lookups"]`에 target season 한 장씩만 실린다(2025 = ≤2024 전체 누적).

행 단위 독립성도 그대로다: 각 행이 **자기 키 값만으로** 표의 한 칸을 찾는다.
표에 없는 키(2025 신인, 미매칭 투수, 결측 키)는 미매칭으로 떨어져 NaN이 되고,
`fill_zero`로 지정된 표본수 컬럼만 0이 된다 — "표본이 없다"는 결측이 아니라
관측된 사실이라서다. tests/test_submit_lookup.py가 src와의 값 동등성과 순열·
부분집합 불변성을 함께 고정한다.

**팀원 LightGBM 스택 멤버 / meta 스키마 v5 (R7 합집합).** 팀 병합 상대의 구성이
멤버 하나로 들어온다. 두 가지가 새롭다.

  1. `groups`에 `"team"`이 있으면 그 멤버의 피처는 우리 파생이 아니라 **저쪽
     전처리**로 만든다 — 행 단위 파생 10개(`team_add_features`) + 값->정수코드
     범주 인코딩(`team_encode_categories`). 매핑표는 `member["team"]
     ["category_maps"]`에 실린 **학습 데이터 유래 상수**이고, 학습에 없던 값은
     NaN(결측)으로 떨어진다. 이 멤버는 우리가 버린 raw pitcher_id/batter_id를
     수치형으로 쓴다 — 그 선택의 차이가 앙상블 다양성의 원천이다.
  2. 부스터 4개가 파일 **하나**(`*.lgbmstack.json`)에 담겨 `TeamLGBMStackPredictor`
     한 개로 로드된다. 저쪽 레시피는 군마다 가중이 다른 3종 평균 + 군별 Platt +
     60:40이라 우리 `model_files` 균등평균 규약으로 표현되지 않기 때문이다.
     바깥에서 보면 `predict_proba`를 가진 보통 멤버라 `predict_blend`는 그대로다.

행 단위 독립성은 그대로다: 파생도 인코딩도 부스터 예측도 전부 자기 행만 본다.
src와의 동등성과 순열·부분집합 불변성은 tests/test_submit_team.py가 고정한다.

**S1 로지스틱 endpoint / meta 스키마 v7 (R8 측정).** 트리·NN과 귀납 편향이 다른
선형 모델 하나가 스테이지로 붙는다. 새로운 것은 둘이다.

  1. `engine: "s1_logistic"` 멤버 — 모델은 평범한 sklearn 로지스틱 `.pkl`이지만
     전처리 상수(rate 결측의 prior 대치, 중앙값 대치, 표준화 평균/표준편차,
     범주 5종의 **고정 레벨 원핫**)가 pickle에 들어 있지 않아 meta의 `s1` 블록에
     실린다. `S1LogisticPredictor`가 그걸 읽어 재현한다. 전부 학습 데이터
     (≤2024) 유래 상수다.
  2. **3번째 스테이지** — `stages`가 셋이 되고 `stage_weights`가
     `[0.9*0.4, 0.9*0.6, 0.1]`이다. 이는 `0.9 * (현 배포 2단) + 0.1 * S1`과
     산술적으로 같고, S1 가중을 0으로 두면 현 배포 예측이 완전히 복원된다
     (endpoint-preserving). `_combine_stages`는 손대지 않았다 — 스테이지 수는
     원래부터 가변이다.

행 단위 독립성은 그대로다: 표준화도 원핫도 로지스틱 내적도 전부 자기 행만 본다.
src와의 동등성·순열·부분집합 불변성은 tests/test_submit_s1.py가 고정한다.

**팀원 5차 스택 재통합 (R10, `submit/model_team2`).** 이 코드에 **새 연산은 없다** —
스키마 v5(팀 스택)와 v6(계층 결합)의 조합만으로 표현된다. 팀 스테이지가 멤버 둘을
갖게 된 것이 전부다.

    P_team_new = 0.9 * TeamLGBMStackPredictor + 0.1 * EmbedNNPredictor(팀원 가중치)

  * 군 가중이 `[0.60, 0.40]`에서 `[0.6090564, 0.3909436]`으로 바뀌었다. 이 값은
    멤버 사양(`team.group_weights`)에서 읽으므로 코드 변경이 없다. 부스터 4개와
    Platt 상수는 R7과 **바이트 단위로 같다**.
  * 팀원 MLP는 우리 `nn_embed`와 **같은 구조**를 저쪽 데이터로 학습한 별개
    가중치다. 전처리 상수와 범주 레벨 **순서**가 우리 것과 다르므로 저쪽
    `nn_meta.json`을 그대로 멤버 사양에 싣는다 — `nn_prep_transform`이 레벨을
    dtype이 아니라 **값**으로 매핑하기 때문에 순서가 달라도 같은 열에 떨어진다.
    zip 안 model/은 평평하므로 파일 이름은 `team_nn_model.npz`로 분리한다.

행 단위 독립성: 두 멤버 다 자기 행만 본다. 다만 float32 GEMM의 잔여 블록
마이크로커널 때문에 **행 순서를 바꾸면 극소수 행의 마지막 자리(1 ulp = 6e-08)가
흔들린다** — 배치 크기를 고정해도 남는 성질이고(고정되는 것은 행렬 모양이지 한
행이 블록 안에서 갖는 위치가 아니다), 우리 기존 NN 멤버(제출 #9~#12)에도 같은
크기로 존재한다. 정보가 새는 것이 아니라 반올림이다.
tests/test_submit_team2.py가 그 상한(1 ulp)과 "트리 멤버는 atol=0"을 함께 고정한다.

**당해 시즌 폼 / meta 스키마 v9 (R11-S2D).** `meta["season_form"]`이 있으면 결합이
전부 끝난 **마지막**에 행 단위 보정을 한 번 얹는다.

    p' = clip(p + alpha * (form - mu), 0, 1)      # 적용 대상 행에만

`asof_pitcher_n`과 `asof_pitcher_success_rate`는 시즌을 관통하는 커리어 누적
카운터다. 그래서 각 행의 누적값에서 **학습 종료 시점의 누적값**을 빼면 그 투수의
당해 시즌 투구수·성공수가 복원된다 — 다른 평가 행을 전혀 보지 않고서.

  * meta에 실리는 것은 `(pitcher_ids, n0, s0, pooled_p0, mu, alpha, m, segment)`로
    **전부 학습 데이터(≤2024) 유래 상수**다. 이 스크립트는 읽기만 한다.
  * `s0`에 학습구간 마지막 행의 `control_success`가 한 번 더해져 있다. 이는
    **학습 데이터 내부 타깃의 누적 카운터를 완결시키는 off-by-one 보정이며 평가
    데이터의 타깃이 아니다** (`asof_*`는 그 투구 직전까지의 누적이므로).
  * 적용 대상은 `game_type == segment`이고 학습 테이블에 있는 투수이고 당해 시즌
    표본이 있는 행뿐이다. 나머지는 보정값이 정확히 0이라 base 예측이 그대로 나온다.

행 독립성은 설계상 보장된다: 각 행이 자기 `pitcher_id`·`asof` 두 값으로 고정
테이블의 한 칸을 찾을 뿐이고, 평가 데이터의 평균·정렬·빈도·rolling이 들어갈 경로가
없다. src와의 동등성과 순열·부분집합 불변성은 tests/test_submit_r11.py가 고정한다.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from season_state_runtime import add_state
from corrections_runtime import apply_corrections
from batter_form_runtime import apply_batter_form
from coldstart_runtime import apply_coldstart_expert

ID_COL = "row_id"
TARGET_COL = "control_success"
EPS = 1e-6

# src/features.py의 고정 레벨과 동일해야 한다 (동등성 테스트가 고정)
COUNT_LEVELS = list(range(12))
HAND_COMBO_LEVELS = ["1-1", "1-2", "2-1", "2-2"]
SMOOTH_MAP = {
    "asof_pitcher_success_rate": "asof_pitcher_n",
    "asof_pitcher_reverse_rate": "asof_pitcher_n",
    "asof_pitcher_middle_rate": "asof_pitcher_n",
    "asof_pitcher_ball_rate": "asof_pitcher_n",
    "asof_pitcher_strike_rate": "asof_pitcher_n",
    "asof_batter_success_rate": "asof_batter_n",
    "asof_batter_middle_rate": "asof_batter_n",
    "asof_pitcher_fastball_rate": "asof_pitcher_pitchmix_n",
    "asof_pitcher_breaking_rate": "asof_pitcher_pitchmix_n",
    "asof_pitcher_offspeed_rate": "asof_pitcher_pitchmix_n",
}
LOW_N_THRESHOLD = 500

# EB v2 수축(src.features의 smooth 그룹)에서 M을 공유하는 컬럼군.
# 같은 분모(표본수 컬럼)를 쓰는 것끼리 묶여 있다 — 분모가 다르면 M의 단위가 달라
# 하나의 상수로 묶을 근거가 없다.
SMOOTH_GROUPS = {
    "pitcher_success": ("asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
                        "asof_pitcher_strike_rate"),
    "batter": ("asof_batter_success_rate", "asof_batter_middle_rate"),
    "pitchmix": ("asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                 "asof_pitcher_offspeed_rate"),
}
SMOOTH_GROUP_OF = {c: g for g, cols in SMOOTH_GROUPS.items() for c in cols}

# modes 그룹: (파생 이름, 재료 rate 컬럼, 어느 국면 플래그와 곱하는가)
MODE_SPEC = (
    ("middle_x_3ball", "asof_pitcher_middle_rate", "is_3ball"),
    ("ball_x_pcount", "asof_pitcher_ball_rate", "is_pitcher_count"),
    ("breaking_x_pcount", "asof_pitcher_breaking_rate", "is_pitcher_count"),
    ("strike_x_3ball", "asof_pitcher_strike_rate", "is_3ball"),
)


def _f(df, col):
    """결측을 NaN으로 보존한 float64 배열."""
    return df[col].to_numpy(dtype="float64", na_value=np.nan)


def _shrink(rate, n, k, prior):
    """(n*rate + k*prior) / (n + k). rate 결측이면 관측 0건 취급 -> 사전값.

    `prior`는 스칼라(legacy 균일 k)여도 되고 **행마다 다른 배열**(EB v2의
    시즌별 μ)이어도 된다. 어느 쪽이든 그 행의 값만으로 계산된다.
    """
    valid = np.isfinite(rate) & np.isfinite(n)
    num = np.where(valid, n * rate, 0.0) + k * prior
    den = np.where(valid, n, 0.0) + k
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num / den).astype("float32")


def season_prior_values(coefs, seasons):
    """직선 계수 (slope, intercept)를 각 행의 season에서 읽어 μ를 만든다.

    [0, 1] 클립. 행 단위 변환이라 어떤 행의 μ도 같은 배치 구성에 의존하지 않는다.
    src.features.season_prior_values와 같은 값을 내야 한다.
    """
    slope, intercept = coefs
    s = np.asarray(seasons, dtype="float64")
    return np.clip(float(slope) * s + float(intercept), 0.0, 1.0)


def _key_codes(frame, col):
    """룩업 키 컬럼 -> int64 배열. 결측/비유한값은 어떤 실제 키와도 안 맞는 -1로.

    학습 경로(src)는 키 컬럼이 항상 정수인 train만 보므로 그대로 int64 캐스팅을
    하지만, 평가 데이터에 결측이 섞이면 그 캐스팅은 예외로 터진다. 여기서는 그
    행을 **미매칭**(= 표본 없음)으로 떨어뜨린다 — 값이 있는 행의 결과는 학습
    경로와 완전히 같고, 값이 없는 행만 cold-start 경로를 탄다.
    """
    v = frame[col].to_numpy(dtype="float64", na_value=np.nan)
    return np.where(np.isfinite(v), v, -1.0).astype("int64")


def apply_lookup(X, frame, spec, name):
    """as-of 룩업 테이블에서 행마다 한 칸씩 읽어 피처를 붙인다.

    `spec`은 meta에 실린 표 한 장이다:
      key_cols  — 키 컬럼 이름 순서 (예: pitcher_id, batter_hand, season)
      keys      — 키 컬럼별 정수 배열 (표의 행)
      columns   — 피처 컬럼 이름
      values    — 피처 컬럼별 값 배열 (결측은 null -> NaN)
      fill_zero — 미매칭일 때 NaN이 아니라 0으로 채울 컬럼 (표본수)

    `src.platoon.lookup` / `src.features`의 tm 블록과 같은 값을 내야 한다
    (tests/test_submit_lookup.py가 고정). 각 행은 자기 키만 보고 위치를 찾으므로
    배치에 무엇이 함께 들어오는지에 의존하지 않는다.
    """
    for key in ("key_cols", "keys", "columns", "values"):
        if key not in spec:
            raise ValueError(f"룩업 사양 {name!r}에 {key!r}가 없다")
    missing = [c for c in spec["key_cols"] if c not in frame.columns]
    if missing:
        raise ValueError(f"룩업 {name!r}에 필요한 키 컬럼이 없다: {missing}")

    table_idx = pd.MultiIndex.from_arrays(
        [np.asarray(spec["keys"][c], dtype="int64") for c in spec["key_cols"]])
    row_idx = pd.MultiIndex.from_arrays(
        [_key_codes(frame, c) for c in spec["key_cols"]])
    # get_indexer는 유일 인덱스에서 행별 위치를 돌려준다(없으면 -1).
    pos = table_idx.get_indexer(row_idx)
    hit = pos >= 0
    fill_zero = set(spec.get("fill_zero", []))
    for c in spec["columns"]:
        vals = np.asarray(spec["values"][c], dtype="float64")
        out = np.full(len(row_idx), np.nan, dtype="float64")
        out[hit] = vals[pos[hit]]
        if c in fill_zero:
            out = np.nan_to_num(out, nan=0.0)
        X[c] = out.astype("float32")


TEAM_LOG1P_COLS = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
TEAM_BUNDLE_SUFFIX = ".lgbmstack.json"


def team_add_features(X):
    """팀원 스택의 행 단위 파생 10개 (`src.team_lgbm.add_features` 이식본).

    이름이 우리 `matchup`/`count` 블록과 겹치지만(`same_hand`, `count_state`)
    값의 dtype과 정의가 다르다 — 팀 멤버는 groups=["team"]만 쓰므로 두 블록이
    한 프레임에서 만날 일이 없다. `tests/test_submit_team.py`가 src와의 동등성을
    고정한다.
    """
    X["count_state"] = (X["balls_before"] * 3 + X["strikes_before"]).astype("int8")
    X["same_hand"] = (X["pitcher_hand"] == X["batter_hand"]).astype("int8")
    X["late_inning"] = (X["inning"] >= 7).astype("int8")
    X["two_strikes"] = (X["strikes_before"] == 2).astype("int8")
    X["three_balls"] = (X["balls_before"] == 3).astype("int8")
    X["pitcher_rate_gap_1_5"] = (X["asof_pitcher_prev1_game_success_rate"]
                                 - X["asof_pitcher_prev5_game_success_rate"])
    X["pitcher_rate_gap_career_5"] = (X["asof_pitcher_prev5_game_success_rate"]
                                      - X["asof_pitcher_success_rate"])
    for c in TEAM_LOG1P_COLS:
        X[f"log1p_{c}"] = np.log1p(X[c].clip(lower=0))
    return X


def team_encode_categories(X, spec):
    """값(문자열화) -> 정수코드. 학습에 없던 값은 NaN = LightGBM의 결측 분기.

    2025 신규 팀·신규 범주가 이 경로로 안전하게 떨어진다. 각 행이 **자기 값만**
    보고 코드를 얻으므로 행 단위 독립성이 유지된다.
    """
    maps = spec["category_maps"]
    for c in spec["cat_cols"]:
        X[c] = X[c].astype(str).map(maps[c]).astype("float32")
    return X


def _require_lookup(lookups, name):
    if not lookups or name not in lookups:
        raise ValueError(
            f"groups에 {name!r}가 있으면 meta['lookups'][{name!r}] 테이블이 "
            f"필요하다 — 학습 데이터에서 만든 as-of 룩업이 실려야 한다")
    return lookups[name]


def build_submit_features(test, meta, lookups=None):
    """meta(또는 멤버 사양) 기준으로 학습 때와 **같은** 파생 피처를 만든다.

    전부 행 단위 변환이다. 평가 데이터의 행 간 통계(평균/랭킹/rolling 등)를
    만드는 코드는 여기에 존재하지 않는다 — 수축에 쓰는 사전확률조차
    meta.json에 박힌 학습 데이터 유래 상수를 읽어 쓸 뿐이다.

    블렌드에서는 멤버마다 groups/feature_columns/수축 상수가 다르므로, 이 함수는
    meta 전체가 아니라 **그 멤버의 사양 dict**를 받아도 똑같이 동작한다
    (키 이름이 단일 구성 meta와 같다).
    """
    groups = meta.get("groups", [])
    X = test.copy()

    if "count" in groups:
        code = (X["balls_before"].to_numpy(dtype="int16") * 3
                + X["strikes_before"].to_numpy(dtype="int16"))
        X["count_state"] = pd.Categorical(code, categories=COUNT_LEVELS)

    if "matchup" in groups:
        ph = X["pitcher_hand"].to_numpy()
        bh = X["batter_hand"].to_numpy()
        X["same_hand"] = (ph == bh).astype("int8")
        combo = np.char.add(np.char.add(ph.astype(str), "-"), bh.astype(str))
        X["hand_combo"] = pd.Categorical(combo, categories=HAND_COMBO_LEVELS)

    if "rates" in groups:
        X["strike_minus_ball"] = (_f(X, "asof_pitcher_strike_rate")
                                  - _f(X, "asof_pitcher_ball_rate")).astype("float32")
        X["success_minus_middle"] = (_f(X, "asof_pitcher_success_rate")
                                     - _f(X, "asof_pitcher_middle_rate")).astype("float32")
        X["prev_vs_career"] = (_f(X, "asof_pitcher_prev5_game_success_rate")
                               - _f(X, "asof_pitcher_success_rate")).astype("float32")
        second = np.fmax(_f(X, "asof_pitcher_breaking_rate"),
                         _f(X, "asof_pitcher_offspeed_rate"))
        X["fastball_dominance"] = (_f(X, "asof_pitcher_fastball_rate")
                                   - second).astype("float32")

    if "modes" in groups:
        balls = X["balls_before"].to_numpy(dtype="int16")
        strikes = X["strikes_before"].to_numpy(dtype="int16")
        flags = {
            "is_3ball": (balls == 3).astype("int8"),
            "is_pitcher_count": ((strikes == 2) & (balls <= 1)).astype("int8"),
        }
        for name, arr in flags.items():
            X[name] = arr
        for name, rate_col, flag in MODE_SPEC:
            # rate가 결측이면 곱도 결측(0*NaN=NaN) — src.features와 같은 처리다.
            X[name] = (_f(X, rate_col) * flags[flag]).astype("float32")

    if "smooth" in groups:
        smooth_M = meta.get("smooth_M")
        if smooth_M is not None:
            # EB v2 (R3b C트랙): 컬럼군별 M + 시즌별 사전확률 μ(season).
            # 두 상수 모두 meta.json에 박힌 **학습 데이터 유래** 값이고, 각 행은
            # 자기 season 값만으로 μ를 읽으므로 여전히 행 단위다.
            priors = meta.get("season_priors")
            if priors is None:
                raise ValueError("smooth_M이 있으면 season_priors도 필요하다")
            seasons = X["season"].to_numpy()
            for rate_col, n_col in SMOOTH_MAP.items():
                mu = season_prior_values(priors[rate_col], seasons)
                X[f"{rate_col}_sm"] = _shrink(
                    _f(X, rate_col), _f(X, n_col),
                    float(smooth_M[SMOOTH_GROUP_OF[rate_col]]), mu)
        elif meta.get("smooth_k") is not None:
            # legacy 균일 k (배포된 submit/model/ 전용) — 사전확률도 pooled 상수.
            k = float(meta["smooth_k"])
            league = meta["league_priors"]
            for rate_col, n_col in SMOOTH_MAP.items():
                X[f"{rate_col}_sm"] = _shrink(_f(X, rate_col), _f(X, n_col), k,
                                              float(league[rate_col]))
        else:
            raise ValueError(
                "groups에 'smooth'가 있으면 smooth_M+season_priors(EB v2) 또는 "
                "smooth_k+league_priors(legacy) 중 하나가 meta에 있어야 한다")
        X["pitcher_low_n"] = (X["asof_pitcher_n"] < LOW_N_THRESHOLD).astype("int8")
        X["batter_low_n"] = (X["asof_batter_n"] < LOW_N_THRESHOLD).astype("int8")

    if "tm" in groups:
        # trackman 물리 집계 as-of 룩업 (src.features의 "tm" 블록과 같은 값).
        apply_lookup(X, X, _require_lookup(lookups, "tm"), "tm")

    if "platoon" in groups:
        # 투수 플래툰 as-of 룩업 (src.features의 "platoon" 블록과 같은 값).
        # 표본수 platoon_n만 미매칭 시 0이다 — spec의 fill_zero가 그것을 정한다.
        apply_lookup(X, X, _require_lookup(lookups, "platoon"), "platoon")

    if "team" in groups:
        # 팀원 LightGBM 스택의 전처리 (R7). 다른 그룹과 섞이지 않는다 —
        # 이 멤버의 groups는 ["team"] 하나뿐이다.
        spec = meta.get("team")
        if not spec:
            raise ValueError("groups에 'team'이 있으면 멤버 사양에 'team'"
                             "(cat_cols/category_maps)이 있어야 한다")
        X = team_encode_categories(team_add_features(X), spec)

    return X[meta["feature_columns"]]


# =========================================================================
# R11-S2D — 당해 시즌 폼 (meta 스키마 v9). `src.season_form`의 인라인 이식본.
# =========================================================================

def season_form_values(frame, spec):
    """보정에 실리는 값 `(form - mu)`를 **적용 대상 행에만**. 나머지는 정확히 0.

    `src.season_form.compute_form_adj`의 이식본이다. 산식은
    `docs/research/2026-08-17-season-to-date-form-audit.md` §2.

        N_season = asof_pitcher_n          - N0[pitcher]
        S_season = round(n * success_rate) - S0[pitcher]
        form     = (S_season + M*p0) / (N_season + M) - p0

    **각 행은 자기 asof 두 값 + train 유래 상수 테이블만 본다.**
    `(N0, S0, pooled_p0, mu)`는 학습 데이터(≤2024)만으로 굳어 meta에 실린 상수이고
    여기서는 읽기만 한다. 다른 평가 행·전체 평균·정렬·빈도를 일절 보지 않으므로
    "1행만 넣은 예측 == 전체를 넣은 예측"이 성립한다(DACON 공지 판정 기준).

    적용 대상 = `game_type == segment` **그리고** 학습 테이블에 있는 투수
    **그리고** 당해 시즌 표본이 있는 행(`N_season > 0`). 나머지 행(F행·신규 투수·
    `N_season=0`)은 `0.0`이라 base 예측이 그대로 나온다 — `-mu`조차 새지 않는다.

    `n * rate`는 **float64**로 곱한다 — CSV가 rate를 소수 6자리로 저장해서 생기는
    절단 오차의 안전 마진(0.5)을 float32로 줄이면 안 된다.
    """
    ids = np.asarray(spec["pitcher_ids"], dtype="int64")
    n0_tab = np.asarray(spec["n0"], dtype="float64")
    s0_tab = np.asarray(spec["s0"], dtype="float64")
    pooled = float(spec["pooled_p0"])
    m = float(spec.get("m", 75.0))
    if m <= 0:
        raise ValueError(f"season_form.m은 양수여야 한다: {m}")
    if len(ids) != len(n0_tab) or len(ids) != len(s0_tab):
        raise ValueError("season_form 테이블의 길이가 어긋난다")

    pid = pd.Series(frame["pitcher_id"]).to_numpy(dtype="float64")
    na_pid = ~np.isfinite(pid)
    key = np.where(na_pid, -1, pid).astype("int64")
    pos = np.searchsorted(ids, key)
    pos_c = np.clip(pos, 0, max(len(ids) - 1, 0))
    seen = (len(ids) > 0) & (pos < len(ids)) & (ids[pos_c] == key) & ~na_pid

    n_row = np.asarray(frame["asof_pitcher_n"], dtype="float64")
    rate = pd.Series(frame["asof_pitcher_success_rate"]).to_numpy(
        dtype="float64")
    s_row = np.rint(n_row * np.where(np.isfinite(rate), rate, 0.0))

    if len(ids):
        n0 = np.where(seen, n0_tab[pos_c], 0.0)
        s0 = np.where(seen, s0_tab[pos_c], 0.0)
    else:
        n0 = np.zeros(len(pid))
        s0 = np.zeros(len(pid))
    p0 = np.where(n0 > 0, np.divide(s0, np.where(n0 > 0, n0, 1.0)), pooled)

    n_season = n_row - n0
    form = (s_row - s0 + m * p0) / (n_season + m) - p0

    # 적용 대상 마스크. **값(form!=0)으로 유추하지 않는다** — form이 정확히 0이 되는
    # 정상 행이 있을 수 있어(N_season=1이고 p0가 0/1) 그런 행이 조용히 빠진다.
    active = np.isfinite(n_season) & (n_season > 0) & np.isfinite(form) & seen
    segment = spec.get("segment")
    if segment is not None:
        labels = pd.Series(frame[spec.get("group_col", "game_type")]).to_numpy(
            dtype=object)
        active = active & (labels == segment)
    return np.where(active, form - float(spec.get("mu", 0.0)), 0.0)


def apply_season_form(p, frame, spec, verbose=True):
    """`p' = clip(p + alpha*(form - mu), 0, 1)`. 결합의 **마지막** 단계.

    spec이 없으면 아무것도 하지 않는다(기존 배포물 하위호환). 이 뒤에 붙는 연산은
    없다 — 생기는 순간 사전 등록된 구성이 아니게 된다.
    """
    if not spec:
        return np.asarray(p, dtype=float)
    alpha = float(spec["alpha"])
    adj = season_form_values(frame, spec)
    out = np.clip(np.asarray(p, dtype=float) + alpha * adj, 0.0, 1.0)
    if verbose:
        nz = int((adj != 0).sum())
        print(f"season form: alpha={alpha} m={spec.get('m')} "
              f"mu={spec.get('mu', 0.0)} segment={spec.get('segment')} "
              f"active {nz}/{len(adj)} mean(adj) {adj.mean():+.6f} "
              f"-> mean pred {out.mean():.4f}")
    return out


def apply_logit_shift(p, b, slope=1.0):
    """p' = sigmoid(a·logit(p) + b). 행 단위 변환 — 행 간 통계 없음.

    logit 전에 클립하므로 p=0/1이 와도 발산하지 않고, sigmoid 출력은 (0,1)이라
    확률 범위가 보장된다. slope(a)는 기본 1.0 = 절편만 보정(기존 동작 그대로).
    b는 스칼라여도 되고 **행마다 다른 배열**이어도 된다(그룹별 절편이 그 경우).
    """
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    z = float(slope) * np.log(p / (1.0 - p)) + np.asarray(b, dtype=float)
    return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)


def apply_group_logit_shift(p, labels, b_by_group, default_b=0.0, slope=1.0):
    """행의 label(game_type)에 해당하는 절편을 골라 적용한다.

    src.calibrate.apply_shift_by_group의 이식본이다. label이 b_by_group에 없으면
    (미지 값·결측) default_b로 떨어진다 — 2025에 새로운 game_type이 오더라도
    pooled 보정으로 안전하게 착지한다. 각 행은 자기 label만 보고 변환되므로
    행 간 통계가 개입할 여지가 없다.
    """
    p = np.asarray(p, dtype=float)
    lab = np.asarray(labels, dtype=object)
    if len(lab) != len(p):
        raise ValueError(f"p/labels 길이 불일치: {len(p)} vs {len(lab)}")
    b = np.full(len(p), float(default_b))
    for key, val in b_by_group.items():
        b[lab == key] = float(val)
    return apply_logit_shift(p, b, slope)


def apply_calibration(p, calib, frame):
    """멤버(또는 단일 구성)의 보정 사양을 확률에 적용한다.

    사양은 둘 중 하나다:
      {"b": 스칼라}                                      — pooled 절편 (c0)
      {"b_by_group": {"F":.., "R":..}, "default_b":..}   — game_type별 절편 (c1/c2)
    둘 다 없거나 사양 자체가 없으면 원 확률을 그대로 돌려준다(보정 없는 멤버).
    """
    if not calib:
        return np.asarray(p, dtype=float)
    slope = float(calib.get("slope", 1.0))
    if "b_by_group" in calib:
        col = calib.get("group_col", "game_type")
        return apply_group_logit_shift(p, frame[col].to_numpy(dtype=object),
                                       calib["b_by_group"],
                                       float(calib.get("default_b", 0.0)), slope)
    return apply_logit_shift(p, float(calib.get("b", 0.0)), slope)


class CatBoostNativePredictor:
    """`.cbm`으로 저장된 CatBoost + 학습 때와 같은 전처리.

    `src.models.CatBoostCategoricalClassifier`의 예측 경로를 그대로 재현한다.
    두 가지가 재현 대상이다.
      1. **범주 결측 -> 고정 토큰.** CatBoost는 범주 컬럼의 NaN을 받지 않는다.
         우리 파이프라인은 미지 레벨을 NaN으로 떨어뜨리는 설계라(cat_levels)
         2025의 신규 값이 그대로 NaN이 되는데, 버리면 그 행이 통째로 막힌다.
         학습 때와 같은 토큰으로 바꿔 "미지"라는 범주로 예측한다.
      2. **열 순서 검사.** cat_features는 이름이 아니라 **위치**로 지정되므로,
         순서가 어긋나면 에러 없이 조용히 틀린 예측이 나온다.

    변환은 전부 행 단위다 — 각 행의 값만 보고 토큰을 정하므로 배치 구성이
    바뀌어도 그 행의 예측은 변하지 않는다.
    """

    def __init__(self, model, feature_columns, cat_features, na_token="__NA__"):
        self.model = model
        self.feature_columns = list(feature_columns)
        self.cat_features = list(cat_features)
        self.na_token = str(na_token)

    @classmethod
    def load(cls, path, spec):
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(path))
        try:
            cols = spec["feature_columns"]
            cats = spec["cat_features"]
        except KeyError as e:
            raise ValueError(
                f"CatBoost 멤버 사양에 {e} 가 없다 — .cbm만으로는 학습 때의 열 "
                f"순서와 범주 컬럼을 알 수 없어 전처리를 재현할 수 없다") from e
        return cls(model, cols, cats, spec.get("na_token", "__NA__"))

    def _prep(self, X):
        if list(X.columns) != self.feature_columns:
            raise ValueError(
                "학습 때와 열 구성/순서가 다르다 — CatBoost의 cat_features는 "
                f"위치로 지정되므로 그대로 쓰면 조용히 틀린다.\n"
                f"  학습: {self.feature_columns}\n  지금: {list(X.columns)}")
        out = X.copy()
        for c in self.cat_features:
            s = out[c].astype("object")
            out[c] = s.where(s.notna(), self.na_token).astype(str)
        return out

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def nn_prep_transform(X, prep):
    """수치/범주 -> 신경망 입력 행렬. `src.nn_embed.TabularPrep.transform` 이식본.

    컬럼 분류(num/cat/결측 플래그 대상)와 상수(중앙값·평균·표준편차·레벨)는 전부
    meta에서 온다 — 여기서 dtype을 다시 추론하지 않는다. 평가 데이터의 dtype이
    조금 달라져도(1행 CSV에서 전부 결측인 컬럼 등) 같은 열이 같은 자리에 떨어져야
    하기 때문이다. 각 값은 자기 행의 값과 학습 상수만으로 정해진다.
    """
    num_cols = prep["num_cols"]
    cat_cols = prep["cat_cols"]
    na_cols = prep["na_cols"]
    missing = [c for c in num_cols + cat_cols if c not in X.columns]
    if missing:
        raise KeyError(f"학습 때 있던 컬럼이 없다: {missing}")

    n = len(X)
    n_feat = len(prep["feature_names"])
    out = np.empty((n, n_feat), dtype="float32")
    j = 0
    for c in num_cols:
        v = X[c].to_numpy(dtype="float64", na_value=np.nan)
        finite = np.isfinite(v)
        v = np.where(finite, v, float(prep["median"][c]))
        out[:, j] = (v - float(prep["mean"][c])) / float(prep["std"][c])
        j += 1
    for c in na_cols:
        v = X[c].to_numpy(dtype="float64", na_value=np.nan)
        out[:, j] = (~np.isfinite(v)).astype("float32")
        j += 1
    for c in cat_cols:
        levels = prep["levels"][c]
        # 값으로 다시 매핑한다 — X의 dtype categories가 학습 때와 달라도
        # (순서가 섞이거나 일부만 관측돼도) 같은 열에 떨어진다. 미지 값/결측은
        # code -1이 되어 그 행의 블록이 전부 0 = "레벨 없음"으로 표현된다.
        codes = pd.Categorical(X[c], categories=levels).codes
        block = np.zeros((n, len(levels)), dtype="float32")
        hit = codes >= 0
        block[np.nonzero(hit)[0], codes[hit]] = 1.0
        out[:, j:j + len(levels)] = block
        j += len(levels)
    if j != n_feat:
        raise ValueError(f"피처 폭이 meta와 다르다: {j} vs {n_feat} — 전처리 "
                         f"사양이 학습 때와 어긋났다")
    return out


def nn_vocab_transform(values, ids):
    """선수 id -> 임베딩 행 번호. `src.nn_embed.IdVocab.transform` 이식본.

    0번은 UNK 고정, 실제 선수는 1번부터다. 학습에 없던 id와 결측은 **조용히 UNK로**
    떨어진다 — 예외를 내지 않는 것이 의도다. 2025에 신인이 섞이는 것은 정상 상황이고
    그 행도 수치 피처만으로 예측이 나와야 한다.
    """
    arr = pd.Series(values).to_numpy(dtype="float64")
    ids = np.asarray(ids, dtype="int64")
    if len(ids) == 0:
        return np.zeros(len(arr), dtype="int64")
    na = ~np.isfinite(arr)
    key = np.where(na, -1, arr).astype("int64")
    pos = np.searchsorted(ids, key)
    pos_clipped = np.clip(pos, 0, len(ids) - 1)
    hit = (pos < len(ids)) & (ids[pos_clipped] == key) & ~na
    return np.where(hit, pos + 1, 0).astype("int64")


def build_nn_module(torch, arch):
    """`src.nn_embed.build_model`과 **같은 구조**를 이 파일 안에서 다시 만든다.

    파라미터 이름(`pitcher_emb.weight`, `mlp.0.weight`, ...)이 state_dict의 키와
    1:1로 맞아야 하므로 레이어 구성 순서까지 원본과 같아야 한다. 초기화는 어차피
    가중치를 덮어쓰므로 재현하지 않는다.

    **BatchNorm이 없다**는 것이 규칙상 중요하다 — 있으면 배치 통계가 예측에
    끼어들어 행 단위 독립성이 깨진다. Dropout은 eval 모드에서 항등이다.
    """
    nn = torch.nn

    class EmbedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            emb_dim = int(arch["emb_dim"])
            self.pitcher_emb = nn.Embedding(int(arch["n_pitcher"]), emb_dim)
            self.batter_emb = nn.Embedding(int(arch["n_batter"]), emb_dim)
            layers, d = [], int(arch["n_features"]) + 2 * emb_dim
            for h in arch["hidden"]:
                layers += [nn.Linear(d, int(h)), nn.ReLU(),
                           nn.Dropout(float(arch["dropout"]))]
                d = int(h)
            layers.append(nn.Linear(d, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x_num, pid, bid):
            z = torch.cat([x_num, self.pitcher_emb(pid), self.batter_emb(bid)],
                          dim=1)
            return self.mlp(z).squeeze(1)

    return EmbedMLP()


class EmbedNNPredictor:
    """`.npz`(state_dict 배열)로 저장된 선수 임베딩 MLP + 학습 때와 같은 전처리.

    `predict_proba`가 (n, 2) 배열을 돌려주므로 블렌드 쪽 `_mean_proba`는 트리
    모델과 구분 없이 이 멤버를 쓴다.

    입력 `X`에는 챔피언 피처 **+ pitcher_id / batter_id**가 함께 들어온다(멤버
    meta의 `feature_columns`가 그렇게 적혀 있다). 수치/범주 전처리는 meta의
    `num_cols`/`cat_cols` 목록만 보고 돌아가므로 id 두 컬럼은 그 경로에 끼어들지
    않고 임베딩 입력으로만 쓰인다.

    추론은 **항상 CPU + eval 모드**다. GPU가 있어도 쓰지 않는다 — 245,789행짜리
    작은 MLP는 CPU로 몇 초면 끝나고, 장치에 따라 값이 미세하게 달라질 여지를
    아예 없애는 편이 재현성에 낫다.

    **고정 크기 배치로 패딩한다.** float32 행렬곱은 결합법칙이 성립하지 않고,
    BLAS는 행렬의 행 수(M)에 따라 다른 커널/블록킹을 고른다. 그래서 같은 행이라도
    1행짜리 입력과 25만행짜리 입력에서 마지막 자리가 ~1e-8 달라진다. 값으로는
    무의미하지만 "1행만 넣은 예측 == 전체를 넣은 예측"이라는 공지 판정 기준은
    기계적으로 깨진다. 매 배치를 **항상 같은 행 수로 0 패딩**해 넣으면 M이 상수라
    커널 선택도 누산 순서도 고정되고, 실제로 atol=0으로 일치한다(스레드 수를
    바꿔도 같다). 패딩 행은 잘라 버리고, eval 모드라 드롭아웃도 없으므로 진짜
    행에 어떤 영향도 주지 않는다.

    그 대가로 **배치 크기는 값의 일부**가 된다(meta의 nn.runtime.batch_size).
    바꾸면 예측이 1e-8 수준으로 흔들리므로 산출물을 만든 뒤에는 건드리지 않는다.
    """

    def __init__(self, model, torch, spec, batch_size=16384):
        self.model = model
        self.torch = torch
        self.prep = spec["prep"]
        self.vocab = spec["vocab"]
        self.batch_size = int(batch_size)

    @classmethod
    def load(cls, path, spec):
        import torch

        nn_spec = spec.get("nn")
        if not nn_spec:
            raise ValueError(
                "NN 멤버 사양에 'nn' 키가 없다 — .npz만으로는 전처리 상수와 "
                "아키텍처를 알 수 없어 예측을 재현할 수 없다")
        for key in ("prep", "vocab", "arch"):
            if key not in nn_spec:
                raise ValueError(f"NN 멤버 사양에 nn.{key}가 없다")
        model = build_nn_module(torch, nn_spec["arch"])
        with np.load(path) as z:
            state = {k: torch.from_numpy(np.asarray(z[k])) for k in z.files}
        # strict=True: 이름/모양이 하나라도 어긋나면 여기서 터진다. 조용히 일부만
        # 로드되면 초기값 그대로인 층이 남아 예측이 통째로 망가진다.
        model.load_state_dict(state, strict=True)
        model.eval()
        rt = nn_spec.get("runtime") or {}
        return cls(model, torch, nn_spec, rt.get("batch_size", 16384))

    def predict_proba(self, X):
        torch = self.torch
        Xn = nn_prep_transform(X, self.prep)
        pid = nn_vocab_transform(X[self.vocab["pitcher_col"]],
                                 self.vocab["pitcher_ids"])
        bid = nn_vocab_transform(X[self.vocab["batter_col"]],
                                 self.vocab["batter_ids"])
        n = len(Xn)
        p = np.empty(n, dtype="float64")
        bs = max(1, int(self.batch_size))
        n_feat = Xn.shape[1]
        with torch.no_grad():
            for s in range(0, n, bs):
                e = min(s + bs, n)
                k = e - s
                # 항상 bs행짜리 입력을 만든다 (부족분은 0 패딩 + UNK id).
                # 행렬 모양이 상수라야 float32 누산 순서가 고정된다 — 클래스
                # docstring 참조.
                xb = torch.zeros((bs, n_feat), dtype=torch.float32)
                xb[:k] = torch.as_tensor(Xn[s:e], dtype=torch.float32)
                pb = torch.zeros(bs, dtype=torch.long)
                pb[:k] = torch.as_tensor(pid[s:e], dtype=torch.long)
                bb = torch.zeros(bs, dtype=torch.long)
                bb[:k] = torch.as_tensor(bid[s:e], dtype=torch.long)
                p[s:e] = torch.sigmoid(self.model(xb, pb, bb)).numpy()[:k]
        return np.column_stack([1.0 - p, p])


class TeamLGBMStackPredictor:
    """팀원 LightGBM 4종 스택 하나를 **멤버 한 개**처럼 보이게 감싼다 (R7).

    이 스택의 내부 연산은 우리 블렌드 규약(`model_files`를 균등 평균)으로는
    표현할 수 없다 — 군(group)마다 **가중이 다른** 3종 평균을 내고, 군마다 Platt를
    얹은 뒤 60:40으로 합친다. 그래서 부스터 4개를 파일 하나(`*.lgbmstack.json`)에
    담고 그 안에서 레시피를 완결시킨다. 바깥에서 보면 `predict_proba`가 있는
    보통의 멤버라 `predict_blend`가 손댈 것이 없다.

    부스터는 `model_str=`로 읽는다. `model_file=`은 LightGBM의 C++ 파일 열기를
    타는데 **경로에 비ASCII 문자가 있으면 실패**한다(이 저장소 경로가 그렇다) —
    평가 서버는 ASCII지만 로컬 패리티 검증이 거기서 죽으므로 경로를 아예 타지 않는다.

    행 단위 독립성: 부스터 predict는 행마다 독립이고, 군 평균·Platt·군 평균은
    전부 같은 행의 값끼리만 섞는다. 다른 행이 개입할 경로가 없다.
    """

    def __init__(self, boosters, spec, feature_columns):
        self.boosters = boosters
        self.groups = spec["groups"]
        self.group_weights = spec["group_weights"]
        self.calib = spec.get("calib")          # None이면 보정 전 raw 블렌드
        self.feature_columns = list(feature_columns)
        if self.calib is not None and len(self.calib) != len(self.groups):
            raise ValueError(
                f"calib 개수({len(self.calib)})가 군 수({len(self.groups)})와 다르다")
        need = {m for g in self.groups for m in g["members"]}
        missing = need - set(self.boosters)
        if missing:
            raise ValueError(f"번들에 없는 부스터: {sorted(missing)}")

    @classmethod
    def load(cls, path, spec):
        import lightgbm as lgb
        team = spec.get("team")
        if not team:
            raise ValueError("팀 스택 멤버에는 meta의 'team' 사양이 필요하다 "
                             "(groups/group_weights/cat_cols/category_maps)")
        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        boosters = {name: lgb.Booster(model_str=text)
                    for name, text in bundle["boosters"].items()}
        return cls(boosters, team, spec["feature_columns"])

    def predict_proba(self, X):
        if list(X.columns) != self.feature_columns:
            raise ValueError(
                f"팀 스택 입력 컬럼이 학습 때와 다르다 "
                f"({len(X.columns)} vs {len(self.feature_columns)})")
        preds = {name: np.asarray(b.predict(X), dtype="float64")
                 for name, b in self.boosters.items()}
        outs = []
        for i, g in enumerate(self.groups):
            raw = np.average([preds[m] for m in g["members"]], axis=0,
                             weights=g["weights"])
            if self.calib is not None:
                raw = apply_logit_shift(raw, self.calib[i]["bias"],
                                        slope=self.calib[i]["scale"])
            outs.append(raw)
        p = np.average(outs, axis=0, weights=self.group_weights)
        return np.column_stack([1.0 - p, p])


def s1_shrink_rate(rate, n, M, prior):
    """λ = n/(n+M), rate_sh = λ·rate + (1−λ)·prior. `src.reliability` 이식본.

    결측·음수 n은 **관측 0건**으로 본다(λ=0 -> prior). rate가 결측이어도 같은
    자리로 떨어진다 — cold-start와 "표본은 있는데 값이 없음"을 구분할 근거가
    데이터에 없기 때문이다. 전부 그 행의 값과 학습 상수만 쓰는 행 단위 변환이다.
    """
    M = float(M)
    if not np.isfinite(M) or M <= 0.0:
        raise ValueError(f"S1 수축의 M은 양의 유한값이어야 한다: {M!r}")
    n = np.asarray(n, dtype="float64")
    n = np.where(np.isfinite(n) & (n > 0.0), n, 0.0)
    lam = n / (n + M)
    rate = np.asarray(rate, dtype="float64")
    valid = np.isfinite(rate)
    lam = np.where(valid, lam, 0.0)
    return lam * np.where(valid, rate, 0.0) + (1.0 - lam) * float(prior)


class S1LogisticPredictor:
    """`.pkl`로 저장된 sklearn 로지스틱 + 학습 때와 같은 전처리 (R8 S1).

    `src.reliability.ReliabilityLogistic`의 예측 경로 이식본이다. 모델 자체는
    평범한 sklearn 추정기라 pickle이 안전하지만(모듈 경로가 sklearn 안에서
    닫힌다), **전처리 상수는 pickle에 없다** — 그래서 meta의 `s1` 블록에서
    읽는다:

      numeric_cols / onehot   열 구성과 **고정 레벨** (열 순서가 곧 계수의 자리다)
      rate_n_map / priors     rate 컬럼과 그 표본수 컬럼, 학습측 prior
      shrink / M              λ(n) 수축 여부와 강도 (배포본은 shrink=false)
      median / center / scale 결측 대치값과 표준화 상수 (학습측에서만 나온 값)
      design_columns          설계행렬 폭 검증용 이름표

    미지 범주값(2025 신규)은 그 블록이 전부 0인 행으로 안전하게 떨어지고,
    rate 결측은 prior 자리로 간다. 전부 **행 단위** 변환이라 배치 구성이 바뀌어도
    그 행의 예측은 변하지 않는다. src와의 값 동등성·행 독립성은
    tests/test_submit_s1.py가 실제 적합된 모델로 고정한다.

    **선형 점수는 행렬곱이 아니라 행별 합으로 낸다.** `model.predict_proba`를
    그대로 쓰면 BLAS가 행 수(M)에 따라 다른 커널을 골라 1행 입력과 25만행 입력의
    마지막 자리가 ~1e-16 달라진다 — 값으로는 무의미하지만 "1행만 넣은 예측 ==
    전체를 넣은 예측"이라는 공지 판정 기준은 기계적으로 깨진다(실측). NN 멤버가
    고정 배치 패딩으로 푼 것과 같은 문제이고, 여기서는 `(Z*w).sum(axis=1)`로
    **누산 순서를 행 길이만의 함수**로 만들어 푼다(배치 크기라는 숨은 상수가
    생기지 않는 대신, sklearn 출력과 ≤2.2e-16 차이가 난다). 계수는 실린 추정기에서
    꺼내 쓴다 — meta로 전사하지 않으므로 두 값이 어긋날 경로가 없다.
    """

    REQUIRED = ("numeric_cols", "onehot", "rate_n_map", "priors", "median",
                "center", "scale", "design_columns")
    #: 점수 계산 청크 (값에 영향 없음 — 행별 합이라 경계와 무관하다). 메모리만 제한.
    SCORE_CHUNK = 65536

    def __init__(self, model, spec, feature_columns):
        missing = [k for k in self.REQUIRED if k not in spec]
        if missing:
            raise ValueError(f"S1 멤버 사양에 없는 상수: {missing} — 전처리를 "
                             f"재현할 수 없다")
        self.model = model
        self.feature_columns = list(feature_columns)
        self.numeric_cols = list(spec["numeric_cols"])
        self.onehot = {c: list(lv) for c, lv in spec["onehot"].items()}
        self.rate_n_map = dict(spec["rate_n_map"])
        self.priors = {c: float(v) for c, v in spec["priors"].items()}
        self.shrink = bool(spec.get("shrink", False))
        self.M = {g: float(v) for g, v in (spec.get("M") or {}).items()}
        self.median = np.asarray(spec["median"], dtype="float64")
        self.center = np.asarray(spec["center"], dtype="float64")
        self.scale = np.asarray(spec["scale"], dtype="float64")
        self.design_columns = list(spec["design_columns"])

        n = len(self.numeric_cols)
        if not (len(self.median) == len(self.center) == len(self.scale) == n):
            raise ValueError(
                f"S1 표준화 상수 길이가 수치 열 수와 다르다: "
                f"median {len(self.median)} / center {len(self.center)} / "
                f"scale {len(self.scale)} vs {n} — 상수가 위치로 걸리므로 "
                f"그대로 쓰면 조용히 틀린다")
        width = n + sum(len(lv) for lv in self.onehot.values())
        if width != len(self.design_columns):
            raise ValueError(f"S1 설계행렬 폭이 meta와 다르다: {width} vs "
                             f"{len(self.design_columns)}")

        self.coef = np.asarray(model.coef_, dtype="float64").reshape(-1)
        self.intercept = float(np.asarray(model.intercept_,
                                          dtype="float64").reshape(-1)[0])
        if len(self.coef) != width:
            raise ValueError(f"로지스틱 계수 개수({len(self.coef)})가 설계행렬 "
                             f"폭({width})과 다르다 — 실린 모델과 meta 상수가 "
                             f"다른 적합에서 나왔다")

    @classmethod
    def load(cls, path, spec):
        s1 = spec.get("s1")
        if not s1:
            raise ValueError(
                "S1 멤버 사양에 's1' 키가 없다 — .pkl만으로는 전처리 상수"
                "(수축/결측대치/표준화/원핫 레벨)를 알 수 없어 예측을 재현할 수 없다")
        return cls(joblib.load(path), s1, spec["feature_columns"])

    def _numeric_block(self, X):
        cols = []
        for c in self.numeric_cols:
            v = X[c].to_numpy(dtype="float64", na_value=np.nan)
            n_col = self.rate_n_map.get(c)
            if n_col is not None:
                prior = self.priors[c]
                if self.shrink:
                    v = s1_shrink_rate(
                        v, X[n_col].to_numpy(dtype="float64", na_value=np.nan),
                        self.M[SMOOTH_GROUP_OF[c]], prior)
                else:
                    # 수축 없는 변형(대조 A = 배포본)도 **결측 채움은 공유**한다.
                    v = np.where(np.isfinite(v), v, prior)
            cols.append(v)
        return (np.column_stack(cols) if cols
                else np.empty((len(X), 0), dtype="float64"))

    def _onehot_block(self, X):
        blocks = []
        for c, levels in self.onehot.items():
            v = np.asarray(X[c].astype("object"))
            blocks.append(np.column_stack(
                [(v == lv) for lv in levels]).astype("float64"))
        return np.hstack(blocks)

    def design_matrix(self, X):
        if list(X.columns) != self.feature_columns:
            raise ValueError(
                "학습 때와 입력 열 구성/순서가 다르다 — 표준화 상수와 계수가 "
                f"위치로 걸리므로 그대로 쓰면 조용히 틀린다 "
                f"({len(X.columns)} vs {len(self.feature_columns)})")
        num = self._numeric_block(X)
        num = np.where(np.isfinite(num), num, self.median)
        safe = np.where(self.scale > 0, self.scale, 1.0)
        num = (num - self.center) / safe
        return np.hstack([num, self._onehot_block(X)]).astype("float32")

    def decision(self, X):
        """행별 선형 점수. 청크 경계와 배치 구성에 값이 의존하지 않는다."""
        Z = self.design_matrix(X)
        out = np.empty(len(Z), dtype="float64")
        step = max(1, int(self.SCORE_CHUNK))
        for s in range(0, len(Z), step):
            block = Z[s:s + step].astype("float64")
            out[s:s + step] = (block * self.coef).sum(axis=1) + self.intercept
        return out

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision(X)))
        return np.column_stack([1.0 - p, p])


def _load_one(model_dir, filename, spec):
    """파일 하나 -> 예측기. **engine**이 먼저, 없으면 확장자가 로더를 고른다
    (.cbm = CatBoost, .npz = 임베딩 NN, .lgbmstack.json = 팀원 LightGBM 스택).

    zip 안 model/은 평평하므로 이름이 유일한 단서다. joblib pickle과 native
    포맷을 섞어 쓰는 이유는 모듈 경로 문제(CatBoost 래퍼 / nn.Module) 때문이다
    (모듈 docstring 참조). S1 로지스틱만 확장자가 아니라 engine으로 고른다 —
    저장 포맷이 평범한 `.pkl`이라 이름으로는 트리 멤버와 구분되지 않는데,
    잘못 고르면 전처리를 건너뛴 채 예측이 나온다.
    """
    path = os.path.join(model_dir, filename)
    if spec.get("engine") == "s1_logistic":
        return S1LogisticPredictor.load(path, spec)
    if filename.endswith(".cbm"):
        return CatBoostNativePredictor.load(path, spec)
    if filename.endswith(".npz"):
        return EmbedNNPredictor.load(path, spec)
    if filename.endswith(TEAM_BUNDLE_SUFFIX):
        return TeamLGBMStackPredictor.load(path, spec)
    return joblib.load(path)


def load_models(model_dir, meta):
    """meta 기준으로 예측기를 읽는다.

    - "members"가 있으면 **멤버마다 모델 리스트**를 담은 리스트를 돌려준다
      (멤버 하나가 seed 앙상블일 수 있으므로 항상 이중 리스트).
      파일 이름은 멤버 사양의 `model_files`에 그대로 적혀 있다 — 제출 zip은
      model/ 아래가 평평하므로 이름만으로 구분한다.
    - "seeds"가 있으면 그 길이만큼 model_0.pkl..model_{n-1}.pkl을 순서대로 읽는다.
    - 둘 다 없으면 model.pkl 하나.

    어느 경로든 파일이 하나라도 없으면 joblib이 FileNotFoundError를 내며 멈춘다 —
    조용히 적은 수로 평균하면 학습 때와 다른 예측기가 되는데 그 사실이 출력만
    봐서는 드러나지 않는다.
    """
    members = meta.get("members")
    if members:
        return [[_load_one(model_dir, f, m) for f in m["model_files"]]
                for m in members]
    seeds = meta.get("seeds")
    if not seeds:
        return [joblib.load(os.path.join(model_dir, "model.pkl"))]
    return [joblib.load(os.path.join(model_dir, f"model_{i}.pkl"))
            for i in range(len(seeds))]


def _mean_proba(models, X):
    """여러 예측기의 양성 확률 평균.

    행 단위 독립성은 그대로다: 각 모델이 같은 행 X[i]를 보고 낸 값들만 합쳐지고,
    다른 행의 값은 어떤 경로로도 섞이지 않는다.
    """
    total = None
    for m in models:
        p = m.predict_proba(X)[:, 1]
        total = p if total is None else total + p
    return total / len(models)


def _warn_unknown_categories(X, verbose, tag=""):
    """미지 범주는 NaN(=결측)으로 떨어진다. 무신호로 조용히 흘러가면 곤란하므로
    발생 건수를 남긴다(2025 신규 값 감지용)."""
    if not verbose:
        return
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            n_nan = int(X[col].isna().sum())
            if n_nan:
                print(f"warn: {tag}{col} unknown->NaN {n_nan} rows")


def normalize_weights(weights, n):
    """가중 목록 -> 합이 1인 배열. None이면 균등 (= v2 동작 그대로).

    합이 1이 아닌 가중을 그대로 쓰면 결과가 확률 범위를 벗어나므로 항상
    재정규화한다. 음수는 "확률의 가중평균"이 아니라 외삽이라 거부한다.
    src.validate.normalize_weights의 이식본이다.
    """
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype="float64")
    if w.shape != (n,):
        raise ValueError(f"가중 개수가 멤버 수와 다르다: {w.shape} vs ({n},) — "
                         f"조용히 브로드캐스트하면 검증한 블렌드와 달라진다")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError(f"가중에 음수/비유한 값이 있다: {list(w)}")
    total = float(w.sum())
    if total <= 0:
        raise ValueError(f"가중의 합이 0 이하다: {list(w)}")
    return w / total


def predict_blend(frame, member_models, meta, verbose=True):
    """멤버별 (모델 평균 -> 그 멤버의 보정) 뒤 **가중평균**, 마지막에 블렌드 보정.

    logit 평균이 아니다 — 확률의 가중평균이다. 가중은
    `meta["blend"]["weights"]`이고 없으면 균등(v2 동작 그대로).
    `meta["blend"]["calibration"]`이 있으면 **가중평균 뒤에 한 번** 얹는다
    (v3의 단일 c0). logit이 비선형이라 "멤버별로 먼저"와 값이 다르므로, 로컬
    검증(exp17)이 실측한 순서를 그대로 지켜야 점수가 제출로 이어진다.

    `frame`은 이미 cat_levels가 적용된 사본이다. 멤버마다 groups·feature_columns가
    다르므로 피처는 멤버별로 다시 만든다(멤버 사양이 곧 그 멤버의 meta다).
    """
    members = meta["members"]
    if len(member_models) != len(members):
        raise ValueError(
            f"모델 묶음 수({len(member_models)})가 meta의 멤버 수"
            f"({len(members)})와 다르다 — 일부 멤버가 빠진 채로 평균하면 "
            f"검증한 블렌드와 다른 예측이 된다")
    blend = meta.get("blend") or {}
    lookups = meta.get("lookups")

    keys = [spec.get("name", i) for i, spec in enumerate(members)]
    if len(set(keys)) != len(keys):
        raise ValueError(f"멤버 이름이 중복이다: {keys} — 스테이지가 어느 멤버를 "
                         f"가리키는지 결정되지 않는다")
    member_p = {}
    for i, (spec, models) in enumerate(zip(members, member_models)):
        models = list(models) if isinstance(models, (list, tuple)) else [models]
        X = build_submit_features(frame, spec, lookups)
        _warn_unknown_categories(X, verbose,
                                 tag=f"[{spec.get('name', '?')}] ")
        p = apply_calibration(_mean_proba(models, X), spec.get("calibration"),
                              frame)
        if verbose:
            print(f"member {spec.get('name', '?')} "
                  f"[{spec.get('engine', 'hgb')}]: {len(models)} model(s), "
                  f"{len(spec['feature_columns'])} features, "
                  f"mean pred {p.mean():.4f}")
        member_p[keys[i]] = p

    stages = blend.get("stages")
    if stages:
        total = _combine_stages(member_p, stages, blend, frame, verbose)
    else:
        w = normalize_weights(blend.get("weights"), len(members))
        total = None
        for key, wi in zip(keys, w):
            contrib = member_p[key] * wi
            total = contrib if total is None else total + contrib

    calib = blend.get("calibration")
    if calib:
        total = apply_calibration(total, calib, frame)
        if verbose:
            print(f"blend calibration {calib} -> mean pred {total.mean():.4f}")
    return total


def _combine_stages(member_p, stages, blend, frame, verbose=True):
    """**계층 결합** (meta 스키마 v6, R7): 스테이지 안에서 먼저 합치고 보정한 뒤,
    스테이지끼리 가중평균한다.

        P_ours = c0( .4 hgb + .4 cat + .2 nn )        # 우리 배포 형태 (LB 938.07)
        P_team = 60:40( Platt(g1), Platt(g2) )        # 팀원 배포 형태 (LB 939.92)
        final  = .4 P_ours + .6 P_team

    7모델 평평 평균과 **값이 다르다**. 보정이 logit 비선형이라 "합치고 보정"과
    "보정하고 합치기"의 순서가 결과를 바꾸기 때문이다. 계층으로 두는 이유는
    가중을 1/0으로 몰면 각 팀이 리더보드에서 실측한 **바로 그 예측**이 그대로
    복원되기 때문이다(endpoint-preserving) — 두 실측 앵커와의 연결이 유지된다.

    각 스테이지는 자기 멤버들의 확률만 가중평균하고 자기 보정을 얹는다. 전부 행
    단위 변환이라 행 독립성은 그대로다.
    """
    stage_p = []
    for st in stages:
        names = st["members"]
        missing = [n for n in names if n not in member_p]
        if missing:
            raise ValueError(f"스테이지 {st.get('name', '?')!r}가 참조하는 "
                             f"멤버가 없다: {missing}")
        sw = normalize_weights(st.get("weights"), len(names))
        p = None
        for n, wi in zip(names, sw):
            contrib = member_p[n] * wi
            p = contrib if p is None else p + contrib
        p = apply_calibration(p, st.get("calibration"), frame)
        if verbose:
            print(f"stage {st.get('name', '?')}: members={names} "
                  f"weights={[round(float(x), 4) for x in sw]} "
                  f"calib={st.get('calibration')} mean pred {p.mean():.4f}")
        stage_p.append(p)
    gw = normalize_weights(blend.get("stage_weights"), len(stages))
    total = None
    for p, wi in zip(stage_p, gw):
        contrib = p * wi
        total = contrib if total is None else total + contrib
    if verbose:
        print(f"stage mix weights={[round(float(x), 4) for x in gw]} "
              f"-> mean pred {total.mean():.4f}")
    return total


def merge_predictions(sub, pred_map):
    """sample_submission 순서대로 예측을 채우고 미매칭 건수를 반환한다.

    미매칭 행은 sample의 placeholder 값이 그대로 제출되므로 조용히 넘어가면
    안 된다. 건수를 세어 호출자가 로그로 남길 수 있게 한다.
    """
    merged, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        if rid in pred_map:
            merged.append(pred_map[rid])
        else:
            merged.append(cur)
            n_missing += 1
    return merged, n_missing


def check_output(preds, test_ids, sub_ids, n_missing, strict):
    """제출 직전 무결성 검사. `strict`면 **경고가 아니라 실패**다 (R7).

    관대한 기본 동작(placeholder 유지 + 경고)은 스키마가 단순하던 시절의 것이다.
    멤버가 넷이고 스테이지가 둘인 지금은 조용히 절반만 채워진 제출이 실제로
    가능하고, 그 사고는 리더보드 점수 한 줄로만 드러난다. 그래서 R7 패키지는
    meta에 `strict_output: true`를 박아 아래를 전부 하드 실패로 바꾼다.

      * 예측에 NaN/Inf
      * 예측이 [0, 1] 밖
      * test.csv의 row_id 중복
      * sample_submission에 있는데 예측이 없는 row_id (missing)
      * 예측에 있는데 sample_submission에 없는 row_id (extra)
    """
    problems = []
    p = np.asarray(preds, dtype="float64")
    n_bad = int((~np.isfinite(p)).sum())
    if n_bad:
        problems.append(f"예측에 NaN/Inf {n_bad}건")
    if len(p) and (p.min() < 0.0 or p.max() > 1.0):
        problems.append(f"예측이 [0,1] 밖: [{p.min():.6g}, {p.max():.6g}]")
    n_dup = len(test_ids) - len(set(test_ids))
    if n_dup:
        problems.append(f"test.csv row_id 중복 {n_dup}건")
    if n_missing:
        problems.append(f"예측이 없는 sample_submission row_id {n_missing}건")
    n_extra = len(set(test_ids) - set(sub_ids))
    if n_extra:
        problems.append(f"sample_submission에 없는 test row_id {n_extra}건")
    if not problems:
        return
    msg = "제출 무결성 검사 실패: " + " / ".join(problems)
    if strict:
        raise ValueError(msg)
    print(f"warn: {msg}")


def predict(test, model, meta, verbose=True):
    """test 프레임 -> 확률 배열. 제출 예측 경로 전체가 이 함수 하나다.

    `model`은 예측기 하나여도 되고 리스트여도 된다(seed 앙상블). 리스트면
    predict_proba를 평균한 **뒤** 보정을 얹는다. meta에 "members"가 있으면
    `model`은 멤버별 모델 리스트의 리스트이고, 멤버 블렌드 경로로 간다.

    **행 단위 독립**이다: 어떤 행의 예측도 같은 배치에 무엇이 들어왔는지에
    의존하지 않는다. 1행만 넣으나 전체를 넣으나 같은 값이 나와야 하며,
    tests/test_row_independence.py가 실제 배포본으로 그걸 확인한다.
    """
    test = test.copy()
    test = add_state(test, "./model")
    # 학습과 동일한 전처리: category 레벨 고정 → 파생 생성 → 컬럼 선택
    for col, levels in meta["cat_levels"].items():
        test[col] = pd.Categorical(test[col], categories=levels)

    if meta.get("members"):
        preds = predict_blend(test, model, meta, verbose=verbose)
        preds = apply_season_form(preds, test, meta.get("season_form"), verbose)
        preds = apply_batter_form(preds, test, "./model")
        preds = apply_coldstart_expert(preds, test, "./model")
        preds = apply_corrections(preds, test, "./model")
        shift = float(meta.get("final_logit_shift", 0.0))
        return apply_logit_shift(preds, shift) if shift else preds

    X = build_submit_features(test, meta, meta.get("lookups"))
    _warn_unknown_categories(X, verbose)

    models = list(model) if isinstance(model, (list, tuple)) else [model]
    preds = _mean_proba(models, X)
    if verbose and len(models) > 1:
        print(f"averaged {len(models)} models (seeds={meta.get('seeds')})")
    shift = float(meta.get("logit_shift", 0.0))
    if shift:
        preds = apply_logit_shift(preds, shift)
        if verbose:
            print(f"applied logit_shift={shift:+.4f} "
                  f"(mean pred {preds.mean():.4f}, "
                  f"range [{preds.min():.4f}, {preds.max():.4f}])")
    preds = apply_season_form(preds, test, meta.get("season_form"), verbose)
    preds = apply_batter_form(preds, test, "./model")
    preds = apply_coldstart_expert(preds, test, "./model")
    preds = apply_corrections(preds, test, "./model")
    shift = float(meta.get("final_logit_shift", 0.0))
    return apply_logit_shift(preds, shift) if shift else preds


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    with open("./model/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    models = load_models("./model", meta)

    preds = predict(test, models, meta)
    test_ids = list(test[ID_COL])
    pred_map = dict(zip(test_ids, preds))
    sub[TARGET_COL], n_missing = merge_predictions(sub, pred_map)
    check_output(preds, test_ids, list(sub[ID_COL]), n_missing,
                 bool(meta.get("strict_output", False)))
    if n_missing:
        print(f"warn: {n_missing} row_ids in sample_submission had no "
              f"prediction (placeholder kept)")

    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"saved ./output/submission.csv rows={len(sub)}")


if __name__ == "__main__":
    main()
