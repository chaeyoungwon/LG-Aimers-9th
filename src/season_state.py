"""각 행에서 그 선수의 **당해 시즌 현재 상태**를 복원한다.

**원리.** `asof_*` 는 커리어 누적이고 `rate x n` 은 정확히 정수 카운트다(검증됨:
10개 컬럼 전부 정수 이탈 <= 0.0077, CSV 소수 자릿수 오차). 그래서 시즌 경계의
누적값 `(n0, c0)` 를 학습 데이터에서 상수로 얼려두면

    n_season = asof_n - n0
    cur_X    = (round(asof_n * rate_X) - c0_X) / n_season

가 **그 행의 값만으로** 당해 시즌 상태를 준다.

**왜 확장하는가.** `season_form` 은 이 복원을 `success_rate` 하나에만 적용해
LB +37.7 을 얻었다. 그런데 같은 산술이 10개 컬럼에 성립하고, 그 컬럼들은 성공률과
**중복이 아니다**:

    success + (middle+ball+reverse) 평균 = 1.2738   (1.0 이 아님 = 분할이 아님)
    success 와의 상관: reverse -0.80, middle -0.43, ball -0.32, strike +0.15

특히 `strike_rate` 는 성공률과 거의 직교한다. 그리고 middle/ball/reverse 는 문제
정의의 **실패 3유형**(가운데 실투 / 크게 벗어남 / 반대 방향)과 1:1 대응한다.

**규칙 준수.** `(n0, c0)` 표는 시즌 s 의 행에 대해 **시즌 < s 의 학습 데이터에서만**
만든다. 평가(2025) 행은 2019~2024 말 누적값을 쓴다. 각 행은 자기 `pitcher_id` /
`batter_id` / `season` / `asof_*` 로 조회할 뿐이라 평가 데이터의 다른 행에 의존하지
않는다 — `season_form` 과 동일한 구조다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (그룹명, id 컬럼, 분모 컬럼, {짧은이름: rate 컬럼})
GROUPS = [
    ("p", "pitcher_id", "asof_pitcher_n", {
        "succ": "asof_pitcher_success_rate",
        "mid": "asof_pitcher_middle_rate",
        "ball": "asof_pitcher_ball_rate",
        "rev": "asof_pitcher_reverse_rate",
        "strk": "asof_pitcher_strike_rate"}),
    ("mix", "pitcher_id", "asof_pitcher_pitchmix_n", {
        "fb": "asof_pitcher_fastball_rate",
        "br": "asof_pitcher_breaking_rate",
        "off": "asof_pitcher_offspeed_rate"}),
    ("b", "batter_id", "asof_batter_n", {
        "succ": "asof_batter_success_rate",
        "mid": "asof_batter_middle_rate"}),
]


def _counts(df, ncol, rates):
    """rate x n -> 정수 카운트. 결측 rate 는 0 카운트로 둔다."""
    n = df[ncol].to_numpy(dtype="float64")
    out = {}
    for k, col in rates.items():
        r = df[col].to_numpy(dtype="float64")
        out[k] = np.rint(n * np.where(np.isfinite(r), r, 0.0))
    return n, out


def build_boundary(df, train_seasons):
    """{(그룹, id): (n0, {짧은이름: c0})} — train_seasons 말의 누적 상태.

    각 선수의 학습 구간 **마지막 투구 직전** 상태를 쓴다. 마지막 한 개의 결과는
    빠지지만 수천 개 중 하나라 무시할 수 있고, 모든 컬럼에 일관되게 적용된다
    (모드별 결과는 train 에 라벨이 없어 `season_form` 의 +1 보정을 쓸 수 없다).
    """
    sub = df[df.season.isin(train_seasons)]
    tables = {}
    for name, idcol, ncol, rates in GROUPS:
        s = sub[sub[ncol].notna()]
        if s.empty:
            tables[name] = None
            continue
        last = s.loc[s.groupby(idcol)[ncol].idxmax()]
        n, cnt = _counts(last, ncol, rates)
        tables[name] = {
            "ids": last[idcol].to_numpy(),
            "n0": n,
            "c0": cnt,
        }
    return tables


def add_features(df, tables, prefix="cur_"):
    """행별 당해 시즌 상태를 컬럼으로 붙여 돌려준다 (원본은 건드리지 않는다)."""
    out = {}
    for name, idcol, ncol, rates in GROUPS:
        t = tables.get(name)
        n = df[ncol].to_numpy(dtype="float64")
        if t is None:
            for k in rates:
                out[f"{prefix}{name}_{k}"] = np.full(len(df), np.nan)
            out[f"{prefix}{name}_n"] = np.full(len(df), np.nan)
            continue

        order = np.argsort(t["ids"])
        ids_sorted = t["ids"][order]
        key = df[idcol].to_numpy(dtype="float64")
        miss = ~np.isfinite(key)
        k = np.where(miss, -1, key).astype("int64")
        pos = np.searchsorted(ids_sorted, k)
        clip = np.clip(pos, 0, len(ids_sorted) - 1)
        seen = (pos < len(ids_sorted)) & (ids_sorted[clip] == k) & ~miss

        n0 = np.where(seen, t["n0"][order][clip], np.nan)
        n_season = n - n0
        # 당해 시즌 표본이 없으면(음수/0) 상태를 정의할 수 없다 -> 결측.
        # 트리는 학습된 결측 분기로 보내므로 cold-start 행이 막히지 않는다.
        valid = seen & np.isfinite(n_season) & (n_season > 0)
        out[f"{prefix}{name}_n"] = np.where(valid, n_season, np.nan)

        _, cnt = _counts(df, ncol, rates)
        for kk, col in rates.items():
            c0 = np.where(seen, t["c0"][kk][order][clip], np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = (cnt[kk] - c0) / n_season
            out[f"{prefix}{name}_{kk}"] = np.where(valid & np.isfinite(v), v, np.nan)
    return pd.DataFrame(out, index=df.index)


def add_features_by_season(df, prefix="cur_"):
    """시즌마다 **그 이전 시즌들로만** 만든 경계표를 써서 피처를 붙인다.

    이게 필요한 이유: 학습 구간 말의 표 하나만 쓰면 train 행은 `n_season <= 0` 이
    되어 전부 결측이 되고, 모델이 이 피처를 쓰는 법을 배울 수 없다. 시즌 s 의
    행에는 시즌 < s 로 만든 표를 물려야 학습과 평가의 의미가 같아진다.

    평가 시즌 T 의 표 = 시즌 < T 전체 = 그 폴드의 학습 구간 전체이므로,
    폴드 밖 데이터가 새어들지 않는다.
    """
    seasons = sorted(df.season.unique())
    parts = []
    for s in seasons:
        prior = [x for x in seasons if x < s]
        tab = build_boundary(df, prior) if prior else {g[0]: None for g in GROUPS}
        rows = df[df.season == s]
        parts.append(add_features(rows, tab, prefix))
    return pd.concat(parts).reindex(df.index)


def feature_names(prefix="cur_"):
    names = []
    for name, _, _, rates in GROUPS:
        names.append(f"{prefix}{name}_n")
        names += [f"{prefix}{name}_{k}" for k in rates]
    return names
