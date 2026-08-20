"""Season-to-date form 보정 — 지금까지 가장 크게 먹힌 레버(+37.7)의 재구현.

**원리.** `asof_pitcher_n` / `asof_pitcher_success_rate` 는 **커리어 누적**이다
(검증됨: 2019년 2757에서 끝나고 2020년 2758로 이어짐, `n * rate` 로 누적 성공
횟수가 정확히 복원됨). 따라서 학습 기간 말의 커리어 총량 `(n0, s0)` 를 투수별
상수로 얼려두면, 평가 행에서

    n_season = asof_n - n0        (평가 시즌에 지금까지 던진 공)
    s_season = round(n*rate) - s0 (그중 성공)

이 각 행의 값만으로 복원된다. 여기에 베이즈 축소를 걸어 커리어 기준선 대비
"올해 폼"을 만든다:

    adj = (s_season + M*p0) / (n_season + M) - p0 - mu,   p0 = s0/n0

**규칙 준수.** `(n0, s0)` 표와 `mu` 는 학습 시즌에서만 만들어 고정한다. 추론
시 각 행은 자기 행의 `pitcher_id`·`asof_*` 값만 쓴다. 평가 데이터의 다른 행,
누적, 분포는 일절 보지 않는다 — 설명서 5절을 만족한다.

`mu` 는 보정을 평균 0으로 맞추는 중심화 상수다. 이것 없이 쓰면 전체 확률
수준이 통째로 밀려 Brier가 나빠진다. **평가 시즌이 아니라 직전 시즌 행에서**
계산해야 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT = "R"   # 정규시즌 행에만 적용. F는 레짐이 불안정해 보정 대상이 아니다.


def build_table(df, train_seasons, who="pitcher"):
    """학습 기간 **말**의 커리어 총량 (n0, s0) 를 선수별로 만든다."""
    key, ncol, rcol = f"{who}_id", f"asof_{who}_n", f"asof_{who}_success_rate"
    sub = df[df.season.isin(train_seasons)]
    last = sub.loc[sub.groupby(key)[ncol].idxmax()]

    n_before = last[ncol].to_numpy(dtype="float64")
    rate = last[rcol].to_numpy(dtype="float64")
    s_before = np.rint(n_before * np.where(np.isfinite(rate), rate, 0.0))
    # 그 마지막 투구 자체의 결과까지 더해야 "기간 말 총량"이 된다.
    return pd.DataFrame({
        key: last[key].to_numpy(),
        "n0": n_before + 1.0,
        "s0": s_before + last["control_success"].to_numpy(dtype="float64"),
    })


def raw_adjustment(rows, table, m, who="pitcher"):
    """행별 (형태 보정 전) form 편차와 적용 대상 마스크를 돌려준다."""
    key, ncol, rcol = f"{who}_id", f"asof_{who}_n", f"asof_{who}_success_rate"
    t = table.set_index(key)
    n0 = t["n0"].reindex(rows[key]).to_numpy(dtype="float64")
    s0 = t["s0"].reindex(rows[key]).to_numpy(dtype="float64")

    n = rows[ncol].to_numpy(dtype="float64")
    rate = rows[rcol].to_numpy(dtype="float64")
    s = np.rint(n * np.where(np.isfinite(rate), rate, 0.0))

    n_season, s_season = n - n0, s - s0
    with np.errstate(invalid="ignore", divide="ignore"):
        p0 = s0 / n0
        adj = (s_season + m * p0) / (n_season + m) - p0

    active = (np.isfinite(n0) & np.isfinite(n_season) & (n_season > 0)
              & np.isfinite(adj)
              & rows["game_type"].eq(SEGMENT).to_numpy())
    return np.where(active, np.nan_to_num(adj), 0.0), active


def fit_mu(df, train_seasons, m, who="pitcher"):
    """중심화 상수 mu — 학습 시즌 안에서 한 걸음 앞선 구조로 계산한다.

    마지막 학습 시즌을 '가상의 평가 시즌'으로 두고, 그 이전까지로 만든 표에서
    편차의 평균을 낸다. 평가 시즌 행은 쓰지 않는다.
    """
    seasons = sorted(train_seasons)
    if len(seasons) < 2:
        return 0.0
    inner, held = seasons[:-1], seasons[-1]
    table = build_table(df, inner, who)
    rows = df[df.season == held]
    adj, active = raw_adjustment(rows, table, m, who)
    return float(adj[active].mean()) if active.any() else 0.0


def apply_form(preds, rows, table, m, alpha, mu, who="pitcher"):
    adj, active = raw_adjustment(rows, table, m, who)
    out = np.asarray(preds, dtype="float64").copy()
    out[active] += alpha * (adj[active] - mu)
    return np.clip(out, 0.0, 1.0)
