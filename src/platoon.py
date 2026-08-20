"""투수별 좌우(platoon) 스플릿 보정.

**왜.** 정보원별 천장을 재보면 `pitcher_id` 단독은 684, `pitcher x batter_hand`
는 **784** 인데 현재 모델은 737 이다. 즉 "이 투수가 좌타/우타 중 누구를 상대할
때 제구가 더 되는가"에 약 47점이 미착취로 남아 있다.

공식 피처에 이 정보가 없다. `asof_pitcher_success_rate` 는 좌우를 합친 값이고,
`same_hand` 는 있지만 **투수별** 스플릿이 아니라 전역 평균 효과만 준다. 트리는
`pitcher_id` 가 수치형이라 선수별 상호작용을 만들 수 없다.

스플릿은 시즌을 건너 안정적이다 — 2019~23 스플릿과 2024 스플릿의 상관 0.482
(투구 100개 이상 투수 156명, 스플릿 표준편차 0.036).

**규칙 준수.** 표는 학습 시즌 잔차에서만 만들어 고정하고, 평가 행은 자기 행의
`pitcher_id` 와 `batter_hand` 로 조회한다. 평가 데이터의 다른 행·분포는 보지
않는다. season_form 과 동일한 구조다.

**설계상 중요한 두 가지**
1. β 는 투수 전체 수준을 뺀 **순수 좌우 차이**다. 안 빼면 season-form 이 이미
   하고 있는 투수 수준 보정과 이중 계산이 된다. 구성상 Σ_h n_{p,h}·β_{p,h} ≈ 0.
2. 잔차는 **out-of-sample** 이어야 한다. 그 시즌을 학습한 모델의 잔차는 이미 그
   투수에 맞춰져 0에 가까워 스플릿이 안 잡힌다. 시즌 s 의 잔차는 시즌 < s 로
   학습한 모델에서 뽑는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT = "R"   # 효과 추정도 적용도 정규시즌에서만. F는 레짐이 불안정하다.


def build_table(df, ref, train_seasons, k=300.0):
    """(pitcher_id, batter_hand) -> β. `ref` 는 season -> out-of-sample 예측."""
    use = [s for s in sorted(ref) if s in set(train_seasons)]
    parts = []
    for s in use:
        m = (df.season == s).to_numpy()
        sub = df.loc[m, ["pitcher_id", "batter_hand", "game_type"]].copy()
        r = df.loc[m, "control_success"].to_numpy(dtype="float64") - np.asarray(ref[s])
        sub["r"] = r
        sub = sub[sub.game_type == SEGMENT]
        # 시즌별 중심화 — 기저율 드리프트가 스플릿으로 새어들지 않게 한다.
        sub["r"] = sub["r"] - sub["r"].mean()
        parts.append(sub[["pitcher_id", "batter_hand", "r"]])
    if not parts:
        return pd.DataFrame(columns=["pitcher_id", "batter_hand", "beta"])
    d = pd.concat(parts, ignore_index=True)

    cell = d.groupby(["pitcher_id", "batter_hand"]).r.agg(["sum", "size"])
    per = d.groupby("pitcher_id").r.agg(["sum", "size"]).rename(
        columns={"sum": "psum", "size": "pn"})
    j = cell.join(per, on="pitcher_id")
    # 투수 전체 평균이 설명하는 몫을 뺀 나머지 = 순수 좌우 편차
    expected = j.psum * (j["size"] / j.pn)
    j["beta"] = (j["sum"] - expected) / (j["size"] + k)
    return j.reset_index()[["pitcher_id", "batter_hand", "beta"]]


def apply_table(preds, rows, table):
    """각 행이 자기 pitcher_id·batter_hand 로 β 를 조회해 더한다."""
    if len(table) == 0:
        return np.asarray(preds, dtype="float64")
    key = table.set_index(["pitcher_id", "batter_hand"]).beta
    idx = pd.MultiIndex.from_arrays([rows.pitcher_id, rows.batter_hand])
    beta = key.reindex(idx).to_numpy(dtype="float64")
    beta = np.nan_to_num(beta)                      # 처음 보는 투수는 보정 없음
    beta = np.where(rows.game_type.eq(SEGMENT).to_numpy(), beta, 0.0)
    return np.clip(np.asarray(preds, dtype="float64") + beta, 0.0, 1.0)
