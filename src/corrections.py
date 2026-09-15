"""잔차 기반 보정표 3종 — 학습 시점에 만들어 얼리고, 추론 때 행별로 조회한다.

**무엇을 왜 잡는가.** 정보원별 천장을 재보면 이 문제의 신호는 거의 전부 투수
정체성이다 (2024 R 기준: pitcher_id 단독 684, 현재 모델 737, 상황 변수는 7~23).
그중 `pitcher x batter_hand` 만 **784** 로 모델보다 높다. 즉 "이 투수가 어떤
상황에서 달라지는가" 에만 여지가 남아 있고, 상황 그 자체(구장·주자·홈원정·
포수 대리)에는 없다 — 전부 실측에서 음수였다.

| 보정 | 키 | k | R 구간 기여 |
| --- | --- | --- | --- |
| beta  (platoon) | (pitcher_id, batter_hand) | 2000 | +12.3 |
| gamma (count)   | balls*3 + strikes | 5000 | +8.8 |
| delta (leverage)| (pitcher_id, 유리/중립/불리) | 2000 | +6.4 |

세 개를 합치면 전체 행 기준 2024 폴드 +33.5, 3폴드 평균 +24.6, 결합 z=6.69.

**설계 규칙 — 넷 다 지키지 않으면 부호가 뒤집힌다.**

1. 잔차는 **out-of-sample** 이어야 한다. 시즌 s 의 잔차는 시즌 < s 로 학습한
   모델에서 뽑는다. 그 시즌을 학습한 모델의 잔차는 이미 그 투수에 맞춰져 있어
   0에 가깝고, 스플릿이 잡히지 않는다.
2. 풀링 전에 **시즌별 중심화**. 기저율이 매년 -0.015 씩 내려가므로, 안 빼면
   구간 구조가 아니라 낡은 수준 이동을 옮기게 된다.
3. 투수 상호작용(beta, delta)은 **투수 주효과를 뺀 순수 편차**다. 안 빼면
   season-form 이 이미 하는 투수 수준 보정과 이중 계산이 된다.
4. **R 행에서만** 추정하고 적용한다. F 는 레짐이 깨진다 (2022 .709 -> 2023 .473).

**규칙 준수 (설명서 5절).** 모든 표는 학습 시즌에서만 만들어 고정되고, 평가
행은 자기 행의 값(`pitcher_id`, `batter_hand`, `balls_before`, `strikes_before`)
으로 조회만 한다. 평가 데이터의 다른 행·누적·분포는 일절 쓰지 않는다.
`season_form` 과 동일한 구조다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT = "R"


# =======================
# 행 지역 키 — 그 행의 값만으로 결정된다
# =======================

def key_batter_hand(rows):
    return np.asarray(rows["batter_hand"], dtype="int64")


def key_count(rows):
    return (np.asarray(rows["balls_before"], dtype="int64") * 3
            + np.asarray(rows["strikes_before"], dtype="int64"))


def key_leverage(rows):
    """투수 유리(0) / 중립(1) / 불리(2).

    3단계가 최적이다 — 2단계(+5.3), 12단계 전체 카운트(+4.8), 볼 개수 기준
    (-4.9) 보다 낫다. 야구에서 의미 있는 구분이 원시 카운트가 아니라 유불리
    라는 뜻이다.
    """
    b = np.asarray(rows["balls_before"], dtype="int64")
    s = np.asarray(rows["strikes_before"], dtype="int64")
    return np.where(s > b, 0, np.where(b > s + 1, 2, 1))


KEYS = {"batter_hand": key_batter_hand, "count": key_count,
        "leverage": key_leverage}


# =======================
# 표 만들기 (학습 시점)
# =======================

def _residual_frames(df, ref, train_seasons):
    """시즌별로 중심화한 out-of-sample 잔차를 R 행만 모아 돌려준다."""
    out = []
    want = set(train_seasons)
    for s in sorted(ref):
        if s not in want:
            continue
        m = (df.season == s).to_numpy()
        seg = (df.game_type == SEGMENT).to_numpy()[m]
        sub = df.loc[m].loc[seg]
        r = (df.loc[m, "control_success"].to_numpy(dtype="float64")
             - np.asarray(ref[s], dtype="float64"))[seg]
        out.append((sub, r - r.mean()))     # 규칙 2: 시즌별 중심화
    return out


def build_flat(df, ref, train_seasons, key, k):
    """평면 구간 보정: key -> 값."""
    num, den = {}, {}
    for sub, r in _residual_frames(df, ref, train_seasons):
        g = pd.Series(r).groupby(KEYS[key](sub))
        for q, v in g.sum().items():
            num[q] = num.get(q, 0.0) + float(v)
        for q, v in g.size().items():
            den[q] = den.get(q, 0.0) + float(v)
    return {str(q): num[q] / (den[q] + k) for q in num}


def build_interaction(df, ref, train_seasons, key, k):
    """(pitcher_id, key) -> 값. **투수 주효과를 뺀** 순수 상호작용 편차.

    구성상 각 투수에 대해 sum_g n_{p,g} * v_{p,g} ~= 0 이 되어 수준 이동을
    만들지 않는다 (좌우의 경우 두 값의 상관 -0.984 로 확인).
    """
    frames = []
    for sub, r in _residual_frames(df, ref, train_seasons):
        frames.append(pd.DataFrame({
            "pid": np.asarray(sub["pitcher_id"], dtype="int64"),
            "g": KEYS[key](sub), "r": r}))
    d = pd.concat(frames, ignore_index=True)

    cell = d.groupby(["pid", "g"]).r.agg(["sum", "size"])
    per = d.groupby("pid").r.agg(["sum", "size"]).rename(
        columns={"sum": "ps", "size": "pn"})
    j = cell.join(per, on="pid")
    expected = j.ps * (j["size"] / j.pn)            # 규칙 3: 주효과 몫 제거
    v = (j["sum"] - expected) / (j["size"] + k)
    return {f"{p}|{g}": float(x) for (p, g), x in v.items()}


# =======================
# 표 쓰기 (추론 시점)
# =======================

def apply_flat(preds, rows, table, key):
    look = KEYS[key](rows).astype(str)
    adj = np.array([table.get(q, 0.0) for q in look], dtype="float64")
    return _add(preds, rows, adj)


def apply_interaction(preds, rows, table, key):
    pid = np.asarray(rows["pitcher_id"], dtype="int64")
    g = KEYS[key](rows)
    adj = np.array([table.get(f"{p}|{q}", 0.0) for p, q in zip(pid, g)],
                   dtype="float64")
    return _add(preds, rows, adj)


def _add(preds, rows, adj):
    """규칙 4: 정규시즌 행에만 적용. 처음 보는 키는 보정 0."""
    adj = np.where(np.asarray(rows["game_type"]) == SEGMENT, adj, 0.0)
    return np.clip(np.asarray(preds, dtype="float64") + adj, 0.0, 1.0)


SPEC = [("beta_platoon", "interaction", "batter_hand", 2000.0),
        ("gamma_count", "flat", "count", 5000.0),
        ("delta_leverage", "interaction", "leverage", 2000.0)]


def build_all(df, ref, train_seasons):
    out = {}
    for name, kind, key, k in SPEC:
        fn = build_interaction if kind == "interaction" else build_flat
        out[name] = {"kind": kind, "key": key, "k": k,
                     "table": fn(df, ref, train_seasons, key, k)}
    return out


def apply_all(preds, rows, spec):
    p = preds
    for name, _, _, _ in SPEC:
        s = spec[name]
        fn = apply_interaction if s["kind"] == "interaction" else apply_flat
        p = fn(p, rows, s["table"], s["key"])
    return p
