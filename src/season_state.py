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


def exact_success_features_by_season(df, prefix="exact_"):
    """라벨로 이전 시즌 마지막 투구까지 완결한 성공률 상태를 복원한다.

    일반 ``build_boundary``는 성공/실패 세부 유형 라벨이 모두 없어서 각 선수의
    마지막 투구 *직전* as-of 값을 경계로 쓴다. 성공률만은 ``control_success``가
    있으므로 마지막 투구 한 개를 정확히 더해 다음 시즌의 첫 행이 거짓 표본 1개를
    가진 것처럼 보이는 off-by-one을 없앨 수 있다.

    시즌 Y 행의 표는 시즌 < Y의 라벨만 사용하고, 평가용 표도 공식 train 마지막
    라벨로 미리 고정할 수 있으므로 행 독립성과 시간 순서를 모두 지킨다.
    """
    specs = (
        ("p", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
        ("b", "batter_id", "asof_batter_n", "asof_batter_success_rate"),
    )
    seasons = sorted(df.season.unique())
    parts = []
    for season in seasons:
        prior = df[df.season < season]
        rows = df[df.season == season]
        out = {}
        for name, id_col, n_col, rate_col in specs:
            eligible = prior[prior[n_col].notna()]
            if eligible.empty:
                out[f"{prefix}{name}_n"] = np.full(len(rows), np.nan)
                out[f"{prefix}{name}_succ"] = np.full(len(rows), np.nan)
                continue
            last = eligible.loc[eligible.groupby(id_col)[n_col].idxmax()]
            order = np.argsort(last[id_col].to_numpy())
            ids = last[id_col].to_numpy(dtype="int64")[order]
            n_before = last[n_col].to_numpy(dtype="float64")[order]
            rate_before = last[rate_col].to_numpy(dtype="float64")[order]
            n0 = n_before + 1.0
            c0 = (
                np.rint(n_before * np.where(np.isfinite(rate_before), rate_before, 0.0))
                + last["control_success"].to_numpy(dtype="float64")[order]
            )

            key = rows[id_col].to_numpy(dtype="float64")
            missing = ~np.isfinite(key)
            key_int = np.where(missing, -1, key).astype("int64")
            pos = np.searchsorted(ids, key_int)
            clipped = np.clip(pos, 0, len(ids) - 1)
            seen = (pos < len(ids)) & (ids[clipped] == key_int) & ~missing
            n = rows[n_col].to_numpy(dtype="float64")
            rate = rows[rate_col].to_numpy(dtype="float64")
            span = n - np.where(seen, n0[clipped], np.nan)
            count = np.rint(n * np.where(np.isfinite(rate), rate, 0.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                success = (count - np.where(seen, c0[clipped], np.nan)) / span
            valid = seen & np.isfinite(span) & (span > 0) & np.isfinite(success)
            out[f"{prefix}{name}_n"] = np.where(valid, span, np.nan)
            out[f"{prefix}{name}_succ"] = np.where(valid, success, np.nan)
        parts.append(pd.DataFrame(out, index=rows.index))
    return pd.concat(parts).reindex(df.index).astype("float32")


def recent_form_feature(df, state, prefix="cur_"):
    """직전 5경기 성공률이 당해 시즌 상태보다 얼마나 높은지 계산한다.

    기존 ``prev_vs_career`` 는 직전 5경기를 커리어 누적 성공률과 비교한다. 시즌
    상태 피처가 있는 모델에서는 같은 행에서 복원한 당해 시즌 성공률을 기준으로
    삼는 편이 "이번 시즌 평소보다 지금 뜨거운가"에 더 직접적으로 대응한다.

    두 입력은 모두 행 단위 값이며, ``state`` 역시 공식 학습 데이터로 만든 경계표와
    해당 행의 as-of 값만 사용한다. 어느 한쪽이 결측이면 파생값도 결측으로 둔다.
    """
    cur_col = f"{prefix}p_succ"
    if cur_col not in state:
        raise KeyError(f"state에 {cur_col!r} 컬럼이 없다")

    prev = pd.to_numeric(
        df["asof_pitcher_prev5_game_success_rate"], errors="coerce"
    ).to_numpy(dtype="float64")
    cur = pd.to_numeric(state[cur_col], errors="coerce").to_numpy(dtype="float64")
    value = np.where(np.isfinite(prev) & np.isfinite(cur), prev - cur, np.nan)
    return pd.DataFrame(
        {"prev5_vs_cur_p_succ": value.astype("float32")}, index=df.index
    )


def state_vs_career_features(df, state, prefix="cur_"):
    """당해 시즌 상태가 커리어 누적 상태에서 얼마나 벗어났는지 명시한다.

    원재료 두 개를 모두 넣은 트리도 제한된 깊이에서는 ``cur - career`` 방향을
    여러 분기로 근사해야 한다. 동일 선수의 장기 수준을 제거한 변화량을 직접 주면
    "원래보다 이번 시즌에 좋아졌는가"를 한 번의 분기로 사용할 수 있다.

    두 입력 모두 자기 행의 공식 ``asof_*`` 값과 train-only 시즌 경계표로 만든
    값이므로 추가 피처 역시 행 단위 독립이다.
    """
    out = {}
    for name, _, _, rates in GROUPS:
        for short, career_col in rates.items():
            state_col = f"{prefix}{name}_{short}"
            if state_col not in state:
                raise KeyError(f"state에 {state_col!r} 컬럼이 없다")
            current = pd.to_numeric(
                state[state_col], errors="coerce"
            ).to_numpy(dtype="float64")
            career = pd.to_numeric(
                df[career_col], errors="coerce"
            ).to_numpy(dtype="float64")
            value = np.where(
                np.isfinite(current) & np.isfinite(career),
                current - career,
                np.nan,
            )
            out[f"{state_col}_vs_career"] = value.astype("float32")
    return pd.DataFrame(out, index=df.index)


def context_interaction_features(df, state, prefix="cur_"):
    """투수 시즌 상태 비율을 카운트와 투·타 좌우 셀 안에 명시적으로 게이팅한다.

    트리가 약한 신호의 상호작용을 제한된 깊이 안에서 찾지 못할 가능성을 검증하기
    위한 후보군이다. 투수 상태의 다섯 비율만 사용하고, 12개 볼-스트라이크 셀과
    4개 투·타 손 조합을 사전에 고정한다. 해당 셀이 아닌 행은 0이 아니라 결측으로
    두어, 값 0과 "이 셀에 속하지 않음"을 구분한다.
    """
    state_cols = [f"{prefix}p_{name}" for name in ("succ", "mid", "ball", "rev", "strk")]
    missing = [c for c in state_cols if c not in state]
    if missing:
        raise KeyError(f"state에 필요한 컬럼이 없다: {missing}")

    balls = pd.to_numeric(df["balls_before"], errors="coerce").to_numpy(float)
    strikes = pd.to_numeric(df["strikes_before"], errors="coerce").to_numpy(float)
    count = balls * 3 + strikes
    pitcher_hand = pd.to_numeric(df["pitcher_hand"], errors="coerce").to_numpy(float)
    batter_hand = pd.to_numeric(df["batter_hand"], errors="coerce").to_numpy(float)

    out = {}
    for col in state_cols:
        short = col.removeprefix(prefix)
        value = pd.to_numeric(state[col], errors="coerce").to_numpy(float)
        for balls_before in range(4):
            for strikes_before in range(3):
                key = balls_before * 3 + strikes_before
                out[f"{short}_x_count_{balls_before}_{strikes_before}"] = np.where(
                    count == key, value, np.nan
                ).astype("float32")
        for ph in (1, 2):
            for bh in (1, 2):
                out[f"{short}_x_hand_{ph}_{bh}"] = np.where(
                    (pitcher_hand == ph) & (batter_hand == bh), value, np.nan
                ).astype("float32")
    return pd.DataFrame(out, index=df.index)


def window_features(df, windows=(1, 2)):
    """직전 N개 시즌만의 성적 — `(선수, 시즌)` 조회로 끝나는 순수 상수 피처.

    `asof_*` 는 커리어 누적이라 2019년을 2024년과 똑같이 섞는다. 시즌 경계 누적값
    두 개를 빼면 그 사이 구간만의 성적이 나온다:

        prev{N} = (c0[Y] - c0[Y-N]) / (n0[Y] - n0[Y-N])

    창을 나눠 주면 "얼마나 최근을 믿을지"를 모델이 직접 학습할 수 있다.
    `cur_*` 와 달리 행의 `asof` 값조차 쓰지 않으므로 표본 크기와 무관하게 안정적이다.

    **측정 결과: 기각.** 3시드 비교에서 `cur_*` 위에 얹으면 세 폴드 전부 악화된다
    (-3.6 / -12.4 / -12.1, 평균 -9.4, z=-1.98). 커리어 누적(`asof_*`)이 장기를,
    `cur_*` 가 당해 시즌을 이미 잡고 있어 그 사이 창은 잉여였다. 재시도 전에 이
    숫자를 볼 것.

    **규칙 준수.** 시즌 Y 의 행에는 시즌 < Y 의 경계값만 들어간다. 평가(2025)는
    2019~2024 경계를 쓰고, 각 행은 자기 `pitcher_id`/`batter_id`/`season` 으로
    조회할 뿐이다.
    """
    seasons = sorted(df.season.unique())
    bounds = {}
    for i, s in enumerate(seasons):
        prior = seasons[:i]
        bounds[s] = build_boundary(df, prior) if prior else None

    out = {}
    for name, idcol, ncol, rates in GROUPS:
        for w in windows:
            out[f"prev{w}_{name}_n"] = np.full(len(df), np.nan)
            for k in rates:
                out[f"prev{w}_{name}_{k}"] = np.full(len(df), np.nan)

    pos_of = {s: (df.season == s).to_numpy() for s in seasons}
    for i, s in enumerate(seasons):
        for w in windows:
            j = i - w
            hi, lo = bounds[s], (bounds[seasons[j]] if j >= 0 else None)
            if hi is None:
                continue
            m = pos_of[s]
            for name, idcol, ncol, rates in GROUPS:
                th = hi[name]
                if th is None:
                    continue
                oh = np.argsort(th["ids"])
                ids_h = th["ids"][oh]
                key = pd.to_numeric(df.loc[m, idcol], errors="coerce").to_numpy(float)
                k_ = np.where(np.isfinite(key), key, -1).astype("int64")
                ph = np.searchsorted(ids_h, k_)
                ch = np.clip(ph, 0, len(ids_h) - 1)
                seen = (ph < len(ids_h)) & (ids_h[ch] == k_)
                nhi = np.where(seen, th["n0"][oh][ch], np.nan)
                nlo = np.zeros(len(k_))
                clo = {kk: np.zeros(len(k_)) for kk in rates}
                if lo is not None and lo[name] is not None:
                    tl = lo[name]
                    ol = np.argsort(tl["ids"])
                    ids_l = tl["ids"][ol]
                    pl = np.searchsorted(ids_l, k_)
                    cl = np.clip(pl, 0, len(ids_l) - 1)
                    sl = (pl < len(ids_l)) & (ids_l[cl] == k_)
                    nlo = np.where(sl, tl["n0"][ol][cl], 0.0)
                    clo = {kk: np.where(sl, v[ol][cl], 0.0) for kk, v in tl["c0"].items()}
                span = nhi - nlo
                ok = seen & np.isfinite(span) & (span > 0)
                out[f"prev{w}_{name}_n"][m] = np.where(ok, span, np.nan)
                for kk in rates:
                    v = (np.where(seen, th["c0"][kk][oh][ch], np.nan) - clo[kk]) / span
                    out[f"prev{w}_{name}_{kk}"][m] = np.where(ok & np.isfinite(v), v, np.nan)
    return pd.DataFrame(out, index=df.index)


def feature_names(prefix="cur_"):
    names = []
    for name, _, _, rates in GROUPS:
        names.append(f"{prefix}{name}_n")
        names += [f"{prefix}{name}_{k}" for k in rates]
    return names
