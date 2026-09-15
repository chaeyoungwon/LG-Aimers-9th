"""Trackman 과거 로그에서 **투수 단위 요약값**을 만든다.

**왜.** 공식 피처 57개는 전부 경기 상황과 `asof_*` 결과 비율이다. 즉 "무엇이
일어났나"만 있고 "어떻게 던졌나"가 없다 — 발표자료 11쪽이 지적한 바로 그 한계
(ERA·BB%는 결과만 알려준다)를 그대로 반복하고 있다. `trackman_history.csv`에는
릴리스 위치·익스텐션·무브먼트가 들어 있고, 야구 분석에서 제구력의 고전적
예측인자는 **릴리스 포인트의 반복성**이다. 같은 지점에서 같은 팔로 나오는
투수가 제구가 좋다.

**ID 연결.** 메인 데이터의 `pitcher_id`(21961)와 Trackman의
`pitcher_trackman_id`(502010)는 서로 다른 익명 공간이다. (season, game_month,
game_dayofweek) 셀별 투구 수 지문 + 좌우 하드 필터로 연결한다. 지문에 쓰지 않은
두 정보로 검증했다 — 팀 일치율 0.997, 구종 비율 상관 0.69(무작위 -0.03).

**규칙 준수 (설명서 3절·5절).**
- 설명서 126행이 이 용도를 명시 허용한다: "투수 단위 요약값 등 추가 피처".
- 매핑과 프로필은 **학습 시점에 고정**되고, 추론 시 각 행은 자기 행의
  `pitcher_id`로 조회만 한다. 평가 데이터의 다른 행은 일절 보지 않는다.
- **시즌 s의 행은 Trackman 시즌 < s 만 사용한다.** 2019년 행에 2020~2024
  로그를 붙이면 그 행이 미래를 보게 되고, 로컬만 오르고 리더보드는 떨어지는
  익숙한 실패로 이어진다. 평가(2025)는 2019~2024 전체를 쓰므로 "직전 시즌
  까지"라는 규칙이 학습과 평가에서 동일하게 성립한다.
- 2025년 Trackman은 존재하지 않으므로 사용 금지 항목에 저촉되지 않는다.

**측정 결과: 기각.** 롤링 백테스트에서 R 구간 델타가 +5.3 / -12.0 / +12.2 로
부호가 엇갈리고 전부 2σ(~18) 안이다. 2024 폴드 전체 델타 +7.3 도 유의하지 않다.
원인은 `asof_pitcher_success_rate` 가 이미 타깃을 **직접** 측정하고 있어서 물리
특성이 잉여 프록시가 되는 것 — `asof_pitcher_success_rate` 를 뺀 잔차와의 상관이
전 피처 |0.016| 이하였다. 특히 "릴리스가 일관되면 제구가 좋다"는 야구 통념은 이
데이터에서 성립하지 않았다 (릴리스 산포 5분위별 성공률 0.4931 vs 0.4920, 평평).

다시 시도하기 전에 위 숫자를 먼저 볼 것. 단, `build_pitcher_map` 이 복원한 ID
연결 자체는 검증되었고(팀 일치율 0.997, 구종 비율 상관 0.69) 재사용 가능하다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 릴리스·무브먼트 원측정치. zone_speed는 rel_speed와 거의 중복이라 뺀다.
METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
           "extension", "rel_height", "rel_side"]

SHORT = {"rel_speed": "velo", "spin_rate": "spin", "induced_vert_break": "ivb",
         "horz_break": "hb", "extension": "ext", "rel_height": "relh",
         "rel_side": "rels"}

# 1군 정규시즌 로그만 지문에 쓴다 (train의 game_type='R'과 대응).
_MINOR_PREFIX = "MIN_"
_NON_KBO = ("KBO_ARM", "KBO_POL", "ACE_MEX")


def first_team_mask(tm):
    return ~(tm.pitcher_team.astype(str).str.startswith(_MINOR_PREFIX)
             | tm.pitcher_team.isin(_NON_KBO))


# =======================
# 1) pitcher_id  <->  pitcher_trackman_id
# =======================

def build_pitcher_map(train_df, tm, seasons):
    """`seasons` 범위에서만 지문을 만들어 투수를 연결한다.

    폴드 Y에서는 seasons = 2019..Y-1 이 들어온다. 그 해에 데뷔한 투수가
    매핑되지 않는 것은 의도된 동작이다 — 실제 평가에서도 신인은 과거 로그가
    없다.
    """
    tr = train_df[(train_df.game_type == "R") & train_df.season.isin(seasons)]
    tk = tm[first_team_mask(tm) & tm.season.isin(seasons)]

    def cell(d):
        return (d.season.astype(int) * 10000 + d.game_month.astype(int) * 100
                + d.game_dayofweek.astype(int))

    tr = tr.assign(_c=cell(tr))
    tk = tk.assign(_c=cell(tk))
    cells = np.union1d(tr._c.unique(), tk._c.unique())
    cidx = {c: i for i, c in enumerate(cells)}

    def fp(d, key):
        g = d.groupby([key, "_c"]).size().reset_index(name="n")
        ids = np.sort(g[key].unique())
        ridx = {p: i for i, p in enumerate(ids)}
        M = np.zeros((len(ids), len(cells)))
        M[g[key].map(ridx).to_numpy(), g._c.map(cidx).to_numpy()] = g.n.to_numpy()
        # sqrt는 투구 수가 많은 셀의 지배력을 낮춰 선발끼리 뭉치는 것을 막는다.
        M = np.sqrt(M)
        return ids, M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)

    a_ids, A = fp(tr, "pitcher_id")
    b_ids, B = fp(tk, "pitcher_trackman_id")
    if len(a_ids) == 0 or len(b_ids) == 0:
        return pd.DataFrame(columns=["pitcher_id", "pitcher_trackman_id"])

    sim = A @ B.T
    ah = tr.groupby("pitcher_id").pitcher_hand.first().map({1: "Left", 2: "Right"})
    bh = tk.groupby("pitcher_trackman_id").pitcher_hand.first()
    sim[ah.reindex(a_ids).to_numpy()[:, None] != bh.reindex(b_ids).to_numpy()[None, :]] = -1.0

    best = np.argmax(sim, axis=1)
    mutual = np.argmax(sim, axis=0)[best] == np.arange(len(a_ids))

    # 팀 일치 — 지문에 쓰지 않은 정보라 독립적인 검증 겸 오매칭 차단이 된다.
    at = tr.groupby(["pitcher_id", "season"]).pitcher_team_id.agg(
        lambda s: s.mode().iloc[0]).rename("t").reset_index()
    bt_ = tk.groupby(["pitcher_trackman_id", "season"]).pitcher_team.agg(
        lambda s: s.mode().iloc[0]).rename("m").reset_index()
    pair = pd.DataFrame({"pitcher_id": a_ids, "pitcher_trackman_id": b_ids[best],
                         "sim": sim[np.arange(len(a_ids)), best], "mutual": mutual,
                         "n_pitch": (A != 0).sum(1)})
    j = pair.merge(at, on="pitcher_id").merge(bt_, on=["pitcher_trackman_id", "season"])
    seed = j[j.pitcher_id.isin(pair.pitcher_id[pair.mutual])]
    consensus = set(map(tuple, seed.groupby(["t", "m"]).size().reset_index(name="n")
                        .query("n >= 3")[["t", "m"]].to_numpy()))
    j["ok"] = [(t, m) in consensus for t, m in zip(j.t, j.m)]
    pair = pair.merge(j.groupby("pitcher_id").ok.mean().rename("team_agree"),
                      on="pitcher_id", how="left")
    keep = pair.mutual & (pair.team_agree.fillna(0) >= 0.5)
    return pair.loc[keep, ["pitcher_id", "pitcher_trackman_id"]].reset_index(drop=True)


# =======================
# 2) 시즌별 as-of 프로필
# =======================

def build_profiles(tm, pmap, target_seasons):
    """(pitcher_id, season) -> 그 시즌 **이전** Trackman 로그의 요약값.

    분산은 반드시 **구종군 내부**에서 잰다. 패스트볼과 커브를 섞는 투수는
    설계상 무브먼트 폭이 크지, 제구가 나쁜 게 아니다. 구종 내 분산을 투구 수로
    가중평균해야 "레퍼토리"와 "반복성"이 분리된다.
    """
    tk = tm[first_team_mask(tm)].merge(pmap, on="pitcher_trackman_id", how="inner")
    key = ["pitcher_id", "season", "pitch_type_group"]
    sq = tk[METRICS].pow(2).rename(columns=lambda c: c + "__sq")
    tk = pd.concat([tk[key + METRICS], sq], axis=1)

    n = tk.groupby(key)[METRICS].count().rename(columns=lambda c: c + "__n")
    s = tk.groupby(key)[METRICS + [m + "__sq" for m in METRICS]].sum()
    cell = pd.concat([n, s], axis=1).reset_index()

    out = []
    for y in sorted(target_seasons):
        past = cell[cell.season < y]
        if past.empty:
            continue
        # (pitcher, pitch_type) 로 시즌을 합산 -> 구종 내부 평균/분산
        g = past.groupby(["pitcher_id", "pitch_type_group"]).sum(numeric_only=True)
        rec = {}
        for m in METRICS:
            cnt = g[m + "__n"].to_numpy(dtype="float64")
            tot = g[m].to_numpy(dtype="float64")
            totsq = g[m + "__sq"].to_numpy(dtype="float64")
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
                var = np.where(cnt > 1, totsq / np.maximum(cnt, 1) - mean ** 2, np.nan)
            rec[m + "_mean"] = mean
            rec[m + "_std"] = np.sqrt(np.clip(var, 0, None))
            rec[m + "_n"] = cnt
        pt = pd.DataFrame(rec, index=g.index).reset_index()

        rows = []
        for m in METRICS:
            w = pt[m + "_n"].fillna(0)
            d = pt[["pitcher_id"]].assign(_w=w, _mu=pt[m + "_mean"] * w,
                                          _sd=pt[m + "_std"] * w)
            a = d.groupby("pitcher_id")[["_w", "_mu", "_sd"]].sum()
            rows.append(pd.DataFrame({
                f"tm_{SHORT[m]}_mean": a._mu / a._w.replace(0, np.nan),
                # 구종 내부 표준편차의 투구수 가중평균 = 반복성 지표
                f"tm_{SHORT[m]}_wstd": a._sd / a._w.replace(0, np.nan)}))
        prof = pd.concat(rows, axis=1)
        prof["tm_n"] = pt.groupby("pitcher_id")[METRICS[0] + "_n"].sum()
        prof["tm_n_types"] = pt.groupby("pitcher_id").pitch_type_group.nunique()
        # 릴리스 산포를 하나로 — 팔 위치가 얼마나 반복되는가
        prof["tm_release_spread"] = np.hypot(prof.tm_relh_wstd, prof.tm_rels_wstd)
        prof["season"] = y
        out.append(prof.reset_index())

    if not out:
        return pd.DataFrame(columns=["pitcher_id", "season"])
    return pd.concat(out, ignore_index=True)


# =======================
# 3) 백테스트 빌더
# =======================

def make_builder(tm, base_builder):
    """baseline 피처 + Trackman 프로필을 붙인 빌더를 만든다."""
    def builder(df, train_seasons):
        X, cols, cat_cols = base_builder(df, train_seasons)
        pmap = build_pitcher_map(df, tm, train_seasons)
        prof = build_profiles(tm, pmap, sorted(df.season.unique()))
        joined = df[["pitcher_id", "season"]].merge(prof, on=["pitcher_id", "season"],
                                                    how="left")
        add = [c for c in prof.columns if c not in ("pitcher_id", "season")]
        X = pd.concat([X.reset_index(drop=True),
                       joined[add].reset_index(drop=True).astype("float32")], axis=1)
        return X, cols + add, cat_cols
    return builder
