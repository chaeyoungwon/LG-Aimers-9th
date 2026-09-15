"""후보 A 스크리닝 — Trackman 의 **카운트별 구종 구성**에 신호가 있는지 30분 안에 판정한다.

**가설.** 공식 피처의 `asof_pitcher_{fastball,breaking,offspeed}_rate` 는 투수의
**전체** 구종 구성이다. 그런데 실제 배합은 카운트에 따라 크게 달라지고(3-0 직구,
0-2 유인구), 구종마다 제구 난이도가 다르다. Trackman 로그에는 `balls_before` /
`strikes_before` 가 그대로 들어 있으므로 `(투수, 카운트) -> 구종 분포` 를 만들 수
있다. 이건 `asof_*` 에 **없는** 정보다.

**측정하는 것.** 모델을 새로 학습하지 않는다. 현재 최고 구성(시즌 상태 피처)의
백테스트 예측 잔차 `y - p` 와, 새 피처의 상관을 본다. 새 피처는 반드시
**전체 구성 대비 편차**로 만든다:

    dev_fb = mix_fb(투수, 카운트) - mix_fb(투수, 전체)

전체 구성은 이미 모델이 알고 있으므로, 편차만이 새 정보다.

**판정 기준.** 기각된 Trackman 물리 피처의 잔차 상관은 전부 |0.016| 이하였고
실제로 백테스트에서 0 이었다. 그러니 **|r| >= 0.02 가 하나라도 나와야** 다음
단계(백테스트 학습 비교)로 갈 가치가 있다. 그 미만이면 여기서 멈춘다.

**규칙 준수.** 시즌 Y 폴드에서는 매핑도 구성표도 Trackman 시즌 < Y 만으로 만든다.
추론 시 각 행은 자기 `pitcher_id` / `balls_before` / `strikes_before` 로 조회할
뿐이라 평가 데이터의 다른 행을 보지 않는다.

    python scripts/probe_trackman_count_mix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import trackman_features as tf  # noqa: E402

FOLD = 2024                       # 잔차가 있는 폴드 중 가장 최근
TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]
GROUPS = ["fastball", "breaking", "offspeed"]
K = 50.0                          # 전체 구성 쪽으로 당기는 축소 상수


def count_key(balls, strikes):
    """볼-스트라이크를 0..11 로. 희소한 칸은 백테스트 단계에서 묶는다."""
    return np.asarray(balls, dtype="int64") * 3 + np.asarray(strikes, dtype="int64")


def build_count_mix(tm, pmap, seasons, k=K):
    """(pitcher_id, count) -> 구종 분포 + 그 투수의 전체 분포. Trackman 시즌 < FOLD 만."""
    tk = tm[tf.first_team_mask(tm) & tm.season.isin(seasons)]
    tk = tk.merge(pmap, on="pitcher_trackman_id", how="inner")
    tk = tk[tk.pitch_type_group.isin(GROUPS)
            & tk.balls_before.between(0, 3) & tk.strikes_before.between(0, 2)]
    tk = tk.assign(_k=count_key(tk.balls_before, tk.strikes_before))

    cnt = (tk.groupby(["pitcher_id", "_k", "pitch_type_group"]).size()
           .unstack("pitch_type_group").reindex(columns=GROUPS).fillna(0.0))
    tot = cnt.sum(axis=1)
    overall = cnt.groupby(level="pitcher_id").sum()
    ov_rate = overall.div(overall.sum(axis=1).replace(0, np.nan), axis=0)

    # 경험적 베이즈 축소 — 표본이 적은 (투수, 카운트) 칸은 전체 구성으로 당긴다.
    prior = ov_rate.reindex(cnt.index.get_level_values("pitcher_id")).to_numpy()
    rate = (cnt.to_numpy() + k * prior) / (tot.to_numpy()[:, None] + k)

    out = pd.DataFrame(rate, index=cnt.index, columns=[f"cm_{g}" for g in GROUPS])
    out["cm_n"] = tot.to_numpy()
    for i, g in enumerate(GROUPS):
        out[f"cm_dev_{g}"] = out[f"cm_{g}"] - prior[:, i]
    return out.reset_index()


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    raw = np.load(ROOT / "artifacts" / "backtest" / "state_raw.npz")
    p = raw[str(FOLD)]
    m = (df.season == FOLD).to_numpy()
    assert m.sum() == len(p), f"{m.sum()} vs {len(p)} — state_raw.npz 를 다시 만들 것"
    print(f"train {df.shape}, fold {FOLD} rows {m.sum():,}", flush=True)

    tm = pd.read_csv(ROOT / "data" / "trackman_history.csv", encoding="utf-8-sig")
    tm.columns = [c.lstrip("﻿") for c in tm.columns]
    print(f"trackman {tm.shape}", flush=True)

    pmap = tf.build_pitcher_map(df, tm, TRAIN_SEASONS)
    print(f"매핑된 투수 {len(pmap):,}명", flush=True)

    mix = build_count_mix(tm, pmap, [s for s in TRAIN_SEASONS])
    print(f"(투수, 카운트) 칸 {len(mix):,}개", flush=True)

    ev = df.loc[m, ["pitcher_id", "balls_before", "strikes_before",
                    "control_success", "game_type"]].reset_index(drop=True)
    ev["_k"] = count_key(ev.balls_before, ev.strikes_before)
    ev = ev.merge(mix, on=["pitcher_id", "_k"], how="left")
    cov = ev.cm_n.notna().mean()
    print(f"커버리지 {cov:.3f}", flush=True)

    resid = ev.control_success.to_numpy(float) - p
    print(f"\n{'피처':>16} {'r(잔차)':>10} {'r(R행)':>10} {'커버':>8}")
    worth = []
    for c in [c for c in ev.columns if c.startswith("cm_")]:
        v = ev[c].to_numpy(float)
        ok = np.isfinite(v)
        r_all = np.corrcoef(v[ok], resid[ok])[0, 1] if ok.sum() > 100 else np.nan
        okr = ok & (ev.game_type.to_numpy() == "R")
        r_r = np.corrcoef(v[okr], resid[okr])[0, 1] if okr.sum() > 100 else np.nan
        print(f"{c:>16} {r_all:>+10.4f} {r_r:>+10.4f} {ok.mean():>8.3f}")
        if max(abs(r_all), abs(r_r)) >= 0.02:
            worth.append(c)

    print(f"\n기각선 |r|>=0.02 를 넘은 피처: {worth or '없음'}")

    # 상관은 선형·주변 효과만 본다. 트리가 쓸 수 있는 비선형 구조가 남아 있는지
    # 5분위 잔차 평균으로 한 번 더 확인한다 — 단조 계단이 보이면 다시 생각한다.
    print("\n[5분위 잔차 평균] 단조 계단이면 트리가 쓸 수 있다")
    for c in ["cm_dev_offspeed", "cm_dev_breaking", "cm_dev_fastball"]:
        v = ev[c].to_numpy(float)
        ok = np.isfinite(v)
        q = pd.qcut(pd.Series(v[ok]), 5, labels=False, duplicates="drop")
        mu = pd.Series(resid[ok]).groupby(q).mean()
        print(f"  {c:>16}: " + "  ".join(f"{x:+.4f}" for x in mu)
              + f"   폭 {mu.max()-mu.min():.4f}")

    # 축소 상수가 편차를 죽였을 가능성 — K 를 낮춰 다시 잰다.
    print("\n[축소 상수 K 민감도] cm_dev_offspeed 의 잔차 상관")
    for k in (5.0, 20.0, 50.0, 200.0):
        mk = build_count_mix(tm, pmap, TRAIN_SEASONS, k=k)
        e2 = ev[["pitcher_id", "_k"]].merge(mk, on=["pitcher_id", "_k"], how="left")
        v = e2.cm_dev_offspeed.to_numpy(float)
        ok = np.isfinite(v)
        print(f"  K={k:>6.0f}: r={np.corrcoef(v[ok], resid[ok])[0,1]:+.4f}")

    # **결정적 확인.** 위 두 표는 카운트를 가로질러 본 것이라 교란된다 — 오프스피드
    # 사용은 2스트라이크에서 늘고, 성공률도 카운트마다 다르며, 카운트는 모델이 이미
    # 안다. 카운트 칸 **안에서** 상관이 살아남아야 새 정보다.
    print("\n[카운트 칸 내부 상관] 여기서 죽으면 위 신호는 카운트 교란이다")
    kk = ev._k.to_numpy()
    v = ev.cm_dev_offspeed.to_numpy(float)
    ok = np.isfinite(v)
    tot_n, acc = 0, 0.0
    for c in range(12):
        sub = ok & (kk == c)
        if sub.sum() < 1000:
            continue
        r = np.corrcoef(v[sub], resid[sub])[0, 1]
        acc += r * sub.sum()
        tot_n += sub.sum()
        print(f"  {c//3}-{c%3}: n={sub.sum():>7,}  r={r:+.4f}")
    print(f"  가중평균 r={acc/tot_n:+.4f}  (칸 밖 전체 +0.0125)")

    print("\n어느 쪽도 기준을 못 넘으면 멈춘다 — 물리 피처(<=0.016)와 같은 결말이다.")


if __name__ == "__main__":
    main()
