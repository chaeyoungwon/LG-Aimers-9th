"""Rolling-origin backtest — 변경안을 제출 전에 판정하는 측정 도구.

**왜 필요한가.** 지금까지 14번 제출 중 5번이 "로컬 개선 → 리더보드 하락"이었다.
단일 2024 홀드아웃은 *한 번의* 시즌 전이만 보므로, 그 한 번에 우연히 맞으면
채택되어 버린다. 여기서는 시즌 전이를 세 번 재현하고 **세 폴드 전부**에서
개선될 때만 채택한다. 평가가 2024 -> 2025 한 걸음이므로, 각 폴드도 정확히
한 걸음(train <= Y-1, valid = Y)으로 맞춘다.

**규칙 준수 (대회 규정 5절).** 피처 빌더는 `train_seasons`를 인자로 받으며
범주형 매핑·상수표·룩업테이블은 **그 시즌들에서만** 만들어야 한다. 검증 시즌의
행은 물론이고, 검증 시즌의 분포·빈도도 학습에 쓰이지 않는다. 캘리브레이션
상수 역시 검증 시즌이 아니라 그 이전 시즌에서 적합한다 (아래 참고).

**캘리브레이션 체인.** 폴드 Y의 캘리브레이션을 폴드 Y에서 적합하면 그 폴드의
점수가 낙관적으로 부풀려진다. 그래서 시즌 Y-1에 대한 예측(= 시즌 <= Y-2로
학습한 모델의 산출물)에서 적합한다. 이는 앞 폴드의 결과를 그대로 재사용하는
것이라 추가 학습 비용이 없다.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.train_base import (CAT_COLS, MODEL_CONFIGS, PARAMS, TARGET_COL,
                            add_features, apply_category_maps,
                            build_category_maps)

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
EVAL_SEASONS = [2022, 2023, 2024]   # 실제로 채점하는 폴드
SEED_SEASON = 2021                  # 2022 폴드의 캘리브레이션을 위해서만 예측

# 빠른 스크리닝용 — 챔피언 앙상블에서 가중치가 가장 큰 단일 모델(47%)
SCREEN_CONFIGS = [c for c in MODEL_CONFIGS if c["name"] == "leaf63"]


# =======================
# 피처 빌더
# =======================

def baseline_builder(df, train_seasons):
    """현행 챔피언과 동일한 행 지역(row-local) 피처.

    범주형 매핑은 `train_seasons` 행에서만 만든다.
    """
    df = add_features(df)
    feature_cols = [c for c in df.columns
                    if c not in ("row_id", TARGET_COL)]
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    maps = build_category_maps(df[df.season.isin(train_seasons)], cat_cols)
    return apply_category_maps(df, cat_cols, maps)[feature_cols], feature_cols, cat_cols


# =======================
# 학습 / 예측
# =======================

def _fit_predict(X, y, seasons, train_seasons, target_season, cat_cols, configs):
    """train_seasons로 학습해 target_season 행의 확률을 반환 (캘리브레이션 전)."""
    tr = seasons.isin(train_seasons).to_numpy()
    te = (seasons == target_season).to_numpy()
    preds, weights = [], []
    for cfg in configs:
        params = dict(PARAMS, seed=cfg["seed"], num_leaves=cfg["num_leaves"],
                      min_data_in_leaf=cfg["min_data_in_leaf"],
                      objective=cfg["objective"], num_threads=6,
                      metric="l2" if cfg["objective"] == "regression" else "binary_logloss")
        w = None
        if cfg["season_decay"] is not None:
            # 감쇠 기준점은 그 폴드가 예측하는 시즌이다.
            w = cfg["season_decay"] ** (target_season - seasons[tr].to_numpy())
        ds = lgb.Dataset(X[tr], label=y[tr], weight=w,
                         categorical_feature=cat_cols, free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=cfg["rounds"])
        preds.append(booster.predict(X[te]))
        weights.append(cfg["weight"])
    return np.average(preds, axis=0, weights=weights)


def _bss(y, p):
    """BSS — train_base와 같은 정의지만 **0에서 클램핑하지 않는다**.

    백테스트에서는 악화도 그대로 보여야 한다. 클램핑하면 "기준선보다 나쁨"이
    전부 0으로 뭉개져 두 안을 비교할 수 없다.
    """
    brier = float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))
    r = float(np.mean(y))
    return 100000 * (1 - brier / (r * (1 - r))), brier


def _fit_calibration(p, y):
    """Brier를 최소화하는 logit 아핀 보정 (scale, bias)을 격자+정밀 탐색으로 적합."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))

    def brier(a, b):
        return np.mean((1 / (1 + np.exp(-(a * z + b))) - y) ** 2)

    lo, hi = 0.20, 2.00
    best = (1.0, 0.0, brier(1.0, 0.0))
    for a in np.arange(lo, hi + 1e-9, 0.05):
        for b in np.arange(-0.30, 0.301, 0.025):
            s = brier(a, b)
            if s < best[2]:
                best = (a, b, s)
    a0, b0 = best[0], best[1]
    for a in np.arange(a0 - 0.05, a0 + 0.051, 0.005):
        for b in np.arange(b0 - 0.025, b0 + 0.0251, 0.0025):
            s = brier(a, b)
            if s < best[2]:
                best = (a, b, s)
    if not (lo + 0.06 < best[0] < hi - 0.06):
        print(f"    ! calibration scale {best[0]:.3f} hit the grid edge "
              f"— 모델이 덜 학습됐거나 격자가 좁다", flush=True)
    return float(best[0]), float(best[1])


def _logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _apply_calibration(p, a, b):
    return _sigmoid(a * _logit(p) + b)


def extrapolate_rate(rates, target):
    """시즌 < target 의 기저율만으로 target 시즌 기저율을 외삽한다.

    기저율이 매년 단조 하락하므로(2019 .565 -> 2024 .486) "직전 시즌 값"을
    그대로 쓰면 매번 높게 잡는다. 직전 값 + 과거 연차 변화량의 평균을 쓴다.
    **target 시즌의 행은 쓰지 않는다.**
    """
    prior = sorted(s for s in rates if s < target)
    if not prior:
        return None
    last = rates[prior[-1]]
    deltas = [rates[b] - rates[a] for a, b in zip(prior, prior[1:])]
    return float(last + (np.mean(deltas) if deltas else 0.0))


def _solve_shift(z, target_mean):
    """mean(sigmoid(z + d)) == target_mean 이 되는 상수 d를 이분 탐색으로 찾는다."""
    lo, hi = -3.0, 3.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _sigmoid(z + mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =======================
# 백테스트 본체
# =======================

def fit_raw(builder, name, configs=None, df=None, out_dir=None, verbose=True):
    """각 폴드의 **캘리브레이션 전** 예측을 만든다 (학습은 여기서 한 번만).

    캘리브레이션 변형은 재학습 없이 `evaluate`에서 갈아끼운다.
    """
    configs = configs or SCREEN_CONFIGS
    if df is None:
        df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    seasons = df["season"].astype(int)
    y = df[TARGET_COL].to_numpy(dtype="float64")

    raw = {}
    for target in [SEED_SEASON] + EVAL_SEASONS:
        train_seasons = [s for s in SEASONS if s < target]
        # 빌더는 자기가 학습해도 되는 시즌만 본다 — 규칙 준수의 강제 지점.
        X, feature_cols, cat_cols = builder(df, train_seasons)
        if verbose:
            print(f"  [{name}] fold -> {target}  train={train_seasons}  "
                  f"n_feat={len(feature_cols)}", flush=True)
        raw[target] = _fit_predict(X, y, seasons, train_seasons, target,
                                   cat_cols, configs)

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{name}_raw.npz", **{str(t): raw[t] for t in raw})
    return raw


def evaluate(raw, df, by_game_type=False, drift=False, name="", verbose=True):
    """캘리브레이션을 적용해 폴드별 BSS를 낸다.

    `by_game_type` — R과 F에 각각 다른 (scale, bias)를 적합한다. 두 구간은
        기저율도 동역학도 다르다(F는 2022 .709 -> 2023 .473로 붕괴). 각 행은
        **자기 행의 game_type** 으로 상수를 고르므로 행 독립이다.
    `drift`      — 기저율의 연차 하락 추세를 외삽해 절편을 미리 내린다.
        이동량은 시즌 <= Y-1 행에서만 계산한 상수 하나다.
    """
    seasons = df["season"].astype(int).to_numpy()
    gt = df["game_type"].to_numpy()
    y = df[TARGET_COL].to_numpy(dtype="float64")

    rows = []
    for target in EVAL_SEASONS:
        prev = target - 1
        mp, mt = seasons == prev, seasons == target
        p = np.empty(int(mt.sum()))
        groups = sorted(set(gt[mp])) if by_game_type else [None]
        for g in groups:
            sp = mp if g is None else (mp & (gt == g))
            st = mt if g is None else (mt & (gt == g))
            a, b = _fit_calibration(raw[prev][gt[mp] == g] if g else raw[prev],
                                    y[sp])
            zt = a * _logit(raw[target][gt[mt] == g] if g else raw[target]) + b
            if drift:
                # 기저율 시계열은 시즌 < target 에서만 만든다.
                rates = {s: y[(seasons == s) & ((gt == g) if g else True)].mean()
                         for s in SEASONS if s < target}
                rhat = extrapolate_rate(rates, target)
                zp = a * _logit(raw[prev][gt[mp] == g] if g else raw[prev]) + b
                zt = zt + _solve_shift(zp, rhat)
            p[(gt[mt] == g) if g else slice(None)] = _sigmoid(zt)
        bss, brier = _bss(y[mt], p)
        row = {"season": target, "bss": bss, "brier": brier, "n": int(mt.sum())}
        for g in ("R", "F"):
            sub = gt[mt] == g
            if sub.any():
                row[f"bss_{g}"] = _bss(y[mt][sub], p[sub])[0]
        rows.append(row)
        if verbose:
            print(f"  [{name}] {target}: BSS={bss:8.1f}  "
                  f"R={row.get('bss_R', float('nan')):8.1f}  "
                  f"F={row.get('bss_F', float('nan')):9.1f}", flush=True)

    res = pd.DataFrame(rows)
    if verbose:
        print(f"  [{name}] MEAN={res.bss.mean():8.1f}  min={res.bss.min():8.1f}"
              f"  |  R mean={res.bss_R.mean():7.1f}", flush=True)
    return res


def run_backtest(builder, name, configs=None, df=None, out_dir=None, verbose=True,
                 **eval_kw):
    raw = fit_raw(builder, name, configs, df, out_dir, verbose)
    if df is None:
        df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    res = evaluate(raw, df, name=name, verbose=verbose, **eval_kw)
    if out_dir:
        res.to_csv(Path(out_dir) / f"{name}_folds.csv", index=False)
    return res


def compare(res_a, res_b, name_a="A", name_b="B", preds_a=None, preds_b=None,
            baseline=0.2498):
    """두 결과를 폴드별로 비교하고, 예측 차이로부터 노이즈 바닥을 추정한다.

    같은 평가셋 위의 두 예측 p_A, p_B에 대해 Brier 차이의 표준오차는
    delta = p_A - p_B 로부터 SE ~= sqrt(sum(delta^2)) / n 로 근사된다
    (y in {0,1}, p ~ 0.5 이므로 (p_A + p_B - 2y)^2 ~= 1). 이 값을 넘지 못하는
    차이는 우연과 구분되지 않으므로 제출 대상이 아니다.
    """
    m = res_a.merge(res_b, on="season", suffixes=("_a", "_b"))
    m["delta"] = m.bss_b - m.bss_a
    print(f"\n{'season':>8} {name_a:>12} {name_b:>12} {'delta':>10} {'~2sigma':>10}")
    for _, r in m.iterrows():
        se = np.nan
        if preds_a is not None and preds_b is not None:
            s = str(int(r.season))
            d = np.asarray(preds_b[s]) - np.asarray(preds_a[s])
            se = 100000 * np.sqrt(np.sum(d ** 2)) / (len(d) * baseline)
        print(f"{int(r.season):>8} {r.bss_a:>12.2f} {r.bss_b:>12.2f} "
              f"{r.delta:>+10.2f} {2*se:>10.2f}")
    print(f"{'MEAN':>8} {m.bss_a.mean():>12.2f} {m.bss_b.mean():>12.2f} "
          f"{m.delta.mean():>+10.2f}")
    verdict = "ACCEPT" if (m.delta > 0).all() else "REJECT"
    print(f"\n판정: {verdict}  (세 폴드 전부 개선이어야 채택)")
    return m


if __name__ == "__main__":
    import sys
    df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    print(f"train {df.shape}")
    res = run_backtest(baseline_builder, "baseline", df=df,
                       out_dir="artifacts/backtest")
    print(res.to_string(index=False))
