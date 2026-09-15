"""Row-independent residual corrections: platoon, count, count-leverage.

세 표 모두 학습(<=2024) 잔차에서만 만들어져 추론 시 읽기 전용이다. 각 행은
자기 행의 `pitcher_id`, `batter_hand`, `balls_before`, `strikes_before` 로만
조회하므로 평가 데이터의 다른 행에 의존하지 않는다 — season_form/batter_form
과 동일한 구조다.

표에 없는 투수·키는 보정 0 = base 예측 그대로. game_type != R 인 행도 보정 0.
"""
import json
import os

import numpy as np
import pandas as pd

SEGMENT = "R"


def _key(name, df):
    b = df["balls_before"].to_numpy(dtype="int64")
    s = df["strikes_before"].to_numpy(dtype="int64")
    if name == "batter_hand":
        return df["batter_hand"].to_numpy(dtype="int64")
    if name == "count":
        return b * 3 + s
    if name == "leverage":
        # 투수 유리(0) / 중립(1) / 불리(2)
        return np.where(s > b, 0, np.where(b > s + 1, 2, 1))
    raise ValueError(f"unknown key: {name}")


def apply_corrections(preds, test, model_dir, verbose=True):
    path = os.path.join(model_dir, "corrections.json")
    if not os.path.exists(path):
        if verbose:
            print("corrections: corrections.json 없음 — 보정 생략")
        return np.asarray(preds, dtype="float64")
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)

    p = np.asarray(preds, dtype="float64").copy()
    seg = test["game_type"].eq(SEGMENT).to_numpy()
    pid = test["pitcher_id"].to_numpy(dtype="int64")

    for item in spec["items"]:
        table = item["table"]
        keys = _key(item["key"], test)
        if item["kind"] == "interaction":
            lookup = [f"{a}|{b}" for a, b in zip(pid, keys)]
        else:
            lookup = [str(k) for k in keys]
        adj = np.fromiter((table.get(q, 0.0) for q in lookup),
                          dtype="float64", count=len(lookup))
        adj = np.where(seg, adj, 0.0)
        before = p
        p = np.clip(p + adj, 0.0, 1.0)
        if verbose:
            hit = int(np.count_nonzero(adj))
            print(f"corrections {item['name']}: key={item['key']} "
                  f"table={len(table)} active {hit}/{len(adj)} "
                  f"mean(adj) {adj.mean():+.6f} "
                  f"max|adj| {np.abs(adj).max():.6f} "
                  f"-> mean pred {p.mean():.4f}")
    return p
