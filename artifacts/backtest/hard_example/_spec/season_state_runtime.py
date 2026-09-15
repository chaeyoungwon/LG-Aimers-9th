"""각 평가 행에서 그 선수의 **당해 시즌 현재 상태**를 복원해 피처로 붙인다.

`asof_*` 는 커리어 누적이고 `rate x n` 이 정확히 정수 카운트다. 학습 구간 말의
누적값 `(n0, c0)` 를 상수로 얼려두면

    n_season = asof_n - n0
    cur_X    = (round(asof_n * rate_X) - c0_X) / n_season

가 그 행의 값만으로 당해 시즌 상태를 준다. `season_form` 이 `success_rate` 하나에
쓰던 산술을 10개 컬럼으로 넓힌 것이다.

**행 단위 독립.** `(n0, c0)` 는 model/season_state.json 에 박힌 학습 데이터 유래
상수이고, 각 행은 자기 `pitcher_id` / `batter_id` / `asof_*` 로만 조회한다.
같은 배치에 무엇이 들어왔는지에 의존하지 않는다.

표에 없는 선수, 또는 당해 시즌 표본이 없는 행(`n_season <= 0`)은 결측으로 둔다 —
트리가 학습된 결측 분기로 보내므로 cold-start 행이 막히지 않는다.
"""
import json
import os

import numpy as np
import pandas as pd


def add_state(test, model_dir, prefix="cur_"):
    path = os.path.join(model_dir, "season_state.json")
    if not os.path.exists(path):
        return test
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)

    out = test.copy()
    for g in spec["groups"]:
        idcol, ncol, rates = g["id_col"], g["n_col"], g["rates"]
        ids = np.asarray(g["ids"], dtype="int64")          # 저장 시 정렬해 둔다
        n0t = np.asarray(g["n0"], dtype="float64")

        key = pd.to_numeric(out[idcol], errors="coerce").to_numpy(dtype="float64")
        miss = ~np.isfinite(key)
        k = np.where(miss, -1, key).astype("int64")
        pos = np.searchsorted(ids, k)
        clip = np.clip(pos, 0, max(len(ids) - 1, 0))
        seen = (pos < len(ids)) & (ids[clip] == k) & ~miss if len(ids) else \
            np.zeros(len(out), dtype=bool)

        n = pd.to_numeric(out[ncol], errors="coerce").to_numpy(dtype="float64")
        n0 = np.where(seen, n0t[clip], np.nan)
        n_season = n - n0
        valid = seen & np.isfinite(n_season) & (n_season > 0)
        out[f"{prefix}{g['name']}_n"] = np.where(valid, n_season, np.nan)

        for short, col in rates.items():
            r = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype="float64")
            c = np.rint(n * np.where(np.isfinite(r), r, 0.0))
            c0 = np.where(seen, np.asarray(g["c0"][short], dtype="float64")[clip],
                          np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = (c - c0) / n_season
            out[f"{prefix}{g['name']}_{short}"] = np.where(
                valid & np.isfinite(v), v, np.nan)
    return out
