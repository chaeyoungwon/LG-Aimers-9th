"""NN 멤버(nn·team_nn)를 시즌 상태 피처로 재학습한다.

    python scripts/train_nn_state.py <base.zip> <out_dir>

NN 은 `.npz`(가중치 배열) + meta 안의 `prep`/`vocab`/`arch` 로 이루어진다. 피처를
바꾸면 그 메타까지 정확히 다시 만들어야 하며, 하나라도 어긋나면 조용히 틀린 예측이
나온다. `src/nn_embed.py` 의 `TabularPrep`/`IdVocab`/`prep_meta`/`state_dict_arrays`
가 배포본이 읽는 것과 같은 형식을 만든다.

`runtime.batch_size` 는 **원본 값을 그대로 유지한다.** 배포본은 float32 행렬곱의
배치 의존성을 없애려고 고정 크기 배치로 0-패딩하는데, 그래서 배치 크기가 값의
일부다 (nn 16384, team_nn 256).

pickle 을 만들지 않으므로 numpy 버전 문제와 무관하다.
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import season_state as ss  # noqa: E402
from src.champion import Champion, _torch_device  # noqa: E402
from src.nn_embed import (IdVocab, TabularPrep, prep_meta,  # noqa: E402
                          state_dict_arrays, train_embed_nn)

SEEDS = {"nn": 42, "team_nn": 43}


def main():
    base, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    y = df["control_success"].to_numpy(dtype="float32")
    print(f"train {df.shape}", flush=True)

    STATE = ss.add_features_by_season(df)
    ch = Champion(base)
    dev = _torch_device()
    print(f"device={dev}", flush=True)

    specs = {}
    for name, seed in SEEDS.items():
        mem = ch.members[name]
        cols = [c for c in mem["feature_columns"]
                if c not in ("pitcher_id", "batter_id")]
        X = pd.concat([ch.frame(df, name)[cols], STATE], axis=1)

        prep = TabularPrep().fit(X)
        pv = IdVocab.fit(df["pitcher_id"])
        bv = IdVocab.fit(df["batter_id"])
        Xn = prep.transform(X)
        pid, bid = pv.transform(df.pitcher_id), bv.transform(df.batter_id)

        p = dict(mem["model_params"])
        p["hidden"] = tuple(p["hidden"])
        p["seed"] = seed
        model, info = train_embed_nn(Xn, pid, bid, y, n_pitcher=pv.size,
                                     n_batter=bv.size, device=dev,
                                     verbose=False, **p)
        np.savez(outdir / f"{name}_model.npz", **state_dict_arrays(model))

        old_rt = mem["nn"]["runtime"]          # batch_size 는 값의 일부다
        specs[name] = {
            "feature_columns": list(X.columns) + ["pitcher_id", "batter_id"],
            "nn": {
                "prep": prep_meta(prep),
                "vocab": {"pitcher_col": "pitcher_id", "batter_col": "batter_id",
                          "pitcher_ids": [int(v) for v in pv.ids_],
                          "batter_ids": [int(v) for v in bv.ids_]},
                "arch": {"n_features": prep.n_features, "n_pitcher": pv.size,
                         "n_batter": bv.size, "emb_dim": p["emb_dim"],
                         "hidden": list(p["hidden"]), "dropout": p["dropout"]},
                "runtime": old_rt,
            },
        }
        print(f"  {name}: {X.shape[1]} raw -> {prep.n_features} prepped, "
              f"best_epoch={info.get('best_epoch')}, seed={seed}", flush=True)

    (outdir / "nn_specs.json").write_text(
        json.dumps(specs, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    print(f"saved -> {outdir}")


if __name__ == "__main__":
    main()
