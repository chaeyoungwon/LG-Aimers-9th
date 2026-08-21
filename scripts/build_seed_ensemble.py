"""트리 멤버(hgb·cat·team)에 시드 복제를 얹어 분산을 줄인다.

**왜 이것만 이득이 확실한가.** Brier 는 볼록 손실이라 앙상블 오차 = 평균 개별
오차 − 평균 불일치도 가 항상 성립한다. 튜닝한 값이 아니라 구조적 보장이므로
LOFO 검증이 필요 없는 유일한 항목이다.

**왜 트리만인가.** 롤링 백테스트 실측 (단일 시드 기대값 702.4 기준):
    트리만 (hgb·cat·team)  706.3   +3.8
    NN만  (nn·team_nn)     703.4   +1.0
    전체 5멤버              704.8   +2.4
트리만 평균하는 쪽이 더 좋고, NN 은 `prep`/`vocab` 메타데이터가 가중치와
결합돼 있어 교체 위험도 크다. 두 이유가 같은 방향을 가리킨다.

**설계.** 배포본 런타임은 이미 멤버당 여러 모델을 평균한다(`model_files` 리스트
-> `_mean_proba`). 그래서 **원본 모델을 그대로 두고 시드 변형 2개씩만 추가**한다.
검증된 원본이 항상 평균에 포함되므로 되돌릴 것이 없다.

멤버별 시드 분산 (예측 표준편차 대비): hgb 11.7% · cat 9.0% · team 5.0%.

**hgb 는 제외한다 — 실제 제출 실패로 배운 것.**
hgb 는 joblib/numpy **pickle** 로 저장되는데, fitted 모델이 `_feature_subsample_rng`
(numpy Generator) 를 품는다. 로컬 numpy 2.x 로 pickle 하면 BitGenerator 가 클래스로
직렬화되고, 평가 서버의 numpy 1.26.4 는 문자열을 기대해서
`ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module`
로 죽는다. 같은 numpy 로 저장하고 읽는 로컬 리허설은 이걸 절대 못 잡는다.

cat(`.cbm` native) 과 team(`model_to_string` 텍스트) 은 pickle 이 아니라서 안전하다.
그리고 hgb 를 빼도 이득이 거의 안 준다 — 실측 +3.8 -> +3.5.

**가드 2개**로 재발을 막는다: (1) 직렬화에 관여하는 라이브러리 버전을
`requirements.txt` 와 대조, (2) 새 `.pkl` 이 생기면 중단.
"""
import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


import lightgbm as lgb
import numpy as np
import pandas as pd

from src.champion import Champion
from src.train_base import (PARAMS, add_features, apply_category_maps,
                            build_category_maps)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "submit_corrections.zip"
OUT = ROOT / "artifacts" / "submit_seed_ensemble_v2.zip"
TRAIN = ROOT / "data" / "train.csv"
EXTRA = [101, 202]          # 원본에 더할 시드 2개
TEAM_OFFSETS = [100, 200]   # team 4종은 각자의 시드에 오프셋을 더한다
POS = {"asof_pitcher_success_rate", "asof_pitcher_strike_rate",
       "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
       "asof_pitcher_prev5_game_success_rate", "asof_batter_success_rate"}
NEG = {"asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
       "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
       "asof_pitcher_prev5_game_middle_rate", "asof_batter_middle_rate"}


def check_versions(work):
    """직렬화에 관여하는 라이브러리가 평가 서버와 같은 버전인지 확인한다.

    numpy 는 제외한다 — pickle 을 만들지 않는 한 산출물 형식에 관여하지 않고,
    1.26.4 는 Python 3.13 에 설치되지 않는다.
    """
    import importlib.metadata as md
    want = {}
    for line in (work / "requirements.txt").read_text().splitlines():
        if "==" in line:
            k, v = line.strip().split("==")
            want[k] = v
    bad = []
    for pkg in ("lightgbm", "catboost", "pandas", "scikit-learn"):
        if pkg not in want:
            continue
        got = md.version(pkg)
        print(f"  {pkg:<14} 요구 {want[pkg]:<9} 로컬 {got:<9} "
              f"{'ok' if got == want[pkg] else '불일치'}")
        if got != want[pkg]:
            bad.append(f"{pkg}: 요구 {want[pkg]}, 로컬 {got}")
    if bad:
        raise SystemExit("라이브러리 버전 불일치 — 산출물이 평가 서버에서 로드되지 "
                         "않을 수 있다:\n  " + "\n  ".join(bad))


def train_cat(ch, df, y, seed, path):
    from catboost import CatBoostClassifier, Pool
    spec = ch.members["cat"]
    X = ch.frame(df, "cat")
    na = spec.get("na_token", "__NA__")
    cats = spec["cat_features"]
    Xs = X.copy()
    for c in cats:
        # 런타임과 **같은 토큰**으로 결측 범주를 채운다. "nan" 으로 두면 평가
        # 시점의 미지 레벨이 학습 때 없던 범주가 되어 조용히 어긋난다.
        Xs[c] = Xs[c].astype("object").where(Xs[c].notna(), na).astype(str)
    idx = [Xs.columns.get_loc(c) for c in cats]
    m = CatBoostClassifier(random_seed=seed, verbose=False, thread_count=6,
                           allow_writing_files=False, **spec["model_params"])
    m.fit(Pool(Xs, y, cat_features=idx))
    m.save_model(str(path))
    return float(m.predict_proba(Pool(Xs.head(2000), cat_features=idx))[:, 1].mean())


def train_team(ch, df, y, offset, path):
    spec = ch.members["team"]["team"]
    cfgs = ch.meta["model_params"]["team"]
    cols = ch.members["team"]["feature_columns"]
    cats = spec["cat_cols"]
    d = add_features(df)
    maps = build_category_maps(d, cats)
    X = apply_category_maps(d, cats, maps)[cols]
    season = df.season.astype(int).to_numpy()
    nxt = int(season.max()) + 1

    boosters = {}
    for nm, c in cfgs.items():
        par = dict(PARAMS, seed=c["seed"] + offset, num_leaves=c["num_leaves"],
                   min_data_in_leaf=c["min_data_in_leaf"], num_threads=6)
        if c.get("monotone"):
            par["monotone_constraints"] = [
                1 if f in POS else -1 if f in NEG else 0 for f in cols]
            par["monotone_constraints_method"] = "advanced"
        w = c["season_decay"] ** (nxt - season) if c.get("season_decay") else None
        ds = lgb.Dataset(X, label=y, weight=w, categorical_feature=cats,
                         free_raw_data=False)
        boosters[nm] = lgb.train(par, ds, num_boost_round=c["rounds"]).model_to_string()
    path.write_text(json.dumps({"boosters": boosters, "order": list(cfgs)}),
                    encoding="utf-8")
    return len(boosters)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not BASE.exists():
        raise FileNotFoundError(f"{BASE} — scripts/build_corrections.py 를 먼저 실행")

    df = pd.read_csv(TRAIN, encoding="utf-8-sig")
    y = df["control_success"].to_numpy(dtype="float64")
    print(f"train {df.shape}", flush=True)

    with tempfile.TemporaryDirectory(prefix="seedens_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(BASE) as z:
            z.extractall(work)
        print("라이브러리 버전 대조:")
        check_versions(work)
        before_pkl = {p.name for p in (work / "model").glob("*.pkl")}

        ch = Champion(BASE, workdir=str(work / "_spec"))
        meta = json.loads((work / "model" / "meta.json").read_text("utf-8"))
        members = {m["name"]: m for m in meta["members"]}

        for i, seed in enumerate(EXTRA, start=1):
            f = f"cat_model_s{i}.cbm"
            mp = train_cat(ch, df, y, seed, work / "model" / f)
            members["cat"]["model_files"].append(f)
            print(f"  cat seed={seed} -> {f}  (probe mean {mp:.4f})", flush=True)

        for i, off in enumerate(TEAM_OFFSETS, start=1):
            f = f"team_model_s{i}.lgbmstack.json"
            n = train_team(ch, df, y, off, work / "model" / f)
            members["team"]["model_files"].append(f)
            print(f"  team offset={off} -> {f}  ({n} boosters)", flush=True)

        meta["seed_ensemble"] = {
            "members": ["cat", "team"],
            "extra_seeds": EXTRA, "team_offsets": TEAM_OFFSETS,
            "rationale": ("Brier is convex: ensemble error = mean individual error "
                          "- mean disagreement. Structural, not tuned."),
            "backtest": ("R-segment mean over 3 folds: single-seed expectation 702.4, "
                         "cat+team 705.9 (+3.5); with hgb 706.3 (+3.8)"),
            "hgb_excluded": ("joblib/numpy pickle carries a numpy Generator "
                             "(_feature_subsample_rng); local numpy 2.x pickles it in a "
                             "form the eval server's numpy 1.26.4 cannot read. "
                             "Gain from including it is only +0.3."),
        }
        (work / "model" / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        shutil.rmtree(work / "_spec", ignore_errors=True)

        after_pkl = {p.name for p in (work / "model").glob("*.pkl")}
        if after_pkl != before_pkl:
            raise SystemExit(f"새 pickle 이 생겼다 — 평가 서버 numpy 와 형식이 "
                             f"어긋날 수 있다: {sorted(after_pkl - before_pkl)}")
        print(f"  pickle 파일 변화 없음 ✓ ({sorted(before_pkl)})")
        for name in ("cat", "team"):
            print(f"  {name}.model_files = {members[name]['model_files']}")

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(work.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(work))
    print(f"built {out}")


if __name__ == "__main__":
    main()
