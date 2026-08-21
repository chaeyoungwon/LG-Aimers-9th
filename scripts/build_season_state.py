"""시즌 상태 복원 피처를 챔피언에 얹는다 — 이번 라운드 최대 변경.

**아이디어.** `season_form` 은 `asof_pitcher_success_rate` 하나를 시즌 단위로 복원해
LB +37.7 을 얻었다. 그런데 같은 산술이 10개 `asof_*` 컬럼 전부에 성립하고, 그것들은
성공률과 **중복이 아니다** (success + middle+ball+reverse 평균 = 1.2738, 즉 분할이
아님; strike_rate 는 success 와 상관 +0.15 로 거의 직교). middle/ball/reverse 는
문제 정의의 실패 3유형과 1:1 대응한다.

**롤링 백테스트 (R구간, 3시드 평균):**
    현행 (baseline + form)      658.9
    state only                  683.6
    state + form                696.7   <- 2024 폴드는 -30.7 (이중 계산)
    state + corr                703.1   <- 2024 폴드 785.5, 최고
    state + form + corr         716.1

**그래서 season_form 을 끈다.** `cur_p_succ` 가 곧 season_form 의 신호이므로 피처로
들어간 뒤에 덧셈 보정을 또 하면 이중 계산이다. β/γ/δ 보정은 좌우·카운트라는 다른
축이라 유지한다 (+19.6, 세 폴드 모두 >=0).

**적용 범위: hgb·cat·team (블렌드의 86%).** NN 두 멤버는 `prep`/`vocab` 메타데이터가
가중치와 결합돼 있어 피처를 바꾸면 그 메타까지 정확히 재생성해야 한다. 위험 대비
이득이 낮아 원본을 유지한다. 배포본은 멤버마다 다른 `feature_columns` 를 허용한다.

**규칙 준수.** 경계표 `(n0, c0)` 는 train(<=2024) 에서만 만들고, 각 행은 자기
`pitcher_id`/`batter_id`/`asof_*` 로 조회한다. 평가 데이터의 다른 행은 보지 않는다.
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

from src import backtest as bt
from src import corrections as co
from src import season_state as ss
from src.champion import Champion
from src.train_base import (PARAMS, add_features, apply_category_maps,
                            build_category_maps)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "submit_corrections.zip"
OUT = ROOT / "artifacts" / "submit_season_state.zip"
TRAIN = ROOT / "data" / "train.csv"
STATE_RAW = ROOT / "artifacts" / "backtest" / "state_raw.npz"
POS = {"asof_pitcher_success_rate", "asof_pitcher_strike_rate",
       "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
       "asof_pitcher_prev5_game_success_rate", "asof_batter_success_rate"}
NEG = {"asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
       "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
       "asof_pitcher_prev5_game_middle_rate", "asof_batter_middle_rate"}


def check_versions(work):
    import importlib.metadata as md
    want = dict(l.strip().split("==") for l in
                (work / "requirements.txt").read_text().splitlines() if "==" in l)
    bad = []
    for pkg in ("lightgbm", "catboost", "pandas", "scikit-learn"):
        got = md.version(pkg)
        print(f"  {pkg:<14} 요구 {want.get(pkg,'-'):<9} 로컬 {got:<9} "
              f"{'ok' if got == want.get(pkg) else '불일치'}")
        if got != want.get(pkg):
            bad.append(pkg)
    if bad:
        raise SystemExit(f"라이브러리 버전 불일치: {bad}")


def state_spec(df):
    """평가(2025)용 경계표 — train 전체 말의 누적 상태."""
    tables = ss.build_boundary(df, sorted(df.season.unique()))
    groups = []
    for name, idcol, ncol, rates in ss.GROUPS:
        t = tables[name]
        order = np.argsort(t["ids"])
        groups.append({
            "name": name, "id_col": idcol, "n_col": ncol, "rates": rates,
            "ids": t["ids"][order].astype(int).tolist(),
            "n0": t["n0"][order].tolist(),
            "c0": {k: v[order].tolist() for k, v in t["c0"].items()},
        })
    return {"groups": groups, "prefix": "cur_"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)

    df = pd.read_csv(TRAIN, encoding="utf-8-sig")
    y = df["control_success"].to_numpy(dtype="float64")
    print(f"train {df.shape}", flush=True)
    print("시즌 상태 피처 생성 ...", flush=True)
    STATE = ss.add_features_by_season(df)
    NEW = list(STATE.columns)
    print(f"  {len(NEW)}개: {NEW}", flush=True)

    with tempfile.TemporaryDirectory(prefix="sstate_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(BASE) as z:
            z.extractall(work)
        print("라이브러리 버전 대조:")
        check_versions(work)
        before_pkl = {p.name for p in (work / "model").glob("*.pkl")}

        ch = Champion(BASE, workdir=str(work / "_spec"))
        meta = json.loads((work / "model" / "meta.json").read_text("utf-8"))
        members = {m["name"]: m for m in meta["members"]}

        # ---- cat : 배포본 빌더 결과에 상태 피처를 붙여 학습 ------------------
        #
        # hgb 는 제외한다. joblib/numpy **pickle** 로 저장되는데 fitted 모델이
        # numpy Generator(`_feature_subsample_rng`)를 품어서, 로컬 numpy 2.x 로
        # 다시 pickle 하면 평가 서버의 numpy 1.26.4 가 읽지 못한다 (실제 제출
        # 실패로 확인). 덮어쓰기는 "새 pkl" 가드에도 안 걸리므로 아예 건드리지
        # 않는다. hgb 는 블렌드의 16% 이고 원본 파일은 이미 두 번 채점됐다.
        from catboost import CatBoostClassifier, Pool
        spec = members["cat"]
        X = pd.concat([ch.frame(df, "cat"), STATE], axis=1)
        na = spec.get("na_token", "__NA__")
        cats_c = spec["cat_features"]
        Xs = X.copy()
        for c in cats_c:
            Xs[c] = Xs[c].astype("object").where(Xs[c].notna(), na).astype(str)
        idx = [Xs.columns.get_loc(c) for c in cats_c]
        m = CatBoostClassifier(random_seed=42, verbose=False, thread_count=6,
                               allow_writing_files=False, **spec["model_params"])
        m.fit(Pool(Xs, y, cat_features=idx))
        m.save_model(str(work / "model" / "cat_model.cbm"))
        spec["feature_columns"] = list(X.columns)
        print(f"  cat 학습 완료 ({X.shape[1]} features)", flush=True)

        # ---- team : LGBM 4종 ------------------------------------------------
        tspec = members["team"]["team"]
        cfgs = meta["model_params"]["team"]
        cats = tspec["cat_cols"]
        d = add_features(df)
        maps = build_category_maps(d, cats)
        cols = members["team"]["feature_columns"] + NEW
        X = pd.concat([apply_category_maps(d, cats, maps), STATE], axis=1)[cols]
        season = df.season.astype(int).to_numpy()
        nxt = int(season.max()) + 1
        boosters = {}
        for nm, c in cfgs.items():
            par = dict(PARAMS, seed=c["seed"], num_leaves=c["num_leaves"],
                       min_data_in_leaf=c["min_data_in_leaf"], num_threads=6)
            if c.get("monotone"):
                par["monotone_constraints"] = [
                    1 if f in POS else -1 if f in NEG else 0 for f in cols]
                par["monotone_constraints_method"] = "advanced"
            w = c["season_decay"] ** (nxt - season) if c.get("season_decay") else None
            ds = lgb.Dataset(X, label=y, weight=w, categorical_feature=cats,
                             free_raw_data=False)
            boosters[nm] = lgb.train(par, ds, c["rounds"]).model_to_string()
        (work / "model" / "team_model.lgbmstack.json").write_text(
            json.dumps({"boosters": boosters, "order": list(cfgs)}), encoding="utf-8")
        members["team"]["feature_columns"] = cols
        members["team"]["model_files"] = [members["team"]["model_files"][0]]
        # hgb 는 원본 유지 — feature_columns 도 그대로다
        print(f"  team 학습 완료 ({len(cols)} features)", flush=True)

        # ---- 경계표 + 런타임 --------------------------------------------------
        (work / "model" / "season_state.json").write_text(
            json.dumps(state_spec(df), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        (work / "season_state_runtime.py").write_text(
            (ROOT / "scripts" / "season_state_runtime.py").read_text("utf-8"),
            encoding="utf-8")

        # ---- season_form 끄기, batter_form 무력화 ------------------------------
        meta["season_form"] = None
        bf = json.loads((work / "model" / "batter_form.json").read_text("utf-8"))
        bf["alpha"] = 0.0
        bf["disabled_reason"] = ("cur_b_succ 가 피처로 들어가 이중 계산이 된다 "
                                 "(백테스트: state 위에 form 을 얹으면 2024 폴드 -30.7)")
        (work / "model" / "batter_form.json").write_text(
            json.dumps(bf, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

        # ---- β/γ/δ 보정표 재추정 (상태 피처 반영된 참조값으로) -----------------
        z = np.load(STATE_RAW)
        raw = {int(k): z[k] for k in z.files}
        se = df.season.astype(int).to_numpy()
        gt = df.game_type.to_numpy()
        ref = {}
        for t in sorted(raw):
            if t - 1 not in raw:
                continue
            mp = se == t - 1
            a, b = bt._fit_calibration(raw[t - 1][gt[mp] == "R"], y[mp][gt[mp] == "R"])
            ref[t] = bt._sigmoid(a * bt._logit(np.clip(raw[t], 1e-6, 1 - 1e-6)) + b)
        tables = co.build_all(df, ref, sorted(df.season.unique()))
        spec = {"items": [{"name": n, **tables[n]} for n, _, _, _ in co.SPEC],
                "source": {"residuals": f"state-feature forward OOF {sorted(ref)}",
                           "note": "season_form 없는 참조값에서 재추정",
                           "row_independent": True}}
        (work / "model" / "corrections.json").write_text(
            json.dumps(spec, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        for it in spec["items"]:
            print(f"  {it['name']:>16}: {len(it['table'])} entries", flush=True)

        # 수준 재캘리브레이션은 scripts/calibrate_zip.py 가 별도로 수행한다.
        # 이유: 그 단계는 NN 예측이 필요해 torch 를 부르는데, OMP_NUM_THREADS>1 로
        # torch + lightgbm + catboost 가 각자의 OpenMP 런타임을 들고 오면 macOS 에서
        # 교착한다. 학습은 6스레드가 필요하고 캘리브레이션은 1스레드여야 해서
        # 한 프로세스에 둘 수 없다.
        meta["final_logit_shift"] = 0.0

        meta["season_state"] = {
            "applied_to": ["cat", "team"],
            "hgb_excluded": "pickle 재직렬화 위험 (numpy 2.x -> 서버 1.26.4)",
            "features": NEW,
            "backtest": ("R-segment 3-fold mean: 현행(baseline+form) 658.9 -> "
                         "state+corr 703.1 (2024 fold 736.5 -> 785.5)"),
            "season_form": "disabled — cur_p_succ 가 피처로 들어가 이중 계산",
        }
        (work / "model" / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        shutil.rmtree(work / "_spec", ignore_errors=True)

        # ---- script.py 패치: 상태 피처 주입 ------------------------------------
        path = work / "script.py"
        s = path.read_text("utf-8")
        anchor = "    test = test.copy()\n"
        if s.count(anchor) != 1:
            raise SystemExit(f"predict() 앵커가 {s.count(anchor)}개 — 구조가 바뀌었다")
        s = s.replace("import pandas as pd\n",
                      "import pandas as pd\n"
                      "from season_state_runtime import add_state\n", 1)
        s = s.replace(anchor, anchor + '    test = add_state(test, "./model")\n', 1)

        # 최종 수준 보정 — 보정층 뒤, 반환 직전에 상수 하나를 더한다.
        call = 'apply_corrections(preds, test, "./model")'
        for ind in ("        ", "    "):
            old = f"\n{ind}return {call}"
            new = (f"\n{ind}preds = {call}"
                   f'\n{ind}shift = float(meta.get("final_logit_shift", 0.0))'
                   f"\n{ind}return apply_logit_shift(preds, shift) if shift else preds")
            if s.count(old) != 1:
                raise SystemExit(f"corrections 반환 앵커가 {s.count(old)}개 ({ind!r})")
            s = s.replace(old, new, 1)
        path.write_text(s, encoding="utf-8")

        after_pkl = {p.name for p in (work / "model").glob("*.pkl")}
        if after_pkl != before_pkl:
            raise SystemExit(f"새 pickle: {sorted(after_pkl - before_pkl)}")

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(work.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(work))
    print(f"built {out}")


if __name__ == "__main__":
    main()
