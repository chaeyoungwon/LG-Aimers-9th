"""챔피언 5멤버(hgb·cat·nn·team·team_nn)를 임의의 학습 구간으로 재현한다.

**왜 필요한가.** 지금까지의 백테스트 기준선은 챔피언의 team 스테이지(블렌드의
60%)만 근사했다. 그래서 블렌드 비중·멤버 구성 같은 결정은 로컬로 판정할 수 없었고,
실제로 `meta.json`에도 "로컬 fold 게이트 없음 — LB가 심판"이라고 적혀 있다.
5멤버를 전부 폴드별로 재현하면 그 결정들을 처음으로 오프라인에서 판정할 수 있다.

**충실도 우선 설계.** 파생 피처는 재구현하지 않고 **배포 ZIP 안의
`build_submit_features` 를 그대로 import 해서 쓴다.** dtype 하나만 어긋나도
트리 분기가 달라지므로, 재구현은 가장 흔한 실패 지점이다. 하이퍼파라미터도
`meta.json` 에 박힌 값을 읽는다 (hgb/cat/nn 은 `model_params`, team 4종은
`model_params.team`, 앙상블 레시피는 team 멤버의 `team` 블록).

**규칙 준수.** 범주 매핑·전처리 상수·임베딩 사전은 전부 그 폴드의 `train_seasons`
에서만 적합한다. 각 행의 예측은 자기 행의 값과 그 상수들만으로 나온다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import season_state as ss

TARGET = "control_success"


def _torch_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Champion:
    """배포 ZIP 한 개를 '사양서'로 삼아 같은 구성을 다시 학습한다."""

    def __init__(self, zip_path, workdir=None):
        self.work = Path(workdir or tempfile.mkdtemp(prefix="champ_spec_"))
        if not (self.work / "script.py").exists():
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(self.work)
        self.meta = json.loads((self.work / "model" / "meta.json").read_text("utf-8"))
        self.script = self._import_script()
        self.members = {m["name"]: m for m in self.meta["members"]}
        self._state_cache = {}

    def _import_script(self):
        spec = importlib.util.spec_from_file_location(
            "champ_script", self.work / "script.py")
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(self.work))
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.pop(0)
        return mod

    # ---- 피처 ------------------------------------------------------------

    def state_frame(self, df):
        """시즌별 train-only 경계로 복원한 상태 프레임을 실행 중 한 번만 만든다."""
        key = id(df)
        cached = self._state_cache.get(key)
        if cached is None or not cached.index.equals(df.index):
            cached = ss.add_features_by_season(df)
            self._state_cache = {key: cached}
        return cached

    def frame(self, df, name):
        """멤버 하나가 먹는 피처 프레임. 배포본의 빌더를 그대로 통과시킨다."""
        X = df.copy()
        required = self.members[name]["feature_columns"]
        if any(c.startswith("cur_") and c not in X for c in required):
            X = pd.concat([X, self.state_frame(df)], axis=1)
        for col, levels in self.meta["cat_levels"].items():
            X[col] = pd.Categorical(X[col], categories=levels)
        X = self.script.build_submit_features(X, self.members[name])
        return X[self.members[name]["feature_columns"]]

    # ---- 멤버별 학습 ------------------------------------------------------

    def _fit_tree(self, name, Xtr, ytr, Xte, seed):
        p = self.members[name].get("model_params", {})
        if name == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(
                categorical_features="from_dtype", random_state=seed, **p)
            m.fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1]
        from catboost import CatBoostClassifier, Pool
        cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        idx = [Xtr.columns.get_loc(c) for c in cats]

        def s(F):
            F = F.copy()
            for c in cats:
                F[c] = F[c].astype(str)
            return F

        m = CatBoostClassifier(random_seed=seed, verbose=False, thread_count=6,
                               allow_writing_files=False, **p)
        m.fit(Pool(s(Xtr), ytr, cat_features=idx))
        return m.predict_proba(Pool(s(Xte), cat_features=idx))[:, 1]

    def _fit_team(self, df, train_seasons, tr, te, seed_offset):
        """LGBM 4종 -> 2개 군 가중평균 -> 군별 Platt -> 60:40. 배포본 레시피 그대로."""
        import lightgbm as lgb
        from src.train_base import PARAMS, add_features, apply_category_maps, build_category_maps

        spec = self.members["team"]["team"]
        cfgs = self.meta["model_params"]["team"]
        cols = self.members["team"]["feature_columns"]
        cats = spec["cat_cols"]

        d = add_features(df)
        if any(c.startswith("cur_") and c not in d for c in cols):
            d = pd.concat([d, self.state_frame(df)], axis=1)
        maps = build_category_maps(d[d.season.isin(train_seasons)], cats)
        X = apply_category_maps(d, cats, maps)[cols]
        y = df[TARGET].to_numpy(dtype="float64")
        season = df.season.astype(int).to_numpy()
        nxt = max(train_seasons) + 1

        # 단조 제약: train_monotone.py 와 같은 부호 배정
        pos = {"asof_pitcher_success_rate", "asof_pitcher_strike_rate",
               "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
               "asof_pitcher_prev5_game_success_rate", "asof_batter_success_rate"}
        neg = {"asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
               "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
               "asof_pitcher_prev5_game_middle_rate", "asof_batter_middle_rate"}

        preds = {}
        for nm, c in cfgs.items():
            par = dict(PARAMS, seed=c["seed"] + seed_offset,
                       num_leaves=c["num_leaves"],
                       min_data_in_leaf=c["min_data_in_leaf"], num_threads=6)
            if c.get("monotone"):
                par["monotone_constraints"] = [
                    1 if f in pos else -1 if f in neg else 0 for f in cols]
                par["monotone_constraints_method"] = "advanced"
            w = None
            if c.get("season_decay"):
                w = c["season_decay"] ** (nxt - season[tr])
            ds = lgb.Dataset(X[tr], label=y[tr], weight=w,
                             categorical_feature=cats, free_raw_data=False)
            preds[nm] = lgb.train(par, ds, num_boost_round=c["rounds"]).predict(X[te])

        outs = []
        for i, g in enumerate(spec["groups"]):
            raw = np.average([preds[m] for m in g["members"]], axis=0,
                             weights=g["weights"])
            cal = spec["calib"][i]
            outs.append(self.script.apply_logit_shift(raw, cal["bias"],
                                                      slope=cal["scale"]))
        return np.average(outs, axis=0, weights=spec["group_weights"])

    def _fit_nn(self, name, df, train_seasons, tr, te, seed_offset):
        from src.nn_embed import IdVocab, TabularPrep, train_embed_nn

        p = dict(self.members[name]["model_params"])
        p["hidden"] = tuple(p["hidden"])
        p["seed"] = p["seed"] + seed_offset
        cols = [c for c in self.members[name]["feature_columns"]
                if c not in ("pitcher_id", "batter_id")]
        X = self.frame(df, name)[cols]
        y = df[TARGET].to_numpy(dtype="float32")

        prep = TabularPrep().fit(X[tr])                 # 학습 구간에서만 적합
        pv = IdVocab.fit(df.loc[tr, "pitcher_id"])
        bv = IdVocab.fit(df.loc[tr, "batter_id"])
        Xn = prep.transform(X)
        pid, bid = pv.transform(df.pitcher_id), bv.transform(df.batter_id)

        model, _ = train_embed_nn(
            Xn[tr], pid[tr], bid[tr], y[tr], n_pitcher=pv.size, n_batter=bv.size,
            device=_torch_device(), verbose=False, **p)
        return _nn_predict(model, Xn[te], pid[te], bid[te])

    # ---- 전체 ------------------------------------------------------------

    def fit_predict(self, df, train_seasons, target_season, seed_offset=0,
                    members=None, verbose=True):
        """{멤버: target_season 행에 대한 확률}."""
        season = df.season.astype(int).to_numpy()
        tr = np.isin(season, list(train_seasons))
        te = season == target_season
        y = df[TARGET].to_numpy(dtype="float64")
        out = {}
        for name in (members or ["hgb", "cat", "nn", "team", "team_nn"]):
            if name in ("hgb", "cat"):
                F = self.frame(df, name)
                out[name] = self._fit_tree(name, F[tr], y[tr], F[te], 11 + seed_offset)
            elif name == "team":
                out[name] = self._fit_team(df, train_seasons, tr, te, seed_offset)
            else:
                out[name] = self._fit_nn(name, df, train_seasons, tr, te, seed_offset)
            if verbose:
                print(f"    {name}: mean={out[name].mean():.4f}", flush=True)
        return out


def _nn_predict(model, Xn, pid, bid, batch=4096):
    """고정 배치 패딩 — 배포본과 같은 이유(float32 행렬곱의 배치 의존성 제거)."""
    import torch
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch):
            xs, ps, bs = Xn[i:i + batch], pid[i:i + batch], bid[i:i + batch]
            k = len(xs)
            if k < batch:
                xs = np.vstack([xs, np.zeros((batch - k, xs.shape[1]), xs.dtype)])
                ps = np.concatenate([ps, np.zeros(batch - k, ps.dtype)])
                bs = np.concatenate([bs, np.zeros(batch - k, bs.dtype)])
            logit = model(torch.as_tensor(xs, dtype=torch.float32, device=dev),
                          torch.as_tensor(ps, dtype=torch.long, device=dev),
                          torch.as_tensor(bs, dtype=torch.long, device=dev))
            outs.append(torch.sigmoid(logit).float().cpu().numpy().ravel()[:k])
    return np.concatenate(outs)


def blend(member_preds, meta, script, stage_calib=None):
    """meta['blend'] 레시피대로 스테이지 합성. stage_calib 로 보정을 갈아끼운다."""
    b = meta["blend"]
    stages = []
    for i, st in enumerate(b["stages"]):
        p = np.average([member_preds[m] for m in st["members"]], axis=0,
                       weights=st["weights"])
        cal = (stage_calib or {}).get(st["name"], st.get("calibration"))
        if cal:
            p = script.apply_logit_shift(p, cal["b"], slope=cal.get("scale", 1.0))
        stages.append(p)
    return np.average(stages, axis=0, weights=b["stage_weights"])
