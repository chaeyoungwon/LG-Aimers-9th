"""Build the gated 25% mean(LGB, Cat) two-strike submission candidate."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.champion import Champion
from src.train_base import PARAMS, add_features, apply_category_maps, build_category_maps
from scripts.validate_two_strike_expert import slice_mask

ROOT=Path(__file__).resolve().parents[1]; TARGET="control_success"; SEEDS=(42,43,44)


def train_lgb(ch,df,work):
    spec=ch.members["team"]["team"]; cols=list(ch.members["team"]["feature_columns"]); cats=list(spec["cat_cols"])
    d=add_features(df)
    if any(c.startswith("cur_") and c not in d for c in cols): d=pd.concat([d,ch.state_frame(df)],axis=1)
    mask=slice_mask(df); maps=build_category_maps(d,cats); X=apply_category_maps(d,cats,maps)[cols]
    files=[]
    for seed in SEEDS:
        model=lgb.LGBMClassifier(**dict(PARAMS,seed=seed,num_leaves=31,min_data_in_leaf=800,num_threads=6,n_estimators=236))
        model.fit(X[mask],df.loc[mask,TARGET],categorical_feature=cats,callbacks=[lgb.log_evaluation(0)])
        name=f"two_strike_lgb_s{seed}.pkl"; joblib.dump(model,work/"model"/name); files.append(name)
    return {"name":"two_strike_lgb","engine":"lightgbm","groups":["team"],"feature_columns":cols,
            "model_files":files,"team":{"cat_cols":cats,"category_maps":maps}}


def train_cat(ch,df,work):
    source=ch.members["cat"]; X=ch.frame(df,"cat"); mask=slice_mask(df)
    cats=[c for c in X if str(X[c].dtype)=="category"]; idx=[X.columns.get_loc(c) for c in cats]
    def clean(x):
        x=x.copy()
        for c in cats:x[c]=x[c].astype(object).where(x[c].notna(),"__NA__").astype(str)
        return x
    files=[]
    for seed in SEEDS:
        model=CatBoostClassifier(random_seed=seed,verbose=False,thread_count=6,allow_writing_files=False,**source.get("model_params",{}))
        model.fit(Pool(clean(X[mask]),df.loc[mask,TARGET],cat_features=idx))
        name=f"two_strike_cat_s{seed}.cbm"; model.save_model(str(work/"model"/name)); files.append(name)
    return {"name":"two_strike_cat","engine":"catboost_native","groups":list(source["groups"]),
            "feature_columns":list(source["feature_columns"]),"model_files":files,
            "cat_features":cats,"na_token":"__NA__"}


def patch_runtime(work):
    path=work/"script.py"; text=path.read_text("utf-8")
    needle="""    calib = blend.get(\"calibration\")
    if calib:
        total = apply_calibration(total, calib, frame)
        if verbose:
            print(f\"blend calibration {calib} -> mean pred {total.mean():.4f}\")
    return total
"""
    replacement="""    calib = blend.get(\"calibration\")
    if calib:
        total = apply_calibration(total, calib, frame)
        if verbose:
            print(f\"blend calibration {calib} -> mean pred {total.mean():.4f}\")
    slice_spec = meta.get(\"two_strike_expert\")
    if slice_spec:
        lgb_name = slice_spec[\"lgb_member\"]
        cat_name = slice_spec[\"cat_member\"]
        expert = 0.5 * member_p[lgb_name] + 0.5 * member_p[cat_name]
        active = ((frame[\"strikes_before\"].to_numpy() == 2)
                  & (frame[\"balls_before\"].to_numpy() < 3))
        weight = float(slice_spec[\"weight\"])
        # Copy first: outside-slice values remain bit-identical to champion.
        gated = np.asarray(total, dtype=\"float64\").copy()
        gated[active] = ((1.0 - weight) * gated[active]
                         + weight * expert[active])
        total = gated
        if verbose:
            print(f\"two-strike expert: weight={weight} active={int(active.sum())}\")
    return total
"""
    if text.count(needle)!=1: raise RuntimeError("runtime insertion point mismatch")
    path.write_text(text.replace(needle,replacement),"utf-8")


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--train",type=Path,default=ROOT.parent/"open/data/train.csv")
    p.add_argument("--base",type=Path,default=ROOT/"artifacts/sub_tree_reblend_w0p300.zip")
    p.add_argument("--output",type=Path,default=ROOT/"artifacts/sub_two_strike_lgbcat_w0p250.zip"); args=p.parse_args(argv)
    df=pd.read_csv(args.train,encoding="utf-8-sig")
    with tempfile.TemporaryDirectory(prefix="two_strike_build_") as td:
        work=Path(td); zipfile.ZipFile(args.base).extractall(work); meta=json.loads((work/"model/meta.json").read_text("utf-8"))
        ch=Champion(args.base,workdir=str(work/"_spec")); lgb_spec=train_lgb(ch,df,work); cat_spec=train_cat(ch,df,work)
        meta["members"].extend([lgb_spec,cat_spec]); meta["two_strike_expert"]={"rule":"strikes_before == 2 and balls_before < 3",
            "weight":.25,"expert_mix":{"lightgbm":.5,"catboost":.5},"lgb_member":lgb_spec["name"],"cat_member":cat_spec["name"],
            "validation":"rolling 2022/2023/2024 seeds 42/43/44; 2024 +4.336 BSS > paired 2SE 4.084"}
        (work/"model/meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),"utf-8"); patch_runtime(work)
        shutil.rmtree(work/"_spec",ignore_errors=True); args.output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.output,"w",zipfile.ZIP_DEFLATED) as z:
            for path in sorted(work.rglob("*")):
                if path.is_file(): z.write(path,path.relative_to(work))
    print(args.output)


if __name__=="__main__":main()
