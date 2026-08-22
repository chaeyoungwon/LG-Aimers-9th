"""Final bounded validation of four two-strike expert candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score

from src.champion import Champion
from src.train_base import PARAMS, add_features, apply_category_maps, build_category_maps
from scripts.validate_hard_example_objective import inner_oof_vector
from scripts.validate_two_strike_expert import slice_mask

ROOT=Path(__file__).resolve().parents[1]; TARGET="control_success"
SEASONS=(2022,2023,2024); SEEDS=(42,43,44); WEIGHTS=(.10,.25,.50)


def champion_pred(path,s): return .3*np.load(path/f"{s}_cat.npy")+.7*np.load(path/f"{s}_team.npy")
def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))


def cat_fit(ch,df,season,seed):
    X=ch.frame(df,"cat"); tr=df.season.lt(season)&slice_mask(df); te=df.season.eq(season)&slice_mask(df)
    cats=[c for c in X if str(X[c].dtype)=="category"]; idx=[X.columns.get_loc(c) for c in cats]
    def clean(x):
        x=x.copy()
        for c in cats: x[c]=x[c].astype(object).where(x[c].notna(),"__NA__").astype(str)
        return x
    params=dict(ch.members["cat"].get("model_params",{}))
    model=CatBoostClassifier(random_seed=seed,verbose=False,thread_count=6,allow_writing_files=False,**params)
    model.fit(Pool(clean(X[tr]),df.loc[tr,TARGET],cat_features=idx))
    return model.predict_proba(Pool(clean(X[te]),cat_features=idx))[:,1]


def team_features(ch,df):
    spec=ch.members["team"]["team"]; cols=list(ch.members["team"]["feature_columns"]); d=add_features(df)
    if any(c.startswith("cur_") and c not in d for c in cols): d=pd.concat([d,ch.state_frame(df)],axis=1)
    return d,cols,list(spec["cat_cols"])


def residual_fit(df,d,cols,cats,inner,season,seed):
    outer=df.season.lt(season).to_numpy(); tr=outer&slice_mask(df).to_numpy()&np.isfinite(inner)
    te=df.season.eq(season).to_numpy()&slice_mask(df).to_numpy(); maps=build_category_maps(d[outer],cats)
    X=apply_category_maps(d,cats,maps)[cols]; residual=df[TARGET].to_numpy(float)[tr]-inner[tr]
    params=dict(PARAMS,objective="regression",metric="l2",seed=seed,num_leaves=15,
                min_data_in_leaf=800,lambda_l1=1.,lambda_l2=20.,num_threads=6)
    model=lgb.train(params,lgb.Dataset(X[tr],label=residual,categorical_feature=cats,free_raw_data=False),num_boost_round=150)
    return model.predict(X[te]),int(tr.sum())


def candidate_rows(df,oof,season,seed,experts):
    part=df[df.season.eq(season)]; y=part[TARGET].to_numpy(float); champ=champion_pred(oof,season); mask=slice_mask(part).to_numpy()
    rows=[]
    candidates={
        "lgb":experts["lgb"],"cat":experts["cat"],
        "lgb_cat_mean":.5*experts["lgb"]+.5*experts["cat"],
    }
    for name,expert in candidates.items():
        for weight in WEIGHTS:
            cand=champ.copy(); cand[mask]=(1-weight)*champ[mask]+weight*expert
            rows.extend(evaluate_groups(part,y,champ,cand,mask,season,seed,name,weight))
    for alpha in WEIGHTS:
        cand=champ.copy(); cand[mask]=np.clip(champ[mask]+alpha*experts["residual"],1e-6,1-1e-6)
        rows.extend(evaluate_groups(part,y,champ,cand,mask,season,seed,"residual",alpha))
    return rows


def evaluate_groups(part,y,champ,cand,mask,season,seed,name,weight):
    out=[]
    count=part.balls_before.astype(str)+"-"+part.strikes_before.astype(str)
    groups={"all":np.ones(len(part),bool),"slice":mask,"outside":~mask,
            "R":part.game_type.astype(str).eq("R").to_numpy(),"F":part.game_type.astype(str).eq("F").to_numpy()}
    for value in ("0-2","1-2","2-2"): groups[value]=count.eq(value).to_numpy()
    for group,m in groups.items():
        if not m.any(): continue
        out.append({"candidate":name,"weight":weight,"seed":seed,"season":season,"group":group,"n":int(m.sum()),
                    "champion_brier":brier(y[m],champ[m]),"candidate_brier":brier(y[m],cand[m]),"gain":brier(y[m],champ[m])-brier(y[m],cand[m]),
                    "target_mean":float(y[m].mean()),"prediction_mean":float(cand[m].mean()),"prediction_std":float(cand[m].std()),
                    "logloss":log_loss(y[m],cand[m]),"auc":roc_auc_score(y[m],cand[m]) if len(np.unique(y[m]))==2 else np.nan,
                    "outside_max_abs_diff":float(np.max(np.abs(cand[~mask]-champ[~mask]))) if (~mask).any() else 0.})
    return out


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--train",type=Path,default=ROOT.parent/"open/data/train.csv")
    p.add_argument("--zip",type=Path,default=ROOT/"artifacts/submit_season_state.zip")
    p.add_argument("--oof",type=Path,default=ROOT/"artifacts/backtest/champion_oof")
    p.add_argument("--out",type=Path,default=ROOT/"artifacts/two_strike_final"); args=p.parse_args(argv); args.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.train,encoding="utf-8-sig"); ch=Champion(args.zip,workdir=str(args.out/"_spec")); d,cols,cats=team_features(ch,df)
    rows=[]; diversity=[]
    for seed in SEEDS:
        for season in SEASONS:
            paths={k:args.out/f"{season}_{k}_s{seed}.npy" for k in ("cat","residual")}
            if paths["cat"].exists(): cat=np.load(paths["cat"])
            else: print(f"fit cat seed={seed} season={season}",flush=True); cat=cat_fit(ch,df,season,seed); np.save(paths["cat"],cat)
            inner=inner_oof_vector(df,args.oof,season)
            if paths["residual"].exists(): residual=np.load(paths["residual"]); residual_n=-1
            else: print(f"fit residual seed={seed} season={season}",flush=True); residual,residual_n=residual_fit(df,d,cols,cats,inner,season,seed); np.save(paths["residual"],residual)
            lgb_pred=np.load(ROOT/f"artifacts/residual_slice/two_strike_expert_{season}_s{seed}.npy")
            experts={"lgb":lgb_pred,"cat":cat,"residual":residual}; rows.extend(candidate_rows(df,args.oof,season,seed,experts))
            part=df[df.season.eq(season)]; y=part.loc[slice_mask(part),TARGET].to_numpy(float); champ=champion_pred(args.oof,season)[slice_mask(part)]
            diversity.append({"seed":seed,"season":season,"residual_train_rows":residual_n,
                              "corr_champion_lgb":np.corrcoef(champ,lgb_pred)[0,1],"corr_champion_cat":np.corrcoef(champ,cat)[0,1],
                              "corr_lgb_cat":np.corrcoef(lgb_pred,cat)[0,1],"lgb_residual_corr":np.corrcoef(y-lgb_pred,y-champ)[0,1],
                              "cat_residual_corr":np.corrcoef(y-cat,y-champ)[0,1]})
    result=pd.DataFrame(rows); result.to_csv(args.out/"candidate_results.csv",index=False); pd.DataFrame(diversity).to_csv(args.out/"expert_diversity.csv",index=False)
    (args.out/"summary.json").write_text(json.dumps({"seeds":SEEDS,"weights":WEIGHTS,"slice":"strikes == 2 and balls < 3","test_read":False},indent=2),"utf-8")


if __name__=="__main__": main()
