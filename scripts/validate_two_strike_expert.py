"""Validate one preregistered two-strike slice LightGBM expert."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.champion import Champion
from src.train_base import PARAMS, add_features, apply_category_maps, build_category_maps

ROOT=Path(__file__).resolve().parents[1]; TARGET="control_success"
SEASONS=(2022,2023,2024); SEEDS=(42,43,44); WEIGHTS=(.10,.25,.50)


def slice_mask(df):
    """Fixed count-family rule: two strikes except 3-2, which is three-ball."""
    return df.strikes_before.eq(2)&df.balls_before.lt(3)


def champion_pred(path,season): return .3*np.load(path/f"{season}_cat.npy")+.7*np.load(path/f"{season}_team.npy")
def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))


def prepare(champion,df):
    spec=champion.members["team"]["team"]; cols=list(champion.members["team"]["feature_columns"])
    d=add_features(df)
    if any(c.startswith("cur_") and c not in d for c in cols): d=pd.concat([d,champion.state_frame(df)],axis=1)
    return d,cols,list(spec["cat_cols"])


def fit_one(df,d,cols,cats,season,seed):
    tr=df.season.lt(season)&slice_mask(df); te=df.season.eq(season)&slice_mask(df)
    maps=build_category_maps(d[tr],cats); X=apply_category_maps(d,cats,maps)[cols]
    params=dict(PARAMS,seed=seed,num_leaves=31,min_data_in_leaf=800,num_threads=6)
    ds=lgb.Dataset(X[tr],label=df.loc[tr,TARGET].to_numpy(float),categorical_feature=cats,free_raw_data=False)
    model=lgb.train(params,ds,num_boost_round=236)
    return model.predict(X[te]),int(tr.sum()),int(te.sum())


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--train",type=Path,default=ROOT.parent/"open/data/train.csv")
    p.add_argument("--zip",type=Path,default=ROOT/"artifacts/submit_season_state.zip")
    p.add_argument("--oof",type=Path,default=ROOT/"artifacts/backtest/champion_oof")
    p.add_argument("--out",type=Path,default=ROOT/"artifacts/residual_slice"); args=p.parse_args(argv); args.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.train,encoding="utf-8-sig"); ch=Champion(args.zip,workdir=str(args.out/"_spec")); d,cols,cats=prepare(ch,df)
    rows=[]; stop=None
    for seed_i,seed in enumerate(SEEDS):
        for season in SEASONS:
            path=args.out/f"two_strike_expert_{season}_s{seed}.npy"
            if path.exists(): expert=np.load(path); tr_n=int((df.season.lt(season)&slice_mask(df)).sum()); te_n=len(expert)
            else:
                print(f"fit seed={seed} season={season}",flush=True); expert,tr_n,te_n=fit_one(df,d,cols,cats,season,seed); np.save(path,expert)
            part=df[df.season.eq(season)]; mask=slice_mask(part).to_numpy(); y=part[TARGET].to_numpy(float); champ=champion_pred(args.oof,season)
            for weight in WEIGHTS:
                cand=champ.copy(); cand[mask]=(1-weight)*champ[mask]+weight*expert
                for group,gmask in (("all",np.ones(len(part),bool)),("slice",mask),("outside",~mask),
                                    ("R",part.game_type.astype(str).eq("R").to_numpy()),("F",part.game_type.astype(str).eq("F").to_numpy())):
                    rows.append({"seed":seed,"season":season,"weight":weight,"group":group,"train_slice_rows":tr_n,"validation_slice_rows":te_n,
                                 "champion_brier":brier(y[gmask],champ[gmask]),"candidate_brier":brier(y[gmask],cand[gmask]),
                                 "gain":brier(y[gmask],champ[gmask])-brier(y[gmask],cand[gmask])})
        if seed_i==1:
            table=pd.DataFrame(rows); recent=table[table.seed.isin(SEEDS[:2])&table.season.isin((2022,2024))&table.group.eq("all")]
            if all((recent[recent.weight.eq(w)].gain<0).all() for w in WEIGHTS):
                stop="seeds 42/43: every weight worsened 2022 and 2024"; break
    out=pd.DataFrame(rows); out.to_csv(args.out/"expert_results.csv",index=False)
    verdict={"expert_tested":True,"fast_stop":stop,"seed_44_run":44 in out.seed.unique().tolist(),"weights":WEIGHTS}
    (args.out/"expert_summary.json").write_text(json.dumps(verdict,indent=2),"utf-8"); print(json.dumps(verdict,indent=2))


if __name__=="__main__": main()
