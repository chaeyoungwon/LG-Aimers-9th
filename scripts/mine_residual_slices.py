"""Mine predeclared, row-local residual slices with 2022/2024 primary gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import season_state

ROOT=Path(__file__).resolve().parents[1]
TARGET="control_success"; SEASONS=(2022,2023,2024)


def champion(path,season):
    return .3*np.load(path/f"{season}_cat.npy")+.7*np.load(path/f"{season}_team.npy")


def history_bucket(x):
    return pd.cut(pd.to_numeric(x,errors="coerce").fillna(0),[-1,0,20,100,500,np.inf],labels=["0","1_20","21_100","101_500","500_plus"])


def slice_frame(df,state,season):
    rows=df[df.season.eq(season)].copy(); prior=df[df.season.lt(season)]
    pair=prior.groupby(["pitcher_id","batter_id"],observed=True).size()
    idx=pd.MultiIndex.from_frame(rows[["pitcher_id","batter_id"]]); n=pair.reindex(idx).fillna(0).to_numpy()
    balls=rows.balls_before.to_numpy(); strikes=rows.strikes_before.to_numpy()
    count=np.select([balls==3,strikes==2,balls>strikes,balls==strikes],
                    ["three_ball","two_strike","batter_ahead","even"],default="pitcher_ahead")
    inning=rows.inning.to_numpy(); phase=np.select([inning<=3,inning<=6,inning<=9],["early","middle","late"],default="extras")
    risp=rows.runner_on_2b.fillna(0).astype(bool)|rows.runner_on_3b.fillna(0).astype(bool)
    base=np.select([rows.num_runners_on.fillna(0).eq(0),risp],["empty","RISP"],default="runner_non_RISP")
    current=state.loc[rows.index]
    return pd.DataFrame({"game_type":rows.game_type.astype(str).to_numpy(),
                         "pitcher_history":history_bucket(rows.asof_pitcher_n).astype(str).to_numpy(),
                         "batter_history":history_bucket(rows.asof_batter_n).astype(str).to_numpy(),
                         "matchup_novelty":np.select([n==0,n<=5],["unseen","low_1_5"],default="established"),
                         "count_family":count,"hand_matchup":np.where(rows.pitcher_hand.astype(str).eq(rows.batter_hand.astype(str)),"same","opposite"),
                         "inning_phase":phase,"base_state_coarse":base,
                         "two_outs":np.where(rows.outs_before.eq(2),"yes","no"),
                         "pitcher_state_available":np.where(current.cur_p_n.notna(),"yes","no"),
                         "batter_state_available":np.where(current.cur_b_n.notna(),"yes","no")},index=rows.index)


def metrics(mask,y,p):
    err=(y-p)**2; share=float(mask.mean()); target=float(y[mask].mean()); pred=float(p[mask].mean())
    return {"rows":int(mask.sum()),"share":share,"target_mean":target,"prediction_mean":pred,
            "brier":float(err[mask].mean()),"global_brier":float(err.mean()),
            "excess":float(err[mask].mean()-err.mean()),"calibration_error":target-pred}


def scan(df,states,preds):
    singles=[]; interactions=[]; frames={}
    for season in SEASONS:
        rows=df[df.season.eq(season)]; y=rows[TARGET].to_numpy(float); p=preds[season]
        f=slice_frame(df,states,season); frames[season]=f
        for axis in f.columns:
            for value in sorted(f[axis].unique()):
                m=f[axis].eq(value).to_numpy(); singles.append({"season":season,"slice":f"{axis}={value}","axis":axis,**metrics(m,y,p)})
        pairs=(("game_type","pitcher_history"),("game_type","batter_history"),("count_family","pitcher_history"),
               ("count_family","batter_history"),("hand_matchup","count_family"),("inning_phase","game_type"))
        for left,right in pairs:
            for (a,b),idx in f.groupby([left,right],observed=True).groups.items():
                m=f.index.isin(idx); interactions.append({"season":season,"slice":f"{left}={a} & {right}={b}",
                                                          "parent_left":f"{left}={a}","parent_right":f"{right}={b}",**metrics(m,y,p)})
    return pd.DataFrame(singles),pd.DataFrame(interactions),frames


def rank(scan,min_share):
    wide=scan.pivot(index="slice",columns="season",values=["share","excess","calibration_error","brier","rows"])
    wide.columns=[f"{x}_{y}" for x,y in wide.columns]; wide=wide.reset_index()
    gate=(wide.share_2022>=min_share)&(wide.share_2024>=min_share)&(wide.excess_2022>0)&(wide.excess_2024>0)
    out=wide[gate].copy(); out["common_score"]=out[["excess_2022","excess_2024"]].min(axis=1)
    out["same_direction"]=np.sign(out.calibration_error_2022)==np.sign(out.calibration_error_2024)
    return out.sort_values("common_score",ascending=False)


def bootstrap_ci(df,frames,preds,candidates,reps=500):
    rng=np.random.default_rng(42); out=[]
    for name in candidates:
        for season in (2022,2024):
            rows=df[df.season.eq(season)]; y=rows[TARGET].to_numpy(float); p=preds[season]; err=(y-p)**2
            # Parse masks from the already frozen slice frame, never labels.
            pieces=name.split(" & "); mask=np.ones(len(rows),bool)
            for piece in pieces:
                axis,value=piece.split("=",1); mask &= frames[season][axis].astype(str).eq(value).to_numpy()
            inside=err[mask]; outside=err
            values=[]
            for _ in range(reps):
                values.append(float(rng.choice(inside,len(inside),replace=True).mean()-rng.choice(outside,len(outside),replace=True).mean()))
            out.append({"slice":name,"season":season,"reps":reps,"excess":inside.mean()-outside.mean(),
                        "ci_lower":np.quantile(values,.025),"ci_upper":np.quantile(values,.975),
                        "stable":np.quantile(values,.025)>0})
    return pd.DataFrame(out)


def classify_error(row):
    bias=max(abs(row.calibration_error_2022),abs(row.calibration_error_2024))
    # Compare squared mean bias with common excess; descriptive only.
    if bias* bias > .5*row.common_score: return "BOTH" if row.common_score>bias*bias else "CALIBRATION"
    return "DISCRIMINATION"


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--train",type=Path,default=ROOT.parent/"open/data/train.csv")
    p.add_argument("--oof",type=Path,default=ROOT/"artifacts/backtest/champion_oof")
    p.add_argument("--out",type=Path,default=ROOT/"artifacts/residual_slice"); args=p.parse_args(argv); args.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.train,encoding="utf-8-sig"); states=season_state.add_features_by_season(df); preds={s:champion(args.oof,s) for s in SEASONS}
    single,interaction,frames=scan(df,states,preds); srank=rank(single,.02); irank=rank(interaction,.01)
    # Interaction must improve materially over both parent common scores.
    parent_score=dict(zip(srank.slice,srank.common_score));
    parents=interaction.drop_duplicates("slice").set_index("slice")[["parent_left","parent_right"]]
    if len(irank):
        irank=irank.join(parents,on="slice"); irank["parent_best_common"]=irank.apply(lambda r:max(parent_score.get(r.parent_left,-np.inf),parent_score.get(r.parent_right,-np.inf)),axis=1)
        irank["improves_parent"]=irank.common_score>irank.parent_best_common+0.001
    pool=pd.concat([srank[srank.same_direction],irank[irank.same_direction & irank.improves_parent] if len(irank) else irank],ignore_index=True).sort_values("common_score",ascending=False)
    top=pool.head(10).slice.tolist(); boot=bootstrap_ci(df,frames,preds,top)
    stable=[]
    for name in top:
        b=boot[boot.slice.eq(name)]
        if len(b)==2 and b.stable.all(): stable.append(name)
    for table,path in ((single,"single_slice_scan.csv"),(interaction,"interaction_slice_scan.csv"),(srank,"single_slice_ranking.csv"),(irank,"interaction_slice_ranking.csv"),(boot,"bootstrap_stability.csv")):
        table.to_csv(args.out/path,index=False)
    diagnostics=pool.copy(); diagnostics["error_type"]=diagnostics.apply(classify_error,axis=1); diagnostics.to_csv(args.out/"stable_gate_diagnostics.csv",index=False)
    # No expert is allowed unless at least one fully stable slice survives.
    verdict={"stable_slice_count":len(stable),"stable_slices":stable,"expert_tested":False,
             "expert_reason":"diagnostic gate must pass before expert training","test_read":False,"submission_created":False}
    pd.DataFrame(columns=["season","candidate","brier","gain"]).to_csv(args.out/"expert_results.csv",index=False)
    (args.out/"summary.json").write_text(json.dumps(verdict,indent=2),"utf-8"); print(json.dumps(verdict,indent=2))


if __name__=="__main__": main()
