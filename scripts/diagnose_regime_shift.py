"""Diagnose the 2023 regime using train seasons and frozen rolling OOF only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src import season_state
from scripts.validate_hierarchical_eb import fit_lookups, prepare as eb_prepare, transform as eb_transform, prior_prediction

ROOT = Path(__file__).resolve().parents[1]
SEASONS = (2020, 2021, 2022, 2023, 2024)
MAIN = (2022, 2023, 2024)
TARGET = "control_success"
PAIRS = ((2022, 2023), (2023, 2024), (2022, 2024))


def champion(oof, season):
    return .3*np.load(oof/f"{season}_cat.npy") + .7*np.load(oof/f"{season}_team.npy")


def ece(y, p):
    total=0.
    for lo in np.arange(0, 1, .1):
        hi=lo+.1; mask=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
        if mask.any(): total += mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(total)


def psi(a, b, bins=10):
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if not len(a) or not len(b): return np.nan
    edges=np.unique(np.quantile(np.concatenate([a,b]), np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.
    pa=np.histogram(a,edges)[0]/len(a); pb=np.histogram(b,edges)[0]/len(b)
    pa=np.clip(pa,1e-6,None); pb=np.clip(pb,1e-6,None)
    return float(np.sum((pa-pb)*np.log(pa/pb)))


def js_cat(a,b):
    aa=pd.Series(a).fillna("__NA__").astype(str).value_counts(normalize=True)
    bb=pd.Series(b).fillna("__NA__").astype(str).value_counts(normalize=True)
    idx=aa.index.union(bb.index)
    return float(jensenshannon(aa.reindex(idx,fill_value=0),bb.reindex(idx,fill_value=0))**2)


def season_summary(df, preds):
    out=[]; calibration=[]
    for s in MAIN:
        rows=df[df.season.eq(s)]; y=rows[TARGET].to_numpy(float); p=preds[s]
        prior=df[df.season.lt(s)]
        known_p=set(prior.pitcher_id.dropna()); known_b=set(prior.batter_id.dropna())
        match=set(zip(prior.pitcher_id,prior.batter_id))
        out.append({"season":s,"rows":len(rows),"target_mean":y.mean(),"target_variance":y.var(),
                    "prediction_mean":p.mean(),"prediction_std":p.std(),"brier":np.mean((y-p)**2),
                    "logloss":log_loss(y,p),"ece":ece(y,p),"unique_pitcher":rows.pitcher_id.nunique(),
                    "new_pitcher_count":len(set(rows.pitcher_id)-known_p),
                    "new_pitcher_row_share":(~rows.pitcher_id.isin(known_p)).mean(),
                    "pitcher_cold_start":rows.asof_pitcher_n.fillna(0).le(0).mean(),
                    "pitcher_low_history":rows.asof_pitcher_n.fillna(0).lt(100).mean(),
                    "pitcher_history_median":rows.asof_pitcher_n.median(),
                    "unique_batter":rows.batter_id.nunique(),"new_batter_count":len(set(rows.batter_id)-known_b),
                    "new_batter_row_share":(~rows.batter_id.isin(known_b)).mean(),
                    "batter_cold_start":rows.asof_batter_n.fillna(0).le(0).mean(),
                    "batter_low_history":rows.asof_batter_n.fillna(0).lt(100).mean(),
                    "batter_history_median":rows.asof_batter_n.median(),
                    "unseen_matchup":np.mean([x not in match for x in zip(rows.pitcher_id,rows.batter_id)])})
        for i,(lo,hi) in enumerate(zip(np.arange(0,1,.1),np.arange(.1,1.1,.1))):
            mask=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
            calibration.append({"season":s,"bin":i,"lower":lo,"upper":hi,"rows":int(mask.sum()),
                                "prediction_mean":p[mask].mean() if mask.any() else np.nan,
                                "target_mean":y[mask].mean() if mask.any() else np.nan,
                                "calibration_error":(y[mask].mean()-p[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(out),pd.DataFrame(calibration)


def distribution_shift(df, features):
    num=[]; cat=[]
    for col in features:
        numeric=pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True)>20
        for a,b in PAIRS:
            x=df.loc[df.season.eq(a),col]; z=df.loc[df.season.eq(b),col]
            if numeric:
                xa=pd.to_numeric(x,errors="coerce").to_numpy(float); za=pd.to_numeric(z,errors="coerce").to_numpy(float)
                finite_x=xa[np.isfinite(xa)]; finite_z=za[np.isfinite(za)]
                num.append({"feature":col,"from":a,"to":b,"ks":ks_2samp(finite_x,finite_z).statistic if len(finite_x) and len(finite_z) else np.nan,
                            "psi":psi(xa,za),"mean_from":np.nanmean(xa),"mean_to":np.nanmean(za),
                            "median_from":np.nanmedian(xa),"median_to":np.nanmedian(za),
                            "missing_from":np.mean(~np.isfinite(xa)),"missing_to":np.mean(~np.isfinite(za))})
            else:
                top=x.fillna("__NA__").astype(str).value_counts(normalize=True).head(1)
                top_name=top.index[0] if len(top) else ""
                cat.append({"feature":col,"from":a,"to":b,"js":js_cat(x,z),
                            "top_category":top_name,"top_share_from":float(top.iloc[0]) if len(top) else np.nan,
                            "top_share_to":float(z.fillna("__NA__").astype(str).eq(top_name).mean()),
                            "unseen_to":float((~z.fillna("__NA__").astype(str).isin(set(x.fillna("__NA__").astype(str)))).mean())})
    return pd.DataFrame(num),pd.DataFrame(cat)


def team_analysis(df,preds):
    rows=[]
    for s in MAIN:
        part=df[df.season.eq(s)]; y=part[TARGET].to_numpy(float); p=preds[s]
        for team in sorted(part.pitcher_team_id.dropna().unique()):
            m=part.pitcher_team_id.eq(team).to_numpy()
            rows.append({"season":s,"team":team,"rows":int(m.sum()),"row_share":m.mean(),
                         "target_rate":y[m].mean(),"prediction_mean":p[m].mean(),
                         "brier":np.mean((y[m]-p[m])**2),"calibration_error":y[m].mean()-p[m].mean()})
    out=pd.DataFrame(rows)
    shares=out.pivot(index="team",columns="season",values="row_share")
    out=out.merge((shares.get(2023)-shares.get(2022)).rename("share_23_minus_22"),on="team",how="left")
    out=out.merge((shares.get(2024)-shares.get(2023)).rename("share_24_minus_23"),on="team",how="left")
    return out


def error_subgroups(df,preds,state):
    rows=[]
    for s in MAIN:
        part=df[df.season.eq(s)].copy(); y=part[TARGET].to_numpy(float); p=preds[s]; err=(y-p)**2
        current=state.loc[part.index]
        descriptors={"target":y,"prediction":p,"pitcher_history":part.asof_pitcher_n.to_numpy(float),
                     "batter_history":part.asof_batter_n.to_numpy(float),"inning":part.inning.to_numpy(float),
                     "cold_pitcher":part.asof_pitcher_n.fillna(0).le(0).to_numpy(float),
                     "cold_batter":part.asof_batter_n.fillna(0).le(0).to_numpy(float),
                     "cur_p_succ":current.cur_p_succ.to_numpy(float),"cur_b_succ":current.cur_b_succ.to_numpy(float)}
        for pct in (1,5,10,20):
            m=err>=np.quantile(err,1-pct/100)
            for name,val in descriptors.items(): rows.append({"season":s,"top_pct":pct,"feature":name,"value":np.nanmean(val[m]),"all_value":np.nanmean(val)})
        # Deployable subgroup error table.
        groups={"pitcher_cold":part.asof_pitcher_n.fillna(0).le(0),"pitcher_low_lt100":part.asof_pitcher_n.fillna(0).lt(100),
                "batter_cold":part.asof_batter_n.fillna(0).le(0),"batter_low_lt100":part.asof_batter_n.fillna(0).lt(100),
                "game_F":part.game_type.astype(str).eq("F"),"same_hand":part.pitcher_hand.astype(str).eq(part.batter_hand.astype(str)),
                "late_inning":part.inning.ge(7)}
        for name,m in groups.items():
            m=m.to_numpy(); rows.append({"season":s,"top_pct":0,"feature":name,"value":np.mean(err[m]) if m.any() else np.nan,"all_value":np.mean(err)})
    return pd.DataFrame(rows)


def load_candidates(df,oof,upstream):
    store={}
    for s in MAIN:
        # Two-seed averages where available.
        role=np.mean([np.load(ROOT/f"artifacts/backtest/role_state/{s}_candidate_s{x}.npy") for x in (0,1)],axis=0)
        hard=np.mean([np.load(ROOT/f"artifacts/backtest/hard_example/{s}_hard_s{x}.npy") for x in (42,43)],axis=0)
        unc=np.mean([np.load(ROOT/f"artifacts/backtest/hard_example/{s}_uncertainty_0p5_s{x}.npy") for x in (42,43)],axis=0)
        z=np.load(upstream/f"artifacts/lupi_aux_oof/cache/season_{s}.npz")
        d=np.load(upstream/f"artifacts/lupi_distill_oof/cache/season_{s}.npz")
        store[s]={"role_state":role,"hard_weight":hard,"uncertainty":unc,
                  "lupi_aux":z["multi_seed42"].astype(float),"lupi_distill":d["student_alpha_050"].astype(float)}
        # Deterministic EB replay.
        prep=eb_prepare(df); lookup=fit_lookups(prep[prep.season.lt(s)])
        store[s]["hierarchical_eb"]=prior_prediction(eb_transform(prep[prep.season.eq(s)].drop(columns=[TARGET]),lookup,500))
    return store


def correction_analysis(df,preds,candidates):
    rows=[]
    for s in MAIN:
        part=df[df.season.eq(s)]; y=part[TARGET].to_numpy(float); p=preds[s]
        buckets={"all":np.ones(len(part),bool),"pitcher_low_lt100":part.asof_pitcher_n.fillna(0).lt(100).to_numpy(),
                 "batter_low_lt100":part.asof_batter_n.fillna(0).lt(100).to_numpy(),"game_F":part.game_type.astype(str).eq("F").to_numpy(),
                 "same_hand":part.pitcher_hand.astype(str).eq(part.batter_hand.astype(str)).to_numpy(),"late_inning":part.inning.ge(7).to_numpy()}
        for name,c in candidates[s].items():
            gain=(y-p)**2-(y-c)**2; correction=c-p
            effective=gain>=np.quantile(gain,.9)
            buckets2={**buckets,"top10_effective":effective,"large_abs_correction":np.abs(correction)>=np.quantile(np.abs(correction),.9)}
            for group,m in buckets2.items():
                rows.append({"season":s,"candidate":name,"subgroup":group,"n":int(m.sum()),"brier_gain":float(gain[m].mean()),
                             "correction_mean":float(correction[m].mean()),"correction_abs_mean":float(np.abs(correction[m]).mean())})
    return pd.DataFrame(rows)


def regime_classifier(df,features,out,filename="regime_classifier_importance.csv"):
    sample=pd.concat([df[df.season.eq(s)].sample(n=min(100000,(df.season==s).sum()),random_state=42) for s in MAIN])
    X=sample[list(features)].copy(); y=sample.season.eq(2023).astype(int).to_numpy()
    cats=[]
    for c in X:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c]=pd.factorize(X[c].fillna("__NA__").astype(str),sort=True)[0]; cats.append(c)
        else: X[c]=pd.to_numeric(X[c],errors="coerce")
    X=X.replace([np.inf,-np.inf],np.nan)
    tr,te=train_test_split(np.arange(len(X)),test_size=.3,random_state=42,stratify=y)
    model=lgb.LGBMClassifier(n_estimators=120,num_leaves=15,max_depth=4,learning_rate=.05,
                             min_child_samples=1000,subsample=.8,colsample_bytree=.8,reg_lambda=10,
                             random_state=42,n_jobs=6,verbosity=-1)
    model.fit(X.iloc[tr],y[tr],categorical_feature=cats)
    pred=model.predict_proba(X.iloc[te])[:,1]; auc=roc_auc_score(y[te],pred)
    imp=pd.DataFrame({"feature":features,"gain_importance":model.booster_.feature_importance("gain")}).sort_values("gain_importance",ascending=False)
    imp["gain_share"]=imp.gain_importance/imp.gain_importance.sum()
    imp.to_csv(out/filename,index=False)
    return float(auc)


def shift_ranking(num, cat):
    rows=[]
    for feature, group in num.groupby("feature"):
        d={(int(r["from"]),int(r["to"])):r for _,r in group.iterrows()}
        a,b,c=d[(2022,2023)],d[(2023,2024)],d[(2022,2024)]
        rows.append({"feature":feature,"type":"numeric","shift_22_23":a.ks,"shift_23_24":b.ks,
                     "shift_22_24":c.ks,"specificity":min(a.ks,b.ks)-.5*c.ks,
                     "direction_2023":a.mean_to-a.mean_from})
    for feature, group in cat.groupby("feature"):
        d={(int(r["from"]),int(r["to"])):r for _,r in group.iterrows()}
        a,b,c=d[(2022,2023)],d[(2023,2024)],d[(2022,2024)]
        rows.append({"feature":feature,"type":"categorical","shift_22_23":a.js,"shift_23_24":b.js,
                     "shift_22_24":c.js,"specificity":min(a.js,b.js)-.5*c.js,"direction_2023":np.nan})
    return pd.DataFrame(rows).sort_values("specificity",ascending=False)


def error_explanation(df,preds):
    rows=[]
    for season in MAIN:
        part=df[df.season.eq(season)]; y=part[TARGET].to_numpy(float); p=preds[season]; error=(y-p)**2
        groups={"game_F":part.game_type.astype(str).eq("F"),"game_R":part.game_type.astype(str).eq("R"),
                "pitcher_low_lt100":part.asof_pitcher_n.fillna(0).lt(100),"batter_low_lt100":part.asof_batter_n.fillna(0).lt(100),
                "same_hand":part.pitcher_hand.astype(str).eq(part.batter_hand.astype(str)),"late_inning":part.inning.ge(7)}
        for team in sorted(part.pitcher_team_id.dropna().unique()): groups[f"pitcher_team_{team}"]=part.pitcher_team_id.eq(team)
        for count in sorted(set(zip(part.balls_before,part.strikes_before))):
            groups[f"count_{count[0]}_{count[1]}"]=(part.balls_before.eq(count[0])&part.strikes_before.eq(count[1]))
        for name,mask in groups.items():
            m=mask.to_numpy();
            if m.any(): rows.append({"season":season,"subgroup":name,"n":int(m.sum()),"error":float(error[m].mean()),
                                     "target_rate":float(y[m].mean()),"prediction_mean":float(p[m].mean())})
    table=pd.DataFrame(rows); pivot=table.pivot(index="subgroup",columns="season",values="error")
    special=(pivot[2023]-(pivot[2022]+pivot[2024])/2).rename("error_2023_excess")
    return table.merge(special,on="subgroup",how="left").sort_values("error_2023_excess",ascending=False)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--train",type=Path,default=ROOT.parent/"open/data/train.csv")
    p.add_argument("--oof",type=Path,default=ROOT/"artifacts/backtest/champion_oof")
    p.add_argument("--upstream",type=Path,default=ROOT.parent/"LG-Aimers-9th")
    p.add_argument("--out",type=Path,default=ROOT/"artifacts/regime_shift"); args=p.parse_args(argv); args.out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.train,encoding="utf-8-sig"); preds={s:champion(args.oof,s) for s in MAIN}
    # ``season`` is the diagnostic label itself, never a valid discriminator input.
    raw=[c for c in df.columns if c not in ("row_id","season",TARGET)]
    summary,cal=season_summary(df,preds); summary.to_csv(args.out/"season_summary.csv",index=False); cal.to_csv(args.out/"calibration_by_season.csv",index=False)
    state=season_state.add_features_by_season(df); derived=pd.concat([df,state],axis=1)
    num,cat=distribution_shift(df[df.season.isin(MAIN)],raw); num.to_csv(args.out/"feature_shift.csv",index=False); cat.to_csv(args.out/"categorical_shift.csv",index=False)
    shift_ranking(num,cat).to_csv(args.out/"shift_ranking.csv",index=False)
    dnum,dcat=distribution_shift(derived[derived.season.isin(MAIN)],list(state.columns)); dnum.to_csv(args.out/"derived_shift.csv",index=False); dcat.to_csv(args.out/"derived_categorical_shift.csv",index=False)
    team_analysis(df,preds).to_csv(args.out/"team_composition.csv",index=False)
    error_subgroups(df,preds,state).to_csv(args.out/"champion_error_subgroups.csv",index=False)
    error_explanation(df,preds).to_csv(args.out/"error_explanation_ranking.csv",index=False)
    candidates=load_candidates(df,args.oof,args.upstream); correction_analysis(df,preds,candidates).to_csv(args.out/"candidate_correction_analysis.csv",index=False)
    auc=regime_classifier(df[df.season.isin(MAIN)],raw,args.out)
    temporal={"pitcher_id","batter_id","game_month","asof_pitcher_n","asof_batter_n","asof_pitcher_pitchmix_n"}
    restricted=[c for c in raw if c not in temporal]
    restricted_auc=regime_classifier(df[df.season.isin(MAIN)],restricted,args.out,"regime_classifier_restricted_importance.csv")
    result={"regime_classifier_auc":auc,"restricted_auc":restricted_auc,
            "restricted_excludes":sorted(temporal),"test_read":False,"submission_created":False,"seasons":MAIN}
    (args.out/"summary.json").write_text(json.dumps(result,indent=2),"utf-8"); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
