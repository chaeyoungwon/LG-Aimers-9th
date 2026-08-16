"""Train the monotone model and configure the scored 40% blend."""
import json
import lightgbm as lgb
import pandas as pd
from src.train_base import add_features, apply_category_maps, PARAMS

root="submission"
with open(f"{root}/model/meta.json",encoding="utf-8") as f: meta=json.load(f)
df=add_features(pd.read_csv("data/train.csv",encoding="utf-8-sig"))
df=apply_category_maps(df,meta["cat_cols"],meta["category_maps"])
features=meta["feature_cols"]
pos={"asof_pitcher_success_rate","asof_pitcher_strike_rate",
     "asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate",
     "asof_pitcher_prev5_game_success_rate","asof_batter_success_rate"}
neg={"asof_pitcher_reverse_rate","asof_pitcher_middle_rate","asof_pitcher_ball_rate",
     "asof_pitcher_prev1_game_middle_rate","asof_pitcher_prev3_game_middle_rate",
     "asof_pitcher_prev5_game_middle_rate","asof_batter_middle_rate"}
params=dict(PARAMS,seed=42,num_leaves=63,min_data_in_leaf=1200,num_threads=6,
            monotone_constraints=[1 if c in pos else -1 if c in neg else 0 for c in features],
            monotone_constraints_method="advanced")
d=lgb.Dataset(df[features],df["control_success"],categorical_feature=meta["cat_cols"])
m=lgb.train(params,d,220)
m.save_model(f"{root}/model/lgbm_mono63.txt")

meta["all_model_files"] = [
    "lgbm_eng31.txt", "lgbm_leaf63.txt", "lgbm_decay85.txt", "lgbm_mono63.txt"
]
meta["ensemble_groups"] = [
    {
        "model_files": ["lgbm_eng31.txt", "lgbm_leaf63.txt", "lgbm_decay85.txt"],
        "model_weights": [0.370886, 0.470361, 0.158753],
        "scale": 1.07022,
        "bias": -0.04495,
    },
    {
        "model_files": ["lgbm_leaf63.txt", "lgbm_decay85.txt", "lgbm_mono63.txt"],
        "model_weights": [0.263430, 0.045001, 0.691570],
        "scale": 1.021134,
        "bias": -0.038516,
    },
]
meta["group_weights"] = [0.60, 0.40]
with open(f"{root}/model/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
