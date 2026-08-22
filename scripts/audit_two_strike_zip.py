"""CRC, isolated execution, and row-independence audit for final candidate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def load_runtime(work):
    spec=importlib.util.spec_from_file_location("two_strike_candidate_runtime",work/"script.py")
    mod=importlib.util.module_from_spec(spec); sys.path.insert(0,str(work))
    try: spec.loader.exec_module(mod)
    finally: sys.path.pop(0)
    return mod


def predict(mod,models,meta,frame,work):
    old=os.getcwd(); os.chdir(work)
    try:return mod.predict(frame,models,meta,verbose=False)
    finally:os.chdir(old)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--zip",type=Path,default=ROOT/"artifacts/sub_two_strike_lgbcat_w0p250.zip")
    p.add_argument("--train",type=Path,default=ROOT.parent/"open/data/train.csv"); p.add_argument("--out",type=Path,default=ROOT/"artifacts/two_strike_final/zip_audit.json")
    args=p.parse_args(); report={"zip":str(args.zip),"sha256":sha256(args.zip)}
    with zipfile.ZipFile(args.zip) as z:
        bad=z.testzip(); report["crc_passed"]=bad is None; report["crc_bad_file"]=bad; report["file_count"]=len(z.namelist())
        with tempfile.TemporaryDirectory(prefix="two_strike_audit_") as td:
            work=Path(td); z.extractall(work); meta=json.loads((work/"model/meta.json").read_text("utf-8")); mod=load_runtime(work)
            models=mod.load_models(str(work/"model"),meta)
            rows=pd.read_csv(args.train,encoding="utf-8-sig").tail(3000).drop(columns=["control_success"]).copy(); rows["season"]=2025
            rows["audit_id"]=np.arange(len(rows)); feature_rows=rows.drop(columns="audit_id")
            full=predict(mod,models,meta,feature_rows,work); reference=dict(zip(rows.audit_id,full)); variants={
                "full":rows,"single":rows.iloc[[0]],"shuffled":rows.sample(frac=1,random_state=42),
                "reversed":rows.iloc[::-1],"subset":rows.iloc[::5],
                "unrelated_mutation":rows.copy(),}
            variants["unrelated_mutation"].loc[variants["unrelated_mutation"].index[-1],"inning"]=99
            detail={}; all_delta=[]
            for name,current in variants.items():
                got=predict(mod,models,meta,current.drop(columns="audit_id"),work)
                delta=np.array([abs(float(v)-float(reference[i])) for i,v in zip(current.audit_id,got)])
                if name=="unrelated_mutation": delta=delta[:-1]
                detail[name]={"rows":len(delta),"max_abs_diff":float(delta.max()) if len(delta) else 0.,
                              "mean_abs_diff":float(delta.mean()) if len(delta) else 0.,"changed_rows":int(np.count_nonzero(delta))}
                all_delta.extend(delta.tolist())
            report["row_independence"]=detail; report["row_independence_max_abs_diff"]=max(all_delta)
            # Isolated CLI execution uses only extracted ZIP plus copied input CSVs.
            data=work/"data"; data.mkdir(); sample=feature_rows.head(5).copy(); sample.to_csv(data/"test.csv",index=False,encoding="utf-8-sig")
            pd.DataFrame({"row_id":sample.row_id,"control_success":0.5}).to_csv(data/"sample_submission.csv",index=False,encoding="utf-8-sig")
            env=dict(os.environ); proc=subprocess.run([sys.executable,"script.py"],cwd=work,env=env,text=True,capture_output=True,timeout=300)
            report["isolated_execution"]={"returncode":proc.returncode,"output_exists":(work/"output/submission.csv").is_file(),"stderr_tail":proc.stderr[-1000:]}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2),"utf-8"); print(json.dumps(report,indent=2))


if __name__=="__main__":main()
