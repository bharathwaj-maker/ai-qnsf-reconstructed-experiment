#!/usr/bin/env python3
"""Evaluate reconstructed synthetic AI-QNSF feature-level datasets.

This program creates run-disjoint splits and evaluates Random Forest,
Isolation Forest and a QBER-only anomaly baseline. All labels are synthetic.
"""
from __future__ import annotations
import argparse, json, hashlib, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score,
    precision_recall_fscore_support)

FEATURES = ["qber", "key_rate", "channel_loss", "detector_efficiency", "dark_count_rate",
            "authentication_failures", "configuration_events", "latency_ms",
            "synchronisation_error", "telemetry_missing_fraction", "telemetry_age_ms",
            "channel_noise", "optical_power"]
CLASSES = ["normal", "channel_disturbance", "intercept_resend_proxy", "detector_anomaly",
           "control_plane_tampering", "telemetry_dropout"]
TARGETS = [0.005, 0.01, 0.02]

def run_stratified_split(df: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in CLASSES:
        ids = np.array(sorted(df.loc[df.scenario == c, "run_id"].unique()))
        rng.shuffle(ids)
        if len(ids) == 20:
            ntr, nv, nte = 12, 4, 4
        elif len(ids) == 30:
            ntr, nv, nte = 18, 6, 6
        else:
            raise ValueError(f"Unexpected runs per class: {c}={len(ids)}")
        train += ids[:ntr].tolist()
        val += ids[ntr:ntr+nv].tolist()
        test += ids[ntr+nv:ntr+nv+nte].tolist()
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return train, val, test

def tune_threshold(scores, y_true, target_fpr):
    normal_scores = np.asarray(scores)[np.asarray(y_true) == "normal"]
    if len(normal_scores) == 0:
        raise ValueError("No normal validation rows for threshold selection")
    return float(np.quantile(normal_scores, 1-target_fpr, method="higher"))

def classify_from_attack_scores(class_prob, threshold, classes):
    attack_cols = [i for i, c in enumerate(classes) if c != "normal"]
    attack_scores = class_prob[:, attack_cols].sum(axis=1)
    predicted_attack_idx = np.argmax(class_prob[:, attack_cols], axis=1)
    attack_names = np.array([classes[i] for i in attack_cols])
    pred = np.full(len(class_prob), "normal", dtype=object)
    mask = attack_scores > threshold
    pred[mask] = attack_names[predicted_attack_idx[mask]]
    return pred, attack_scores

def metric_row(y_true, y_pred, scores, threshold, model, split, target=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    truth = (y_true != "normal").astype(int); pred = (y_pred != "normal").astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    try: auc = roc_auc_score(truth, scores)
    except Exception: auc = float("nan")
    try: ap = average_precision_score(truth, scores)
    except Exception: ap = float("nan")
    return {"split": split, "model": model, "target_validation_fpr": target,
            "threshold": float(threshold), "accuracy_multiclass": accuracy_score(y_true, y_pred),
            "binary_accuracy": accuracy_score(truth, pred),
            "macro_f1_multiclass": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
            "binary_precision": precision_score(truth, pred, zero_division=0),
            "binary_attack_recall": recall_score(truth, pred, zero_division=0),
            "binary_f1": f1_score(truth, pred, zero_division=0),
            "binary_fpr": fp/(fp+tn) if (fp+tn) else float("nan"),
            "binary_roc_auc": auc, "binary_average_precision": ap,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="data/primary_36000_reconstructed.csv")
    ap.add_argument("--independent", default="data/independent_54000_reconstructed.csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7319)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--repeat-offset", type=int, default=0)
    ap.add_argument("--skip-transfer", action="store_true")
    args = ap.parse_args()
    start = time.time(); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(args.primary); independent = pd.read_csv(args.independent)
    for name, df, n in [("primary", primary, 120), ("independent", independent, 180)]:
        if len(df) != n*300 or df.run_id.nunique() != n:
            raise ValueError(f"{name} dataset does not match expected rows/runs")
        if not set(FEATURES).issubset(df.columns):
            raise ValueError(f"Missing features in {name}")
    rows=[]; class_rows=[]
    for rep in range(args.repeats):
        repeat_id = args.repeat_offset + rep + 1
        repeat_seed = args.seed + args.repeat_offset + rep
        print(f"Starting primary repeat {repeat_id}", flush=True)
        tr, va, te = run_stratified_split(primary, repeat_seed)
        split_manifest = {"repeat": repeat_id, "train_runs": tr, "validation_runs": va,
                          "test_runs": te, "pairwise_overlap": 0}
        train=primary[primary.run_id.isin(tr)]; val=primary[primary.run_id.isin(va)]
        test=primary[primary.run_id.isin(te)]
        Xtr=train[FEATURES].to_numpy(); Xv=val[FEATURES].to_numpy(); Xt=test[FEATURES].to_numpy()
        ytr=train.label.to_numpy(); yv=val.label.to_numpy(); yt=test.label.to_numpy()
        rf=RandomForestClassifier(n_estimators=args.trees,max_features="sqrt",min_samples_leaf=2,
                                  class_weight="balanced_subsample",random_state=repeat_seed,n_jobs=1)
        rf.fit(Xtr,ytr); pval=rf.predict_proba(Xv); ptest=rf.predict_proba(Xt)
        model_classes=rf.classes_.tolist()
        probs_val=np.zeros((len(val),len(CLASSES))); probs_test=np.zeros((len(test),len(CLASSES)))
        for j,c in enumerate(model_classes):
            probs_val[:,CLASSES.index(c)]=pval[:,j]
            probs_test[:,CLASSES.index(c)]=ptest[:,j]
        rf_val_attack=probs_val[:,1:].sum(axis=1)
        for target in TARGETS:
            thr=tune_threshold(rf_val_attack,yv,target)
            pred,scores=classify_from_attack_scores(probs_test,thr,CLASSES)
            rows.append(metric_row(yt,pred,scores,thr,"RandomForest",repeat_id,target))
        normal_train=train[train.label=="normal"][FEATURES].to_numpy()
        iso=IsolationForest(n_estimators=args.trees,contamination="auto",random_state=repeat_seed,n_jobs=1)
        iso.fit(normal_train); s_val=-iso.score_samples(Xv); s_test=-iso.score_samples(Xt)
        for target in TARGETS:
            thr=tune_threshold(s_val,yv,target); binary=s_test>thr
            pred=np.where(binary,"attack_proxy","normal")
            row=metric_row(yt,pred,s_test,thr,"IsolationForest",repeat_id,target)
            row["macro_f1_multiclass"]=float("nan"); rows.append(row)
        qv=val.qber.to_numpy(); qt=test.qber.to_numpy()
        for target in TARGETS:
            thr=tune_threshold(qv,yv,target); binary=qt>thr
            pred=np.where(binary,"attack_proxy","normal")
            row=metric_row(yt,pred,qt,thr,"QBER_only",repeat_id,target)
            row["macro_f1_multiclass"]=float("nan"); rows.append(row)
        p1=probs_val[:,1:].sum(axis=1); thr1=tune_threshold(p1,yv,0.01)
        pred1,_=classify_from_attack_scores(probs_test,thr1,CLASSES)
        from sklearn.metrics import precision_recall_fscore_support
        p,r,f,s=precision_recall_fscore_support(yt,pred1,labels=CLASSES,zero_division=0)
        for c,pp,rr,ff,ss in zip(CLASSES,p,r,f,s):
            class_rows.append({"split":repeat_id,"class":c,"precision":pp,"recall":rr,"f1":ff,"support":int(ss)})
        (out/f"split_{repeat_id:02d}.json").write_text(json.dumps(split_manifest,indent=2))
        print(f"Finished primary repeat {repeat_id}", flush=True)
    metrics=pd.DataFrame(rows); metrics.to_csv(out/"primary_metrics_by_split.csv",index=False)
    class_df=pd.DataFrame(class_rows); class_df.to_csv(out/"primary_classwise_by_split.csv",index=False)
    numeric=["accuracy_multiclass","macro_f1_multiclass","binary_accuracy","binary_precision",
             "binary_attack_recall","binary_f1","binary_fpr","binary_roc_auc","binary_average_precision"]
    aggregate=metrics.groupby(["model","target_validation_fpr"],dropna=False)[numeric].agg(["mean","std"]).reset_index()
    aggregate.columns=["_".join([str(z) for z in c if z]) if isinstance(c,tuple) else c for c in aggregate.columns]
    aggregate.to_csv(out/"primary_metrics_aggregate.csv",index=False)
    class_agg=class_df.groupby("class")[["precision","recall","f1","support"]].mean().reset_index()
    class_agg.to_csv(out/"primary_classwise_aggregate.csv",index=False)
    split_files=sorted(out.glob("split_*.json"))
    (out/"split_manifest.json").write_text(json.dumps([json.loads(p.read_text()) for p in split_files],indent=2))
    cross=None; threshold=None
    if not args.skip_transfer:
        tr,va,_=run_stratified_split(primary,args.seed)
        train=primary[primary.run_id.isin(tr)]; val=primary[primary.run_id.isin(va)]
        cross_rf=RandomForestClassifier(n_estimators=args.trees,max_features="sqrt",min_samples_leaf=2,
                     class_weight="balanced_subsample",random_state=args.seed,n_jobs=1)
        cross_rf.fit(train[FEATURES],train.label); classes=cross_rf.classes_.tolist()
        vp=cross_rf.predict_proba(val[FEATURES]); ip=cross_rf.predict_proba(independent[FEATURES])
        pv=np.zeros((len(val),len(CLASSES))); pi=np.zeros((len(independent),len(CLASSES)))
        for j,c in enumerate(classes):
            pv[:,CLASSES.index(c)]=vp[:,j]; pi[:,CLASSES.index(c)]=ip[:,j]
        val_scores=pv[:,1:].sum(axis=1); threshold=tune_threshold(val_scores,val.label.to_numpy(),0.01)
        pred,scores=classify_from_attack_scores(pi,threshold,CLASSES); yind=independent.label.to_numpy()
        cross=metric_row(yind,pred,scores,threshold,"RandomForest_cross_generator",0,0.01)
        cross["source_validation_threshold_target"]="1%"
        cross["source_training_runs"]=len(tr); cross["source_validation_runs"]=len(va)
        cross["independent_test_runs"]=independent.run_id.nunique()
        (out/"cross_generator_summary.json").write_text(json.dumps(cross,indent=2))
        report=classification_report(yind,pred,labels=CLASSES,output_dict=True,zero_division=0)
        pd.DataFrame(report).T.to_csv(out/"cross_generator_classification_report.csv")
        rng=np.random.default_rng(args.seed+999); y_true_bin=(yind!="normal").astype(int)
        y_pred_bin=(pred!="normal").astype(int); run_confusion={}
        run_scenario=independent.drop_duplicates("run_id").set_index("run_id").scenario.to_dict()
        for rid, idx in independent.groupby("run_id").indices.items():
            vals=confusion_matrix(y_true_bin[idx],y_pred_bin[idx],labels=[0,1]).ravel()
            run_confusion[rid]=vals
        by_class={c:[rid for rid in run_confusion if run_scenario[rid]==c] for c in CLASSES}
        boot=[]
        for b in range(args.bootstrap):
            sampled=[]
            for c,ids in by_class.items():
                sampled.extend(rng.choice(ids,size=len(ids),replace=True).tolist())
            tn,fp,fn,tp=np.sum([run_confusion[rid] for rid in sampled],axis=0)
            f1v=(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.0
            fprv=(fp/(fp+tn)) if (fp+tn) else float("nan")
            recv=(tp/(tp+fn)) if (tp+fn) else 0.0
            boot.append({"replicate":b+1,"attack_f1":f1v,"fpr":fprv,"attack_recall":recv})
        boot_df=pd.DataFrame(boot); boot_df.to_csv(out/"cross_generator_bootstrap.csv",index=False)
        ci={m:{"lower_2_5":float(boot_df[m].quantile(.025)),"upper_97_5":float(boot_df[m].quantile(.975)),
               "mean":float(boot_df[m].mean())} for m in ["attack_f1","fpr","attack_recall"]}
        (out/"cross_generator_bootstrap_ci.json").write_text(json.dumps(ci,indent=2))
        cross["bootstrap_ci"]=ci; (out/"cross_generator_summary.json").write_text(json.dumps(cross,indent=2))
    summary={"status":"RECONSTRUCTED_NEW_SYNTHETIC_EXPERIMENT_NOT_ORIGINAL_REPRODUCTION",
      "python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,
      "pandas":pd.__version__,"scikit_learn":sklearn.__version__,
      "primary_rows":len(primary),"primary_runs":primary.run_id.nunique(),
      "independent_rows":len(independent),"independent_runs":independent.run_id.nunique(),
      "repeats":args.repeats,"trees":args.trees,"bootstrap_resamples":args.bootstrap,
      "split_seed":args.seed,"repeat_offset":args.repeat_offset,"cross_generator_threshold":threshold,
      "cross_generator":cross,"elapsed_seconds":round(time.time()-start,2),
      "interpretation":"This is a newly reconstructed synthetic benchmark based on the manuscript's documented design. It is not the missing original experiment and does not reproduce the manuscript's former reported metrics. Results must be reviewed and the manuscript's numbers updated before submission."}
    (out/"execution_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
