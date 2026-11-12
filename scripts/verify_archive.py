#!/usr/bin/env python3
"""Validate checksums and internal consistency of the AI-QNSF reconstruction."""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["normal", "channel_disturbance", "intercept_resend_proxy",
           "detector_anomaly", "control_plane_tampering", "telemetry_dropout"]
FEATURES = ["qber", "key_rate", "channel_loss", "detector_efficiency", "dark_count_rate",
            "authentication_failures", "configuration_events", "latency_ms",
            "synchronisation_error", "telemetry_missing_fraction", "telemetry_age_ms",
            "channel_noise", "optical_power"]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def require(condition: bool, message: str) -> None:
    if not condition: raise AssertionError(message)

def check_checksums() -> int:
    sums=ROOT/"SHA256SUMS.txt"
    require(sums.exists(),"SHA256SUMS.txt is missing; run scripts/write_checksums.py")
    checked=0
    for line_no,raw in enumerate(sums.read_text(encoding="utf-8").splitlines(),1):
        if not raw.strip(): continue
        try: expected,rel=raw.split("  ",1)
        except ValueError as exc: raise AssertionError(f"Malformed checksum entry on line {line_no}") from exc
        path=(ROOT/rel).resolve()
        require(path.is_relative_to(ROOT.resolve()),f"Unsafe checksum path: {rel}")
        require(path.exists() and path.is_file(),f"Missing packaged file: {rel}")
        require(sha256(path)==expected,f"SHA-256 mismatch: {rel}")
        checked+=1
    require(checked>0,"Checksum file contains no entries")
    return checked

def scenario_from_run_id(run_id: str) -> str:
    match=re.match(r"^primary_(.+)_\d{3}$",str(run_id))
    require(match is not None,f"Unrecognised primary run identifier: {run_id}")
    scenario=match.group(1)
    require(scenario in CLASSES,f"Unexpected scenario in run identifier: {scenario}")
    return scenario

def main() -> None:
    checksums=check_checksums()
    spec=json.loads((ROOT/"metadata"/"experiment_spec.json").read_text(encoding="utf-8"))
    manifest=json.loads((ROOT/"data"/"dataset_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("status")=="RECONSTRUCTED_NEW_SYNTHETIC_EXPERIMENT_NOT_ORIGINAL_REPRODUCTION",
            "Dataset manifest does not carry the reconstruction provenance marker")
    primary_path=ROOT/"data"/"primary_36000_reconstructed.csv"
    independent_path=ROOT/"data"/"independent_54000_reconstructed.csv"
    primary=pd.read_csv(primary_path); independent=pd.read_csv(independent_path)
    for name,frame,expected_rows,expected_runs in [
        ("primary",primary,36000,120),("independent",independent,54000,180)]:
        require(len(frame)==expected_rows,f"{name} row count mismatch")
        require(frame["run_id"].nunique()==expected_runs,f"{name} run count mismatch")
        require(set(FEATURES).issubset(frame.columns),f"{name} is missing one or more model features")
        require(set(CLASSES)==set(frame["scenario"].unique()),f"{name} scenario set mismatch")
        require(set(CLASSES)==set(frame["label"].unique()),f"{name} label set mismatch")
        require(frame["label"].value_counts().to_dict()==manifest[name]["class_counts"],
                f"{name} class counts do not match dataset_manifest.json")
        filename="primary_36000_reconstructed.csv" if name=="primary" else "independent_54000_reconstructed.csv"
        require(sha256(ROOT/"data"/filename)==manifest[name]["sha256"],
                f"{name} dataset hash does not match dataset_manifest.json")
    split_manifest=json.loads((ROOT/"results"/"split_manifest.json").read_text(encoding="utf-8"))
    require(len(split_manifest)==10,"Expected ten split manifests")
    for entry in split_manifest:
        train=set(entry["train_runs"]); val=set(entry["validation_runs"]); test=set(entry["test_runs"])
        require((len(train),len(val),len(test))==(72,24,24),f"Split {entry.get('repeat')} has unexpected sizes")
        require(not (train&val or train&test or val&test),f"Split {entry.get('repeat')} contains run overlap")
        require(entry.get("pairwise_overlap")==0,f"Split {entry.get('repeat')} overlap marker is not zero")
        require(train|val|test==set(primary["run_id"].unique()),f"Split {entry.get('repeat')} does not partition all runs")
        for subset,expected_per_class in [(train,12),(val,4),(test,4)]:
            counts={}
            for run_id in subset:
                c=scenario_from_run_id(run_id); counts[c]=counts.get(c,0)+1
            require(all(counts.get(c,0)==expected_per_class for c in CLASSES),
                    f"Split {entry.get('repeat')} has non-stratified run allocation")
    metrics=pd.read_csv(ROOT/"results"/"primary_metrics_by_split.csv")
    require(len(metrics)==90,"Expected 90 primary model/threshold metric rows")
    require(metrics["split"].nunique()==10,"Primary metric output does not contain ten repeats")
    require(set(metrics["model"].unique())=={"RandomForest","IsolationForest","QBER_only"},
            "Primary metric output has unexpected models")
    require(set(metrics["target_validation_fpr"].round(3).unique())=={0.005,0.01,0.02},
            "Primary metric output has unexpected validation targets")
    require(not metrics.duplicated(["split","model","target_validation_fpr"]).any(),
            "Duplicate split/model/target metric rows found")
    classwise=pd.read_csv(ROOT/"results"/"primary_classwise_by_split.csv")
    require(len(classwise)==60,"Expected 60 primary class-wise rows")
    require(classwise["split"].nunique()==10,"Class-wise metrics do not contain ten repeats")
    require(set(classwise["class"].unique())==set(CLASSES),"Class-wise metrics have unexpected classes")
    cross=json.loads((ROOT/"results"/"cross_generator_summary.json").read_text(encoding="utf-8"))
    total=sum(int(cross[k]) for k in ["tn","fp","fn","tp"])
    require(total==54000,"Cross-generator confusion counts do not sum to 54,000")
    fpr=cross["fp"]/(cross["fp"]+cross["tn"])
    recall=cross["tp"]/(cross["tp"]+cross["fn"])
    require(abs(fpr-cross["binary_fpr"])<1e-10,"Cross-generator FPR mismatch")
    require(abs(recall-cross["binary_attack_recall"])<1e-10,"Cross-generator recall mismatch")
    require(cross.get("source_validation_threshold_target")=="1%",
            "Cross-generator threshold target should be 1% source-validation FPR")
    print(f"PASS: {checksums} SHA-256 entries verified.")
    print("PASS: both datasets, feature lists, run/class counts, and dataset hashes verified.")
    print("PASS: ten stratified 72/24/24 run-disjoint splits verified.")
    print("PASS: 90 primary metric rows and 60 class-wise rows verified.")
    print("PASS: independent-generator confusion counts and binary metrics verified.")
    print("PROVENANCE: this validates the reconstructed package only, not the missing original experiment.")

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"FAIL: {exc}",file=sys.stderr); raise
