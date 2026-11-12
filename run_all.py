#!/usr/bin/env python3
"""End-to-end deterministic AI-QNSF reconstruction runner (chunked for reliability)."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

def run(args: list[object]) -> None:
    cmd = [sys.executable, *[str(x) for x in args]]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), check=True)

def aggregate(metrics, classwise):
    import pandas as pd
    numeric = [
        "accuracy_multiclass", "macro_f1_multiclass", "binary_accuracy", "binary_precision",
        "binary_attack_recall", "binary_f1", "binary_fpr", "binary_roc_auc",
        "binary_average_precision",
    ]
    agg = metrics.groupby(["model", "target_validation_fpr"], dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([str(z) for z in c if z]) if isinstance(c, tuple) else c for c in agg.columns]
    class_agg = classwise.groupby("class")[["precision", "recall", "f1", "support"]].mean().reset_index()
    return agg, class_agg

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=3, help="Primary splits per evaluation subprocess (default: 3).")
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 10:
        parser.error("--chunk-size must be between 1 and 10")
    started = time.time()
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    run([ROOT / "scripts" / "generate_synthetic_data.py", "--out-dir", DATA,
         "--primary-seed", "20261104", "--independent-seed", "20261105"])
    for item in RESULTS.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    with tempfile.TemporaryDirectory(prefix="ai_qnsf_eval_chunks_", dir=ROOT) as tmp:
        tmp_root = Path(tmp)
        chunk_dirs: list[Path] = []
        offsets = list(range(0, 10, args.chunk_size))
        for ix, offset in enumerate(offsets):
            count = min(args.chunk_size, 10 - offset)
            out = tmp_root / f"chunk_{ix + 1:02d}"
            chunk_dirs.append(out)
            cmd: list[object] = [
                ROOT / "scripts" / "evaluate.py",
                "--primary", DATA / "primary_36000_reconstructed.csv",
                "--independent", DATA / "independent_54000_reconstructed.csv",
                "--out-dir", out, "--repeats", count, "--repeat-offset", offset,
                "--trees", 100, "--bootstrap", 200, "--seed", 7319,
            ]
            if offset + count < 10:
                cmd.append("--skip-transfer")
            run(cmd)
        import pandas as pd
        metric_frames = [pd.read_csv(p / "primary_metrics_by_split.csv") for p in chunk_dirs]
        class_frames = [pd.read_csv(p / "primary_classwise_by_split.csv") for p in chunk_dirs]
        metrics = pd.concat(metric_frames, ignore_index=True).sort_values(
            ["split", "model", "target_validation_fpr"], kind="stable")
        classwise = pd.concat(class_frames, ignore_index=True).sort_values(
            ["split", "class"], kind="stable")
        split_manifest = []
        for p in chunk_dirs:
            split_manifest.extend(json.loads((p / "split_manifest.json").read_text(encoding="utf-8")))
        split_manifest.sort(key=lambda x: x["repeat"])
        metrics.to_csv(RESULTS / "primary_metrics_by_split.csv", index=False)
        classwise.to_csv(RESULTS / "primary_classwise_by_split.csv", index=False)
        metrics_agg, class_agg = aggregate(metrics, classwise)
        metrics_agg.to_csv(RESULTS / "primary_metrics_aggregate.csv", index=False)
        class_agg.to_csv(RESULTS / "primary_classwise_aggregate.csv", index=False)
        (RESULTS / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
        final_chunk = chunk_dirs[-1]
        for name in ["cross_generator_summary.json", "cross_generator_classification_report.csv",
                     "cross_generator_bootstrap.csv", "cross_generator_bootstrap_ci.json"]:
            shutil.copy2(final_chunk / name, RESULTS / name)
        final_summary = json.loads((final_chunk / "execution_summary.json").read_text(encoding="utf-8"))
    final_summary.update({
        "repeats": 10, "repeat_offset": 0,
        "batch_repeats": [min(args.chunk_size, 10 - off) for off in offsets],
        "full_pipeline_elapsed_seconds": round(time.time() - started, 2),
        "interpretation": "The complete v1.0.0 reconstructed synthetic pipeline completed in chunked subprocesses. This is a new reconstruction based on the manuscript's documented design, not the missing original experiment. It does not establish physical QKD attack detection or validate the integrated AI-QNSF risk/response framework.",
    })
    (RESULTS / "execution_summary.json").write_text(json.dumps(final_summary, indent=2) + "\n", encoding="utf-8")
    env_file = ROOT / "metadata" / "environment.json"
    if env_file.exists():
        env = json.loads(env_file.read_text(encoding="utf-8"))
        env["full_pipeline_command"] = "python run_all.py"
        env["primary_run_repeats"] = 10
        env["evaluation_batch_repeats"] = final_summary["batch_repeats"]
        env["independent_generator_bootstrap_resamples"] = 200
        env_file.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    run([ROOT / "scripts" / "write_checksums.py"])
    run([ROOT / "scripts" / "verify_archive.py"])
    print("Completed: deterministic generation, all ten primary run-level repeats, independent-generator transfer, bootstrap, aggregation, checksums, and archive verification.", flush=True)

if __name__ == "__main__":
    main()
