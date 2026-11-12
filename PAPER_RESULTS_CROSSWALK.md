# Manuscript-to-package results crosswalk

This file maps the reconstructed experiment outputs to the manuscript. Keep the package and manuscript at the same version.

| Manuscript location | Reproducibility source | Contents |
|---|---|---|
| §7.4 and Tables VI–VII | `data/primary_36000_reconstructed.csv`, `data/independent_54000_reconstructed.csv`, `data/dataset_manifest.json` | Generated datasets, row/run/class counts, feature set and dataset hashes. |
| §7.5 and Table VIII | `metadata/environment.json`, `metadata/experiment_spec.json`, `requirements.txt` | Runtime versions, seeds, estimator settings, dependency pins and non-claims. |
| §7.6 and Table IX | `results/split_manifest.json` | Ten stratified 72/24/24 train/validation/test run allocations and overlap checks. |
| §8.2 and Table XII | `results/primary_metrics_by_split.csv`, `results/primary_metrics_aggregate.csv` | Per-split and aggregate primary metrics for Random Forest, Isolation Forest and QBER-only baselines. |
| §8.2 and Table XIII | `results/primary_classwise_by_split.csv`, `results/primary_classwise_aggregate.csv` | Per-class precision, recall, F1 and support. |
| §8.10 and Table XIV | `results/cross_generator_summary.json`, `results/cross_generator_classification_report.csv`, `results/cross_generator_bootstrap.csv`, `results/cross_generator_bootstrap_ci.json` | Independent-generator transfer metrics, class-wise results and 200 run-cluster bootstrap resamples. |
| §7.8 / Section 6 status summary | `metadata/experiment_spec.json`, `REPRODUCTION_STATUS.md` | Clear boundary between evaluated synthetic classification and the proposed, unimplemented AI-QNSF risk/response design. |

## Key reconstructed values

At the 1% normal-validation-FPR target, primary Random Forest mean attack-proxy recall is 55.89% (SD 9.15 percentage points), observed test FPR is 1.80% (SD 1.27 pp), binary accuracy is 90.71%, binary F1 is 67.65%, and multiclass macro-F1 is 60.60%.

On the independent synthetic generator, attack-proxy recall is 60.30%, binary F1 is 45.76%, and observed FPR is 25.13%. Control-plane proxy recall is 6.15%. This is poor threshold transfer, not successful generalisation.

These are generated-label classification results only. They do not demonstrate physical QKD attack detection, sensor validity, full AI-QNSF risk fusion, calibrated uncertainty, abstention, response safety, simulator performance, latency, or operational cost.
