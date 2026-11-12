# AI-QNSF reconstructed synthetic experiment (v1.0.0)

> **Provenance:** This is a newly reconstructed synthetic experiment based on the feature list, scenario definitions, dataset sizes, and split design documented in the AI-QNSF manuscript. The earlier original generator/evaluation scripts and CSV files were not available. This package is not an exact reproduction of that earlier unarchived experiment.

> **Evidence boundary:** The data contain generated scenario labels and synthetic feature relationships. They do not model photon behaviour, BB84/E91 execution, optical propagation, quantum detector physics, a QKD simulator, live network packets, deployed telemetry, or physical QKD hardware. Results demonstrate learnability within this reconstructed synthetic process; they are not evidence of physical QKD attack detection.

## Contents
- `scripts/generate_synthetic_data.py`: deterministic generation of primary and independent datasets.
- `scripts/evaluate.py`: run-grouped evaluation, baselines, source-validation threshold selection, transfer evaluation and bootstrap.
- `scripts/verify_archive.py`: checksums, counts, split separation and results completeness.
- `run_all.py`: full pipeline entry point.
- `data/`: two reconstructed synthetic datasets and dataset manifest.
- `results/`: split IDs, per-split/aggregate metrics, class-wise metrics, transfer results and bootstrap outputs.
- `metadata/`: experiment specification and software environment.
- `PAPER_RESULTS_CROSSWALK.md`: link between paper tables and result files.
- `REPRODUCTION_STATUS.md`: reproduction and provenance status.
- `SHA256SUMS.txt`: package integrity manifest.

## Datasets
Primary dataset: 36,000 observations, 120 runs, six scenario classes, 13 numerical features.
Independent dataset: 54,000 observations, 180 runs, same six class definitions and feature names, generated using a different stochastic process.

The 13 features are QBER, key rate, channel loss, detector efficiency, dark-count rate, authentication failures, configuration events, latency, synchronisation error, telemetry missing fraction, telemetry age, channel noise and optical power. Run ID, time index, distance, scenario and label are metadata and are excluded from model inputs.

## Reproduce
Use Python 3.13.5 and install the pinned dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_all.py
```

Verify the package:

```bash
python scripts/verify_archive.py
```

The runner regenerates both datasets, executes ten scenario-stratified run-level splits (72/24/24 runs per split), performs the independent-generator evaluation and 200 run-cluster bootstrap resamples, merges results, writes SHA-256 checksums, and runs the verifier. See `REPRODUCTION_STATUS.md` for the exact recorded environment and execution status.

## Interpretation
The independent-generator false-positive rate is high (25.13%) and control-plane proxy recall is low (6.15%). This is a transfer failure, not successful generalisation. The experiments do not validate the full AI-QNSF A/K/E/V risk-fusion design, uncertainty U, abstention mechanism, response controller, simulator integration, operational latency/cost, or physical QKD operation.

## Citation
See `CITATION.cff`. Please cite the manuscript and identify the exact package version used.

## License
No open-source licence has been assigned. All rights are reserved by the copyright holder unless a separate licence is added.
