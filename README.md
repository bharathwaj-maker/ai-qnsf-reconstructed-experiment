# AI-QNSF reconstructed synthetic experiment (v1.0.0)

> **Provenance:** This is a new synthetic reconstruction based on the feature list, scenario definitions, dataset sizes and split design documented in the AI-QNSF manuscript. The earlier original generator/evaluation scripts and CSV files were not available. This is not an exact reproduction of the earlier unarchived experiment.

> **Evidence boundary:** Generated scenario labels and synthetic feature relationships are used. The data do not model photon behaviour, BB84/E91 execution, optical propagation, quantum detector physics, a QKD simulator, live network packets, deployed telemetry or physical QKD hardware. Results demonstrate learnability within this reconstructed synthetic process; they are not evidence of physical QKD attack detection.

## Repository contents

- `scripts/generate_synthetic_data.py`: deterministic generation of both datasets.
- `scripts/evaluate.py`: run-grouped evaluation with Random Forest, Isolation Forest and QBER-only baselines, validation-only threshold selection, cross-generator transfer and run-cluster bootstrap.
- `scripts/write_checksums.py` and `scripts/verify_archive.py`: checksum and internal-consistency verification.
- `run_all.py`: complete end-to-end pipeline.
- `metadata/`: experiment specification and recorded software environment.
- `PAPER_RESULTS_CROSSWALK.md`: mapping from manuscript tables to result files.
- `REPRODUCTION_STATUS.md`: provenance and reproduction status.
- `REPOSITORY_UPLOAD_CHECKLIST.md`: reviewer-access checklist.
- `CITATION.cff`: citation metadata.

The workflow generates and commits `data/`, `results/` and `SHA256SUMS.txt` after a successful run.

## Run the experiment on GitHub

1. Open the **Actions** tab.
2. Select **Reproduce datasets and results**.
3. Choose **Run workflow** on the `main` branch.
4. Wait for the workflow to finish. It installs pinned dependencies, regenerates both datasets, executes all ten primary run-level splits and the independent-generator test, performs 200 bootstrap resamples, verifies outputs and commits generated files to the repository.
5. Confirm that `data/`, `results/` and `SHA256SUMS.txt` appear on the default branch.

The workflow needs repository Actions write access to commit the generated outputs.

## Reproduce locally

Python 3.13.5 is the recorded environment. Install pinned dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_all.py
python scripts/verify_archive.py
```

## Dataset design

Primary dataset: 36,000 observations, 120 runs, 20 runs per each of six scenario classes.
Independent dataset: 54,000 observations, 180 runs, 30 runs per each of six scenario classes.

The 13 model features are QBER, key rate, channel loss, detector efficiency, dark-count rate, authentication failures, configuration events, latency, synchronisation error, telemetry missing fraction, telemetry age, channel noise and optical power. Run ID, time index, distance, scenario and label are metadata excluded from model inputs.

## Interpretation

The independent-generator false-positive rate is high (25.13%) and control-plane proxy recall is low (6.15%). This is a threshold-transfer failure, not successful generalisation. The experiments do not validate the full AI-QNSF A/K/E/V risk-fusion design, uncertainty U, abstention mechanism, response controller, simulator integration, operational latency/cost or physical QKD operation.

## Citation and licensing

See `CITATION.cff`. No open-source licence has been assigned; no additional reuse rights are granted by this repository.
