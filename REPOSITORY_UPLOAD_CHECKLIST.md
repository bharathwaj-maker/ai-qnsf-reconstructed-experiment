# Repository upload and reviewer-access checklist

This repository contains the source code and documentation for a new reconstructed synthetic experiment. It is not an exact reproduction of the missing earlier experiment.

1. Open the **Actions** tab.
2. Select **Reproduce datasets and results**.
3. Click **Run workflow** on the `main` branch.
4. Wait for the workflow to complete successfully. It regenerates both datasets, runs all ten primary splits and the independent-generator evaluation, verifies outputs and commits generated data/results to `main`.
5. Confirm that `data/primary_36000_reconstructed.csv`, `data/independent_54000_reconstructed.csv`, `results/`, and `SHA256SUMS.txt` are present on the public default branch.
6. Open the repository in a private/incognito browser session to confirm reviewer access.
7. Add the actual repository URL to §7.5 and Appendix B of the manuscript. Record a DOI only after a trusted repository service issues it.
8. Ask an independent reviewer to clone the repository and run `python run_all.py` with the pinned dependencies.

Suggested local commands:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py
python scripts/verify_archive.py
```

The repository must retain the provenance note that these data and results belong to a new reconstruction, not the missing original experiment.
