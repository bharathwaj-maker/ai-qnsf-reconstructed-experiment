#!/usr/bin/env python3
"""Generate transparent, reconstructed synthetic AI-QNSF feature-level datasets.

IMPORTANT: This generator is a new reconstruction from the published manuscript
specification. It is NOT the missing original generator and cannot reproduce the
original CSV hashes or metrics. Labels are synthetic scenario proxies, not
physical QKD attack observations.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

FEATURES = [
    "qber", "key_rate", "channel_loss", "detector_efficiency",
    "dark_count_rate", "authentication_failures", "configuration_events",
    "latency_ms", "synchronisation_error", "telemetry_missing_fraction",
    "telemetry_age_ms", "channel_noise", "optical_power",
]
CLASSES = ["normal", "channel_disturbance", "intercept_resend_proxy",
           "detector_anomaly", "control_plane_tampering", "telemetry_dropout"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def make_run(rng: np.random.Generator, run_id: str, scenario: str,
             n_seconds: int, generator: str, run_index: int) -> pd.DataFrame:
    """Create one independent run with run-level heterogeneity and local events."""
    shift = generator == "independent"
    base_scale = 1.20 if shift else 1.0
    rho = 0.78 if shift else 0.62
    t = np.arange(n_seconds)
    qber0 = rng.normal(0.018 if not shift else 0.024, 0.0045 * base_scale)
    key0 = rng.normal(1.00 if not shift else 0.91, 0.12 * base_scale)
    loss0 = rng.normal(3.0 if not shift else 3.45, 0.65 * base_scale)
    eff0 = rng.normal(0.92 if not shift else 0.885, 0.035 * base_scale)
    dark0 = rng.normal(0.0009 if not shift else 0.0013, 0.00035 * base_scale)
    lat0 = rng.normal(18 if not shift else 23, 4.0 * base_scale)
    sync0 = rng.normal(0.20 if not shift else 0.28, 0.09 * base_scale)
    noise0 = rng.normal(0.012 if not shift else 0.017, 0.004 * base_scale)
    power0 = rng.normal(1.00 if not shift else 0.96, 0.08 * base_scale)

    def ar_noise(sd: float) -> np.ndarray:
        z = rng.normal(0, sd * base_scale, n_seconds)
        out = np.zeros(n_seconds)
        for i in range(n_seconds):
            out[i] = (rho * out[i-1] if i else 0.0) + z[i]
        return out

    qber = qber0 + ar_noise(0.0028)
    key_rate = key0 + ar_noise(0.055)
    channel_loss = loss0 + ar_noise(0.28)
    detector_eff = eff0 + ar_noise(0.012)
    dark_count = dark0 + ar_noise(0.00012)
    auth_fail = np.maximum(0, rng.poisson(0.025, n_seconds).astype(float))
    config_events = np.maximum(0, rng.poisson(0.018, n_seconds).astype(float))
    latency = lat0 + ar_noise(1.8)
    sync = sync0 + ar_noise(0.035)
    missing = np.clip(rng.beta(1.0, 45.0, n_seconds), 0, 0.18)
    age = np.maximum(0, rng.normal(25 if not shift else 34, 8 * base_scale, n_seconds))
    noise = noise0 + ar_noise(0.0025)
    optical = power0 + ar_noise(0.035)

    labels = np.full(n_seconds, "normal", dtype=object)
    event_start = None
    event_end = None
    if scenario != "normal":
        lo, hi = (25, 100) if not shift else (30, 115)
        duration = int(rng.integers(lo, hi + 1))
        event_start = int(rng.integers(0, n_seconds - duration + 1))
        event_end = event_start + duration
        mask = np.zeros(n_seconds, dtype=bool)
        mask[event_start:event_end] = True
        labels[mask] = scenario
        strength = rng.lognormal(mean=-0.05 if not shift else -0.18, sigma=0.34, size=n_seconds)
        if scenario == "channel_disturbance":
            qber[mask] += rng.normal(0.020, 0.012, mask.sum()) * strength[mask]
            channel_loss[mask] += rng.normal(1.10, 0.72, mask.sum()) * strength[mask]
            noise[mask] += rng.normal(0.012, 0.010, mask.sum()) * strength[mask]
            key_rate[mask] -= rng.normal(0.12, 0.10, mask.sum()) * strength[mask]
            optical[mask] += rng.normal(-0.015, 0.055, mask.sum())
        elif scenario == "intercept_resend_proxy":
            qber[mask] += rng.normal(0.014, 0.014, mask.sum()) * strength[mask]
            channel_loss[mask] += rng.normal(0.48, 0.80, mask.sum()) * strength[mask]
            noise[mask] += rng.normal(0.010, 0.012, mask.sum()) * strength[mask]
            key_rate[mask] -= rng.normal(0.07, 0.11, mask.sum()) * strength[mask]
        elif scenario == "detector_anomaly":
            detector_eff[mask] -= rng.normal(0.075 if not shift else 0.060, 0.050, mask.sum()) * strength[mask]
            dark_count[mask] += np.abs(rng.normal(0.0009, 0.0009, mask.sum())) * strength[mask]
            qber[mask] += rng.normal(0.012, 0.012, mask.sum()) * strength[mask]
            optical[mask] += rng.normal(0.03, 0.10, mask.sum())
        elif scenario == "control_plane_tampering":
            auth_fail[mask] += rng.poisson(0.42 if not shift else 0.30, mask.sum())
            config_events[mask] += rng.poisson(0.30 if not shift else 0.22, mask.sum())
            latency[mask] += rng.normal(8.0, 7.5, mask.sum()) * strength[mask]
            sync[mask] += rng.normal(0.08, 0.10, mask.sum())
        elif scenario == "telemetry_dropout":
            missing[mask] = np.clip(missing[mask] + rng.beta(2.0, 2.2, mask.sum()) * 0.58, 0, 0.98)
            age[mask] += np.abs(rng.normal(110 if not shift else 145, 95, mask.sum()))
            qber[mask] += rng.normal(0, 0.012, mask.sum())
            latency[mask] += np.abs(rng.normal(4, 7, mask.sum()))

    n_bursts = int(rng.integers(1, 5))
    for _ in range(n_bursts):
        burst_len = int(rng.integers(5, 24))
        start = int(rng.integers(0, max(1, n_seconds - burst_len + 1)))
        ix = np.arange(start, min(start + burst_len, n_seconds))
        ix = ix[labels[ix] == "normal"]
        if not len(ix):
            continue
        kind = int(rng.integers(0, 4))
        if kind == 0:
            qber[ix] += rng.normal(0.010, 0.009, len(ix))
            noise[ix] += rng.normal(0.006, 0.006, len(ix))
        elif kind == 1:
            latency[ix] += rng.normal(5.0, 5.0, len(ix))
            sync[ix] += rng.normal(0.04, 0.05, len(ix))
        elif kind == 2:
            optical[ix] += rng.normal(0.0, 0.11, len(ix))
            channel_loss[ix] += rng.normal(0.30, 0.35, len(ix))
        else:
            missing[ix] = np.clip(missing[ix] + rng.beta(1.4, 8.0, len(ix))*0.24, 0, 0.5)
            age[ix] += np.abs(rng.normal(20, 22, len(ix)))

    frame = pd.DataFrame({
        "run_id": run_id, "time_index": t,
        "distance_km": max(1.0, rng.normal(25 if not shift else 35, 8)),
        "scenario": scenario, "label": labels,
        "qber": np.clip(qber, 0, 0.35),
        "key_rate": np.clip(key_rate, 0.01, 2.0),
        "channel_loss": np.clip(channel_loss, 0, 15),
        "detector_efficiency": np.clip(detector_eff, 0.20, 1.0),
        "dark_count_rate": np.clip(dark_count, 0, 0.03),
        "authentication_failures": np.clip(auth_fail, 0, 20),
        "configuration_events": np.clip(config_events, 0, 20),
        "latency_ms": np.clip(latency, 1, 500),
        "synchronisation_error": np.clip(sync, 0, 3),
        "telemetry_missing_fraction": np.clip(missing, 0, 1),
        "telemetry_age_ms": np.clip(age, 0, 10000),
        "channel_noise": np.clip(noise, 0, 0.35),
        "optical_power": np.clip(optical, 0.1, 2.5),
    })
    frame.attrs["event_start"] = event_start
    frame.attrs["event_end"] = event_end
    return frame

def generate(total_runs_per_class: int, seed: int, generator: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    for scenario in CLASSES:
        for i in range(total_runs_per_class):
            run_id = f"{generator}_{scenario}_{i+1:03d}"
            pieces.append(make_run(rng, run_id, scenario, 300, generator, i))
    return pd.concat(pieces, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--primary-seed", type=int, default=20261104)
    ap.add_argument("--independent-seed", type=int, default=20261105)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    primary = generate(20, args.primary_seed, "primary")
    independent = generate(30, args.independent_seed, "independent")
    p1 = out / "primary_36000_reconstructed.csv"
    p2 = out / "independent_54000_reconstructed.csv"
    primary.to_csv(p1, index=False, float_format="%.8f")
    independent.to_csv(p2, index=False, float_format="%.8f")
    manifest = {
        "status": "RECONSTRUCTED_NEW_SYNTHETIC_EXPERIMENT_NOT_ORIGINAL_REPRODUCTION",
        "generator_version": "1.0.0",
        "primary": {"rows": len(primary), "runs": primary.run_id.nunique(), "runs_per_class": 20,
                    "seed": args.primary_seed, "sha256": sha256(p1),
                    "class_counts": primary.label.value_counts().sort_index().to_dict()},
        "independent": {"rows": len(independent), "runs": independent.run_id.nunique(), "runs_per_class": 30,
                        "seed": args.independent_seed, "sha256": sha256(p2),
                        "class_counts": independent.label.value_counts().sort_index().to_dict()},
        "features": FEATURES,
        "metadata_excluded_from_model_inputs": ["run_id", "time_index", "distance_km", "scenario", "label"],
        "note": "Generated feature-level assumptions; no QKD physics, photons, live telemetry, network simulator, or hardware are modelled."
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
