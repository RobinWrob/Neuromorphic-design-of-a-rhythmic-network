"""
fig6_edf_example.py - Figure 6 Single-Neuron Event Describing Function (eDF) Calculation.

Computes the single-neuron Event Describing Function (eDF) by measuring steady-state
spike onset delay delta(T) as a function of presynaptic input period T.

Exports:
  results/data/plotdata_edf_single_curve.csv
  Columns: period, delay, relative_phase
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.edf_engine import compute_edf_curve, extract_edf_metrics
from src.helper import export_to_csv

# Paths
OUTPUT_CSV = PROJECT_ROOT / "results" / "data" / "plotdata_edf_single_curve.csv"
OUTPUT_FIG = PROJECT_ROOT / "results" / "figures" / "fig6_edf_example.png"

# ============================================================
# Simulation parameters
# ============================================================
DT = 0.0001
IMPULSE_WIDTH = 0.1
IMPULSE_AMPLITUDE = 50.0
SPIKE_THRESHOLD = 50.0

PERIOD_MIN = 10.0
PERIOD_MAX = 50.0
PERIOD_STEP = 0.1

N_CYCLES_SETTLE = 5
N_CYCLES_MEASURE = 10
MIN_T = 400.0              # Minimum total simulation time (ms)
DURATION_MULTIPLIER = 5.0  # Duration factor per period

# Parameter overrides for single-neuron eDF benchmark
NEURON_OVERRIDES = {"g_a": 0.0}
SYNAPSE_OVERRIDES = {"tau_decay": 2.15, "tau_rise": 0.1, "g_max": 3.0}


def run():
    periods = np.arange(PERIOD_MIN, PERIOD_MAX + 0.5 * PERIOD_STEP, PERIOD_STEP)

    all_samples = []
    for p in periods:
        samps = compute_edf_curve(
            periods=[p],
            dt=DT,
            impulse_width=IMPULSE_WIDTH,
            impulse_amplitude=IMPULSE_AMPLITUDE,
            spike_threshold=SPIKE_THRESHOLD,
            neuron_params=NEURON_OVERRIDES,
            synapse_params=SYNAPSE_OVERRIDES,
            parallel=False,
        )
        all_samples.extend(samps)

    crits = extract_edf_metrics(all_samples)
    if crits["t_min"] is not None:
        print(f"[fig6] T_min={crits['t_min']:.2f} ms  delta_inf={crits['delta_inf']:.4f} ms")
    else:
        print("[fig6] Warning: no valid 1:1 phase-locked points found.")

    rows = []
    for s in all_samples:
        if s.is_valid and s.onset_delay is not None:
            rel_phase = s.onset_delay / s.input_period if s.input_period > 0 else 0.0
            rows.append({
                "period": s.input_period,
                "delay": s.onset_delay,
                "relative_phase": rel_phase,
            })

    df = pd.DataFrame(rows)
    export_to_csv(df, OUTPUT_CSV)
    print(f"[fig6] Wrote CSV data to: {OUTPUT_CSV} ({len(df)} valid points)")

    # Validation plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if not df.empty:
        ax.plot(df["period"], df["delay"], marker="o", markersize=4, label=r"eDF Delay $\delta(T)$")
        if crits["delta_inf"] is not None:
            ax.axhline(crits["delta_inf"], color="red", linestyle="--", alpha=0.7, label=r"$\delta_\infty$")
        if crits["t_min"] is not None:
            ax.axvline(
                crits["t_min"],
                color="green",
                linestyle=":",
                lw=2,
                alpha=0.8,
                label=f"T_min={crits['t_min']:.1f} ms",
            )
    ax.set_title("Figure 6: Single-Neuron Event Describing Function (eDF)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Input Period T (ms)")
    ax.set_ylabel(r"Onset Delay $\delta(T)$ (ms)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG)
    plt.close(fig)
    print(f"[fig6] Wrote plot figure to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
