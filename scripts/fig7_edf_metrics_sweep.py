"""
fig7_edf_metrics_sweep.py - Figure 7 eDF Parameter Sweeps.

Sweeps neuron.g_a and synapse.tau_decay, extracting T_min, T_r, and delta_inf metrics.

Exports:
- plotdata_metrics_sweep_edf_neuron_g_a.csv
- plotdata_metrics_sweep_edf_synapse_tau_decay.csv
  Columns: param_val, t_min, t_r, delta_inf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.edf_engine import compute_edf_adaptive, extract_edf_metrics
from src.helper import export_to_csv

# Paths
DATA_DIR = PROJECT_ROOT / "results" / "data"
PLOTDATA_DIR = PROJECT_ROOT / "plotdata"
OUTPUT_FIG = PROJECT_ROOT / "results" / "figures" / "fig7_edf_metrics_sweep.png"

# ============================================================
# Simulation parameters
# ============================================================
DT = 0.0001
IMPULSE_WIDTH = 0.1
IMPULSE_AMPLITUDE = 50.0
SPIKE_THRESHOLD = 50.0

PERIOD_MIN = 10.0
PERIOD_MAX = 200.0

# Base parameter overrides
BASE_PARAM_OVERRIDES = {
    "synapse.tau_decay": 2,
    "synapse.tau_rise": 0.1,
    "synapse.g_max": 3.0,
    "neuron.g_a": 0.0,
}

# Sweep value ranges: g_a sampled so z = 1.65 - g_a is geometrically spaced from 1.6 to 0.01
n_points = 11
z_vals = np.geomspace(1.6, 0.01, n_points)
G_A_VALUES = 1.6 - z_vals
G_A_VALUES[0] = 0.01
G_A_VALUES[-1] = 1.6
TAU_D_VALUES = np.linspace(1.0, 10.0, 10)     # 10 points

DURATION_MULTIPLIER = 5.0
MIN_T = 1000.0
PARALLELIZE = True


def run_sweep(param_name: str, values: np.ndarray) -> pd.DataFrame:
    """Evaluates adaptive eDF sweep over a target parameter."""
    print(f"\n--- Running sweep for {param_name} ({len(values)} points) ---", flush=True)
    records = []

    for i, val in enumerate(values, 1):
        overrides = BASE_PARAM_OVERRIDES.copy()
        overrides[param_name] = float(val)

        print(f"  [{i}/{len(values)}] {param_name} = {val:.4f} ...", end=" ", flush=True)

        samples = compute_edf_adaptive(
            p_min=PERIOD_MIN,
            p_max=PERIOD_MAX,
            coarse_step=2.0,
            medium_step=1.0,
            fine_step=0.1,
            dt=DT,
            impulse_width=IMPULSE_WIDTH,
            impulse_amplitude=IMPULSE_AMPLITUDE,
            spike_threshold=SPIKE_THRESHOLD,
            duration_multiplier=DURATION_MULTIPLIER,
            min_t=MIN_T,
            parallel=PARALLELIZE,
            neuron_params={"g_a": overrides["neuron.g_a"]},
            synapse_params={
                "tau_decay": overrides["synapse.tau_decay"],
                "tau_rise": overrides["synapse.tau_rise"],
                "g_max": overrides["synapse.g_max"],
            },
        )

        metrics = extract_edf_metrics(samples)
        t_min = metrics["t_min"]
        t_r = metrics["t_r"]
        delta_inf = metrics["delta_inf"]

        t_min_str = f"{t_min:.4f}" if t_min is not None else "N/A"
        t_r_str = f"{t_r:.4f}" if t_r is not None else "N/A"
        d_inf_str = f"{delta_inf:.4f}" if delta_inf is not None else "N/A"
        print(f"T_min={t_min_str}, T_r={t_r_str}, delta_inf={d_inf_str}", flush=True)

        records.append({
            "param_val": float(val),
            "t_min": t_min,
            "t_r": t_r,
            "delta_inf": delta_inf,
        })

    return pd.DataFrame(records)


def export_metrics_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Exports critical metrics to CSV matching reference schema."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("param_val,t_min,t_r,delta_inf\n")
        for _, row in df.iterrows():
            p_val = row["param_val"]
            t_min = f"{row['t_min']:.4f}" if pd.notna(row['t_min']) else "NaN"
            t_r = f"{row['t_r']:.4f}" if pd.notna(row['t_r']) else "NaN"
            d_inf = f"{row['delta_inf']:.6f}" if pd.notna(row['delta_inf']) else "NaN"
            f.write(f"{p_val:.4f},{t_min},{t_r},{d_inf}\n")
    print(f"[fig7] Wrote metrics sweep CSV to: {filepath}")


def run():
    start_t = time.perf_counter()

    # 1. Sweep neuron.g_a
    df_ga = run_sweep("neuron.g_a", G_A_VALUES)
    export_metrics_csv(df_ga, DATA_DIR / "plotdata_metrics_sweep_edf_neuron_g_a.csv")
    export_metrics_csv(df_ga, PLOTDATA_DIR / "plotdata_metrics_sweep_edf_neuron_g_a.csv")

    # 2. Sweep synapse.tau_decay
    df_tau = run_sweep("synapse.tau_decay", TAU_D_VALUES)
    export_metrics_csv(df_tau, DATA_DIR / "plotdata_metrics_sweep_edf_synapse_tau_decay.csv")
    export_metrics_csv(df_tau, PLOTDATA_DIR / "plotdata_metrics_sweep_edf_synapse_tau_decay.csv")

    print(f"\n[fig7] Completed parameter sweeps in {time.perf_counter() - start_t:.2f}s. Plotting Figure 7...", flush=True)

    # 3. Validation plot matching LaTeX TikZ groupplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Subplot 1 (Left): Sweep over Synaptic Decay Time tau_decay
    if not df_tau.empty:
        ax1.plot(df_tau["param_val"], df_tau["t_min"], "o-", color="#008000", lw=1.5, ms=5, label=r"$T_{\min}$")
        ax1.plot(df_tau["param_val"], df_tau["delta_inf"], "^-", color="#B22222", lw=1.5, ms=5, label=r"$\delta_{\infty}$")
    ax1.set_xlabel(r"$\tau_{\mathrm{decay}}$ (ms)", fontsize=10)
    ax1.set_ylabel("Period (ms)", fontsize=10)
    ax1.set_xticks([1, 5, 10])
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax1.grid(False)

    # Subplot 2 (Right): Sweep over A-type Potassium Conductance g_a
    # Matches TikZ: xmode=log, x dir=reverse, z = 1.65 - g_a
    if not df_ga.empty:
        z_plot = 1.65 - df_ga["param_val"].values
        ax2.plot(z_plot, df_ga["t_min"], "o-", color="#008000", lw=1.5, ms=5, label=r"$T_{\min}$")
        ax2.plot(z_plot, df_ga["delta_inf"], "^-", color="#B22222", lw=1.5, ms=5, label=r"$\delta_{\infty}$")
        ax2.set_xscale("log")
        ax2.invert_xaxis()  # Keeps g_a increasing from left to right (z=1.65 [g_a=0] on left to z=0.03 [g_a=1.62] on right)
        ax2.set_xticks([1.65, 1.0, 0.3, 0.1, 0.03])
        ax2.set_xticklabels(["0", "0.65", "1.35", "1.55", "1.62"])
    ax2.set_xlabel(r"$g_A$ (mS/cm$^2$)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax2.grid(False)

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.close(fig)
    print(f"[fig7] Saved Figure 7 plot to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
