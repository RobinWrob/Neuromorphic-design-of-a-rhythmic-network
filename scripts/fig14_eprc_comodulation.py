"""
fig14_eprc_comodulation.py - Figure 14 ePRC Co-Modulation Covaried Sweep.

Sweeps two physical parameters simultaneously in opposite directions (neuron.g_a increasing
while synapse_nom.tau_decay decreases), computes the Event Phase Response Curve (ePRC) at each
paired step, plots all curves overlaid, and exports the sweep dataset to CSV.

Export CSVs:
- plotdata/plotdata_eprc_covaried_exc_neuron_g_a_vs_synapse_nom_tau_decay.csv
- results/data/plotdata_eprc_covaried_exc_neuron_g_a_vs_synapse_nom_tau_decay.csv
- results/data/fig14_eprc_comodulation_data.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eprc_engine import compute_eprc_curve
from src.helper import export_to_csv

# Paths
DATA_DIR     = PROJECT_ROOT / "results" / "data"
PLOTDATA_DIR = PROJECT_ROOT / "plotdata"
OUTPUT_FIG   = PROJECT_ROOT / "results" / "figures" / "fig14_eprc_comodulation.png"

# ============================================================================
# Covaried Sweep & Simulation Configuration
# ============================================================================

NUM_STEPS = 5

PARAM_1_KEY = "neuron.g_a"
PARAM_1_LABEL = r"$g_a$"
PARAM_1_MIN = 0.0
PARAM_1_MAX = 1.5

def hill_sigmoid_scale(x, a=1.26, mid_val=1.11, max_val=1.5):
    """Sigmoid scaling that anchors g_a(0)=0, g_a(0.5)=mid_val, g_a(1)=max_val."""
    b = (max_val / mid_val) - 1.0
    return max_val * (x**a / (x**a + b * (1.0 - x)**a))

x = np.linspace(0.0, 1.0, NUM_STEPS)
PARAM_1_VALUES = hill_sigmoid_scale(x, a=1.26)

PARAM_2_KEY = "synapse_nom.tau_decay"
PARAM_2_LABEL = r"$\tau_{\mathrm{decay}}$"
PARAM_2_MIN = 3.0
PARAM_2_MAX = 20.5

PARAM_2_VALUES = np.linspace(PARAM_2_MAX, PARAM_2_MIN, NUM_STEPS)

BASE_PARAM_OVERRIDES = {
    "synapse_sens.g_max": 0.01,
}

DT = 0.001
SPIKE_THRESHOLD = 50.0
IMPULSE_WIDTH = 3.0
IMPULSE_AMPLITUDE = 100.0

INPUT_PERIOD = 0.0
TP_MIN = 0.0
TP_MAX = 80.0
TP_STEP = 0.1

SINGLE_EVENT = True
T_NOM_START = INPUT_PERIOD

DURATION_MULTIPLIER = 6.0
MIN_T = 350.0
SYN_TYPE = "exc"

PARALLELIZE = True

def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTDATA_DIR.mkdir(parents=True, exist_ok=True)

    tp_values = np.arange(TP_MIN, TP_MAX + 0.5 * TP_STEP, TP_STEP)

    sweep_results = {}
    export_data = {"tp": tp_values} # Base column for x-axis

    for i in range(NUM_STEPS):
        val1 = float(PARAM_1_VALUES[i])
        val2 = float(PARAM_2_VALUES[i])

        current_overrides = BASE_PARAM_OVERRIDES.copy()
        current_overrides[PARAM_1_KEY] = val1
        current_overrides[PARAM_2_KEY] = val2

        samples = compute_eprc_curve(
            tp_values=tp_values,
            input_period=INPUT_PERIOD,
            dt=DT,
            impulse_width=IMPULSE_WIDTH,
            impulse_amplitude=IMPULSE_AMPLITUDE,
            spike_threshold=SPIKE_THRESHOLD,
            single_event=SINGLE_EVENT,
            duration_multiplier=DURATION_MULTIPLIER,
            min_t=MIN_T,
            t_nom_start=T_NOM_START,
            parallelize=PARALLELIZE,
            param_overrides=current_overrides,
            syn_type=SYN_TYPE,
        )

        # Use np.nan instead of None so pandas writes empty CSV cells that PGFPlots can natively discard
        delta_vals = [s.delta_t if (s.is_valid and s.delta_t is not None) else np.nan for s in samples]
        sweep_results[(val1, val2)] = delta_vals
        
        # Save as a simplified column name for TikZ
        export_data[f"step{i+1}"] = delta_vals

    df = pd.DataFrame(export_data)
    p1_clean = PARAM_1_KEY.replace(".", "_")
    p2_clean = PARAM_2_KEY.replace(".", "_")

    export_to_csv(df, PLOTDATA_DIR / f"plotdata_eprc_covaried_{SYN_TYPE}_{p1_clean}_vs_{p2_clean}.csv")
    export_to_csv(df, DATA_DIR / f"plotdata_eprc_covaried_{SYN_TYPE}_{p1_clean}_vs_{p2_clean}.csv")
    export_to_csv(df, DATA_DIR / "fig14_eprc_comodulation_data.csv")
    print(f"[fig14] Exported covaried ePRC sweep CSVs to PLOTDATA_DIR and DATA_DIR")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    colors = plt.cm.plasma(np.linspace(0, 1, NUM_STEPS))

    for i in range(NUM_STEPS):
        val1 = float(PARAM_1_VALUES[i])
        val2 = float(PARAM_2_VALUES[i])
        delta_vals = sweep_results[(val1, val2)]

        delta_line = [val if val is not None else np.nan for val in delta_vals]
        label_str = rf"{PARAM_1_LABEL}={val1:.2f}, {PARAM_2_LABEL}={val2:.2f}"

        if any(not np.isnan(v) for v in delta_line):
            ax.plot(tp_values, delta_line, "-", lw=2, color=colors[i], label=label_str)

    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Sensory Perturbation Delay $t_p$ (ms)", fontsize=11)
    ax.set_ylabel(r"Output Spike Timing Shift $\Delta t$ (ms)", fontsize=11)
    ax.set_title(rf"Covaried ePRC Sweep: {PARAM_1_LABEL} vs {PARAM_2_LABEL}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200)
    plt.close(fig)
    print(f"[fig14] Saved figure to: {OUTPUT_FIG}")
    

if __name__ == "__main__":
    run()
