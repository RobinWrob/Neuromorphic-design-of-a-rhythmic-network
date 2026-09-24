"""
fig9_endogenous_tuning_hom_het.py - Figure 9 Endogenous Tuning via Parameter Modulation.

Simulates a 4-neuron ring with a parameter switch at T/2 (T_END = 1400 ms):
  - Homogeneous (left column): synapse tau_decay change (2.15 -> 6.0) applied to ALL nodes
  - Heterogeneous (right column): MODULATE_NODE_0_ONLY=True, incoming synapse tau_decay change applied to Node 0 only

Produces a 2x2 multi-panel figure:
  - Top row: Absolute inter-spike ring delay (ms) for all neurons (0 to 1400 ms)
  - Bottom row: Membrane potential traces V (mV) for all neurons (500 to 1100 ms)

Exports CSVs:
  - ring_network_voltage.csv        (Homogeneous voltage traces, all nodes)
  - ring_network_delay.csv          (Homogeneous absolute delays)
  - ring_network_voltage_n0.csv     (Heterogeneous voltage traces, all nodes)
  - ring_network_delay_n0.csv       (Heterogeneous absolute delays)
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

from src.helper import export_to_csv, opheim_simplify_trace

# Paths
DATA_DIR   = PROJECT_ROOT / "results" / "data"
OUTPUT_FIG = PROJECT_ROOT / "results" / "figures" / "fig9_endogenous_tuning_hom_het.png"

# ============================================================
# Network and simulation configuration
# ============================================================
N_NEURONS = 4
DT        = 0.01
T_END     = 1400.0
T_HALF    = T_END / 2.0

PULSE_START     = 0.0
PULSE_DURATION  = 5.0
PULSE_AMPLITUDE = -5.0

SPIKE_THRESHOLD = 30.0

EXPORT_OPHEIM_MIN_TOL = 0.002
EXPORT_OPHEIM_MAX_TOL = 0.02

BASE_NEURON_PARAMS = {
    "cm": 1.0, "tau_s": 5.0, "input_gain": 1.0, "i_base": 0.0,
    "g_na": 50.0, "e_na": 100.0, "na_act_bias": 10.0, "na_act_slope": 0.02,
    "na_inact_bias": 30.0, "na_inact_slope": 0.05,
    "g_na_reb": 2.0, "e_na_reb": 100.0, "na_reb_act_bias": -10.0,
    "na_reb_act_slope": 0.1, "na_reb_inact_bias": -5.0, "na_reb_inact_slope": 0.1,
    "g_k": 1.0, "e_k": -20.0, "k_act_bias": 10.0, "k_act_slope": 0.05,
    "g_l": 0.1, "e_l": 0.0,
    "g_a": 0.0, "e_a": -20.0, "a_act_bias": -15.0, "a_act_slope": 0.1,
    "a_inact_bias": -10.0, "a_inact_slope": 0.15,
}

BASE_SYNAPSE_PARAMS = {
    "tau_decay": 2.15, "tau_rise": 0.1, "g_max": 2.15,
    "e_rev": -20.0, "threshold": 30.0, "k": 0.1,
}

# Post-modulation parameter values
GA_PRE,  GA_POST  = 0.0, 0.0
TAU_PRE, TAU_POST = 2.15, 6.0


def satrelu(v, threshold, slope):
    return np.clip(slope * (v - threshold), 0.0, 1.0)


def simulate_modulated_ring(modulate_node_0_only=False):
    """Forward Euler numerical integration of ring network under parameter modulation."""
    time    = np.arange(0.0, T_END + 0.5 * DT, DT)
    n_steps = len(time)

    syn_src = np.arange(N_NEURONS, dtype=int)
    syn_dst = (syn_src + 1) % N_NEURONS

    n_pre  = {k: np.full(N_NEURONS, v, dtype=float) for k, v in BASE_NEURON_PARAMS.items()}
    n_post = {k: np.full(N_NEURONS, v, dtype=float) for k, v in BASE_NEURON_PARAMS.items()}
    s_pre  = {k: np.full(N_NEURONS, v, dtype=float) for k, v in BASE_SYNAPSE_PARAMS.items()}
    s_post = {k: np.full(N_NEURONS, v, dtype=float) for k, v in BASE_SYNAPSE_PARAMS.items()}

    if modulate_node_0_only:
        n_post["g_a"][0]         = GA_POST
        s_post["tau_decay"][-1]  = TAU_POST   # synapse N-1 -> 0
    else:
        n_post["g_a"][:]         = GA_POST
        s_post["tau_decay"][:]   = TAU_POST

    vf    = np.zeros((n_steps, N_NEURONS))
    vs    = np.zeros((n_steps, N_NEURONS))
    g_syn = np.zeros((n_steps, N_NEURONS))

    slope = max(abs(BASE_SYNAPSE_PARAMS["k"]), 1e-9)

    for k in range(n_steps - 1):
        t        = time[k]
        active_n = n_post if t >= T_HALF else n_pre
        active_s = s_post if t >= T_HALF else s_pre

        v_f = vf[k]
        v_s = vs[k]
        g_s = g_syn[k]

        ext_input = np.zeros(N_NEURONS)
        if PULSE_START <= t < PULSE_START + PULSE_DURATION:
            ext_input[0] = PULSE_AMPLITUDE

        m_na     = satrelu(v_f, active_n["na_act_bias"],        active_n["na_act_slope"])
        h_na     = satrelu(v_s, active_n["na_inact_bias"],     -active_n["na_inact_slope"])
        m_na_reb = satrelu(v_f, active_n["na_reb_act_bias"],    active_n["na_reb_act_slope"])
        h_na_reb = satrelu(v_s, active_n["na_reb_inact_bias"], -active_n["na_reb_inact_slope"])
        n_k      = satrelu(v_s, active_n["k_act_bias"],         active_n["k_act_slope"])
        m_a      = satrelu(v_f, active_n["a_act_bias"],         active_n["a_act_slope"])
        h_a      = satrelu(v_s, active_n["a_inact_bias"],      -active_n["a_inact_slope"])

        i_ionic = (
            -active_n["g_na"]      * m_na     * h_na     * (v_f - active_n["e_na"])
            - active_n["g_a"]      * m_a      * h_a      * (v_f - active_n["e_a"])
            - active_n["g_na_reb"] * m_na_reb * h_na_reb * (v_f - active_n["e_na_reb"])
            - active_n["g_k"]      * n_k                  * (v_f - active_n["e_k"])
            - active_n["g_l"]                              * (v_f - active_n["e_l"])
        )

        input_currents = ext_input.copy()
        dg_syn         = np.zeros(N_NEURONS)

        for j in range(N_NEURONS):
            src, dst  = syn_src[j], syn_dst[j]
            drive     = 1.0 / (1.0 + np.exp(-(v_f[src] - active_s["threshold"][j]) / slope))
            inv_tau_r = 1.0 / float(active_s["tau_rise"][j])
            inv_tau_d = 1.0 / float(active_s["tau_decay"][j])
            alpha     = max(0.0, (inv_tau_r - inv_tau_d) * drive)
            dg_syn[j] = alpha * (1.0 - g_s[j]) - inv_tau_d * g_s[j]
            input_currents[dst] += active_s["g_max"][j] * g_s[j] * (active_s["e_rev"][j] - v_f[dst])

        i_app = -active_n["input_gain"] * input_currents - active_n["i_base"]
        dvf   = (i_ionic - i_app) / active_n["cm"]
        dvs   = (v_f - v_s) / (active_n["tau_s"] * active_n["cm"])

        vf[k + 1]    = v_f + DT * dvf
        vs[k + 1]    = v_s + DT * dvs
        g_syn[k + 1] = g_s + DT * dg_syn

    return time, vf


def detect_spike_times(time, vf, threshold=SPIKE_THRESHOLD):
    spikes = []
    for i in range(vf.shape[1]):
        above     = vf[:, i] >= threshold
        crossings = np.where((~above[:-1]) & above[1:])[0] + 1
        spikes.append(time[crossings])
    return spikes


def compute_absolute_delay_series(spike_times):
    delay_data = {}
    n_neurons  = len(spike_times)
    for i, times_i in enumerate(spike_times):
        ref_times   = spike_times[(i - 1) % n_neurons]
        if len(times_i) == 0 or len(ref_times) == 0:
            delay_data[i] = (np.array([]), np.array([]))
            continue
        ref_indices = np.searchsorted(ref_times, times_i, side="left") - 1
        valid       = ref_indices >= 0
        valid_times = times_i[valid]
        delays      = valid_times - ref_times[ref_indices[valid]]
        delay_data[i] = (valid_times, delays)
    return delay_data


def run():
    # Colors matching standard publication figure (n0..n3)
    colors = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E"]

    # Run simulations for both conditions
    time_hom, vf_hom = simulate_modulated_ring(modulate_node_0_only=False)
    spikes_hom = detect_spike_times(time_hom, vf_hom)
    delays_hom = compute_absolute_delay_series(spikes_hom)

    time_het, vf_het = simulate_modulated_ring(modulate_node_0_only=True)
    spikes_het = detect_spike_times(time_het, vf_het)
    delays_het = compute_absolute_delay_series(spikes_het)

    # ------------------------------------------------------------
    # Export CSVs
    # ------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Homogeneous Voltage & Delay CSVs
    skip = max(1, len(time_hom) // 10000)
    idx  = np.arange(0, len(time_hom), skip)
    df_v_hom = pd.DataFrame({"time": time_hom[idx], **{f"n{i}": vf_hom[idx, i] for i in range(N_NEURONS)}})
    export_to_csv(df_v_hom, DATA_DIR / "ring_network_voltage.csv")

    d_rows_hom = []
    for i, (times, delays) in delays_hom.items():
        for t, d in zip(times, delays):
            d_rows_hom.append({"time": float(t), "neuron": i, "delay_ms": float(d)})
    df_d_hom = pd.DataFrame(d_rows_hom)
    export_to_csv(df_d_hom, DATA_DIR / "ring_network_delay.csv")

    # 2. Heterogeneous Voltage & Delay CSVs
    df_v_het = pd.DataFrame({"time": time_het[idx], **{f"n{i}": vf_het[idx, i] for i in range(N_NEURONS)}})
    export_to_csv(df_v_het, DATA_DIR / "ring_network_voltage_n0.csv")

    d_rows_het = []
    for i, (times, delays) in delays_het.items():
        for t, d in zip(times, delays):
            d_rows_het.append({"time": float(t), "neuron": i, "delay_ms": float(d)})
    df_d_het = pd.DataFrame(d_rows_het)
    export_to_csv(df_d_het, DATA_DIR / "ring_network_delay_n0.csv")

    print("[fig9] CSV exports complete.")

    # ------------------------------------------------------------
    # Plotting Figure 9 (2x2 Grid Layout)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=150, sharex='row', gridspec_kw={'height_ratios': [1, 1.2]})
    (ax_d_hom, ax_d_het), (ax_v_hom, ax_v_het) = axes

    # --- TOP ROW: DELAYS ---
    for node in range(N_NEURONS):
        t_d, d_vals = delays_hom[node]
        ax_d_hom.plot(t_d, d_vals, color=colors[node], marker='|', markersize=6, lw=1.2, label=f"$n_{node}$")

        t_d_het, d_vals_het = delays_het[node]
        ax_d_het.plot(t_d_het, d_vals_het, color=colors[node], marker='|', markersize=6, lw=1.2, label=f"$n_{node}$")

    ax_d_hom.set_title("Homogeneous", fontsize=13, fontweight='bold', pad=10)
    ax_d_het.set_title("Heterogeneous", fontsize=13, fontweight='bold', pad=10)

    ax_d_hom.set_ylabel("Delay (ms)", fontsize=11)
    ax_d_hom.set_ylim(8, 32)
    ax_d_het.set_ylim(8, 32)
    ax_d_hom.set_xlim(0, 1400)
    ax_d_het.set_xlim(0, 1400)

    ax_d_hom.grid(True, linestyle="--", alpha=0.3)
    ax_d_het.grid(True, linestyle="--", alpha=0.3)

    # --- BOTTOM ROW: VOLTAGE TRACES ---
    # Zoom in on 500 - 1100 ms as in reference image
    mask_hom = (time_hom >= 500.0) & (time_hom <= 1100.0)
    mask_het = (time_het >= 500.0) & (time_het <= 1100.0)

    for node in range(N_NEURONS):
        ax_v_hom.plot(time_hom[mask_hom], vf_hom[mask_hom, node], color=colors[node], lw=1.0)
        ax_v_het.plot(time_het[mask_het], vf_het[mask_het, node], color=colors[node], lw=1.0)

    ax_v_hom.set_ylabel("$V$ (mV)", fontsize=11)
    ax_v_hom.set_xlabel("Time (ms)", fontsize=11)
    ax_v_het.set_xlabel("Time (ms)", fontsize=11)

    ax_v_hom.set_xlim(500, 1100)
    ax_v_het.set_xlim(500, 1100)
    ax_v_hom.set_ylim(-30, 105)
    ax_v_het.set_ylim(-30, 105)

    ax_v_hom.grid(True, linestyle="--", alpha=0.3)
    ax_v_het.grid(True, linestyle="--", alpha=0.3)

    # --- SHARED LEGEND ---
    # Place legend centered between top and bottom plots
    handles, labels = ax_d_hom.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.46), ncol=N_NEURONS, frameon=True, fontsize=10)

    plt.subplots_adjust(hspace=0.45, wspace=0.15)
    
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig9] Saved Figure 9 plot to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
