"""
fig8_network_period.py - Figure 8 Ring Network Period Sweep.

Simulates a 4-neuron excitable ring network and sweeps neuron.g_a and
synapse.tau_decay for both Node 0 only (heterogeneous/local) and All Nodes (homogeneous/global),
recording network period and per-neuron delays.

Exports:
- ring_sweep_neuron_g_a_node0.csv
- ring_sweep_neuron_g_a_all_nodes.csv
- ring_sweep_synapse_tau_decay_node0.csv
- ring_sweep_synapse_tau_decay_all_nodes.csv
  Columns: param_val, network_period, delay_node_0 .. delay_node_3
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

from src.helper import export_to_csv

# Paths
DATA_DIR = PROJECT_ROOT / "results" / "data"
PLOTDATA_DIR = PROJECT_ROOT / "plotdata"
OUTPUT_FIG = PROJECT_ROOT / "results" / "figures" / "fig8_network_period.png"

# ============================================================
# Network and simulation configuration
# ============================================================
N_NEURONS = 4
DT = 0.01
T_END = 2000.0
TRANSIENT_CUTOFF = 200.0
SPIKE_THRESHOLD = 40.0

PULSE_START = 0.0
PULSE_DURATION = 5.0
PULSE_AMPLITUDE = -5.0

BASE_NEURON_PARAMS = {
    "cm": 1.0,
    "tau_s": 5.0,
    "input_gain": 1.0,
    "i_base": 0.0,
    "g_na": 50.0,
    "e_na": 100.0,
    "na_act_bias": 10.0,
    "na_act_slope": 0.02,
    "na_inact_bias": 30.0,
    "na_inact_slope": 0.05,
    "g_na_reb": 2.0,
    "e_na_reb": 100.0,
    "na_reb_act_bias": -10.0,
    "na_reb_act_slope": 0.1,
    "na_reb_inact_bias": -5.0,
    "na_reb_inact_slope": 0.1,
    "g_k": 1.0,
    "e_k": -20.0,
    "k_act_bias": 10.0,
    "k_act_slope": 0.05,
    "g_l": 0.1,
    "e_l": 0.0,
    "g_a": 0.0,
    "e_a": -20.0,
    "a_act_bias": -15.0,
    "a_act_slope": 0.1,
    "a_inact_bias": -10.0,
    "a_inact_slope": 0.15,
}

BASE_SYNAPSE_PARAMS = {
    "tau_decay": 2.15,
    "tau_rise": 0.1,
    "g_max": 3.0,
    "e_rev": -20.0,
    "threshold": 30.0,
    "k": 0.1,
}

n_points = 11
z_vals = np.geomspace(1.6, 0.01, n_points)
GA_SWEEP_VALUES = 1.6 - z_vals
GA_SWEEP_VALUES[0] = 0.01
GA_SWEEP_VALUES[-1] = 1.6
TAUD_SWEEP_VALUES = np.concatenate([[1.0], np.linspace(2.0, 14.0, 7)])

def satrelu(v, threshold, slope):
    return np.clip(slope * (v - threshold), 0.0, 1.0)


def compute_gates(v_f, v_s, p):
    """Compute gating variables using piecewise-linear SatReLU formulations."""
    return {
        "m_na": satrelu(v_f, p["na_act_bias"], p["na_act_slope"]),
        "h_na": satrelu(v_s, p["na_inact_bias"], -p["na_inact_slope"]),
        "m_na_reb": satrelu(v_f, p["na_reb_act_bias"], p["na_reb_act_slope"]),
        "h_na_reb": satrelu(v_s, p["na_reb_inact_bias"], -p["na_reb_inact_slope"]),
        "n_k": satrelu(v_s, p["k_act_bias"], p["k_act_slope"]),
        "m_a": satrelu(v_f, p["a_act_bias"], p["a_act_slope"]),
        "h_a": satrelu(v_s, p["a_inact_bias"], -p["a_inact_slope"]),
    }


def simulate_excitable_ring(n_params, s_params):
    """Forward Euler numerical integration of an excitable ring network."""
    time = np.arange(0.0, T_END + 0.5 * DT, DT)
    n_steps = len(time)

    syn_src = np.arange(N_NEURONS, dtype=int)
    syn_dst = (syn_src + 1) % N_NEURONS

    vf = np.zeros((n_steps, N_NEURONS))
    vs = np.zeros((n_steps, N_NEURONS))
    g_syn = np.zeros((n_steps, N_NEURONS))

    syn_k = s_params["k"]
    slope = max(
        abs(float(syn_k[0]) if hasattr(syn_k, "__len__") else float(syn_k)), 1e-9
    )

    for k in range(n_steps - 1):
        t = time[k]
        v_f = vf[k]
        v_s = vs[k]
        g_s = g_syn[k]

        ext_input = np.zeros(N_NEURONS)
        if PULSE_START <= t < PULSE_START + PULSE_DURATION:
            ext_input[0] = PULSE_AMPLITUDE

        gates = compute_gates(v_f, v_s, n_params)

        i_ionic = (
            -n_params["g_na"] * gates["m_na"] * gates["h_na"] * (v_f - n_params["e_na"])
            - n_params["g_a"] * gates["m_a"] * gates["h_a"] * (v_f - n_params["e_a"])
            - n_params["g_na_reb"] * gates["m_na_reb"] * gates["h_na_reb"] * (v_f - n_params["e_na_reb"])
            - n_params["g_k"] * gates["n_k"] * (v_f - n_params["e_k"])
            - n_params["g_l"] * (v_f - n_params["e_l"])
        )

        input_currents = ext_input.copy()
        dg_syn = np.zeros(N_NEURONS)

        for j in range(N_NEURONS):
            src, dst = syn_src[j], syn_dst[j]
            drive = 1.0 / (
                1.0 + np.exp(-(v_f[src] - s_params["threshold"][j]) / slope)
            )
            inv_tau_r = 1.0 / float(s_params["tau_rise"][j])
            inv_tau_d = 1.0 / float(s_params["tau_decay"][j])
            alpha = max(0.0, (inv_tau_r - inv_tau_d) * drive)
            dg_syn[j] = alpha * (1.0 - g_s[j]) - inv_tau_d * g_s[j]
            input_currents[dst] += (
                s_params["g_max"][j] * g_s[j] * (s_params["e_rev"][j] - v_f[dst])
            )

        i_app = -n_params["input_gain"] * input_currents - n_params["i_base"]
        dvf = (i_ionic - i_app) / n_params["cm"]
        dvs = (v_f - v_s) / (n_params["tau_s"] * n_params["cm"])

        vf[k + 1] = v_f + DT * dvf
        vs[k + 1] = v_s + DT * dvs
        g_syn[k + 1] = g_s + DT * dg_syn

    return time, vf


def detect_spikes(time, vf, threshold):
    spikes = []
    for i in range(N_NEURONS):
        above = vf[:, i] >= threshold
        crossings = np.where((~above[:-1]) & above[1:])[0] + 1
        spikes.append(time[crossings])
    return spikes


def compute_network_metrics(
    time, vf, threshold=SPIKE_THRESHOLD, t_start=TRANSIENT_CUTOFF
):
    """Extract network period and per-neuron inter-spike delays after transient cutoff."""
    spikes = detect_spikes(time, vf, threshold)

    delays_per_neuron = []
    for x in range(N_NEURONS):
        prev_node = (x - 1) % N_NEURONS
        spikes_x = spikes[x][spikes[x] >= t_start]
        spikes_prev = spikes[prev_node]
        if len(spikes_x) == 0 or len(spikes_prev) == 0:
            delays_per_neuron.append(np.nan)
            continue
        delays_node = []
        for t_x in spikes_x:
            prior = spikes_prev[spikes_prev < t_x]
            if len(prior) > 0:
                delays_node.append(t_x - prior[-1])
        delays_per_neuron.append(
            float(np.mean(delays_node)) if delays_node else np.nan
        )

    spikes_n0 = spikes[0][spikes[0] >= t_start]
    if len(spikes_n0) >= 2:
        network_period = float(np.mean(np.diff(spikes_n0)))
    else:
        network_period = (
            float(np.sum(delays_per_neuron))
            if not np.any(np.isnan(delays_per_neuron))
            else np.nan
        )

    return {"network_period": network_period, "delays": delays_per_neuron}


def run_param_sweep(param_category, param_key, sweep_values, node_0_only=True):
    """Run the parameter sweep for global or local parameter variation."""
    results = []
    for val in sweep_values:
        n_params = {
            k: np.full(N_NEURONS, v, dtype=float)
            for k, v in BASE_NEURON_PARAMS.items()
        }
        s_params = {
            k: np.full(N_NEURONS, v, dtype=float)
            for k, v in BASE_SYNAPSE_PARAMS.items()
        }

        if param_category == "neuron":
            if node_0_only:
                n_params[param_key][0] = float(val)
            else:
                n_params[param_key][:] = float(val)
        elif param_category == "synapse":
            if node_0_only:
                s_params[param_key][0] = float(val)
            else:
                s_params[param_key][:] = float(val)

        time, vf = simulate_excitable_ring(n_params, s_params)
        metrics = compute_network_metrics(time, vf)
        period_str = (
            f"{metrics['network_period']:.3f} ms"
            if not np.isnan(metrics["network_period"])
            else "N/A"
        )
        print(f"  {param_key}={val:.4f} | Period: {period_str}")

        res_dict = {
            "param_val": float(val),
            "network_period": metrics["network_period"],
        }
        for node in range(N_NEURONS):
            res_dict[f"delay_node_{node}"] = metrics["delays"][node]
        results.append(res_dict)

    return pd.DataFrame(results)


def run():
    start_t = time.perf_counter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Sweep neuron.g_a (Node 0 only / Local & All Nodes / Global)
    print("[fig8] Sweeping neuron.g_a (Node 0 only - Local) ...")
    df_ga_n0 = run_param_sweep("neuron", "g_a", GA_SWEEP_VALUES, node_0_only=True)
    export_to_csv(df_ga_n0, DATA_DIR / "ring_sweep_neuron_g_a_node0.csv")
    export_to_csv(df_ga_n0, PLOTDATA_DIR / "ring_sweep_neuron_g_a_node0.csv")

    print("[fig8] Sweeping neuron.g_a (All Nodes - Global) ...")
    df_ga_all = run_param_sweep("neuron", "g_a", GA_SWEEP_VALUES, node_0_only=False)
    export_to_csv(df_ga_all, DATA_DIR / "ring_sweep_neuron_g_a_all_nodes.csv")
    export_to_csv(df_ga_all, PLOTDATA_DIR / "ring_sweep_neuron_g_a_all_nodes.csv")

    # 2. Sweep synapse.tau_decay (Node 0 only / Local & All Nodes / Global)
    print("[fig8] Sweeping synapse.tau_decay (Node 0 only - Local) ...")
    df_tau_n0 = run_param_sweep(
        "synapse", "tau_decay", TAUD_SWEEP_VALUES, node_0_only=True
    )
    export_to_csv(df_tau_n0, DATA_DIR / "ring_sweep_synapse_tau_decay_node0.csv")
    export_to_csv(df_tau_n0, PLOTDATA_DIR / "ring_sweep_synapse_tau_decay_node0.csv")

    print("[fig8] Sweeping synapse.tau_decay (All Nodes - Global) ...")
    df_tau_all = run_param_sweep(
        "synapse", "tau_decay", TAUD_SWEEP_VALUES, node_0_only=False
    )
    export_to_csv(df_tau_all, DATA_DIR / "ring_sweep_synapse_tau_decay_all_nodes.csv")
    export_to_csv(df_tau_all, PLOTDATA_DIR / "ring_sweep_synapse_tau_decay_all_nodes.csv")

    print(
        f"[fig8] Wrote 4 ring sweep CSVs to: {DATA_DIR} in {time.perf_counter() - start_t:.2f}s."
    )

    # 3. Figure 8 Plot (split into left tau_decay, right log-scaled g_A)
    print("[fig8] Plotting Figure 8 (split into left tau_decay, right log-scaled g_A)...")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Color Palette matching paper style
    blue_color = "#0000B3"   # Blue for g_A sweeps
    brown_color = "#C05000"  # Orange/Brown for tau_decay sweeps

    # =========================================================================
    # LEFT SUBPLOT: Synaptic Decay tau_decay Sweep
    # =========================================================================

    # Global tau_decay (Solid Brown, Filled Squares) - Left Y
    l_tau_global, = ax_left.plot(
        df_tau_all["param_val"],
        df_tau_all["network_period"],
        color=brown_color,
        linestyle="-",
        marker="s",
        markerfacecolor=brown_color,
        markeredgecolor=brown_color,
        markersize=5,
        linewidth=1.6,
        label=r"Global $\tau_{\mathrm{decay}}$",
    )

    # Local tau_decay (Dashed Brown, Open Squares) - Right Y
    l_tau_local, = ax_left.plot(
        df_tau_n0["param_val"],
        df_tau_n0["network_period"],
        color=brown_color,
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markeredgecolor=brown_color,
        markeredgewidth=1.2,
        markersize=5,
        linewidth=1.5,
        label=r"Local $\tau_{\mathrm{decay}}$",
    )

    ax_left.set_xlim(-0.5, 14.5)
    ax_left.set_xticks([0, 5, 10, 14])
    ax_left.set_xticklabels(["0", "5", "10", "14"], fontsize=10)
    ax_left.set_xlabel(r"Synaptic Decay $\tau_{\mathrm{decay}}$ (ms)", fontsize=11)

    ax_left.set_ylim(35, 320)
    ax_left.set_yticks([50, 100, 150, 200, 250, 300])
    ax_left.set_ylabel(r"$T_{\mathrm{tot}}$ (ms)", fontsize=10, color=brown_color)

    lines_left = [l_tau_global, l_tau_local]
    labels_left = [l.get_label() for l in lines_left]
    ax_left.legend(lines_left, labels_left, loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    ax_left.set_title(r"(a) Synaptic Decay Sweep ($\tau_{\mathrm{decay}}$)", fontsize=11)

    # =========================================================================
    # RIGHT SUBPLOT: A-Conductance g_A Sweep (Log Scaled z = 1.65 - g_A)
    # =========================================================================

    z_all = 1.65 - df_ga_all["param_val"]
    z_n0 = 1.65 - df_ga_n0["param_val"]

    # Global g_A (Solid Blue, Filled Circles) - Left Y
    l_ga_global, = ax_right.plot(
        z_all,
        df_ga_all["network_period"],
        color=blue_color,
        linestyle="-",
        marker="o",
        markerfacecolor=blue_color,
        markeredgecolor=blue_color,
        markersize=5,
        linewidth=1.6,
        label=r"Global $g_{\mathrm{A}}$",
    )

    # Local g_A (Dashed Blue, Open Circles) - Right Y
    l_ga_local, = ax_right.plot(
        z_n0,
        df_ga_n0["network_period"],
        color=blue_color,
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        markeredgecolor=blue_color,
        markeredgewidth=1.2,
        markersize=5,
        linewidth=1.5,
        label=r"Local $g_{\mathrm{A}}$",
    )

    ax_right.set_xscale("log")
    ax_right.invert_xaxis()  # Reverses z so g_a = 0 (z=1.65) is on left and g_a = 1.6 (z=0.05) is on right
    ax_right.set_xticks([1.65, 1.0, 0.5, 0.2, 0.05])
    ax_right.set_xticklabels(["0", "0.65", "1.15", "1.45", "1.60"], fontsize=10)
    ax_right.set_xlabel(r"A-conductance $g_{\mathrm{A}}$ (mS/cm$^2$)", fontsize=11)

    # Expand Y-limits slightly on Global right plot to accommodate high g_a asymptote period (~500 ms)
    ax_right.set_ylim(35, 520)
    ax_right.set_yticks([50, 150, 250, 350, 450])
    ax_right.set_ylabel(r"$T_{\mathrm{tot}}$ (ms)", fontsize=10, color=blue_color)

    lines_right = [l_ga_global, l_ga_local]
    labels_right = [l.get_label() for l in lines_right]
    ax_right.legend(lines_right, labels_right, loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    ax_right.set_title(r"(b) Potassium Conductance Sweep ($g_{\mathrm{A}}$)", fontsize=11)

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.close(fig)
    print(f"[fig8] Saved Figure 8 plot to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
