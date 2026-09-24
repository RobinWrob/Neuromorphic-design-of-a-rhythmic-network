"""
fig10_compare_fixed_period_vs_fixed_node.py - Figure 10 Duty Modulation Comparison.

Simulates a 4-neuron ring under two phase profile modulation strategies:
  - Approach 1 (Fixed Period T_tot, left column): Global modulation of g_a and synapse tau_decay.
  - Approach 2 (Fixed Node delta_i, right column): Local modulation of nodes 0 and 2.

Renders a 3x2 multi-panel layout matching the reference figure:
  - Top row: Spike raster plots (neurons 0..3 vs time)
  - Middle row: Absolute nodal delays delta_i (ms)
  - Bottom row: Relative phase delays (normalized to total cycle period T_tot)

Exports CSVs:
  - plotdata_duty_modulation_approach1_raster.csv
  - plotdata_duty_modulation_approach1_delay.csv
  - plotdata_duty_modulation_approach1_reldelay.csv
  - plotdata_duty_modulation_approach2_raster.csv
  - plotdata_duty_modulation_approach2_delay.csv
  - plotdata_duty_modulation_approach2_reldelay.csv
  - fig10_duty_modulation_data.csv
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

from src.helper import export_to_csv

# Paths
DATA_DIR   = PROJECT_ROOT / "results" / "data"
OUTPUT_FIG = PROJECT_ROOT / "results" / "figures" / "fig10_compare_fixed_period_vs_fixed_node.png"

# Shared parameters
N_NEURONS = 4
DT        = 0.01
T_END     = 4000.0
T_HALF    = T_END / 2.0   # 2000.0 ms

PULSE_START     = 0.0
PULSE_DURATION  = 5.0
PULSE_AMPLITUDE = -5.0

SPIKE_THRESHOLD = 30.0

# Approach 1: Fixed Period
BASE_NEURON_A1 = {
    "cm": 1.0, "tau_s": 5.0, "input_gain": 1.0, "i_base": 0.0,
    "g_na": 50.0, "e_na": 100.0, "na_act_bias": 10.0, "na_act_slope": 0.02,
    "na_inact_bias": 30.0, "na_inact_slope": 0.05,
    "g_na_reb": 2.0, "e_na_reb": 100.0, "na_reb_act_bias": -10.0,
    "na_reb_act_slope": 0.1, "na_reb_inact_bias": -5.0, "na_reb_inact_slope": 0.1,
    "g_k": 1.0, "e_k": -20.0, "k_act_bias": 10.0, "k_act_slope": 0.05,
    "g_l": 0.1, "e_l": 0.0,
    "g_a": 1.2, "e_a": -20.0, "a_act_bias": -15.0, "a_act_slope": 0.1,
    "a_inact_bias": -10.0, "a_inact_slope": 0.15,
}
BASE_SYNAPSE_A1 = {
    "tau_decay": 2.15, "tau_rise": 0.1, "g_max": 2.5,
    "e_rev": -20.0, "threshold": 30.0, "k": 0.1,
}
NEURON_MODS_A1  = [{"node": i, "param": "g_a", "val": 0.0}   for i in [0, 2]] + \
                  [{"node": i, "param": "g_a", "val": 1.45}  for i in [1, 3]]
SYNAPSE_MODS_A1 = [
    {"src": 3, "dst": 0, "param": "tau_decay", "val": 2.4},
    {"src": 1, "dst": 2, "param": "tau_decay", "val": 2.4},
]

# Approach 2: Fixed Node
BASE_NEURON_A2 = {**BASE_NEURON_A1, "g_a": 1.2}
BASE_SYNAPSE_A2 = {**BASE_SYNAPSE_A1, "tau_decay": 2.5, "g_max": 2.0}
NEURON_MODS_A2  = [{"node": i, "param": "g_a", "val": 0.0}  for i in [0, 2]]
SYNAPSE_MODS_A2 = [
    {"src": 3, "dst": 0, "param": "tau_decay", "val": 0.5},
    {"src": 1, "dst": 2, "param": "tau_decay", "val": 0.5},
]


def satrelu(v, threshold, slope):
    return np.clip(slope * (v - threshold), 0.0, 1.0)


def apply_modulations(n_base, s_base, n_mods, s_mods):
    n_post = {k: np.copy(v) for k, v in n_base.items()}
    s_post = {k: np.copy(v) for k, v in s_base.items()}
    for mod in n_mods:
        n_post[mod["param"]][mod["node"]] = mod["val"]
    for mod in s_mods:
        src, dst = mod["src"], mod["dst"]
        idx      = src
        if (src + 1) % N_NEURONS == dst:
            s_post[mod["param"]][idx] = mod["val"]
    return n_post, s_post


def simulate_modulated_ring(base_neuron, base_synapse, neuron_mods, synapse_mods):
    time    = np.arange(0.0, T_END + 0.5 * DT, DT)
    n_steps = len(time)

    syn_src = np.arange(N_NEURONS, dtype=int)
    syn_dst = (syn_src + 1) % N_NEURONS

    n_pre  = {k: np.full(N_NEURONS, v, dtype=float) for k, v in base_neuron.items()}
    s_pre  = {k: np.full(N_NEURONS, v, dtype=float) for k, v in base_synapse.items()}
    n_post, s_post = apply_modulations(n_pre, s_pre, neuron_mods, synapse_mods)

    vf    = np.zeros((n_steps, N_NEURONS))
    vs    = np.zeros((n_steps, N_NEURONS))
    g_syn = np.zeros((n_steps, N_NEURONS))

    slope = max(abs(float(base_synapse["k"])), 1e-9)

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


def compute_delays_and_reldelays(spike_times):
    n_neurons  = len(spike_times)
    delay_data = {}

    for i in range(n_neurons):
        times_i   = spike_times[i]
        ref_times = spike_times[(i - 1) % n_neurons]
        if len(times_i) == 0 or len(ref_times) == 0:
            delay_data[i] = (np.array([]), np.array([]), np.array([]))
            continue
        ref_indices = np.searchsorted(ref_times, times_i, side="left") - 1
        valid       = ref_indices >= 0
        valid_times = times_i[valid]
        delays      = valid_times - ref_times[ref_indices[valid]]

        # Compute relative phase delay (delays / T_tot around that cycle)
        rel_delays = np.zeros_like(delays)
        for idx, (t_val, d_val) in enumerate(zip(valid_times, delays)):
            # cycle delays sum of most recent delays across all 4 neurons
            cycle_d = [d_val]
            for other in range(n_neurons):
                if other == i:
                    continue
                o_times = spike_times[other]
                o_ref   = spike_times[(other - 1) % n_neurons]
                idx_o   = np.searchsorted(o_times, t_val, side="right") - 1
                if idx_o >= 0:
                    t_o = o_times[idx_o]
                    idx_r = np.searchsorted(o_ref, t_o, side="left") - 1
                    if idx_r >= 0:
                        cycle_d.append(t_o - o_ref[idx_r])
            t_tot = sum(cycle_d) if len(cycle_d) == n_neurons else 0.0
            rel_delays[idx] = (d_val / t_tot) if t_tot > 0 else 0.25

        delay_data[i] = (valid_times, delays, rel_delays)

    return delay_data


def run():
    colors = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E"]  # n0..n3

    # Simulate both approaches
    time_a1, vf_a1 = simulate_modulated_ring(BASE_NEURON_A1, BASE_SYNAPSE_A1, NEURON_MODS_A1, SYNAPSE_MODS_A1)
    spikes_a1      = detect_spike_times(time_a1, vf_a1)
    delays_a1      = compute_delays_and_reldelays(spikes_a1)

    time_a2, vf_a2 = simulate_modulated_ring(BASE_NEURON_A2, BASE_SYNAPSE_A2, NEURON_MODS_A2, SYNAPSE_MODS_A2)
    spikes_a2      = detect_spike_times(time_a2, vf_a2)
    delays_a2      = compute_delays_and_reldelays(spikes_a2)

    # ------------------------------------------------------------
    # Export CSVs
    # ------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def export_approach_csvs(app_label, spikes, delay_data):
        # 1. Raster CSV
        r_rows = []
        for node, s_times in enumerate(spikes):
            for t in s_times:
                r_rows.append({"time": float(t), "neuron": node})
        df_r = pd.DataFrame(r_rows)
        export_to_csv(df_r, DATA_DIR / f"plotdata_duty_modulation_{app_label}_raster.csv")

        # 2. Delay CSV
        d_rows = []
        rd_rows = []
        for node, (times, d_vals, rd_vals) in delay_data.items():
            for t, d, rd in zip(times, d_vals, rd_vals):
                d_rows.append({"time": float(t), "neuron": node, "delay_ms": float(d)})
                rd_rows.append({"time": float(t), "neuron": node, "rel_delay": float(rd)})

        df_d = pd.DataFrame(d_rows)
        export_to_csv(df_d, DATA_DIR / f"plotdata_duty_modulation_{app_label}_delay.csv")

        df_rd = pd.DataFrame(rd_rows)
        export_to_csv(df_rd, DATA_DIR / f"plotdata_duty_modulation_{app_label}_reldelay.csv")
        return df_d, df_rd

    df_d1, df_rd1 = export_approach_csvs("approach1", spikes_a1, delays_a1)
    df_d2, df_rd2 = export_approach_csvs("approach2", spikes_a2, delays_a2)

    # Combined CSV
    df_comb1 = df_d1.merge(df_rd1, on=["time", "neuron"])
    df_comb1["approach"] = "fixed_period"
    df_comb2 = df_d2.merge(df_rd2, on=["time", "neuron"])
    df_comb2["approach"] = "fixed_node"
    df_all = pd.concat([df_comb1, df_comb2], ignore_index=True)
    export_to_csv(df_all, DATA_DIR / "fig10_duty_modulation_data.csv")

    print("[fig10] CSV exports complete.")

    # ------------------------------------------------------------
    # Plotting Figure 10 (3x2 Grid Layout)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(10, 6), dpi=150, sharex='col', gridspec_kw={'height_ratios': [1, 1, 1]})
    (ax_r1, ax_r2), (ax_d1, ax_d2), (ax_rd1, ax_rd2) = axes

    TIME_MIN, TIME_MAX = 1500.0, 2500.0

    # --- TOP ROW: RASTER PLOTS ---
    for node in range(N_NEURONS):
        t_s1 = spikes_a1[node]
        t_s1_v = t_s1[(t_s1 >= TIME_MIN) & (t_s1 <= TIME_MAX)]
        ax_r1.vlines(t_s1_v, node - 0.35, node + 0.35, colors=colors[node], lw=1.5, label=f"$n_{node}$")

        t_s2 = spikes_a2[node]
        t_s2_v = t_s2[(t_s2 >= TIME_MIN) & (t_s2 <= TIME_MAX)]
        ax_r2.vlines(t_s2_v, node - 0.35, node + 0.35, colors=colors[node], lw=1.5, label=f"$n_{node}$")

    ax_r1.set_title(r"Fixed Period ($T_{\mathrm{tot}}$)", fontsize=13, fontweight='bold', pad=10)
    ax_r2.set_title(r"Fixed Node ($\delta_i$)", fontsize=13, fontweight='bold', pad=10)

    ax_r1.set_ylabel("Neuron", fontsize=11)
    ax_r1.set_yticks(range(N_NEURONS))
    ax_r2.set_yticks(range(N_NEURONS))
    ax_r1.set_ylim(-0.5, 3.5)
    ax_r2.set_ylim(-0.5, 3.5)
    ax_r1.grid(True, linestyle="--", alpha=0.3)
    ax_r2.grid(True, linestyle="--", alpha=0.3)

    # --- MIDDLE ROW: ABSOLUTE DELAY (ms) ---
    for node in range(N_NEURONS):
        t_d1, d1_vals, _ = delays_a1[node]
        ax_d1.plot(t_d1, d1_vals, color=colors[node], marker='|', markersize=6, lw=1.2)

        t_d2, d2_vals, _ = delays_a2[node]
        ax_d2.plot(t_d2, d2_vals, color=colors[node], marker='|', markersize=6, lw=1.2)

    ax_d1.set_ylabel("Delay (ms)", fontsize=11)
    ax_d1.set_ylim(-2, 70)
    ax_d2.set_ylim(-2, 70)
    ax_d1.grid(True, linestyle="--", alpha=0.3)
    ax_d2.grid(True, linestyle="--", alpha=0.3)

    # --- BOTTOM ROW: RELATIVE DELAY ---
    for node in range(N_NEURONS):
        t_d1, _, rd1_vals = delays_a1[node]
        ax_rd1.plot(t_d1, rd1_vals, color=colors[node], marker='|', markersize=6, lw=1.2)

        t_d2, _, rd2_vals = delays_a2[node]
        ax_rd2.plot(t_d2, rd2_vals, color=colors[node], marker='|', markersize=6, lw=1.2)

    ax_rd1.set_ylabel("Rel. Delay", fontsize=11)
    ax_rd1.set_xlabel("Time (ms)", fontsize=11)
    ax_rd2.set_xlabel("Time (ms)", fontsize=11)

    ax_rd1.set_ylim(-0.02, 0.48)
    ax_rd2.set_ylim(-0.02, 0.48)
    ax_rd1.set_xlim(TIME_MIN, TIME_MAX)
    ax_rd2.set_xlim(TIME_MIN, TIME_MAX)
    ax_rd1.grid(True, linestyle="--", alpha=0.3)
    ax_rd2.grid(True, linestyle="--", alpha=0.3)

    # --- SHARED LEGEND ---
    handles, labels = ax_r1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), frameon=True, fontsize=11)

    plt.subplots_adjust(hspace=0.25, wspace=0.15, right=0.86)

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig10] Saved Figure 10 plot to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
