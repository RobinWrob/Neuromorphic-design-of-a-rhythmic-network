"""
fig15_exo_tuning_hom_het.py - Figure 15 Exogenous Tuning (Homogeneous vs Heterogeneous Ring).

Simulates a 4-neuron ring driven by an entraining sensory signal while undergoing a parameter
modulation at T_MOD (g_a -> 1.5 and network tau_decay -> 3.0 ms):
  - Homogeneous (left panel): modulation applied globally across all neurons/synapses
  - Heterogeneous (right panel): modulation applied to Node 0 only

Export CSVs:
- plotdata/entrain_ring_modulation.csv
- plotdata/entrain_ring_modulation_homo.csv
- plotdata/entrain_ring_modulation_hetero.csv
- results/data/fig15_exo_tuning_hom_het_data.csv
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

from src.eprc_engine import (
    default_eprc_neuron_params,
    default_nominal_synapse_params,
    default_sensory_synapse_params,
)
from src.helper import export_to_csv

# Paths
DATA_DIR     = PROJECT_ROOT / "results" / "data"
PLOTDATA_DIR = PROJECT_ROOT / "plotdata"
OUTPUT_FIG   = PROJECT_ROOT / "results" / "figures" / "fig15_exo_tuning_hom_het.png"

# Network and simulation configuration
N_NEURONS = 4
DT = 0.01
T_END = 20000.0           # Total simulation time (ms)
T_MOD = T_END / 2.0       # Modulation onset time (ms)

NEW_G_A = 1.5             # Updated g_a conductance after T_MOD
NEW_TAU_DECAY = 3.0       # Updated network synapse tau_decay after T_MOD

INIT_PULSE_DURATION = 5.0 # Kickstart pulse duration (ms)
INIT_PULSE_CURRENT = 5.0  # Kickstart pulse current (mA)

SENSORY_PERIOD_MISMATCH = -15.0
SENSORY_NULL = 10.0
IMPULSE_WIDTH = 3.0
IMPULSE_AMPLITUDE = 100.0

SPIKE_THRESHOLD = 30.0
G_MAX_SENS = 0.01


def satrelu(v, threshold, slope):
    return np.clip(slope * (v - threshold), 0.0, 1.0)


def compute_gates(vf, vs, p):
    return {
        "m_na": satrelu(vf, p["na_act_bias"], p["na_act_slope"]),
        "h_na": satrelu(vs, p["na_inact_bias"], -p["na_inact_slope"]),
        "m_na_reb": satrelu(vf, p["na_reb_act_bias"], p["na_reb_act_slope"]),
        "h_na_reb": satrelu(vs, p["na_reb_inact_bias"], -p["na_reb_inact_slope"]),
        "n_k": satrelu(vs, p["k_act_bias"], p["k_act_slope"]),
        "m_a": satrelu(vf, p["a_act_bias"], p["a_act_slope"]),
        "h_a": satrelu(vs, p["a_inact_bias"], -p["a_inact_slope"]),
    }


def detect_spike_times(time, vf, threshold):
    spikes = []
    for i in range(vf.shape[1]):
        above = vf[:, i] >= threshold
        crossings = np.where((~above[:-1]) & above[1:])[0] + 1
        spikes.append(time[crossings])
    return spikes


def estimate_baseline_period(spikes, t_start, t_end):
    isis = []
    for node_spikes in spikes:
        valid_spikes = node_spikes[(node_spikes >= t_start) & (node_spikes <= t_end)]
        if len(valid_spikes) > 1:
            isis.extend(np.diff(valid_spikes))
    if len(isis) == 0:
        return 60.8
    return float(np.mean(isis))


def simulate_sensory_ring(sensory_period, g_max_sensory=G_MAX_SENS, modulate_target="none", apply_modulation=True, t_end=T_END):
    time = np.arange(0.0, t_end + 0.5 * DT, DT)
    n_steps = len(time)

    syn_src = np.arange(N_NEURONS, dtype=int)
    syn_dst = (syn_src + 1) % N_NEURONS

    base_p = default_eprc_neuron_params()
    net_syn = default_nominal_synapse_params()
    net_syn["tau_decay"] = 20.6

    sens_syn = default_sensory_synapse_params("inh")
    sens_syn.update({"e_rev": 50.0, "tau_decay": 1.0, "g_max": g_max_sensory})

    p_n = {k: np.full(N_NEURONS, v, dtype=float) for k, v in base_p.items()}
    p_s_net = {k: np.full(N_NEURONS, v, dtype=float) for k, v in net_syn.items()}
    p_s_sens = {k: np.full(N_NEURONS, v, dtype=float) for k, v in sens_syn.items()}

    vf = np.zeros((n_steps, N_NEURONS))
    vs = np.zeros((n_steps, N_NEURONS))
    g_syn_net = np.zeros((n_steps, N_NEURONS))
    g_syn_sens = np.zeros((n_steps, N_NEURONS))

    v_pre_sens_array = np.zeros(n_steps)
    slope_net = max(abs(net_syn["k"]), 1e-9)
    slope_sens = max(abs(sens_syn["k"]), 1e-9)

    for k in range(n_steps - 1):
        t = time[k]

        if apply_modulation and t >= T_MOD:
            if modulate_target == "node0":
                p_n["g_a"][0] = NEW_G_A
                p_s_net["tau_decay"][-1] = NEW_TAU_DECAY
            elif modulate_target == "all":
                p_n["g_a"][:] = NEW_G_A
                p_s_net["tau_decay"][:] = NEW_TAU_DECAY

        v_f, v_s = vf[k], vs[k]
        g_s_net, g_s_sens = g_syn_net[k], g_syn_sens[k]

        phase_sens = t % sensory_period if sensory_period > 0 else 0.0
        v_pre_sens = IMPULSE_AMPLITUDE if phase_sens < IMPULSE_WIDTH and t > SENSORY_NULL else 0.0
        v_pre_sens_array[k] = v_pre_sens

        gates = compute_gates(v_f, v_s, p_n)
        i_ionic = (
            -p_n["g_na"] * gates["m_na"] * gates["h_na"] * (v_f - p_n["e_na"])
            - p_n["g_a"] * gates["m_a"] * gates["h_a"] * (v_f - p_n["e_a"])
            - p_n["g_na_reb"] * gates["m_na_reb"] * gates["h_na_reb"] * (v_f - p_n["e_na_reb"])
            - p_n["g_k"] * gates["n_k"] * (v_f - p_n["e_k"])
            - p_n["g_l"] * (v_f - p_n["e_l"])
        )

        input_currents = np.zeros(N_NEURONS)
        dg_syn_net = np.zeros(N_NEURONS)
        dg_syn_sens = np.zeros(N_NEURONS)

        if t <= INIT_PULSE_DURATION:
            input_currents[3] -= INIT_PULSE_CURRENT

        for j in range(N_NEURONS):
            src, dst = syn_src[j], syn_dst[j]
            drive_net = 1.0 / (1.0 + np.exp(-(v_f[src] - p_s_net["threshold"][j]) / slope_net))
            inv_tau_r_net = 1.0 / p_s_net["tau_rise"][j]
            inv_tau_d_net = 1.0 / p_s_net["tau_decay"][j]

            alpha_net = max(0.0, (inv_tau_r_net - inv_tau_d_net) * drive_net)
            dg_syn_net[j] = alpha_net * (1.0 - g_s_net[j]) - inv_tau_d_net * g_s_net[j]
            input_currents[dst] += p_s_net["g_max"][j] * g_s_net[j] * (p_s_net["e_rev"][j] - v_f[dst])

            drive_sens = 1.0 / (1.0 + np.exp(-(v_pre_sens - p_s_sens["threshold"][j]) / slope_sens))
            inv_tau_r_sens = 1.0 / p_s_sens["tau_rise"][j]
            inv_tau_d_sens = 1.0 / p_s_sens["tau_decay"][j]

            alpha_sens = max(0.0, (inv_tau_r_sens - inv_tau_d_sens) * drive_sens)
            dg_syn_sens[j] = alpha_sens * (1.0 - g_s_sens[j]) - inv_tau_d_sens * g_s_sens[j]
            input_currents[j] += p_s_sens["g_max"][j] * g_s_sens[j] * (p_s_sens["e_rev"][j] - v_f[j])

        i_app = -p_n["input_gain"] * input_currents - p_n["i_base"]
        dvf = (i_ionic - i_app) / p_n["cm"]
        dvs = (v_f - v_s) / (p_n["tau_s"] * p_n["cm"])

        vf[k + 1] = v_f + DT * dvf
        vs[k + 1] = v_s + DT * dvs
        g_syn_net[k + 1] = g_s_net + DT * dg_syn_net
        g_syn_sens[k + 1] = g_s_sens + DT * dg_syn_sens

    pulse_starts_idx = np.where((v_pre_sens_array[:-1] == 0) & (v_pre_sens_array[1:] > 0))[0] + 1
    sensory_event_times = time[pulse_starts_idx]

    return time, vf, sensory_event_times


def compute_sensory_phase_data(spikes, sensory_event_times):
    records = []
    sensory_event_times = np.sort(sensory_event_times)
    for node, node_spikes in enumerate(spikes):
        for t_spike in node_spikes:
            past_events = sensory_event_times[sensory_event_times <= t_spike]
            if len(past_events) > 0:
                last_sensory_t = past_events[-1]
                delay = t_spike - last_sensory_t
                records.append({
                    "spike_time": float(t_spike),
                    "neuron_id": int(node),
                    "last_sensory_time": float(last_sensory_t),
                    "delay_since_sensory": float(delay),
                    "is_valid": True,
                })
    if len(records) == 0:
        return pd.DataFrame(columns=["spike_time", "neuron_id", "last_sensory_time", "delay_since_sensory", "is_valid"])
    return pd.DataFrame(records)


def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTDATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Estimate unmodulated baseline period
    dummy_period = 100.0
    t_base_end = 2000.0
    time_base, vf_base, _ = simulate_sensory_ring(dummy_period, g_max_sensory=0.0, modulate_target="none", apply_modulation=False, t_end=t_base_end)
    spikes_base = detect_spike_times(time_base, vf_base, SPIKE_THRESHOLD)
    baseline_period = estimate_baseline_period(spikes_base, t_base_end / 2.0, t_base_end)
    sensory_period = baseline_period + SENSORY_PERIOD_MISMATCH

    print(f"[fig15] Estimated baseline period: {baseline_period:.2f} ms")
    print(f"[fig15] Sensory period: {sensory_period:.2f} ms")

    # 2. Run Homogeneous Modulation ("all")
    time_homo, vf_homo, sens_t_homo = simulate_sensory_ring(sensory_period, g_max_sensory=G_MAX_SENS, modulate_target="all", apply_modulation=True)
    spikes_homo = detect_spike_times(time_homo, vf_homo, SPIKE_THRESHOLD)
    df_homo = compute_sensory_phase_data(spikes_homo, sens_t_homo)

    # 3. Run Heterogeneous Modulation ("node0")
    time_hetero, vf_hetero, sens_t_hetero = simulate_sensory_ring(sensory_period, g_max_sensory=G_MAX_SENS, modulate_target="node0", apply_modulation=True)
    spikes_hetero = detect_spike_times(time_hetero, vf_hetero, SPIKE_THRESHOLD)
    df_hetero = compute_sensory_phase_data(spikes_hetero, sens_t_hetero)

    df_homo["mode"] = "homogeneous"
    df_hetero["mode"] = "heterogeneous"
    df_combined = pd.concat([df_homo, df_hetero], ignore_index=True)

    export_to_csv(df_homo, PLOTDATA_DIR / "entrain_ring_modulation.csv")
    export_to_csv(df_homo, PLOTDATA_DIR / "entrain_ring_modulation_homo.csv")
    export_to_csv(df_hetero, PLOTDATA_DIR / "entrain_ring_modulation_hetero.csv")

    export_to_csv(df_homo, DATA_DIR / "entrain_ring_modulation_homo.csv")
    export_to_csv(df_hetero, DATA_DIR / "entrain_ring_modulation_hetero.csv")
    export_to_csv(df_combined, DATA_DIR / "fig15_exo_tuning_hom_het_data.csv")
    print(f"[fig15] Exported exogenous tuning CSVs to DATA_DIR and PLOTDATA_DIR")

    # 4. Render Figure 15 (1x2 Multi-Panel Grid matching publication design 1:1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True, dpi=150)

    # publication colors matching Figure 15 screenshot (n0: Red, n1: Blue, n2: Green, n3: Orange)
    node_colors = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E"]
    node_labels = [r"$n_0$", r"$n_1$", r"$n_2$", r"$n_3$"]

    t_mod_sec = T_MOD / 1000.0

    # --- LEFT PANEL: HOMOGENEOUS ---
    if not df_homo.empty and "neuron_id" in df_homo.columns:
        for node in range(N_NEURONS):
            sub = df_homo[df_homo["neuron_id"] == node]
            if not sub.empty:
                t_sec = sub["spike_time"] / 1000.0
                ax1.plot(t_sec, sub["delay_since_sensory"], "|-", color=node_colors[node], label=node_labels[node], ms=5, lw=1.0)

    ax1.axvline(t_mod_sec, color="#555555", linestyle="--", lw=2.0)
    ax1.set_title("Homogeneous", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_ylabel("Delay (ms)", fontsize=10)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 300)
    ax1.set_xticks([0, 2.5, 5, 7.5, 10])
    ax1.grid(True, linestyle="--", alpha=0.3)

    # --- RIGHT PANEL: HETEROGENEOUS ---
    if not df_hetero.empty and "neuron_id" in df_hetero.columns:
        for node in range(N_NEURONS):
            sub = df_hetero[df_hetero["neuron_id"] == node]
            if not sub.empty:
                t_sec = sub["spike_time"] / 1000.0
                ax2.plot(t_sec, sub["delay_since_sensory"], "|-", color=node_colors[node], label=node_labels[node], ms=5, lw=1.0)

    ax2.axvline(t_mod_sec, color="#555555", linestyle="--", lw=2.0)
    ax2.set_title("Heterogeneous", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_xlim(0, 10)
    ax2.set_xticks([0, 2.5, 5, 7.5, 10])
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Shared bottom legend matching publication design
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig15] Saved Figure 15 plot to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
