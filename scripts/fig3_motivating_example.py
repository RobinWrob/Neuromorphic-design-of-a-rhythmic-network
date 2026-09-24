"""
fig3_motivating_example.py - Figure 3 Simulation & Data Export.

Generates raster and inter-node delay CSVs for both Excitable Ring and
Equivalent Oscillator Network under parameter modulation at T_OVERRIDE = T_END / 2:
- plotdata_excitable_raster.csv
- plotdata_excitable_delay.csv
- plotdata_oscillatory_raster.csv
- plotdata_oscillatory_delay.csv

  (with manual_synapse_overrides: tau_decay={"excitable": 8.0, "oscillatory": 4.0})
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
OUTPUT_FIG = PROJECT_ROOT / "results" / "figures" / "fig3_motivating_example.png"

# Global network and simulation parameters
DT = 0.025
T_END = 3000.0
T_OVERRIDE = 2000
TIME = np.arange(0.0, T_END + 0.5 * DT, DT)
N_NEURONS = 4

SPIKE_THRESHOLD = 50.0
T_PLOT_START = 1500.0
T_PLOT_END =  2500.0

PULSE_START     = 0.0
PULSE_DURATION  = 20.0
PULSE_AMPLITUDE = -5.0
I_TONIC_OSC     = 4

SYN_TAU_DECAY   = 3.25
SYN_TAU_RISE    = 0.1
SYN_G_MAX       = 3.0
SYN_E_REV_INHIB = -20.0
SYN_E_REV_EXC   = 100.0
SYN_THRESHOLD   = 30.0
SYN_K           = 0.1

OSC_TOPOLOGY_MODE = "a2a"
A2A_G_MAX         = 0.11
A2A_RING_G_MAX    = 0.05
GA_OSC            = 5.076
GNAREB_OSC        = 0.0

MANUAL_SYNAPSE_IDX = 0
MANUAL_SYNAPSE_OVERRIDES = {
    "g_max": None,
    "tau_decay": {"oscillatory": 4, "excitable": 9.5},
}


def satrelu_from_threshold(v, threshold, slope):
    return np.clip(slope * (v - threshold), 0.0, 1.0)


def satrelu_from_threshold_decreasing(v, threshold, slope):
    return np.clip(slope * (threshold - v), 0.0, 1.0)


def build_ring_topology(n_neurons):
    src = np.arange(n_neurons, dtype=int)
    dst = (src + 1) % n_neurons
    return src, dst


def build_topology_for_case(case_name, n_neurons):
    ring_src, ring_dst = build_ring_topology(n_neurons)
    ring_g = np.full(len(ring_src), SYN_G_MAX, dtype=float)
    ring_e = np.full(len(ring_src), SYN_E_REV_INHIB, dtype=float)
    ring_tau = np.full(len(ring_src), SYN_TAU_DECAY, dtype=float)

    if case_name == "excitable":
        return ring_src, ring_dst, ring_g, ring_e, ring_tau

    a2a_src, a2a_dst = [], []
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i != j:
                a2a_src.append(i)
                a2a_dst.append(j)

    src = np.asarray(a2a_src, dtype=int)
    dst = np.asarray(a2a_dst, dtype=int)
    g = np.full(len(src), A2A_G_MAX, dtype=float)
    e_rev = np.full(len(src), SYN_E_REV_INHIB, dtype=float)
    tau_decay = np.full(len(src), SYN_TAU_DECAY, dtype=float)

    src = np.concatenate((src, ring_src))
    dst = np.concatenate((dst, ring_dst))
    g = np.concatenate((g, np.full(len(ring_src), A2A_RING_G_MAX)))
    e_rev = np.concatenate((e_rev, np.full(len(ring_src), SYN_E_REV_EXC)))
    tau_decay = np.full(len(src), SYN_TAU_DECAY, dtype=float)
    return src, dst, g, e_rev, tau_decay


def build_params(n_neurons):
    p = {}
    p["cm"] = np.full(n_neurons, 1.0)
    p["tau_s"] = np.full(n_neurons, 5.0)
    p["input_gain"] = np.full(n_neurons, 1.0)
    p["i_base"] = np.full(n_neurons, 0.0)

    p["g_na"] = np.full(n_neurons, 50.0)
    p["e_na"] = np.full(n_neurons, 100.0)
    p["na_act_bias"] = np.full(n_neurons, 10.0)
    p["na_act_slope"] = np.full(n_neurons, 0.02)
    p["na_inact_bias"] = np.full(n_neurons, 30.0)
    p["na_inact_slope"] = np.full(n_neurons, 0.05)

    p["g_na_reb"] = np.full(n_neurons, 2.0)
    p["e_na_reb"] = np.full(n_neurons, 100.0)
    p["na_reb_act_bias"] = np.full(n_neurons, -10.0)
    p["na_reb_act_slope"] = np.full(n_neurons, 0.1)
    p["na_reb_inact_bias"] = np.full(n_neurons, -5.0)
    p["na_reb_inact_slope"] = np.full(n_neurons, 0.1)

    p["g_k"] = np.full(n_neurons, 1.0)
    p["e_k"] = np.full(n_neurons, -20.0)
    p["k_act_bias"] = np.full(n_neurons, 10.0)
    p["k_act_slope"] = np.full(n_neurons, 0.05)

    p["g_l"] = np.full(n_neurons, 0.1)
    p["e_l"] = np.full(n_neurons, 0.0)

    p["g_a"] = np.full(n_neurons, 0.0)
    p["e_a"] = np.full(n_neurons, -20.0)
    p["a_act_bias"] = np.full(n_neurons, 2.0)
    p["a_act_slope"] = np.full(n_neurons, 0.01)
    p["a_inact_bias"] = np.full(n_neurons, 10.0)
    p["a_inact_slope"] = np.full(n_neurons, 0.15)
    return p


def compute_gates(vf, vs, params):
    return {
        "m_na": satrelu_from_threshold(vf, params["na_act_bias"], params["na_act_slope"]),
        "h_na": satrelu_from_threshold_decreasing(vs, params["na_inact_bias"], params["na_inact_slope"]),
        "m_na_reb": satrelu_from_threshold(vf, params["na_reb_act_bias"], params["na_reb_act_slope"]),
        "h_na_reb": satrelu_from_threshold_decreasing(vs, params["na_reb_inact_bias"], params["na_reb_inact_slope"]),
        "n_k": satrelu_from_threshold(vs, params["k_act_bias"], params["k_act_slope"]),
        "m_a": satrelu_from_threshold(vf, params["a_act_bias"], params["a_act_slope"]),
        "h_a": satrelu_from_threshold_decreasing(vs, params["a_inact_bias"], params["a_inact_slope"]),
    }


def applied_current_excitable(t, n_neurons):
    current = np.zeros(n_neurons, dtype=float)
    if PULSE_START <= t < PULSE_START + PULSE_DURATION:
        current[0] = PULSE_AMPLITUDE
    return current


def applied_current_oscillatory(n_neurons):
    return np.full(n_neurons, I_TONIC_OSC, dtype=float)


def compute_rhs(
    t, vf, vs, g_syn, gates, params, syn_src, syn_dst,
    syn_g_max, syn_e_rev, syn_tau_decay, external_input,
):
    n_neurons = len(vf)
    n_syn = len(syn_src)

    i_ionic = np.zeros(n_neurons, dtype=float)
    i_ionic += -params["g_na"] * gates["m_na"] * gates["h_na"] * (vf - params["e_na"])
    i_ionic += -params["g_a"] * gates["m_a"] * gates["h_a"] * (vf - params["e_a"])
    i_ionic += -params["g_na_reb"] * gates["m_na_reb"] * gates["h_na_reb"] * (vf - params["e_na_reb"])
    i_ionic += -params["g_k"] * gates["n_k"] * (vf - params["e_k"])
    i_ionic += -params["g_l"] * (vf - params["e_l"])

    input_currents = np.array(external_input, copy=True, dtype=float)
    dg_syn = np.zeros(n_syn, dtype=float)

    slope = max(abs(SYN_K), 1e-9)
    inv_tau_r = 1.0 / max(SYN_TAU_RISE, 1e-9)

    for j in range(n_syn):
        src = syn_src[j]
        dst = syn_dst[j]
        v_pre = vf[src]
        v_post = vf[dst]

        drive = 1.0 / (1.0 + np.exp(-(v_pre - SYN_THRESHOLD) / slope))
        inv_tau_d = 1.0 / max(float(syn_tau_decay[j]), 1e-9)
        alpha = max(0.0, (inv_tau_r - inv_tau_d) * drive)
        dg_syn[j] = alpha * (1.0 - g_syn[j]) - inv_tau_d * g_syn[j]
        input_currents[dst] += syn_g_max[j] * g_syn[j] * (syn_e_rev[j] - v_post)

    i_app = -params["input_gain"] * input_currents - params["i_base"]
    dvf = (i_ionic - i_app) / params["cm"]
    dvs = (vf - vs) / (params["tau_s"] * params["cm"])
    return dvf, dvs, dg_syn


def simulate_case(
    case_name, params_pre, params_post, time, syn_src, syn_dst,
    syn_g_max_pre, syn_g_max_post, syn_e_rev,
    syn_tau_decay_pre, syn_tau_decay_post, t_override,
):
    n_neurons = len(params_pre["cm"])
    params_case_pre = {k: np.array(v, copy=True) for k, v in params_pre.items()}
    params_case_post = {k: np.array(v, copy=True) for k, v in params_post.items()}

    if case_name == "excitable":
        params_case_pre["g_a"] = np.zeros(n_neurons)
        params_case_post["g_a"] = np.zeros(n_neurons)
    elif case_name == "oscillatory":
        params_case_pre["g_a"] = np.full(n_neurons, GA_OSC)
        params_case_post["g_a"] = np.full(n_neurons, GA_OSC)
        params_case_pre["g_na_reb"] = np.full(n_neurons, GNAREB_OSC)
        params_case_post["g_na_reb"] = np.full(n_neurons, GNAREB_OSC)

    n_steps = len(time)
    n_syn = len(syn_src)
    vf = np.zeros((n_steps, n_neurons), dtype=float)
    vs = np.zeros((n_steps, n_neurons), dtype=float)
    g_syn = np.zeros((n_steps, n_syn), dtype=float)

    if case_name == "excitable":
        vf[0, 0] = 0.0
        vs[0, 0] = 0.0
    else:
        vf[0, 0] = -5.0
        vs[0, 0] = -10.0

    for k in range(n_steps - 1):
        t = time[k]
        use_post = t >= t_override
        active_params = params_case_post if use_post else params_case_pre
        active_g = syn_g_max_post if use_post else syn_g_max_pre
        active_tau = syn_tau_decay_post if use_post else syn_tau_decay_pre

        gates = compute_gates(vf[k], vs[k], active_params)

        if case_name == "excitable":
            external_input = applied_current_excitable(t, n_neurons)
        else:
            external_input = applied_current_oscillatory(n_neurons)

        dvf, dvs, dg = compute_rhs(
            t, vf[k], vs[k], g_syn[k], gates, active_params,
            syn_src, syn_dst, active_g, syn_e_rev, active_tau, external_input,
        )
        vf[k + 1] = vf[k] + DT * dvf
        vs[k + 1] = vs[k] + DT * dvs
        g_syn[k + 1] = g_syn[k] + DT * dg

    return {"vf": vf, "vs": vs, "g_syn": g_syn}


def detect_spike_times(time, vf, threshold=SPIKE_THRESHOLD):
    spikes = []
    for i in range(vf.shape[1]):
        above = vf[:, i] >= threshold
        crossings = np.where((~above[:-1]) & above[1:])[0] + 1
        spikes.append(time[crossings])
    return spikes


def compute_absolute_delay_series(spike_times):
    delay_data = {}
    n_neurons = len(spike_times)
    for i, times_i in enumerate(spike_times):
        ref_times = spike_times[(i - 1) % n_neurons]
        if len(times_i) == 0 or len(ref_times) == 0:
            delay_data[i] = (np.array([]), np.array([]))
            continue
        ref_indices = np.searchsorted(ref_times, times_i, side="left") - 1
        valid = ref_indices >= 0
        valid_times = times_i[valid]
        delays = valid_times - ref_times[ref_indices[valid]]
        delay_data[i] = (valid_times, delays)
    return delay_data


def build_raster_df(case_name, spike_times):
    rows = []
    for neuron, times in enumerate(spike_times):
        times = np.asarray(times, dtype=float)
        if T_PLOT_START is not None:
            times = times[times >= T_PLOT_START]
        rows.extend((float(t), neuron) for t in times)
    return pd.DataFrame(rows, columns=["time", "neuron"])


def build_delay_df(case_name, delay_data):
    rows = []
    for i, (times, delays) in delay_data.items():
        for t, delay in zip(times, delays):
            if T_PLOT_START is None or t >= T_PLOT_START:
                rows.append({"time": float(t), "neuron": i, "delay_ms": float(delay)})
    return pd.DataFrame(rows)


def run():
    params_pre = build_params(N_NEURONS)
    params_post = {k: np.array(v, copy=True) for k, v in params_pre.items()}

    run_order = ["excitable", "oscillatory"]
    results = {}

    for name in run_order:
        syn_src, syn_dst, g_base, syn_e_rev, tau_base = build_topology_for_case(name, N_NEURONS)
        g_pre = np.array(g_base, copy=True, dtype=float)
        g_post = np.array(g_base, copy=True, dtype=float)
        tau_pre = np.array(tau_base, copy=True, dtype=float)
        tau_post = np.array(tau_base, copy=True, dtype=float)

        if MANUAL_SYNAPSE_IDX is not None:
            syn_idx = int(MANUAL_SYNAPSE_IDX)
            tau_override_val = MANUAL_SYNAPSE_OVERRIDES["tau_decay"]
            if isinstance(tau_override_val, dict):
                tau_override_val = tau_override_val.get(name)
            if tau_override_val is not None:
                tau_post[syn_idx] = max(float(tau_override_val), 1e-9)

        sim_res = simulate_case(
            name, params_pre, params_post, TIME, syn_src, syn_dst,
            g_pre, g_post, syn_e_rev, tau_pre, tau_post, T_OVERRIDE,
        )
        spikes = detect_spike_times(TIME, sim_res["vf"], SPIKE_THRESHOLD)
        delay_data = compute_absolute_delay_series(spikes)

        df_raster = build_raster_df(name, spikes)
        df_delay = build_delay_df(name, delay_data)

        export_to_csv(df_raster, DATA_DIR / f"plotdata_{name}_raster.csv")
        export_to_csv(df_delay, DATA_DIR / f"plotdata_{name}_delay.csv")
        print(f"[fig3] {name}: wrote raster ({len(df_raster)} spikes) and delay CSVs.")

        results[name] = {
            "spikes": spikes,
            "delay_data": delay_data,
            "df_raster": df_raster,
            "df_delay": df_delay,
        }

    # Validation plot: 2x2 grid matching paper layout
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, N_NEURONS))

    # Column 0: Excitable Ring
    df_r_exc = results["excitable"]["df_raster"]
    df_d_exc = results["excitable"]["df_delay"]

    if not df_r_exc.empty:
        for i in range(N_NEURONS):
            sub = df_r_exc[df_r_exc["neuron"] == i]
            axes[0, 0].vlines(sub["time"], i - 0.35, i + 0.35, colors=colors[i], lw=1.2)
    axes[0, 0].set_title("Ring of excitable nodes", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Neuron", fontsize=10)
    axes[0, 0].set_yticks(range(N_NEURONS))
    axes[0, 0].set_ylim(-0.5, N_NEURONS - 0.5)
    axes[0, 0].grid(True, linestyle="--", alpha=0.3)
    axes[0, 0].axvline(T_OVERRIDE, color="gray", linestyle=":", lw=1.0)

    if not df_d_exc.empty:
        for i in range(N_NEURONS):
            sub = df_d_exc[df_d_exc["neuron"] == i]
            if not sub.empty:
                axes[1, 0].plot(sub["time"], sub["delay_ms"], "o-", color=colors[i],
                                label=f"$n_{i}$", ms=3, lw=1.0)
    axes[1, 0].set_xlabel("Time (ms)", fontsize=10)
    axes[1, 0].set_ylabel("Delay (ms)", fontsize=10)
    axes[1, 0].set_xlim(T_PLOT_START, T_PLOT_END)
    axes[1, 0].set_ylim(0, 50)
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)
    axes[1, 0].axvline(T_OVERRIDE, color="gray", linestyle=":", lw=1.0)

    # Column 1: Oscillator Network
    df_r_osc = results["oscillatory"]["df_raster"]
    df_d_osc = results["oscillatory"]["df_delay"]

    if not df_r_osc.empty:
        for i in range(N_NEURONS):
            sub = df_r_osc[df_r_osc["neuron"] == i]
            axes[0, 1].vlines(sub["time"], i - 0.35, i + 0.35, colors=colors[i], lw=1.2)
    axes[0, 1].set_title("Equivalent network of oscillators", fontsize=11, fontweight="bold")
    axes[0, 1].set_yticks(range(N_NEURONS))
    axes[0, 1].set_ylim(-0.5, N_NEURONS - 0.5)
    axes[0, 1].grid(True, linestyle="--", alpha=0.3)
    axes[0, 1].axvline(T_OVERRIDE, color="gray", linestyle=":", lw=1.0)

    if not df_d_osc.empty:
        for i in range(N_NEURONS):
            sub = df_d_osc[df_d_osc["neuron"] == i]
            if not sub.empty:
                axes[1, 1].plot(sub["time"], sub["delay_ms"], "o-", color=colors[i],
                                label=f"$n_{i}$", ms=3, lw=1.0)
    axes[1, 1].set_xlabel("Time (ms)", fontsize=10)
    axes[1, 1].set_xlim(T_PLOT_START, T_PLOT_END)
    axes[1, 1].set_ylim(0, 50)
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)
    axes[1, 1].axvline(T_OVERRIDE, color="gray", linestyle=":", lw=1.0)

    # Common Legend
    axes[1, 0].legend(loc="upper left", fontsize=8, ncol=N_NEURONS)

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG)
    plt.close(fig)
    print(f"[fig3] Wrote plot figure to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
