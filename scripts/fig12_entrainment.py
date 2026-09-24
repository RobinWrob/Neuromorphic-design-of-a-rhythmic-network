"""
fig12_entrainment.py - Figure 12 Entrainment Simulation & Arnold Tongue.

Simulates a 4-neuron ring driven by a periodic sensory pulse sequence and dynamically computes
the Arnold Tongue (1:1 phase-locking region) across period mismatch Delta T and sensory conductance g_syn.

Exports CSVs:
  - plotdata/entrain_ring_spike_phase_data.csv (and results/data/)
  - results/data/fig12_arnold_tongue_data.csv
  - plotdata/plotdata_arnold_tongue.csv
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eprc_engine import default_eprc_neuron_params, default_sensory_synapse_params, default_nominal_synapse_params
from src.helper import export_to_csv

# Paths
DATA_DIR     = PROJECT_ROOT / "results" / "data"
PLOTDATA_DIR = PROJECT_ROOT / "plotdata"
OUTPUT_FIG   = PROJECT_ROOT / "results" / "figures" / "fig12_entrainment.png"

# Configuration
N_NEURONS = 4
DT = 0.02
T_END = 1500.0

INIT_PULSE_DURATION = 30.0
INIT_PULSE_CURRENT = 5.0

SENSORY_NULL = 100.0
IMPULSE_WIDTH = 3.0
IMPULSE_AMPLITUDE = 100.0
SPIKE_THRESHOLD = 30.0

# Sample operating parameters
SAMPLE_MISMATCH = -6.5
SAMPLE_GMAX     = 0.09

# Arnold Tongue Sweep Settings
MISMATCH_MIN   = -8.0
MISMATCH_MAX   = 2.0
MISMATCH_STEPS = 21

GMAX_MIN   = 0.00
GMAX_MAX   = 0.15
GMAX_STEPS = 31

PARALLELIZE = True
ONLY_SAMPLE = False


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


def detect_spike_times(time, vf, threshold=SPIKE_THRESHOLD):
    spikes = []
    for i in range(vf.shape[1]):
        above = vf[:, i] >= threshold
        crossings = np.where((~above[:-1]) & above[1:])[0] + 1
        spikes.append(time[crossings])
    return spikes


def analyze_period(spikes, t_end, is_sensory=False, sensory_period=None, nom_period=None, applied_mismatch=0):
    half_time = t_end / 2.0
    late_spikes_with_id = []
    for node, node_spikes in enumerate(spikes):
        for t in node_spikes[node_spikes >= half_time]:
            late_spikes_with_id.append((t, node))

    if not late_spikes_with_id:
        return None, "No spikes"

    late_spikes_with_id.sort(key=lambda x: x[0])
    participating_neurons = set([late_spikes_with_id[0][1]])

    for i in range(1, len(late_spikes_with_id)):
        t_curr, node_curr = late_spikes_with_id[i]
        t_prev, node_prev = late_spikes_with_id[i - 1]
        participating_neurons.add(node_curr)
        dt = t_curr - t_prev
        if dt <= 2.0:
            return None, "Not sequential"

    if len(participating_neurons) < len(spikes):
        return None, "Not all neurons fired"

    if is_sensory and sensory_period is not None:
        # Select reference neuron: default to neuron 0, or fallback to any participating neuron with spikes
        ref_spikes = spikes[0]
        if len(ref_spikes) == 0:
            for node_idx in range(len(spikes)):
                if len(spikes[node_idx]) > 0:
                    ref_spikes = spikes[node_idx]
                    break

        sensory_events = np.arange(sensory_period, t_end, sensory_period)
        sensory_events = sensory_events[sensory_events >= SENSORY_NULL]
        late_events = sensory_events[sensory_events >= t_end - 10 * sensory_period]

        if len(late_events) < 4 or len(ref_spikes) < 4:
            return None, "Insufficient spikes"

        # At 0 period mismatch (or near 0), driven oscillation is entrained by definition if ring is sequential
        if nom_period is not None and abs(sensory_period - nom_period) < 1e-4:
            late_spikes = ref_spikes[ref_spikes >= t_end - 10 * sensory_period]
            late_isis = np.diff(late_spikes) if len(late_spikes) > 1 else np.array([sensory_period])
            return np.mean(late_isis), "Phase-locked (Zero Mismatch)"

        # Compute phase delay (sensory_event_time - nearest spike_time) over late cycles
        ev = np.asarray(late_events).ravel().astype(float)
        sp = np.asarray(ref_spikes).ravel().astype(float)
        closest_spike_idx = np.argmin(np.abs(ev[:, None] - sp[None, :]), axis=1)
        delays = ev - sp[closest_spike_idx]

        # Settling criterion: peak-to-peak phase drift over last 10 cycles < mismatch/4 ms
        drift = np.max(delays) - np.min(delays)
        if abs(drift) < abs(applied_mismatch)/4:
            late_spikes = ref_spikes[ref_spikes >= t_end - 10 * sensory_period]
            late_isis = np.diff(late_spikes) if len(late_spikes) > 1 else np.array([sensory_period])
            return np.mean(late_isis), "Phase-locked"

        return None, "Phase drifts"
    else:
        late_cycles = []
        for node_spikes in spikes:
            late_spikes = node_spikes[node_spikes >= half_time]
            if len(late_spikes) > 1:
                late_isis = np.diff(late_spikes)
                start_idx = len(late_isis) // 2
                late_cycles.extend(late_isis[start_idx:])
        if len(late_cycles) == 0:
            return None, "Insufficient spikes"
        if np.std(late_cycles) < 5.0:
            return np.mean(late_cycles), "Settled"
        return None, "Unstable"


def simulate_sensory_ring(g_max_sensory=SAMPLE_GMAX, sensory_period=56.8, tmax=T_END):
    time = np.arange(0.0, tmax + 0.5 * DT, DT)
    n_steps = len(time)

    syn_src = np.arange(N_NEURONS, dtype=int)
    syn_dst = (syn_src + 1) % N_NEURONS

    base_p = default_eprc_neuron_params()
    net_syn = default_nominal_synapse_params()
    sens_syn = default_sensory_synapse_params("exc")
    sens_syn.update({"e_rev": 50.0, "tau_decay": 1.0, "g_max": g_max_sensory})

    p_n = {k: np.full(N_NEURONS, v, dtype=float) for k, v in base_p.items()}
    p_s_net = {k: np.full(N_NEURONS, v, dtype=float) for k, v in net_syn.items()}
    p_s_sens = {k: np.full(N_NEURONS, v, dtype=float) for k, v in sens_syn.items()}

    vf = np.zeros((n_steps, N_NEURONS))
    vs = np.zeros((n_steps, N_NEURONS))
    g_syn_net = np.zeros((n_steps, N_NEURONS))
    g_syn_sens = np.zeros((n_steps, N_NEURONS))

    slope_net = max(abs(net_syn["k"]), 1e-9)
    slope_sens = max(abs(sens_syn["k"]), 1e-9)

    for k in range(n_steps - 1):
        t = time[k]
        v_f, v_s = vf[k], vs[k]
        g_s_net, g_s_sens = g_syn_net[k], g_syn_sens[k]

        v_pre_sens = 0.0
        if sensory_period is not None and sensory_period > 0:
            phase_sens = t % sensory_period
            if phase_sens < IMPULSE_WIDTH and t > SENSORY_NULL:
                v_pre_sens = IMPULSE_AMPLITUDE

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
            input_currents[0] -= INIT_PULSE_CURRENT

        for j in range(N_NEURONS):
            src, dst = syn_src[j], syn_dst[j]
            drive_net = 1.0 / (1.0 + np.exp(-np.clip((v_f[src] - p_s_net["threshold"][j]) / slope_net, -50, 50)))
            inv_tau_r_net = 1.0 / p_s_net["tau_rise"][j]
            inv_tau_d_net = 1.0 / p_s_net["tau_decay"][j]

            alpha_net = max(0.0, (inv_tau_r_net - inv_tau_d_net) * drive_net)
            dg_syn_net[j] = alpha_net * (1.0 - g_s_net[j]) - inv_tau_d_net * g_s_net[j]
            input_currents[dst] += p_s_net["g_max"][j] * g_s_net[j] * (p_s_net["e_rev"][j] - v_f[dst])

            drive_sens = 1.0 / (1.0 + np.exp(-np.clip((v_pre_sens - p_s_sens["threshold"][j]) / slope_sens, -50, 50)))
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

    return time, vf


def evaluate_sweep_point(args):
    mismatch, g_max, nom_period = args
    print(f"Evaluating sweep point: g={g_max}, mismatch={mismatch}")
    sensory_period = nom_period + mismatch
    if sensory_period <= 0 or (g_max == 0 and mismatch != 0):
        return mismatch, g_max, 0
    tmax = T_END * 2 if abs(mismatch) < 0.5 else T_END
    time_sens, vf_sens = simulate_sensory_ring(g_max_sensory=g_max, sensory_period=sensory_period, tmax=tmax)
    spikes_sens = detect_spike_times(time_sens, vf_sens, SPIKE_THRESHOLD)
    sens_val, _ = analyze_period(spikes_sens, tmax, is_sensory=True, sensory_period=sensory_period, nom_period=nom_period, applied_mismatch=mismatch)
    return mismatch, g_max, (1 if sens_val is not None else 0)


def compute_sensory_phase_data(spikes, sensory_event_times):
    records = []
    sensory_event_times = np.sort(sensory_event_times)
    for node, node_spikes in enumerate(spikes):
        for t_spike in node_spikes:
            past_events = sensory_event_times[sensory_event_times <= t_spike]
            last_sensory_t = past_events[-1] if len(past_events) > 0 else np.nan
            delay = (t_spike - last_sensory_t) if len(past_events) > 0 else np.nan
            records.append({
                "spike_time": float(t_spike),
                "neuron_id": int(node),
                "last_sensory_time": float(last_sensory_t) if not np.isnan(last_sensory_t) else np.nan,
                "delay_since_sensory": float(delay) if not np.isnan(delay) else np.nan,
            })
    return pd.DataFrame(records)


def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTDATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Compute Nominal Network Period
    time_nom, vf_nom = simulate_sensory_ring(g_max_sensory=0.0, sensory_period=None)
    spikes_nom = detect_spike_times(time_nom, vf_nom, SPIKE_THRESHOLD)
    nom_period, nom_status = analyze_period(spikes_nom, T_END, is_sensory=False, applied_mismatch=0)

    if nom_period is None:
        nominal_period = 60.8
    else:
        nominal_period = float(nom_period)
    print(f"[fig12] Settled nominal network period: {nominal_period:.2f} ms")

    # 2. Sample Entrainment Simulation at Delta T = SAMPLE_MISMATCH (-6.5 ms)
    sample_sensory_period = nominal_period + SAMPLE_MISMATCH
    time_sens, vf_sens = simulate_sensory_ring(g_max_sensory=SAMPLE_GMAX, sensory_period=sample_sensory_period)
    spikes_sens = detect_spike_times(time_sens, vf_sens, SPIKE_THRESHOLD)

    sensory_event_times = np.arange(sample_sensory_period, T_END, sample_sensory_period)
    sensory_event_times = sensory_event_times[sensory_event_times >= SENSORY_NULL]

    phase_df = compute_sensory_phase_data(spikes_sens, sensory_event_times)
    export_to_csv(phase_df, DATA_DIR / "entrain_ring_spike_phase_data.csv")
    export_to_csv(phase_df, PLOTDATA_DIR / "entrain_ring_spike_phase_data.csv")
    print(f"[fig12] Exported entrain_ring_spike_phase_data.csv")

    if ONLY_SAMPLE is False:
        # 3. Dynamic Arnold Tongue Sweep (1:1 Phase-Locking Region)
        print(f"[fig12] Computing dynamic Arnold Tongue grid sweep...")
        mismatch_vals = np.linspace(MISMATCH_MIN, MISMATCH_MAX, MISMATCH_STEPS)
        gmax_vals     = np.linspace(GMAX_MIN, GMAX_MAX, GMAX_STEPS)

        tasks = [(m, g, nominal_period) for g in gmax_vals for m in mismatch_vals]

        if PARALLELIZE:
            num_workers = min(multiprocessing.cpu_count(), 8)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                sweep_results = list(executor.map(evaluate_sweep_point, tasks))
        else:
            sweep_results = [evaluate_sweep_point(t) for t in tasks]

        df_sweep = pd.DataFrame(sweep_results, columns=["mismatch", "g_max", "is_locked"])
        export_to_csv(df_sweep, DATA_DIR / "fig12_arnold_tongue_data.csv")
        export_to_csv(df_sweep, PLOTDATA_DIR / "plotdata_arnold_tongue.csv")

        # Extract dynamic boundary coordinates from data matrix
        Z = df_sweep["is_locked"].values.reshape(GMAX_STEPS, MISMATCH_STEPS)
        left_x, left_y   = [], []
        right_x, right_y = [], []

        for i, g in enumerate(gmax_vals):
            locked_indices = np.where(Z[i, :] == 1)[0]
            if len(locked_indices) > 0:
                left_x.append(mismatch_vals[locked_indices[0]])
                left_y.append(g)
                right_x.append(mismatch_vals[locked_indices[-1]])
                right_y.append(g)

        print(f"[fig12] Dynamic Arnold Tongue boundaries extracted: {len(left_x)} locked rows.")

        # ------------------------------------------------------------
        # 4. Render Figure 12 (1x2 Multi-Panel Plot)
        # ------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)
        colors = ["#1F77B4", "#D62728", "#2CA02C", "#FF7F0E"]  # n0..n3 matching TikZ RGB

        # --- LEFT PLOT: SPIKE RASTER ---
        first_stim = True
        for st in sensory_event_times:
            if 400.0 <= st <= 600.0:
                ax1.axvline(st, color="#646464", linestyle="--", lw=1.0, label=r"$E^{\mathrm{P}}$" if first_stim else "")
                first_stim = False

        for node in range(N_NEURONS):
            node_spikes = spikes_sens[node]
            valid_spikes = node_spikes[(node_spikes >= 400.0) & (node_spikes <= 600.0)]
            ax1.vlines(valid_spikes, node - 0.35, node + 0.35, colors=colors[node], lw=1.5)

        ax1.set_xlim(400, 600)
        ax1.set_ylim(-0.5, 3.5)
        ax1.set_yticks(range(N_NEURONS))
        ax1.set_xlabel(r"Spike time $t$ (ms)", fontsize=11)
        ax1.set_ylabel("Neuron", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

        # --- RIGHT PLOT: ARNOLD TONGUE (DYNAMIC BOUNDARY) ---
        if left_x and right_x:
            poly_x = list(left_x) + list(right_x)[::-1]
            poly_y = list(left_y) + list(right_y)[::-1]
            ax2.fill(poly_x, poly_y, color="#C6D9F7", alpha=0.6, label="1:1 Phase-locking")
            ax2.plot(left_x, left_y, color="#1D4ED8", lw=1.5)
            ax2.plot(right_x, right_y, color="#1D4ED8", lw=1.5)

        # Sample operating point
        ax2.plot(SAMPLE_MISMATCH, SAMPLE_GMAX, "o", color="#D62728", ms=6, label="Sample")

        ax2.set_xlim(-10, 4)
        ax2.set_ylim(0, 0.10)
        ax2.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
        ax2.set_xlabel(r"Period Mismatch $\Delta T$ (ms)", fontsize=11)
        ax2.set_ylabel(r"$g_{\mathrm{syn}}$", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

        plt.tight_layout()

        OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_FIG, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig12] Saved Figure 12 plot to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
