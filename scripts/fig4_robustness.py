"""
fig4_robustness.py - Figure 4 Robustness Heterogeneity Sweep.

Runs the sweep over heterogeneity levels for both Oscillatory and Excitable regimes,
exports the results to plotdata CSVs, and plots Valid (%) vs Heterogeneity h (%).

Exports:
- plotdata_excitable_robustness.csv
- plotdata_oscillatory_robustness.csv
  Columns: heterogeneity, mean, ci_lower, ci_upper
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
OUTPUT_FIG = PROJECT_ROOT / "results" / "figures" / "fig4_robustness.png"

# ============================================================
# Simulation configuration
# ============================================================

DT = 0.02
T_END = 1000.0
TIME = np.arange(0.0, T_END + 0.5 * DT, DT)
N_NEURONS = 4

HETERO_PARAMS = ("g_na", "g_k", "syn_tau_decay")
HETERO_LEVELS_PCT = np.linspace(0.0, 100.0, 11)  # Heterogeneity sweep levels: 0% to 100% (11 points)
HETERO_N_TRIALS = 400                             # Number of Monte Carlo trials per heterogeneity level
HETERO_RNG_SEED = 42
HETERO_TYPE = "relative"

SPIKE_THRESHOLD = 40.0

PULSE_START = 0.0
PULSE_DURATION = 5.0
PULSE_AMPLITUDE = -5.0
I_TONIC_OSC = 4.0

SYN_TAU_DECAY = 2.15
SYN_TAU_RISE = 0.1
SYN_G_MAX = 3.0
SYN_E_REV_INHIB = -20.0
SYN_E_REV_EXC = 100.0
SYN_THRESHOLD = 30.0
SYN_K = 0.1

OSC_TOPOLOGY_MODE = "a2a"
A2A_G_MAX = 0.1
A2A_RING_G_MAX = 0.05
GA_OSC = 5.076
GNAREB_OSC = 0.0


def satrelu_from_threshold(v, threshold, slope):
    return np.clip(slope * (v - threshold), 0.0, 1.0)


def satrelu_from_threshold_decreasing(v, threshold, slope):
    return np.clip(slope * (threshold - v), 0.0, 1.0)


def build_topology_for_case(case_name, n_neurons):
    ring_src = np.arange(n_neurons, dtype=int)
    ring_dst = (ring_src + 1) % n_neurons

    if case_name == "excitable":
        return (
            ring_src,
            ring_dst,
            np.full(len(ring_src), SYN_G_MAX),
            np.full(len(ring_src), SYN_E_REV_INHIB),
            np.full(len(ring_src), SYN_TAU_DECAY),
        )

    # Oscillatory A2A topology
    a2a_src, a2a_dst = [], []
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i != j:
                a2a_src.append(i)
                a2a_dst.append(j)

    src = np.concatenate((a2a_src, ring_src))
    dst = np.concatenate((a2a_dst, ring_dst))
    g = np.concatenate(
        (
            np.full(len(a2a_src), A2A_G_MAX),
            np.full(len(ring_src), A2A_RING_G_MAX),
        )
    )
    e_rev = np.concatenate(
        (
            np.full(len(a2a_src), SYN_E_REV_INHIB),
            np.full(len(ring_src), SYN_E_REV_EXC),
        )
    )
    tau_decay = np.concatenate(
        (
            np.full(len(a2a_src), SYN_TAU_DECAY),
            np.full(len(ring_src), SYN_TAU_DECAY),
        )
    )
    return src, dst, g, e_rev, tau_decay


def build_base_params(n_neurons, case_name):
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

    p["g_na_reb"] = np.full(
        n_neurons, 2.0 if case_name == "excitable" else GNAREB_OSC
    )
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

    p["g_a"] = np.full(n_neurons, 0.0 if case_name == "excitable" else GA_OSC)
    p["e_a"] = np.full(n_neurons, -20.0)
    p["a_act_bias"] = np.full(n_neurons, 2.0)
    p["a_act_slope"] = np.full(n_neurons, 0.01)
    p["a_inact_bias"] = np.full(n_neurons, 10.0)
    p["a_inact_slope"] = np.full(n_neurons, 0.15)
    return p


def compute_gates(vf, vs, p):
    return {
        "m_na": satrelu_from_threshold(vf, p["na_act_bias"], p["na_act_slope"]),
        "h_na": satrelu_from_threshold_decreasing(
            vs, p["na_inact_bias"], p["na_inact_slope"]
        ),
        "m_na_reb": satrelu_from_threshold(
            vf, p["na_reb_act_bias"], p["na_reb_act_slope"]
        ),
        "h_na_reb": satrelu_from_threshold_decreasing(
            vs, p["na_reb_inact_bias"], p["na_reb_inact_slope"]
        ),
        "n_k": satrelu_from_threshold(vs, p["k_act_bias"], p["k_act_slope"]),
        "m_a": satrelu_from_threshold(vf, p["a_act_bias"], p["a_act_slope"]),
        "h_a": satrelu_from_threshold_decreasing(
            vs, p["a_inact_bias"], p["a_inact_slope"]
        ),
    }


def _run_single_trial(args):
    case_name, params, syn_src, syn_dst, syn_g_max, syn_e_rev, syn_tau_decay, t_start = args
    n_steps = len(TIME)
    vf, vs = np.zeros(N_NEURONS), np.zeros(N_NEURONS)
    g_syn = np.zeros(len(syn_src))
    vf_hist = np.zeros((n_steps, N_NEURONS))

    if case_name == "excitable":
        vf[0] = 0.0
        vs[0] = 0.0
    else:
        vf[0] = -5.0 
        vs[0] = -10.0
        
    ext_input = (
        np.full(N_NEURONS, I_TONIC_OSC)
        if case_name == "oscillatory"
        else np.zeros(N_NEURONS)
    )
    slope = max(abs(SYN_K), 1e-9)
    inv_tau_r = 1.0 / SYN_TAU_RISE

    for k in range(n_steps - 1):
        vf_hist[k] = vf
        t = TIME[k]

        if case_name == "excitable":
            ext_input[0] = (
                PULSE_AMPLITUDE
                if PULSE_START <= t < PULSE_START + PULSE_DURATION
                else 0.0
            )

        gates = compute_gates(vf, vs, params)

        i_ionic = (
            -params["g_na"] * gates["m_na"] * gates["h_na"] * (vf - params["e_na"])
            - params["g_a"] * gates["m_a"] * gates["h_a"] * (vf - params["e_a"])
            - params["g_na_reb"] * gates["m_na_reb"] * gates["h_na_reb"] * (vf - params["e_na_reb"])
            - params["g_k"] * gates["n_k"] * (vf - params["e_k"])
            - params["g_l"] * (vf - params["e_l"])
        )

        input_currents = np.copy(ext_input)
        dg_syn = np.zeros(len(syn_src))

        for j in range(len(syn_src)):
            src, dst = syn_src[j], syn_dst[j]
            drive = 1.0 / (1.0 + np.exp(-(vf[src] - SYN_THRESHOLD) / slope))
            inv_tau_d = 1.0 / syn_tau_decay[j]
            alpha = max(0.0, (inv_tau_r - inv_tau_d) * drive)
            dg_syn[j] = alpha * (1.0 - g_syn[j]) - inv_tau_d * g_syn[j]
            input_currents[dst] += syn_g_max[j] * g_syn[j] * (syn_e_rev[j] - vf[dst])

        i_app = -params["input_gain"] * input_currents - params["i_base"]
        dvf = (i_ionic - i_app) / params["cm"]
        dvs = (vf - vs) / (params["tau_s"] * params["cm"])

        vf += DT * dvf
        vs += DT * dvs
        g_syn += DT * dg_syn

    vf_hist[-1] = vf

    # Spike detection
    spikes = []
    for i in range(N_NEURONS):
        above = vf_hist[:, i] >= SPIKE_THRESHOLD
        crossings = np.where((~above[:-1]) & above[1:])[0] + 1
        spikes.append(TIME[crossings])

    # Sequence validation
    events = []
    for neuron_idx, sp in enumerate(spikes):
        sp_arr = np.asarray(sp, dtype=float)
        mask = sp_arr >= t_start
        for t in sp_arr[mask]:
            events.append((t, neuron_idx))

    if not events:
        return 0

    events.sort(key=lambda x: x[0])
    expected_neuron = events[0][1]
    for t, neuron in events:
        if neuron != expected_neuron:
            return 0
        expected_neuron = (expected_neuron + 1) % N_NEURONS

    return 1


def apply_heterogeneity(
    base_params, base_syn_g_max, base_syn_e_rev, base_syn_tau_decay, het_val, rng
):
    """Applies noise to both intrinsic dictionary parameters and synaptic arrays."""
    p = {k: np.copy(v) for k, v in base_params.items()}
    syn_g_max = np.copy(base_syn_g_max)
    syn_tau_decay = np.copy(base_syn_tau_decay)

    factor = (het_val / 100.0) if HETERO_TYPE == "relative" else het_val

    for param in HETERO_PARAMS:
        if param == "syn_g_max":
            scale = factor * np.abs(syn_g_max) if HETERO_TYPE == "relative" else factor
            noise = rng.normal(loc=0.0, scale=scale, size=len(syn_g_max))
            syn_g_max = np.maximum(syn_g_max + noise, 1e-9)

        elif param == "syn_tau_decay":
            scale = (
                factor * np.abs(base_syn_tau_decay)
                if HETERO_TYPE == "relative"
                else factor
            )
            noise = rng.normal(loc=0.0, scale=scale, size=len(base_syn_tau_decay))
            base_syn_tau_decay = base_syn_tau_decay + noise

        elif param in p:
            base_vals = p[param]
            scale = factor * np.abs(base_vals) if HETERO_TYPE == "relative" else factor
            noise = rng.normal(loc=0.0, scale=scale, size=N_NEURONS)
            p[param] = base_vals + noise

            if param.startswith("g_") or param in ("cm", "tau_s"):
                p[param] = np.maximum(p[param], 1e-9)
        else:
            print(f"Warning: Heterogeneity parameter '{param}' not recognized.")

    return p, syn_g_max, base_syn_e_rev, syn_tau_decay


def run():
    start_t = time.perf_counter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(HETERO_RNG_SEED)

    # 1. Pre-generate trial parameters
    all_trials = []
    # Simulation loop order preserved to ensure identical pseudo-random number generator sequencing
    for case in ["oscillatory", "excitable"]:
        base_params = build_base_params(N_NEURONS, case)
        syn_src, syn_dst, syn_g_base, syn_e_base, syn_tau_base = build_topology_for_case(
            case, N_NEURONS
        )
        t_start = T_END / 2.0 if case == "oscillatory" else 0.0

        for het_pct in HETERO_LEVELS_PCT:
            for trial in range(HETERO_N_TRIALS):
                p_trial, sg_trial, se_trial, st_trial = apply_heterogeneity(
                    base_params, syn_g_base, syn_e_base, syn_tau_base, het_pct, rng
                )
                all_trials.append(
                    (case, het_pct, p_trial, syn_src, syn_dst, sg_trial, se_trial, st_trial, t_start)
                )

    print(f"Generated {len(all_trials)} trials. Running in parallel across CPU cores...", flush=True)

    # 2. Run ODE simulations in parallel across worker processes
    worker_args = [
        (t[0], t[2], t[3], t[4], t[5], t[6], t[7], t[8]) for t in all_trials
    ]
    with ProcessPoolExecutor() as executor:
        valid_results = list(executor.map(_run_single_trial, worker_args, chunksize=10))

    # 3. Aggregate results and export CSVs
    all_dfs = {}
    trial_idx = 0

    for case in ["oscillatory", "excitable"]:
        results = []
        for het_pct in HETERO_LEVELS_PCT:
            valid_count = sum(valid_results[trial_idx : trial_idx + HETERO_N_TRIALS])
            trial_idx += HETERO_N_TRIALS

            p_val = valid_count / HETERO_N_TRIALS
            se = (
                np.sqrt(p_val * (1 - p_val) / HETERO_N_TRIALS)
                if HETERO_N_TRIALS > 0
                else 0.0
            )
            mean_pct = p_val * 100.0
            ci_hw = 1.96 * se * 100.0  # 95% CI

            lower_ci = max(0.0, mean_pct - ci_hw)
            upper_ci = min(100.0, mean_pct + ci_hw)

            results.append((het_pct, mean_pct, lower_ci, upper_ci))

        df = pd.DataFrame(
            results, columns=["heterogeneity", "mean", "ci_lower", "ci_upper"]
        )
        all_dfs[case] = df
        export_to_csv(df, DATA_DIR / f"plotdata_{case}_robustness.csv")
        print(f"[fig4] Wrote plotdata_{case}_robustness.csv to {DATA_DIR}")

    print(
        f"\nParallel sweep complete in {time.perf_counter() - start_t:.2f}s. Plotting Figure 4...",
        flush=True,
    )

    # 4. Plot Figure 4
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=300)

    blue_color = "#0000FF"  # Oscillatory (Blue)
    red_color = "#FF0000"   # Excitable (Red)

    df_osc = all_dfs["oscillatory"]
    df_exc = all_dfs["excitable"]

    # Plot Oscillatory (blue circle markers)
    ax.plot(
        df_osc["heterogeneity"],
        df_osc["mean"],
        marker="o",
        color=blue_color,
        linewidth=1.8,
        markersize=5,
        label="Oscillatory",
    )
    ax.fill_between(
        df_osc["heterogeneity"],
        df_osc["ci_lower"],
        df_osc["ci_upper"],
        color=blue_color,
        alpha=0.2,
    )

    # Plot Excitable (red square markers)
    ax.plot(
        df_exc["heterogeneity"],
        df_exc["mean"],
        marker="s",
        color=red_color,
        linewidth=1.8,
        markersize=5,
        label="Excitable",
    )
    ax.fill_between(
        df_exc["heterogeneity"],
        df_exc["ci_lower"],
        df_exc["ci_upper"],
        color=red_color,
        alpha=0.2,
    )

    ax.set_xlabel("Heterogeneity $h$ (%)", fontsize=11)
    ax.set_ylabel("Valid (%)", fontsize=11)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 105)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 50))

    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.close(fig)
    print(f"[fig4] Saved Figure 4 to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
