"""
fig2_ring_voltage.py - Figure 2 Excitable Ring Network Voltage.

Simulates a 4-neuron excitable ring network and exports voltage traces to:
- results/data/plotdata_excitable_voltage.csv

Validation Plot:
- All neuron traces in grey, with Node 0 highlighted in orange.
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
OUTPUT_EXC_CSV = PROJECT_ROOT / "results" / "data" / "plotdata_excitable_voltage.csv"
OUTPUT_FIG    = PROJECT_ROOT / "results" / "figures" / "fig2_ring_voltage.png"

# Simulation parameters
DT = 0.01
T_END = 700.0
N_NEURONS = 4

PULSE_START     = 0.0
PULSE_DURATION  = 20.0
PULSE_AMPLITUDE = -5.0

SYN_TAU_DECAY   = 2.15
SYN_TAU_RISE    = 0.1
SYN_G_MAX       = 3.0
SYN_E_REV_INHIB = -20.0
SYN_THRESHOLD   = 30.0
SYN_K           = 0.1

BASE_NEURON = {
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


def satrelu_inc(v, threshold, slope):
    return np.clip(slope * (v - threshold), 0.0, 1.0)


def satrelu_dec(v, threshold, slope):
    return np.clip(slope * (threshold - v), 0.0, 1.0)


def run_excitable_ring():
    time = np.arange(0.0, T_END + 0.5 * DT, DT)
    n_steps = len(time)

    syn_src = np.arange(N_NEURONS, dtype=int)
    syn_dst = (syn_src + 1) % N_NEURONS
    n_syn = len(syn_src)

    p = {k: np.full(N_NEURONS, v, dtype=float) for k, v in BASE_NEURON.items()}
    syn_g_max = np.full(n_syn, SYN_G_MAX)
    syn_e_rev = np.full(n_syn, SYN_E_REV_INHIB)
    syn_tau_d = np.full(n_syn, SYN_TAU_DECAY)

    slope     = max(abs(SYN_K), 1e-9)
    inv_tau_r = 1.0 / max(SYN_TAU_RISE, 1e-9)

    vf    = np.zeros((n_steps, N_NEURONS))
    vs    = np.zeros((n_steps, N_NEURONS))
    g_syn = np.zeros((n_steps, n_syn))

    # Initial conditions
    vf[0, 0] = 0.0
    vs[0, 0] = 0.0

    for k in range(n_steps - 1):
        t = time[k]
        v_f = vf[k]
        v_s = vs[k]
        g_s = g_syn[k]

        ext_input = np.zeros(N_NEURONS)
        if PULSE_START <= t < PULSE_START + PULSE_DURATION:
            ext_input[0] = PULSE_AMPLITUDE

        m_na     = satrelu_inc(v_f, p["na_act_bias"],       p["na_act_slope"])
        h_na     = satrelu_dec(v_s, p["na_inact_bias"],     p["na_inact_slope"])
        m_na_reb = satrelu_inc(v_f, p["na_reb_act_bias"],   p["na_reb_act_slope"])
        h_na_reb = satrelu_dec(v_s, p["na_reb_inact_bias"], p["na_reb_inact_slope"])
        n_k      = satrelu_inc(v_s, p["k_act_bias"],        p["k_act_slope"])
        m_a      = satrelu_inc(v_f, p["a_act_bias"],        p["a_act_slope"])
        h_a      = satrelu_dec(v_s, p["a_inact_bias"],      p["a_inact_slope"])

        i_ionic = (
            -p["g_na"]      * m_na     * h_na     * (v_f - p["e_na"])
            - p["g_a"]      * m_a      * h_a      * (v_f - p["e_a"])
            - p["g_na_reb"] * m_na_reb * h_na_reb * (v_f - p["e_na_reb"])
            - p["g_k"]      * n_k                  * (v_f - p["e_k"])
            - p["g_l"]                              * (v_f - p["e_l"])
        )

        input_currents = ext_input.copy()
        dg_syn = np.zeros(n_syn)

        for j in range(n_syn):
            src, dst  = syn_src[j], syn_dst[j]
            exp_arg   = np.clip(-(v_f[src] - SYN_THRESHOLD) / slope, -50.0, 50.0)
            drive     = 1.0 / (1.0 + np.exp(exp_arg))
            inv_tau_d = 1.0 / max(syn_tau_d[j], 1e-9)
            alpha     = max(0.0, (inv_tau_r - inv_tau_d) * drive)
            dg_syn[j] = alpha * (1.0 - g_s[j]) - inv_tau_d * g_s[j]
            input_currents[dst] += syn_g_max[j] * g_s[j] * (syn_e_rev[j] - v_f[dst])

        i_app = -p["input_gain"] * input_currents - p["i_base"]
        dvf   = (i_ionic - i_app) / p["cm"]
        dvs   = (v_f - v_s) / (p["tau_s"] * p["cm"])

        vf[k + 1]    = v_f + DT * dvf
        vs[k + 1]    = v_s + DT * dvs
        g_syn[k + 1] = g_s + DT * dg_syn

    return time, vf


def run():
    t_exc, vf_exc = run_excitable_ring()

    # Downsample for CSV export to keep file size reasonable for PGFPlots
    skip_csv = max(1, len(t_exc) // 10000)
    idx_csv = np.arange(0, len(t_exc), skip_csv)

    df_exc = pd.DataFrame({"time": t_exc[idx_csv]})
    for i in range(N_NEURONS):
        df_exc[f"n{i}"] = vf_exc[idx_csv, i]

    export_to_csv(df_exc, OUTPUT_EXC_CSV)
    print(f"[fig2] Wrote excitable ring voltage CSV to: {OUTPUT_EXC_CSV}")

    # Validation plot: all traces grey, except n0 in orange
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    skip_plot = max(1, len(t_exc) // 20000)

    # Plot n1..n3 in grey first
    for i in range(1, N_NEURONS):
        ax.plot(
            t_exc[::skip_plot],
            vf_exc[::skip_plot, i],
            color="grey",
            alpha=0.6,
            linewidth=1.0,
            label=f"Other Neurons (n1-n{N_NEURONS - 1})" if i == 1 else "",
        )

    # Plot n0 in orange on top
    ax.plot(
        t_exc[::skip_plot],
        vf_exc[::skip_plot, 0],
        color="tab:orange",
        linewidth=1.5,
        label="Node 0 (n0)",
    )

    ax.set_title("Figure 2: Excitable Ring Network Voltage Traces", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Voltage (mV)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(500, 650)

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG)
    plt.close(fig)
    print(f"[fig2] Wrote plot figure to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
