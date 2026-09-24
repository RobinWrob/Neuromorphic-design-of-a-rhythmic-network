"""
fig1_exc_vs_osc.py - Figure 1 Simulation & Data Export.

Generates simulation data comparing single excitable vs oscillatory neuron dynamics.
Uses Hodgkin-Huxley style alpha/beta gating kinetics (shifted +65 mV convention).

Exports data to results/data/neuron_simulation_data.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.helper import export_to_csv

# Paths
OUTPUT_CSV = PROJECT_ROOT / "results" / "data" / "neuron_simulation_data.csv"
OUTPUT_FIG = PROJECT_ROOT / "results" / "figures" / "fig1_exc_vs_osc.png"

# ============================================================
# Simulation parameters
# ============================================================
DT = 0.02   # ms
T_END = 60.0  # ms

# Reversal potentials & conductances (Shifted by +65 mV)
E_Na, E_K, E_L = 115.0, -12.0, 10.6
G_Na, G_K, G_A, G_L = 120.0, 36.0, 10.0, 0.3
C_m = 1.0

# Synaptic parameters (Shifted by +65 mV)
E_SYN = 65.0        # Excitatory synapse
G_SYN_MAX = 0.5
TAU_SYN = 2.0

# Pulse times for excitable scenario
T_I_PULSES = [5.0, 40.0]


def get_rates(v):
    """Gate rate functions (alpha/beta) with +65 mV voltage shift."""
    v_orig = v - 65.0

    # Na activation (m)
    a_m = (
        0.1 * (v_orig + 40) / (1 - np.exp(-(v_orig + 40) / 10))
        if v_orig != -40
        else 1.0
    )
    b_m = 4.0 * np.exp(-(v_orig + 65) / 18)
    # Na inactivation (h)
    a_h = 0.07 * np.exp(-(v_orig + 65) / 20)
    b_h = 1.0 / (1 + np.exp(-(v_orig + 35) / 10))
    # K activation (n)
    a_n = (
        0.01 * (v_orig + 55) / (1 - np.exp(-(v_orig + 55) / 10))
        if v_orig != -55
        else 0.1
    )
    b_n = 0.125 * np.exp(-(v_orig + 65) / 80)
    # A-current activation (a) & inactivation (b)
    a_inf = 1.0 / (1.0 + np.exp(-(v_orig + 30) / 15))
    tau_a = 2.0
    b_inf = 1.0 / (1.0 + np.exp((v_orig + 60) / 10))
    tau_b = 20.0

    return a_m, b_m, a_h, b_h, a_n, b_n, a_inf, tau_a, b_inf, tau_b


def run_simulation(I_app, pulse_times):
    """
    Forward Euler numerical integration of a Hodgkin-Huxley style neuron with A-type potassium
    conductance and dynamic synaptic input.
    """
    time = np.arange(0, T_END, DT)

    # Initial conditions (identical to source)
    v = 0.0
    m, h, n, a, b = 0.05, 0.6, 0.32, 0.1, 0.6
    g_syn = 0.0

    v_hist = []

    for t in time:
        # Check for synaptic pulse events
        syn_event = False
        if pulse_times and any(
            np.isclose(t, t_i, atol=DT / 2) for t_i in pulse_times
        ):
            syn_event = True

        if syn_event:
            g_syn += G_SYN_MAX

        # Synaptic decay
        g_syn += DT * (-g_syn / TAU_SYN)

        # Gate kinetics
        am, bm, ah, bh, an, bn, a_inf, tau_a, b_inf, tau_b = get_rates(v)
        m += DT * (am * (1 - m) - bm * m)
        h += DT * (ah * (1 - h) - bh * h)
        n += DT * (an * (1 - n) - bn * n)
        a += DT * ((a_inf - a) / tau_a)
        b += DT * ((b_inf - b) / tau_b)

        # Currents (Na, K, A, Leak, Synaptic)
        i_na = G_Na * (m ** 3) * h * (v - E_Na)
        i_k  = G_K  * (n ** 4) * (v - E_K)
        i_a  = G_A  * (a ** 3) * b * (v - E_K)
        i_l  = G_L  * (v - E_L)
        i_syn = g_syn * (v - E_SYN)

        # Voltage update
        dv = (I_app - i_na - i_k - i_a - i_l - i_syn) / C_m
        v += DT * dv
        v_hist.append(v)

    return np.array(v_hist)


def run():
    time = np.arange(0, T_END, DT)

    # Scenario A: Oscillatory Neuron — constant tonic current, no pulses
    v_osc = run_simulation(I_app=15.0, pulse_times=[])

    # Scenario B: Excitable Neuron — no tonic current, with synaptic pulses
    v_exc = run_simulation(I_app=0.0, pulse_times=T_I_PULSES)

    # Build pulse indicator column
    pulse_indicator = np.array([
        1 if any(np.isclose(t, t_i, atol=DT / 2) for t_i in T_I_PULSES) else 0
        for t in time
    ])

    # Export CSV
    df = pd.DataFrame({
        "time_ms":           time,
        "v_oscillatory_mV":  v_osc,
        "v_excitable_mV":    v_exc,
        "input_pulse":       pulse_indicator,
    })
    export_to_csv(df, OUTPUT_CSV)

    pulses_df = pd.DataFrame({
        "pulse_index": np.arange(1, len(T_I_PULSES) + 1),
        "t_i_pulse_ms": T_I_PULSES,
    })
    export_to_csv(pulses_df, PROJECT_ROOT / "results" / "data" / "input_pulse_times.csv")
    print(f"[fig1] Wrote CSV data to: {OUTPUT_CSV} and input_pulse_times.csv")

    # Validation plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, dpi=150)

    ax1.plot(time, v_osc, "b", label="V_m", linewidth=1.2)
    for t_i in T_I_PULSES:
        ax1.axvline(x=t_i, color="k", linestyle="--", alpha=0.5)
    ax1.set_title("A) Oscillatory Neuron (I_app=15 µA/cm²)", fontsize=10)
    ax1.set_ylabel("Voltage (mV)")
    ax1.grid(True, alpha=0.4)

    ax2.plot(time, v_exc, "r", label="V_m", linewidth=1.2)
    for t_i in T_I_PULSES:
        ax2.axvline(x=t_i, color="k", linestyle="--", alpha=0.5, label="Pulse")
    ax2.set_title("B) Excitable Neuron (I_app=0, synaptic pulses)", fontsize=10)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Voltage (mV)")
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG)
    plt.close(fig)
    print(f"[fig1] Wrote plot figure to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
