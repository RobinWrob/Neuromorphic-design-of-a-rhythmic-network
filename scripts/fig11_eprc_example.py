"""
fig11_eprc_example.py - Figure 11 Event Phase Response Curve (ePRC) & Voltage Traces.

Renders a 2x2 multi-panel layout matching the reference figure:
  - Top Left: ePRC curve for Excitatory perturbation (g_syn = 0.05)
  - Top Right: ePRC curve for Inhibitory perturbation (g_syn = 0.15)
  - Bottom Left: Voltage traces V_nom, V_p and pulse lines E_nom, E_p for sample delay tp=10 ms (Excitatory)
  - Bottom Right: Voltage traces V_nom, V_p and pulse lines E_nom, E_p for sample delay tp=10 ms (Inhibitory)

Exports CSVs:
  - plotdata_eprc_exc_curve.csv
  - plotdata_eprc_inh_curve.csv
  - plotdata_eprc_exc_trace_tp5.csv
  - plotdata_eprc_inh_trace_tp5.csv
  - fig11_eprc_data.csv
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

from src.eprc_engine import compute_eprc_curve, run_eprc_simulation_trace
from src.helper import export_to_csv

# Paths
DATA_DIR   = PROJECT_ROOT / "results" / "data"
OUTPUT_FIG = PROJECT_ROOT / "results" / "figures" / "fig11_eprc_example.png"

# Parameters
DT                = 0.001
SPIKE_THRESHOLD   = 50.0
IMPULSE_WIDTH     = 3.0
IMPULSE_AMPLITUDE = 100.0

INPUT_PERIOD      = 240.0
T_NOM_START       = 240.0
TP_MIN            = -10.0
TP_MAX            = 20.0
TP_STEP           = 0.1

SINGLE_EVENT      = True
DURATION_MULTIPLIER = 10.0
MIN_T               = 350.0

G_MAX_EXC = 0.05
G_MAX_INH = 0.15


def run():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tp_values = np.arange(TP_MIN, TP_MAX + 0.5 * TP_STEP, TP_STEP)

    # ============================================================
    # 1. Compute ePRC Curves
    # ============================================================
    # Excitatory curve
    samples_exc = compute_eprc_curve(
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
        parallelize=True,
        param_overrides={"sens_g_max": G_MAX_EXC},
        syn_type="exc",
    )

    # Inhibitory curve
    samples_inh = compute_eprc_curve(
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
        parallelize=True,
        param_overrides={"sens_g_max": G_MAX_INH},
        syn_type="inh",
    )

    rows_exc = []
    for s in samples_exc:
        if s.is_valid and s.delta_t is not None:
            rows_exc.append({"tp": s.tp, "delta_t": s.delta_t, "relative_phase": s.tp / INPUT_PERIOD})

    rows_inh = []
    for s in samples_inh:
        if s.is_valid and s.delta_t is not None:
            rows_inh.append({"tp": s.tp, "delta_t": s.delta_t, "relative_phase": s.tp / INPUT_PERIOD})

    df_exc = pd.DataFrame(rows_exc)
    df_inh = pd.DataFrame(rows_inh)

    export_to_csv(df_exc, DATA_DIR / "plotdata_eprc_exc_curve.csv")
    export_to_csv(df_inh, DATA_DIR / "plotdata_eprc_inh_curve.csv")

    # ============================================================
    # 2. Compute Sample Voltage Traces for tp = 10.0 ms
    # ============================================================
    sample_tp = 10.0

    t_exc, vn_exc, vp_exc, gn_exc, gs_exc, pn_exc, ps_exc, _, _, _ = run_eprc_simulation_trace(
        tp=sample_tp,
        input_period=INPUT_PERIOD,
        dt=DT,
        impulse_width=IMPULSE_WIDTH,
        impulse_amplitude=IMPULSE_AMPLITUDE,
        spike_threshold=SPIKE_THRESHOLD,
        single_event=SINGLE_EVENT,
        duration_multiplier=DURATION_MULTIPLIER,
        min_t=MIN_T,
        t_nom_start=T_NOM_START,
        param_overrides={ "sens_g_max": G_MAX_EXC},
        syn_type="exc",
    )

    t_inh, vn_inh, vp_inh, gn_inh, gs_inh, pn_inh, ps_inh, _, _, _ = run_eprc_simulation_trace(
        tp=sample_tp,
        input_period=INPUT_PERIOD,
        dt=DT,
        impulse_width=IMPULSE_WIDTH,
        impulse_amplitude=IMPULSE_AMPLITUDE,
        spike_threshold=SPIKE_THRESHOLD,
        single_event=SINGLE_EVENT,
        duration_multiplier=DURATION_MULTIPLIER,
        min_t=MIN_T,
        t_nom_start=T_NOM_START,
        param_overrides={"sens_g_max": G_MAX_INH},
        syn_type="inh",
    )
    
    # ============================================================
    # Export Discrete Event Timings (single t_k per pulse)
    # ============================================================
    Y_MIN, Y_MAX = -25.0, 110.0
    t_nom = T_NOM_START
    t_pert = T_NOM_START + sample_tp

    # Two rows define the line segment from bottom to top of the plot
    df_events = pd.DataFrame({
        "y": [Y_MIN, Y_MAX],
        "t_nom": [t_nom, t_nom],
        "t_pert": [t_pert, t_pert]
    })
    export_to_csv(df_events, DATA_DIR / "plotdata_eprc_events.csv")
    
    # ============================================================
    # Prepare Data for LaTeX Export (Cropped & Downsampled)
    # ============================================================
    
    # 1. Define the time window you actually plot in LaTeX (with a small margin)
    PLOT_TMIN = 225.0
    PLOT_TMAX = 285.0

    # 2. Find indices for the data inside this window
    valid_idx_exc = np.where((t_exc >= PLOT_TMIN) & (t_exc <= PLOT_TMAX))[0]
    valid_idx_inh = np.where((t_inh >= PLOT_TMIN) & (t_inh <= PLOT_TMAX))[0]

    # 3. Downsample only the relevant window to ~800 points (very safe for LaTeX)
    TARGET_PTS = 800
    skip_exc = max(1, len(valid_idx_exc) // TARGET_PTS)
    skip_inh = max(1, len(valid_idx_inh) // TARGET_PTS)

    idx_exc = valid_idx_exc[::skip_exc]
    idx_inh = valid_idx_inh[::skip_inh]

    # 4. Export Excitatory trace
    df_tr_exc = pd.DataFrame({
        "time": t_exc[idx_exc], 
        "v_nom": vn_exc[idx_exc], 
        "v_pert": vp_exc[idx_exc],
        "g_syn_nom": gn_exc[idx_exc], 
        "g_syn_sens": gs_exc[idx_exc],
        "pulse_nom": pn_exc[idx_exc],
        "pulse_sens": ps_exc[idx_exc]
    })
    export_to_csv(df_tr_exc, DATA_DIR / "plotdata_eprc_exc_trace_tp5.csv")

    # 5. Export Inhibitory trace
    df_tr_inh = pd.DataFrame({
        "time": t_inh[idx_inh], 
        "v_nom": vn_inh[idx_inh], 
        "v_pert": vp_inh[idx_inh],
        "g_syn_nom": gn_inh[idx_inh], 
        "g_syn_sens": gs_inh[idx_inh],
        "pulse_nom": pn_inh[idx_inh],
        "pulse_sens": ps_inh[idx_inh]
    })
    export_to_csv(df_tr_inh, DATA_DIR / "plotdata_eprc_inh_trace_tp5.csv")

    df_exc["type"] = "exc"
    df_inh["type"] = "inh"
    df_comb = pd.concat([df_exc, df_inh], ignore_index=True)
    export_to_csv(df_comb, DATA_DIR / "fig11_eprc_data.csv")

    print("[fig11] CSV exports complete.")

    # ============================================================
    # 3. Render 2x2 Multi-Panel Plot
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=150, gridspec_kw={'height_ratios': [1, 1]})
    (ax_c_exc, ax_c_inh), (ax_t_exc, ax_t_inh) = axes

    # --- TOP ROW: ePRC CURVES ---
    ax_c_exc.plot(df_exc["tp"], df_exc["delta_t"], color="#2B2B2B", lw=1.5)
    ax_c_exc.set_title("Excitatory Perturbation", fontsize=13, fontweight='bold', pad=10)
    ax_c_exc.set_ylabel("Delay (ms)", fontsize=11)
    ax_c_exc.set_xlabel(r"Phase offset $t_p$ (ms)", fontsize=11)
    ax_c_exc.set_xlim(-12, 22)
    ax_c_exc.set_ylim(-7.0, 0.5)
    ax_c_exc.grid(True, linestyle="--", alpha=0.3)

    ax_c_inh.plot(df_inh["tp"], df_inh["delta_t"], color="#2B2B2B", lw=1.5)
    ax_c_inh.set_title("Inhibitory Perturbation", fontsize=13, fontweight='bold', pad=10)
    ax_c_inh.set_xlabel(r"Phase offset $t_p$ (ms)", fontsize=11)
    ax_c_inh.set_xlim(-12, 22)
    ax_c_inh.set_ylim(-0.5, 6.8)
    ax_c_inh.grid(True, linestyle="--", alpha=0.3)

    # --- BOTTOM ROW: VOLTAGE TRACES (tp = 10 ms) ---
    TIME_MIN, TIME_MAX = 225.0, 285.0
    mask_exc = (t_exc >= TIME_MIN) & (t_exc <= TIME_MAX)
    mask_inh = (t_inh >= TIME_MIN) & (t_inh <= TIME_MAX)

    # Excitatory trace
    ax_t_exc.plot(t_exc[mask_exc], vn_exc[mask_exc], color="black", lw=1.2, label=r"$V_{\mathrm{nom}}$")
    ax_t_exc.plot(t_exc[mask_exc], vp_exc[mask_exc], color="#D62728", lw=1.2, label=r"$V_p$")
    ax_t_exc.axvline(T_NOM_START, color="black", linestyle="--", lw=1.5, label=r"$E_{\mathrm{nom}}^{\mathrm{in}}$")
    ax_t_exc.axvline(T_NOM_START + sample_tp, color="#D62728", linestyle=":", lw=1.5, label=r"$E_p^{\mathrm{in}}$")

    ax_t_exc.set_xlabel(r"Time $t$ (ms)", fontsize=11)
    ax_t_exc.set_ylabel("Voltage (mV)", fontsize=11)
    ax_t_exc.set_xlim(TIME_MIN, TIME_MAX)
    ax_t_exc.set_ylim(-25, 110)
    ax_t_exc.grid(True, linestyle="--", alpha=0.3)
    ax_t_exc.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Inhibitory trace
    ax_t_inh.plot(t_inh[mask_inh], vn_inh[mask_inh], color="black", lw=1.2, label=r"$V_{\mathrm{nom}}$")
    ax_t_inh.plot(t_inh[mask_inh], vp_inh[mask_inh], color="#D62728", lw=1.2, label=r"$V_p$")
    ax_t_inh.axvline(T_NOM_START, color="black", linestyle="--", lw=1.5, label=r"$E_{\mathrm{nom}}^{\mathrm{in}}$")
    ax_t_inh.axvline(T_NOM_START + sample_tp, color="#D62728", linestyle=":", lw=1.5, label=r"$E_p^{\mathrm{in}}$")

    ax_t_inh.set_xlabel(r"Time $t$ (ms)", fontsize=11)
    ax_t_inh.set_xlim(TIME_MIN, TIME_MAX)
    ax_t_inh.set_ylim(-25, 110)
    ax_t_inh.grid(True, linestyle="--", alpha=0.3)
    ax_t_inh.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.subplots_adjust(hspace=0.35, wspace=0.15)

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig11] Saved Figure 11 plot to: {OUTPUT_FIG}")


if __name__ == "__main__":
    run()
