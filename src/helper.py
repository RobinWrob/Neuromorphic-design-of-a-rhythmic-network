"""
helper.py - Core Biophysics & Numerical Solvers for Neuromorphic Node Models.

This module provides common functions for single-node and multi-node (ring network)
simulations, biophysical ionic current derivatives, Numba JIT-accelerated
Forward Euler and RK4 numerical integrators, threshold detection, Opheim trace simplification,
and TikZ-compatible CSV data exporters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Try importing numba for JIT compilation, fallback to identity decorator if missing
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# Configuration Loader
# ============================================================================

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration dictionary from a YAML file, with fallback if PyYAML is missing."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        res: Dict[str, Any] = {}
        curr_section: Dict[str, Any] = res
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    line = line.split("#")[0].strip()
                indent = len(raw_line) - len(raw_line.lstrip(" "))
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if not v:
                        res[k] = {}
                        curr_section = res[k]
                        continue
                    try:
                        v_parsed: Any = float(v) if ("." in v or "e" in v.lower()) else int(v)
                    except ValueError:
                        if v.lower() in ("true", "false"):
                            v_parsed = (v.lower() == "true")
                        else:
                            v_parsed = v.strip("'\"")
                    
                    if indent > 0 and isinstance(curr_section, dict):
                        curr_section[k] = v_parsed
                    else:
                        res[k] = v_parsed
                        curr_section = res
        return res




# ============================================================================
# Biophysical Helpers & Solvers
# ============================================================================

@jit(nopython=True, fastmath=True)
def _clip_scalar(val: float, low: float = 0.0, high: float = 1.0) -> float:
    """Scalar SatReLU clipping helper."""
    return max(low, min(high, val))


@jit(nopython=True, fastmath=True)
def _single_node_derivatives(
    vf_val: float,
    vs_val: float,
    gs_val: float,
    v_pre: float,
    # Biophysical parameters
    cm: float,
    tau_s: float,
    input_gain: float,
    i_base: float,
    g_na: float,
    e_na: float,
    na_act_bias: float,
    na_act_slope: float,
    na_inact_bias: float,
    na_inact_slope: float,
    g_na_reb: float,
    e_na_reb: float,
    na_reb_act_bias: float,
    na_reb_act_slope: float,
    na_reb_inact_bias: float,
    na_reb_inact_slope: float,
    g_k: float,
    e_k: float,
    k_act_bias: float,
    k_act_slope: float,
    g_l: float,
    e_l: float,
    g_a: float,
    e_a: float,
    a_act_bias: float,
    a_act_slope: float,
    a_inact_bias: float,
    a_inact_slope: float,
    # Synapse parameters
    tau_decay: float,
    tau_rise: float,
    g_max: float,
    e_rev: float,
    syn_threshold: float,
    syn_k: float,
) -> Tuple[float, float, float]:
    """Compute state derivatives (dvf, dvs, dgs) for a single node."""
    # SatReLU Gating
    m_na = _clip_scalar(na_act_slope * (vf_val - na_act_bias), 0.0, 1.0)
    h_na = _clip_scalar(na_inact_slope * (na_inact_bias - vs_val), 0.0, 1.0)
    m_na_reb = _clip_scalar(na_reb_act_slope * (vf_val - na_reb_act_bias), 0.0, 1.0)
    h_na_reb = _clip_scalar(na_reb_inact_slope * (na_reb_inact_bias - vs_val), 0.0, 1.0)
    n_k = _clip_scalar(k_act_slope * (vs_val - k_act_bias), 0.0, 1.0)
    m_a = _clip_scalar(a_act_slope * (vf_val - a_act_bias), 0.0, 1.0)
    h_a = _clip_scalar(a_inact_slope * (a_inact_bias - vs_val), 0.0, 1.0)

    # Ionic currents
    i_ionic = (
        -g_na * m_na * h_na * (vf_val - e_na)
        - g_a * m_a * h_a * (vf_val - e_a)
        - g_na_reb * m_na_reb * h_na_reb * (vf_val - e_na_reb)
        - g_k * n_k * (vf_val - e_k)
        - g_l * (vf_val - e_l)
    )

    # Synaptic drive with overflow protection
    slope = max(abs(syn_k), 1e-9)
    inv_tau_r = 1.0 / max(tau_rise, 1e-9)
    inv_tau_d = 1.0 / max(tau_decay, 1e-9)

    exp_arg = _clip_scalar(-(v_pre - syn_threshold) / slope, -50.0, 50.0)
    drive = 1.0 / (1.0 + np.exp(exp_arg))
    alpha = max(0.0, (inv_tau_r - inv_tau_d) * drive)
    dg_syn = alpha * (1.0 - gs_val) - inv_tau_d * gs_val

    i_syn = g_max * gs_val * (e_rev - vf_val)
    i_app = -input_gain * i_syn - i_base

    dvf = (i_ionic - i_app) / cm
    dvs = (vf_val - vs_val) / (tau_s * cm)

    return dvf, dvs, dg_syn


@jit(nopython=True, fastmath=True)
def _solve_euler_jit(
    n_steps: int,
    dt: float,
    input_period: float,
    impulse_width: float,
    impulse_amplitude: float,
    # Biophysical parameters
    cm: float, tau_s: float, input_gain: float, i_base: float,
    g_na: float, e_na: float, na_act_bias: float, na_act_slope: float, na_inact_bias: float, na_inact_slope: float,
    g_na_reb: float, e_na_reb: float, na_reb_act_bias: float, na_reb_act_slope: float, na_reb_inact_bias: float, na_reb_inact_slope: float,
    g_k: float, e_k: float, k_act_bias: float, k_act_slope: float,
    g_l: float, e_l: float,
    g_a: float, e_a: float, a_act_bias: float, a_act_slope: float, a_inact_bias: float, a_inact_slope: float,
    # Synapse parameters
    tau_decay: float, tau_rise: float, g_max: float, e_rev: float, syn_threshold: float, syn_k: float,
) -> np.ndarray:
    """Forward Euler solver loop for single node integration."""
    vf = np.zeros(n_steps)
    vs = np.zeros(n_steps)
    g_syn = np.zeros(n_steps)

    for k in range(n_steps - 1):
        t_curr = k * dt
        phase = t_curr % input_period if input_period > 0 else 0.0
        v_pre = impulse_amplitude if phase < impulse_width else 0.0

        dvf, dvs, dg_syn = _single_node_derivatives(
            vf[k], vs[k], g_syn[k], v_pre,
            cm, tau_s, input_gain, i_base,
            g_na, e_na, na_act_bias, na_act_slope, na_inact_bias, na_inact_slope,
            g_na_reb, e_na_reb, na_reb_act_bias, na_reb_act_slope, na_reb_inact_bias, na_reb_inact_slope,
            g_k, e_k, k_act_bias, k_act_slope,
            g_l, e_l,
            g_a, e_a, a_act_bias, a_act_slope, a_inact_bias, a_inact_slope,
            tau_decay, tau_rise, g_max, e_rev, syn_threshold, syn_k,
        )

        vf[k + 1] = _clip_scalar(vf[k] + dt * dvf, -100.0, 200.0)
        vs[k + 1] = _clip_scalar(vs[k] + dt * dvs, -100.0, 200.0)
        g_syn[k + 1] = _clip_scalar(g_syn[k] + dt * dg_syn, 0.0, 10.0)

    return vf


@jit(nopython=True, fastmath=True)
def _solve_rk4_jit(
    n_steps: int,
    dt: float,
    input_period: float,
    impulse_width: float,
    impulse_amplitude: float,
    # Biophysical parameters
    cm: float, tau_s: float, input_gain: float, i_base: float,
    g_na: float, e_na: float, na_act_bias: float, na_act_slope: float, na_inact_bias: float, na_inact_slope: float,
    g_na_reb: float, e_na_reb: float, na_reb_act_bias: float, na_reb_act_slope: float, na_reb_inact_bias: float, na_reb_inact_slope: float,
    g_k: float, e_k: float, k_act_bias: float, k_act_slope: float,
    g_l: float, e_l: float,
    g_a: float, e_a: float, a_act_bias: float, a_act_slope: float, a_inact_bias: float, a_inact_slope: float,
    # Synapse parameters
    tau_decay: float, tau_rise: float, g_max: float, e_rev: float, syn_threshold: float, syn_k: float,
) -> np.ndarray:
    """Runge-Kutta 4th order (RK4) solver loop for single node integration."""
    vf = np.zeros(n_steps)
    vs = np.zeros(n_steps)
    g_syn = np.zeros(n_steps)

    for k in range(n_steps - 1):
        t_curr = k * dt
        phase = t_curr % input_period if input_period > 0 else 0.0
        v_pre = impulse_amplitude if phase < impulse_width else 0.0

        # k1
        k1_vf, k1_vs, k1_gs = _single_node_derivatives(
            vf[k], vs[k], g_syn[k], v_pre,
            cm, tau_s, input_gain, i_base,
            g_na, e_na, na_act_bias, na_act_slope, na_inact_bias, na_inact_slope,
            g_na_reb, e_na_reb, na_reb_act_bias, na_reb_act_slope, na_reb_inact_bias, na_reb_inact_slope,
            g_k, e_k, k_act_bias, k_act_slope,
            g_l, e_l,
            g_a, e_a, a_act_bias, a_act_slope, a_inact_bias, a_inact_slope,
            tau_decay, tau_rise, g_max, e_rev, syn_threshold, syn_k,
        )

        # k2
        k2_vf, k2_vs, k2_gs = _single_node_derivatives(
            vf[k] + 0.5 * dt * k1_vf, vs[k] + 0.5 * dt * k1_vs, g_syn[k] + 0.5 * dt * k1_gs, v_pre,
            cm, tau_s, input_gain, i_base,
            g_na, e_na, na_act_bias, na_act_slope, na_inact_bias, na_inact_slope,
            g_na_reb, e_na_reb, na_reb_act_bias, na_reb_act_slope, na_reb_inact_bias, na_reb_inact_slope,
            g_k, e_k, k_act_bias, k_act_slope,
            g_l, e_l,
            g_a, e_a, a_act_bias, a_act_slope, a_inact_bias, a_inact_slope,
            tau_decay, tau_rise, g_max, e_rev, syn_threshold, syn_k,
        )

        # k3
        k3_vf, k3_vs, k3_gs = _single_node_derivatives(
            vf[k] + 0.5 * dt * k2_vf, vs[k] + 0.5 * dt * k2_vs, g_syn[k] + 0.5 * dt * k2_gs, v_pre,
            cm, tau_s, input_gain, i_base,
            g_na, e_na, na_act_bias, na_act_slope, na_inact_bias, na_inact_slope,
            g_na_reb, e_na_reb, na_reb_act_bias, na_reb_act_slope, na_reb_inact_bias, na_reb_inact_slope,
            g_k, e_k, k_act_bias, k_act_slope,
            g_l, e_l,
            g_a, e_a, a_act_bias, a_act_slope, a_inact_bias, a_inact_slope,
            tau_decay, tau_rise, g_max, e_rev, syn_threshold, syn_k,
        )

        # k4
        k4_vf, k4_vs, k4_gs = _single_node_derivatives(
            vf[k] + dt * k3_vf, vs[k] + dt * k3_vs, g_syn[k] + dt * k3_gs, v_pre,
            cm, tau_s, input_gain, i_base,
            g_na, e_na, na_act_bias, na_act_slope, na_inact_bias, na_inact_slope,
            g_na_reb, e_na_reb, na_reb_act_bias, na_reb_act_slope, na_reb_inact_bias, na_reb_inact_slope,
            g_k, e_k, k_act_bias, k_act_slope,
            g_l, e_l,
            g_a, e_a, a_act_bias, a_act_slope, a_inact_bias, a_inact_slope,
            tau_decay, tau_rise, g_max, e_rev, syn_threshold, syn_k,
        )

        dvf = (k1_vf + 2.0 * k2_vf + 2.0 * k3_vf + k4_vf) / 6.0
        dvs = (k1_vs + 2.0 * k2_vs + 2.0 * k3_vs + k4_vs) / 6.0
        dg_syn = (k1_gs + 2.0 * k2_gs + 2.0 * k3_gs + k4_gs) / 6.0

        vf[k + 1] = _clip_scalar(vf[k] + dt * dvf, -100.0, 200.0)
        vs[k + 1] = _clip_scalar(vs[k] + dt * dvs, -100.0, 200.0)
        g_syn[k + 1] = _clip_scalar(g_syn[k] + dt * dg_syn, 0.0, 10.0)

    return vf


# ============================================================================
# High-Level Modular Simulators
# ============================================================================

def simulate_single_node(
    params: Dict[str, float],
    syn_params: Dict[str, float],
    dt: float,
    t_end: float,
    input_period: float,
    impulse_width: float,
    impulse_amplitude: float,
    solver: str = "euler",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run simulation for a single neuromorphic node (Simple or Simple+H).
    Returns (time, v_trace).
    """
    n_steps = int(round(t_end / dt))
    time = np.linspace(0.0, t_end, n_steps)

    solver_fn = _solve_rk4_jit if solver.lower() == "rk4" else _solve_euler_jit

    v_trace = solver_fn(
        n_steps, dt, input_period, impulse_width, impulse_amplitude,
        # Biophysics
        params["cm"], params["tau_s"], params["input_gain"], params["i_base"],
        params["g_na"], params["e_na"], params["na_act_bias"], params["na_act_slope"], params["na_inact_bias"], params["na_inact_slope"],
        params.get("g_na_reb", 0.0), params.get("e_na_reb", 100.0), params.get("na_reb_act_bias", -10.0), params.get("na_reb_act_slope", 0.1), params.get("na_reb_inact_bias", -5.0), params.get("na_reb_inact_slope", 0.1),
        params["g_k"], params["e_k"], params["k_act_bias"], params["k_act_slope"],
        params["g_l"], params["e_l"],
        params.get("g_a", 0.0), params.get("e_a", -20.0), params.get("a_act_bias", -15.0), params.get("a_act_slope", 0.1), params.get("a_inact_bias", -10.0), params.get("a_inact_slope", 0.15),
        # Synapse
        syn_params["tau_decay"], syn_params["tau_rise"], syn_params["g_max"], syn_params["e_rev"], syn_params["threshold"], syn_params.get("syn_k", syn_params.get("k", 0.1)),
    )

    return time, v_trace


# ============================================================================
# Signal Processing & Event Detection
# ============================================================================

def detect_threshold_crossings(
    t: np.ndarray, v: np.ndarray, threshold: float = 30.0
) -> np.ndarray:
    """Detect upward threshold crossings with linear interpolation."""
    above = v >= threshold
    crossings_idx = np.where((~above[:-1]) & above[1:])[0]
    crossing_times = []
    for idx in crossings_idx:
        v0, v1 = v[idx], v[idx + 1]
        t0, t1 = t[idx], t[idx + 1]
        if abs(v1 - v0) > 1e-12:
            t_cross = t0 + (threshold - v0) * (t1 - t0) / (v1 - v0)
        else:
            t_cross = t0
        crossing_times.append(t_cross)
    return np.array(crossing_times, dtype=np.float64)


def validate_1to1_phase_locking(
    input_times: np.ndarray, output_times: np.ndarray, input_period: float
) -> Tuple[bool, Optional[float]]:
    """
    Validate 1:1 phase locking between input events and output spikes.
    Returns (is_valid, mean_onset_delay).
    """
    if len(input_times) == 0 or len(output_times) == 0:
        return False, None
    if len(input_times) != len(output_times):
        return False, None

    delays = output_times - input_times
    if np.any(delays < 0) or np.any(delays > input_period):
        return False, None

    return True, float(np.mean(delays))


def opheim_simplify_trace(
    t: np.ndarray, v: np.ndarray, min_tol: float = 0.002, max_tol: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Opheim polyline simplification algorithm for reducing time-series trace point count
    specifically for clean LaTeX/TikZ CSV export.
    """
    if len(t) <= 2:
        return t, v

    # Normalize independently to [0, 1] for error bounds
    t_min, t_max = t[0], t[-1]
    v_min, v_max = np.min(v), np.max(v)
    v_range = max(v_max - v_min, 1e-9)
    t_range = max(t_max - t_min, 1e-9)

    t_norm = (t - t_min) / t_range
    v_norm = (v - v_min) / v_range

    kept_indices = [0]
    anchor = 0

    n = len(t)
    while anchor < n - 1:
        # Search for furthest valid point within tolerance
        last_valid = anchor + 1
        for check in range(anchor + 2, n):
            # Check maximum distance from segment (anchor, check)
            p1 = np.array([t_norm[anchor], v_norm[anchor]])
            p2 = np.array([t_norm[check], v_norm[check]])
            vec = p2 - p1
            norm_vec = np.linalg.norm(vec)

            if norm_vec < 1e-12:
                continue

            # Perpendicular distances of intermediate points
            sub_pts = np.column_stack((t_norm[anchor + 1 : check], v_norm[anchor + 1 : check]))
            proj_lengths = np.dot(sub_pts - p1, vec) / norm_vec
            perp_dists = np.linalg.norm((sub_pts - p1) - np.outer(proj_lengths, vec / norm_vec), axis=1)

            if np.max(perp_dists) <= max_tol:
                last_valid = check
            else:
                break

        kept_indices.append(last_valid)
        anchor = last_valid

    idx = np.array(kept_indices, dtype=int)
    return t[idx], v[idx]


# ============================================================================
# Output Data & Plotting Utilities
# ============================================================================

def export_to_csv(
    data: Union[Dict[str, np.ndarray], pd.DataFrame], filepath: Union[str, Path]
) -> Path:
    """Export data dictionary or DataFrame to CSV format for TikZ/LaTeX plotting."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data
    df.to_csv(path, index=False)
    return path


def plot_validation_trace(
    time: np.ndarray,
    v_dict: Dict[str, np.ndarray],
    title: str,
    filepath: Union[str, Path],
    xlabel: str = "Time (ms)",
    ylabel: str = "Membrane Potential (mV)",
) -> Path:
    """Generate Matplotlib validation figure and save to results/figures/."""
    import matplotlib.pyplot as plt

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for label, v_trace in v_dict.items():
        ax.plot(time, v_trace, label=label, linewidth=1.2)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    if len(v_dict) > 1:
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
