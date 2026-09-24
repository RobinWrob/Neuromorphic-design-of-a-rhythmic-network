"""
eprc_engine.py - Core Engine for Event Phase Response Curve (ePRC) Computation.

Computes event phase response curves by measuring output spike timing shifts
caused by secondary synaptic perturbations.
Accelerated using Numba JIT compilation and ProcessPoolExecutor.
"""

from __future__ import annotations

import os
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Optional, List, Dict

import numpy as np
import pandas as pd

try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@dataclass
class EPRCSample:
    """Result of one ePRC calculation at a given perturbation delay t_p."""
    tp: float
    delta_t: float | None
    t_nom_spike: float | None
    t_pert_spike: float | None
    is_valid: bool
    runtime_s: float | None = None


def default_eprc_neuron_params() -> dict[str, float]:
    return {
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


def default_nominal_synapse_params() -> dict[str, float]:
    """Default parameters for the primary (nominal) input synapse."""
    return {
        "tau_decay": 2.0,
        "tau_rise": 0.1,
        "g_max": 3.0,
        "e_rev": -20.0,
        "threshold": 30.0,
        "k": 0.1,
    }


def default_sensory_synapse_params(syn_type: str = "inh") -> dict[str, float]:
    """Default parameters for the secondary (sensory perturbation) input synapse."""
    if syn_type == "inh":
        return {
            "tau_decay": 2.0,
            "tau_rise": 0.1,
            "g_max": 0.01,
            "e_rev": -20.0,
            "threshold": 30.0,
            "k": 0.1,
        }
    return {
        "tau_decay": 2.0,
        "tau_rise": 0.1,
        "g_max": 0.01,
        "e_rev": 50.0,
        "threshold": 30.0,
        "k": 0.1,
    }


def default_eprc_synapse_params(syn_type: str = "inh") -> dict[str, float]:
    return default_sensory_synapse_params(syn_type)


@jit(nopython=True, fastmath=True)
def _clip_scalar(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


@jit(nopython=True, fastmath=True)
def _solve_eprc_euler_jit(
    n_steps: int,
    dt: float,
    input_period: float,
    impulse_width: float,
    impulse_amplitude: float,
    tp: float,
    single_event: bool,
    t_nom_start: float,
    cm: float, tau_s: float, input_gain: float, i_base: float,
    g_na: float, e_na: float, na_act_bias: float, na_act_slope: float, na_inact_bias: float, na_inact_slope: float,
    g_na_reb: float, e_na_reb: float, na_reb_act_bias: float, na_reb_act_slope: float, na_reb_inact_bias: float, na_reb_inact_slope: float,
    g_k: float, e_k: float, k_act_bias: float, k_act_slope: float,
    g_l: float, e_l: float,
    g_a: float, e_a: float, a_act_bias: float, a_act_slope: float, a_inact_bias: float, a_inact_slope: float,
    nom_tau_decay: float, nom_tau_rise: float, nom_g_max: float, nom_e_rev: float, nom_threshold: float, nom_k: float,
    sens_tau_decay: float, sens_tau_rise: float, sens_g_max: float, sens_e_rev: float, sens_threshold: float, sens_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dual simulation solver: computes unperturbed (nominal) and perturbed traces simultaneously."""
    vf_nom = np.zeros(n_steps)
    vs_nom = np.zeros(n_steps)
    gs_nom = np.zeros(n_steps)

    vf_pert = np.zeros(n_steps)
    vs_pert = np.zeros(n_steps)
    gs_nom_pert = np.zeros(n_steps)
    gs_sens = np.zeros(n_steps)

    nom_slope = max(abs(nom_k), 1e-9)
    nom_inv_tr = 1.0 / max(nom_tau_rise, 1e-9)
    nom_inv_td = 1.0 / max(nom_tau_decay, 1e-9)

    sens_slope = max(abs(sens_k), 1e-9)
    sens_inv_tr = 1.0 / max(sens_tau_rise, 1e-9)
    sens_inv_td = 1.0 / max(sens_tau_decay, 1e-9)

    for k in range(n_steps - 1):
        t_curr = k * dt

        if single_event:
            v_pre_nom = impulse_amplitude if (t_nom_start <= t_curr < t_nom_start + impulse_width) else 0.0
            t_sens_start = t_nom_start + tp
            v_pre_sens = impulse_amplitude if (t_sens_start <= t_curr < t_sens_start + impulse_width) else 0.0
        else:
            phase_nom = t_curr % input_period if input_period > 0 else 0.0
            v_pre_nom = impulse_amplitude if phase_nom < impulse_width else 0.0
            phase_sens = (t_curr - tp) % input_period if input_period > 0 else 0.0
            v_pre_sens = impulse_amplitude if (t_curr >= tp and phase_sens < impulse_width) else 0.0

        # Nominal step
        m_na = _clip_scalar(na_act_slope * (vf_nom[k] - na_act_bias), 0.0, 1.0)
        h_na = _clip_scalar(na_inact_slope * (na_inact_bias - vs_nom[k]), 0.0, 1.0)
        m_na_reb = _clip_scalar(na_reb_act_slope * (vf_nom[k] - na_reb_act_bias), 0.0, 1.0)
        h_na_reb = _clip_scalar(na_reb_inact_slope * (na_reb_inact_bias - vs_nom[k]), 0.0, 1.0)
        n_k = _clip_scalar(k_act_slope * (vs_nom[k] - k_act_bias), 0.0, 1.0)
        m_a = _clip_scalar(a_act_slope * (vf_nom[k] - a_act_bias), 0.0, 1.0)
        h_a = _clip_scalar(a_inact_slope * (a_inact_bias - vs_nom[k]), 0.0, 1.0)

        i_ionic_nom = (
            -g_na * m_na * h_na * (vf_nom[k] - e_na)
            - g_a * m_a * h_a * (vf_nom[k] - e_a)
            - g_na_reb * m_na_reb * h_na_reb * (vf_nom[k] - e_na_reb)
            - g_k * n_k * (vf_nom[k] - e_k)
            - g_l * (vf_nom[k] - e_l)
        )

        exp_arg_nom = _clip_scalar(-(v_pre_nom - nom_threshold) / nom_slope, -50.0, 50.0)
        drive_nom = 1.0 / (1.0 + np.exp(exp_arg_nom))
        alpha_nom = max(0.0, (nom_inv_tr - nom_inv_td) * drive_nom)
        dgs_nom = alpha_nom * (1.0 - gs_nom[k]) - nom_inv_td * gs_nom[k]

        i_syn_nom = nom_g_max * gs_nom[k] * (nom_e_rev - vf_nom[k])
        i_app_nom = -input_gain * i_syn_nom - i_base

        dvf_nom = (i_ionic_nom - i_app_nom) / cm
        dvs_nom = (vf_nom[k] - vs_nom[k]) / (tau_s * cm)

        vf_nom[k + 1] = _clip_scalar(vf_nom[k] + dt * dvf_nom, -100.0, 200.0)
        vs_nom[k + 1] = _clip_scalar(vs_nom[k] + dt * dvs_nom, -100.0, 200.0)
        gs_nom[k + 1] = _clip_scalar(gs_nom[k] + dt * dgs_nom, 0.0, 10.0)

        # Perturbed step
        m_na_p = _clip_scalar(na_act_slope * (vf_pert[k] - na_act_bias), 0.0, 1.0)
        h_na_p = _clip_scalar(na_inact_slope * (na_inact_bias - vs_pert[k]), 0.0, 1.0)
        m_na_reb_p = _clip_scalar(na_reb_act_slope * (vf_pert[k] - na_reb_act_bias), 0.0, 1.0)
        h_na_reb_p = _clip_scalar(na_reb_inact_slope * (na_reb_inact_bias - vs_pert[k]), 0.0, 1.0)
        n_k_p = _clip_scalar(k_act_slope * (vs_pert[k] - k_act_bias), 0.0, 1.0)
        m_a_p = _clip_scalar(a_act_slope * (vf_pert[k] - a_act_bias), 0.0, 1.0)
        h_a_p = _clip_scalar(a_inact_slope * (a_inact_bias - vs_pert[k]), 0.0, 1.0)

        i_ionic_pert = (
            -g_na * m_na_p * h_na_p * (vf_pert[k] - e_na)
            - g_a * m_a_p * h_a_p * (vf_pert[k] - e_a)
            - g_na_reb * m_na_reb_p * h_na_reb_p * (vf_pert[k] - e_na_reb)
            - g_k * n_k_p * (vf_pert[k] - e_k)
            - g_l * (vf_pert[k] - e_l)
        )

        exp_arg_nom_p = _clip_scalar(-(v_pre_nom - nom_threshold) / nom_slope, -50.0, 50.0)
        drive_nom_p = 1.0 / (1.0 + np.exp(exp_arg_nom_p))
        alpha_nom_p = max(0.0, (nom_inv_tr - nom_inv_td) * drive_nom_p)
        dgs_nom_p = alpha_nom_p * (1.0 - gs_nom_pert[k]) - nom_inv_td * gs_nom_pert[k]

        exp_arg_sens = _clip_scalar(-(v_pre_sens - sens_threshold) / sens_slope, -50.0, 50.0)
        drive_sens = 1.0 / (1.0 + np.exp(exp_arg_sens))
        alpha_sens = max(0.0, (sens_inv_tr - sens_inv_td) * drive_sens)
        dgs_sens = alpha_sens * (1.0 - gs_sens[k]) - sens_inv_td * gs_sens[k]

        i_syn_nom_p = nom_g_max * gs_nom_pert[k] * (nom_e_rev - vf_pert[k])
        i_syn_sens_p = sens_g_max * gs_sens[k] * (sens_e_rev - vf_pert[k])
        i_app_pert = -input_gain * (i_syn_nom_p + i_syn_sens_p) - i_base

        dvf_p = (i_ionic_pert - i_app_pert) / cm
        dvs_p = (vf_pert[k] - vs_pert[k]) / (tau_s * cm)

        vf_pert[k + 1] = _clip_scalar(vf_pert[k] + dt * dvf_p, -100.0, 200.0)
        vs_pert[k + 1] = _clip_scalar(vs_pert[k] + dt * dvs_p, -100.0, 200.0)
        gs_nom_pert[k + 1] = _clip_scalar(gs_nom_pert[k] + dt * dgs_nom_p, 0.0, 10.0)
        gs_sens[k + 1] = _clip_scalar(gs_sens[k] + dt * dgs_sens, 0.0, 10.0)

    return vf_nom, vf_pert, gs_nom, gs_sens, vs_nom, vs_pert


def detect_threshold_crossings(t: np.ndarray, v: np.ndarray, threshold: float) -> np.ndarray:
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


def compute_spike_shift(
    nom_spikes: np.ndarray,
    pert_spikes: np.ndarray,
    single_event: bool = False,
    t_nom_start: float = 50.0,
) -> tuple[float | None, float | None, float | None]:
    if len(nom_spikes) == 0 or len(pert_spikes) == 0:
        return None, None, None

    if single_event:
        t_nom = nom_spikes[0]
        t_pert = pert_spikes[0]
        return float(t_pert - t_nom), float(t_nom), float(t_pert)
    else:
        valid_nom = nom_spikes[nom_spikes >= t_nom_start]
        if len(valid_nom) == 0:
            return None, None, None
        t_nom = valid_nom[0]
        idx = np.argmin(np.abs(pert_spikes - t_nom))
        t_pert = pert_spikes[idx]
        return float(t_pert - t_nom), float(t_nom), float(t_pert)


def _apply_param_overrides(p_n: dict, p_s_nom: dict, p_s_sens: dict, param_overrides: Mapping[str, Any] | None) -> None:
    if not param_overrides:
        return
    for k, v in param_overrides.items():
        val = float(v)
        key = str(k)
        if key.startswith("neuron."):
            key = key[7:]
        elif key.startswith("synapse_nom."):
            key = "nom_" + key[12:]
        elif key.startswith("synapse_sens."):
            key = "sens_" + key[13:]

        if key in p_n:
            p_n[key] = val
        elif key.startswith("nom_") and key[4:] in p_s_nom:
            p_s_nom[key[4:]] = val
        elif key.startswith("sens_") and key[5:] in p_s_sens:
            p_s_sens[key[5:]] = val
        elif key in p_s_sens:
            p_s_sens[key] = val


def run_eprc_sample(
    tp: float,
    input_period: float = 50.0,
    dt: float = 0.0001,
    impulse_width: float = 0.1,
    impulse_amplitude: float = 50.0,
    spike_threshold: float = 50.0,
    single_event: bool = False,
    duration_multiplier: float = 6.0,
    min_t: float = 400.0,
    t_nom_start: float = 50.0,
    param_overrides: Mapping[str, Any] | None = None,
    syn_type: str = "inh",
) -> EPRCSample:
    start_wall = time.time()
    t_sim = max(min_t, duration_multiplier * input_period) if not single_event else min_t
    n_steps = int(round(t_sim / dt))

    p_n = default_eprc_neuron_params()
    p_s_nom = default_nominal_synapse_params()
    p_s_sens = default_sensory_synapse_params(syn_type)

    _apply_param_overrides(p_n, p_s_nom, p_s_sens, param_overrides)

    vf_nom, vf_pert, _, _, _, _ = _solve_eprc_euler_jit(
        n_steps, dt, input_period, impulse_width, impulse_amplitude, tp, single_event, t_nom_start,
        p_n["cm"], p_n["tau_s"], p_n["input_gain"], p_n["i_base"],
        p_n["g_na"], p_n["e_na"], p_n["na_act_bias"], p_n["na_act_slope"], p_n["na_inact_bias"], p_n["na_inact_slope"],
        p_n.get("g_na_reb", 2.0), p_n.get("e_na_reb", 100.0), p_n.get("na_reb_act_bias", -10.0), p_n.get("na_reb_act_slope", 0.1), p_n.get("na_reb_inact_bias", -5.0), p_n.get("na_reb_inact_slope", 0.1),
        p_n["g_k"], p_n["e_k"], p_n["k_act_bias"], p_n["k_act_slope"], p_n["g_l"], p_n["e_l"],
        p_n.get("g_a", 0.0), p_n.get("e_a", -20.0), p_n.get("a_act_bias", -15.0), p_n.get("a_act_slope", 0.1), p_n.get("a_inact_bias", -10.0), p_n.get("a_inact_slope", 0.15),
        p_s_nom["tau_decay"], p_s_nom["tau_rise"], p_s_nom["g_max"], p_s_nom["e_rev"], p_s_nom["threshold"], p_s_nom.get("k", 0.1),
        p_s_sens["tau_decay"], p_s_sens["tau_rise"], p_s_sens["g_max"], p_s_sens["e_rev"], p_s_sens["threshold"], p_s_sens.get("k", 0.1),
    )

    t = np.linspace(0.0, t_sim, n_steps)
    nom_spikes = detect_threshold_crossings(t, vf_nom, spike_threshold)
    pert_spikes = detect_threshold_crossings(t, vf_pert, spike_threshold)

    delta_t, t_nom_spike, t_pert_spike = compute_spike_shift(
        nom_spikes, pert_spikes, single_event=single_event, t_nom_start=t_nom_start
    )

    return EPRCSample(
        tp=tp, delta_t=delta_t, t_nom_spike=t_nom_spike, t_pert_spike=t_pert_spike,
        is_valid=(delta_t is not None), runtime_s=time.time() - start_wall,
    )


def run_eprc_simulation_trace(
    tp: float,
    input_period: float = 50.0,
    dt: float = 0.0001,
    impulse_width: float = 0.1,
    impulse_amplitude: float = 50.0,
    spike_threshold: float = 50.0,
    single_event: bool = False,
    duration_multiplier: float = 6.0,
    min_t: float = 400.0,
    t_nom_start: float = 50.0,
    param_overrides: Mapping[str, Any] | None = None,
    syn_type: str = "inh",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | None, float | None, float | None]:
    t_sim = max(min_t, duration_multiplier * input_period) if not single_event else min_t
    n_steps = int(round(t_sim / dt))

    p_n = default_eprc_neuron_params()
    p_s_nom = default_nominal_synapse_params()
    p_s_sens = default_sensory_synapse_params(syn_type)

    _apply_param_overrides(p_n, p_s_nom, p_s_sens, param_overrides)

    vf_nom, vf_pert, gs_nom, gs_sens, _, _ = _solve_eprc_euler_jit(
        n_steps, dt, input_period, impulse_width, impulse_amplitude, tp, single_event, t_nom_start,
        p_n["cm"], p_n["tau_s"], p_n["input_gain"], p_n["i_base"],
        p_n["g_na"], p_n["e_na"], p_n["na_act_bias"], p_n["na_act_slope"], p_n["na_inact_bias"], p_n["na_inact_slope"],
        p_n.get("g_na_reb", 2.0), p_n.get("e_na_reb", 100.0), p_n.get("na_reb_act_bias", -10.0), p_n.get("na_reb_act_slope", 0.1), p_n.get("na_reb_inact_bias", -5.0), p_n.get("na_reb_inact_slope", 0.1),
        p_n["g_k"], p_n["e_k"], p_n["k_act_bias"], p_n["k_act_slope"], p_n["g_l"], p_n["e_l"],
        p_n.get("g_a", 0.0), p_n.get("e_a", -20.0), p_n.get("a_act_bias", -15.0), p_n.get("a_act_slope", 0.1), p_n.get("a_inact_bias", -10.0), p_n.get("a_inact_slope", 0.15),
        p_s_nom["tau_decay"], p_s_nom["tau_rise"], p_s_nom["g_max"], p_s_nom["e_rev"], p_s_nom["threshold"], p_s_nom.get("k", 0.1),
        p_s_sens["tau_decay"], p_s_sens["tau_rise"], p_s_sens["g_max"], p_s_sens["e_rev"], p_s_sens["threshold"], p_s_sens.get("k", 0.1),
    )

    t = np.linspace(0.0, t_sim, n_steps)

    pulse_nom = np.zeros(n_steps)
    pulse_sens = np.zeros(n_steps)
    for k in range(n_steps):
        t_curr = k * dt
        if single_event:
            pulse_nom[k] = impulse_amplitude if (0.0 <= t_curr < impulse_width) else 0.0
            pulse_sens[k] = impulse_amplitude if (tp <= t_curr < tp + impulse_width) else 0.0
        else:
            phase_nom = t_curr % input_period if input_period > 0 else 0.0
            pulse_nom[k] = impulse_amplitude if phase_nom < impulse_width else 0.0
            phase_sens = (t_curr - tp) % input_period if input_period > 0 else 0.0
            pulse_sens[k] = impulse_amplitude if (t_curr >= tp and phase_sens < impulse_width) else 0.0

    nom_spikes = detect_threshold_crossings(t, vf_nom, spike_threshold)
    pert_spikes = detect_threshold_crossings(t, vf_pert, spike_threshold)

    delta_t, t_nom_spike, t_pert_spike = compute_spike_shift(
        nom_spikes, pert_spikes, single_event=single_event, t_nom_start=t_nom_start
    )

    return t, vf_nom, vf_pert, gs_nom, gs_sens, pulse_nom, pulse_sens, t_nom_spike, t_pert_spike, delta_t



def compute_eprc_curve(
    tp_values: Sequence[float],
    input_period: float = 50.0,
    dt: float = 0.0001,
    impulse_width: float = 0.1,
    impulse_amplitude: float = 50.0,
    spike_threshold: float = 50.0,
    single_event: bool = False,
    duration_multiplier: float = 6.0,
    min_t: float = 400.0,
    t_nom_start: float = 50.0,
    parallelize: bool = True,
    num_workers: int | None = None,
    param_overrides: Mapping[str, Any] | None = None,
    syn_type: str = "inh",
) -> list[EPRCSample]:
    worker_fn = partial(
        run_eprc_sample,
        input_period=input_period, dt=dt, impulse_width=impulse_width,
        impulse_amplitude=impulse_amplitude, spike_threshold=spike_threshold,
        single_event=single_event, duration_multiplier=duration_multiplier,
        min_t=min_t, t_nom_start=t_nom_start, param_overrides=param_overrides,
        syn_type=syn_type
    )

    if parallelize and len(tp_values) > 1:
        _ = worker_fn(tp_values[0])
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            chunksize = max(1, len(tp_values) // ((num_workers or 4) * 4))
            results = list(executor.map(worker_fn, tp_values, chunksize=chunksize))
    else:
        results = [worker_fn(tp) for tp in tp_values]

    results.sort(key=lambda s: s.tp)
    return results
