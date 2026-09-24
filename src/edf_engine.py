"""
edf_engine.py - Core Engine for Event Describing Function Computation.

Contains single excitable neuron model, dynamic synapse integration,
spike detection, 1:1 phase-locking validation, adaptive period sweeps,
and metric extraction (T_min, T_r, delta_inf).
Accelerated using Numba JIT compilation and ProcessPoolExecutor.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
class EDFSample:
    """Result of one eDF calculation at a given input period."""
    input_period: float       # ms / time units
    onset_delay: float | None # absolute onset delay delta(T), None if invalid
    num_input_events: int
    num_output_events: int
    is_valid: bool            # True if 1:1 phase-locked
    runtime_s: float | None = None


def default_neuron_params() -> dict[str, float]:
    """Default single excitable neuron model parameters."""
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


def default_synapse_params() -> dict[str, float]:
    """Default dynamic synapse parameters."""
    return {
        "tau_decay": 2.15,
        "tau_rise": 0.1,
        "g_max": 3.0,
        "e_rev": -20.0,
        "threshold": 30.0,
        "syn_k": 0.1,
    }


@jit(nopython=True, fastmath=True)
def _clip_scalar(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


@jit(nopython=True, fastmath=True)
def _solve_euler_jit(
    n_steps: int,
    dt: float,
    input_period: float,
    impulse_width: float,
    impulse_amplitude: float,
    cm: float, tau_s: float, input_gain: float, i_base: float,
    g_na: float, e_na: float, na_act_bias: float, na_act_slope: float, na_inact_bias: float, na_inact_slope: float,
    g_na_reb: float, e_na_reb: float, na_reb_act_bias: float, na_reb_act_slope: float, na_reb_inact_bias: float, na_reb_inact_slope: float,
    g_k: float, e_k: float, k_act_bias: float, k_act_slope: float,
    g_l: float, e_l: float,
    g_a: float, e_a: float, a_act_bias: float, a_act_slope: float, a_inact_bias: float, a_inact_slope: float,
    tau_decay: float, tau_rise: float, g_max: float, e_rev: float, syn_threshold: float, syn_k: float,
) -> np.ndarray:
    """JIT-compiled Forward Euler solver loop for dynamic neuron-synapse integration."""
    vf = np.zeros(n_steps)
    vs = np.zeros(n_steps)
    g_syn = np.zeros(n_steps)

    vf[0] = 0.0
    vs[0] = 0.0

    slope = max(abs(syn_k), 1e-9)
    inv_tau_r = 1.0 / max(tau_rise, 1e-9)
    inv_tau_d = 1.0 / max(tau_decay, 1e-9)

    for k in range(n_steps - 1):
        t_curr = k * dt
        phase = t_curr % input_period if input_period > 0 else 0.0
        v_pre = impulse_amplitude if phase < impulse_width else 0.0

        v_f_val = vf[k]
        v_s_val = vs[k]
        g_s_val = g_syn[k]

        m_na = _clip_scalar(na_act_slope * (v_f_val - na_act_bias), 0.0, 1.0)
        h_na = _clip_scalar(na_inact_slope * (na_inact_bias - v_s_val), 0.0, 1.0)
        m_na_reb = _clip_scalar(na_reb_act_slope * (v_f_val - na_reb_act_bias), 0.0, 1.0)
        h_na_reb = _clip_scalar(na_reb_inact_slope * (na_reb_inact_bias - v_s_val), 0.0, 1.0)
        n_k = _clip_scalar(k_act_slope * (v_s_val - k_act_bias), 0.0, 1.0)
        m_a = _clip_scalar(a_act_slope * (v_f_val - a_act_bias), 0.0, 1.0)
        h_a = _clip_scalar(a_inact_slope * (a_inact_bias - v_s_val), 0.0, 1.0)

        i_ionic = (
            -g_na * m_na * h_na * (v_f_val - e_na)
            - g_a * m_a * h_a * (v_f_val - e_a)
            - g_na_reb * m_na_reb * h_na_reb * (v_f_val - e_na_reb)
            - g_k * n_k * (v_f_val - e_k)
            - g_l * (v_f_val - e_l)
        )

        exp_arg = _clip_scalar(-(v_pre - syn_threshold) / slope, -50.0, 50.0)
        drive = 1.0 / (1.0 + np.exp(exp_arg))
        alpha = max(0.0, (inv_tau_r - inv_tau_d) * drive)
        dg_syn = alpha * (1.0 - g_s_val) - inv_tau_d * g_s_val

        i_syn = g_max * g_s_val * (e_rev - v_f_val)
        i_app = -input_gain * i_syn - i_base

        dvf = (i_ionic - i_app) / cm
        dvs = (v_f_val - v_s_val) / (tau_s * cm)

        vf[k + 1] = _clip_scalar(v_f_val + dt * dvf, -100.0, 200.0)
        vs[k + 1] = _clip_scalar(v_s_val + dt * dvs, -100.0, 200.0)
        g_syn[k + 1] = _clip_scalar(g_s_val + dt * dg_syn, 0.0, 10.0)

    return vf


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


def extract_input_event_times(t_start: float, t_end: float, period: float) -> np.ndarray:
    first_pulse = math.ceil(t_start / period) * period if period > 0 else t_start
    times = []
    curr = first_pulse
    while curr <= t_end:
        times.append(curr)
        curr += period
    return np.array(times, dtype=float)


def validate_phase_locking(
    input_times: np.ndarray,
    output_times: np.ndarray,
    period: float,
    tolerance: float = 5.0,
    min_events: int = 3,
) -> bool:
    if len(input_times) < min_events or len(output_times) < min_events:
        return False
    if abs(len(input_times) - len(output_times)) > 1:
        return False
    if len(output_times) > 1:
        output_intervals = np.diff(output_times)
        start_idx = len(output_intervals) // 2
        intervals_check = output_intervals[start_idx:]
        if len(intervals_check) > 0 and not np.allclose(intervals_check, period, atol=tolerance):
            return False
    return True


def compute_onset_delay(input_times: np.ndarray, output_times: np.ndarray) -> float:
    input_times = np.asarray(input_times)
    output_times = np.asarray(output_times)
    if len(input_times) == 0 or len(output_times) == 0:
        return np.nan

    valid_inputs = input_times[input_times <= output_times[-1]]
    if len(valid_inputs) == 0:
        return np.nan

    matched_input_indices = np.searchsorted(valid_inputs, output_times, side='right') - 1
    valid_mask = matched_input_indices >= 0
    if not np.any(valid_mask):
        return np.nan

    matched_inputs = valid_inputs[matched_input_indices[valid_mask]]
    matched_outputs = output_times[valid_mask]

    delays = matched_outputs - matched_inputs
    n_paired = len(delays)
    steady_state_delays = delays[n_paired // 2:]
    return float(np.mean(steady_state_delays))


def compute_edf_single_period(
    input_period: float,
    dt: float = 0.0001,
    impulse_width: float = 0.1,
    impulse_amplitude: float = 50.0,
    spike_threshold: float = 50.0,
    duration_multiplier: float = 5.0,
    min_t: float = 1000.0,
    neuron_params: dict[str, float] | None = None,
    synapse_params: dict[str, float] | None = None,
) -> EDFSample:
    """Compute eDF for a single input period."""
    t_start = time.perf_counter()

    p_neu = default_neuron_params()
    if neuron_params:
        p_neu.update(neuron_params)

    p_syn = default_synapse_params()
    if synapse_params:
        p_syn.update(synapse_params)

    t_end = max(input_period * duration_multiplier, min_t)
    n_steps = int(np.ceil(t_end / dt)) + 1
    t = np.linspace(0.0, t_end, n_steps)

    syn_k = p_syn.get("syn_k", p_syn.get("k", 0.1))

    vf = _solve_euler_jit(
        n_steps, dt, input_period, impulse_width, impulse_amplitude,
        p_neu["cm"], p_neu["tau_s"], p_neu["input_gain"], p_neu["i_base"],
        p_neu["g_na"], p_neu["e_na"], p_neu["na_act_bias"], p_neu["na_act_slope"], p_neu["na_inact_bias"], p_neu["na_inact_slope"],
        p_neu.get("g_na_reb", 2.0), p_neu.get("e_na_reb", 100.0), p_neu.get("na_reb_act_bias", -10.0), p_neu.get("na_reb_act_slope", 0.1), p_neu.get("na_reb_inact_bias", -5.0), p_neu.get("na_reb_inact_slope", 0.1),
        p_neu["g_k"], p_neu["e_k"], p_neu["k_act_bias"], p_neu["k_act_slope"],
        p_neu["g_l"], p_neu["e_l"],
        p_neu.get("g_a", 0.0), p_neu.get("e_a", -20.0), p_neu.get("a_act_bias", -15.0), p_neu.get("a_act_slope", 0.1), p_neu.get("a_inact_bias", -10.0), p_neu.get("a_inact_slope", 0.15),
        p_syn["tau_decay"], p_syn["tau_rise"], p_syn["g_max"], p_syn["e_rev"], p_syn["threshold"], syn_k,
    )

    output_times = detect_threshold_crossings(t, vf, spike_threshold)
    input_times = extract_input_event_times(0.0, t_end, input_period)

    is_valid = validate_phase_locking(input_times, output_times, input_period)
    onset_delay = compute_onset_delay(input_times, output_times) if is_valid else None
    if onset_delay is not None and onset_delay < 1.0:
        is_valid = False
        onset_delay = None

    runtime = time.perf_counter() - t_start
    return EDFSample(
        input_period=input_period, onset_delay=onset_delay,
        num_input_events=len(input_times), num_output_events=len(output_times),
        is_valid=is_valid, runtime_s=runtime,
    )


def compute_edf_curve(
    periods: Sequence[float],
    dt: float = 0.0001,
    impulse_width: float = 0.1,
    impulse_amplitude: float = 50.0,
    spike_threshold: float = 50.0,
    duration_multiplier: float = 5.0,
    min_t: float = 1000.0,
    neuron_params: dict[str, float] | None = None,
    synapse_params: dict[str, float] | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
) -> list[EDFSample]:
    """Compute eDF across a sequence of input periods."""
    fn = partial(
        compute_edf_single_period,
        dt=dt, impulse_width=impulse_width, impulse_amplitude=impulse_amplitude,
        spike_threshold=spike_threshold, duration_multiplier=duration_multiplier,
        min_t=min_t, neuron_params=neuron_params, synapse_params=synapse_params,
    )
    if parallel and len(periods) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(fn, periods))
    else:
        results = [fn(p) for p in periods]
    results.sort(key=lambda s: s.input_period)
    return results


def compute_edf_adaptive(
    p_min: float = 10.0,
    p_max: float = 200.0,
    coarse_step: float = 5.0,
    medium_step: float = 0.5,
    fine_step: float = 0.05,
    dt: float = 0.0001,
    parallel: bool = True,
    max_workers: int | None = None,
    **sim_kwargs,
) -> list[EDFSample]:
    """Multi-stage adaptive sampling to resolve T_min and curve boundaries accurately."""
    cache: dict[float, EDFSample] = {}

    def get_aligned_periods(start: float, stop: float, step: float) -> np.ndarray:
        raw_periods = np.arange(start, stop + step * 0.1, step)
        aligned_periods = np.round(raw_periods / dt) * dt
        valid_idx = (aligned_periods >= p_min) & (aligned_periods <= p_max)
        return np.unique(aligned_periods[valid_idx])

    def evaluate_batch(periods_to_run: Sequence[float]):
        needed = [p for p in periods_to_run if p not in cache]
        if not needed:
            return
        results = compute_edf_curve(
            periods=needed, dt=dt, parallel=parallel, max_workers=max_workers, **sim_kwargs
        )
        for r in results:
            cache[r.input_period] = r

    p_coarse = get_aligned_periods(p_min, p_max, coarse_step)
    evaluate_batch(p_coarse)

    valid_samples = [s for s in cache.values() if s.is_valid and s.onset_delay is not None]
    if not valid_samples:
        return [cache[k] for k in sorted(cache.keys())]

    valid_samples.sort(key=lambda x: x.input_period)
    t_min_approx = valid_samples[0].input_period

    delta_inf = valid_samples[-1].onset_delay
    delay_span = max(s.onset_delay for s in valid_samples) - min(s.onset_delay for s in valid_samples)
    t_r_approx = p_max

    if delay_span > 1e-4:
        for s in valid_samples:
            if abs(s.onset_delay - delta_inf) < 0.02 * delay_span:
                t_r_approx = max(t_min_approx + coarse_step, s.input_period)
                break

    p_medium = get_aligned_periods(
        max(p_min, t_min_approx - coarse_step), t_r_approx + coarse_step, medium_step
    )
    evaluate_batch(p_medium)

    valid_samples = [s for s in cache.values() if s.is_valid and s.onset_delay is not None]
    valid_samples.sort(key=lambda x: x.input_period)
    t_min_approx = valid_samples[0].input_period

    p_fine = get_aligned_periods(
        max(p_min, t_min_approx - medium_step * 2), t_min_approx + medium_step * 4, fine_step
    )
    evaluate_batch(p_fine)

    return [cache[k] for k in sorted(cache.keys())]


def extract_edf_metrics(
    samples: list[EDFSample], deriv_threshold: float = 0.001
) -> dict[str, float | None]:
    """Extract T_min, T_r (settling period), and delta_inf metrics."""
    valid_samples = [s for s in samples if s.is_valid and s.onset_delay is not None]
    if not valid_samples:
        return {"t_min": None, "t_r": None, "delta_inf": None}

    valid_samples.sort(key=lambda s: s.input_period)

    periods = np.array([s.input_period for s in valid_samples], dtype=float)
    delays = np.array([s.onset_delay for s in valid_samples], dtype=float)

    t_min = float(np.min(periods))
    delta_inf = float(delays[-1])

    if len(periods) > 1:
        d_delta_dT = np.gradient(delays, periods)
        abs_deriv = np.abs(d_delta_dT)
        below_thresh = abs_deriv <= deriv_threshold
        sustained_settled = np.cumprod(below_thresh[::-1])[::-1].astype(bool)

        if np.any(sustained_settled):
            t_r = float(periods[sustained_settled][0])
        else:
            t_r = float(periods[-1])
    else:
        t_r = t_min

    return {"t_min": t_min, "t_r": t_r, "delta_inf": delta_inf}
