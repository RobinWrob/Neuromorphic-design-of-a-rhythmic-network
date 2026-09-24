"""
NCE Neuromorphic Node Core Package.

Provides numerical solvers, biophysical node models, dynamic synapse kinetics,
Event Describing Function (eDF) engines, and Event Phase Response Curve (ePRC) engines.
"""

from src.edf_engine import compute_edf_adaptive, compute_edf_curve, extract_edf_metrics
from src.eprc_engine import compute_eprc_curve, run_eprc_sample, run_eprc_simulation_trace
from src.helper import (
    detect_threshold_crossings,
    export_to_csv,
    load_config,
    opheim_simplify_trace,
    plot_validation_trace,
    simulate_single_node,
    validate_1to1_phase_locking,
)

__all__ = [
    "compute_edf_adaptive",
    "compute_edf_curve",
    "extract_edf_metrics",
    "compute_eprc_curve",
    "run_eprc_sample",
    "run_eprc_simulation_trace",
    "detect_threshold_crossings",
    "export_to_csv",
    "load_config",
    "opheim_simplify_trace",
    "plot_validation_trace",
    "simulate_single_node",
    "validate_1to1_phase_locking",
]
