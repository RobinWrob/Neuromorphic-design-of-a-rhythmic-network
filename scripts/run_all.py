"""
run_all.py - Master reproduction runner for all paper simulation scripts.

Allows executing individual figure scripts, specific paper sections, or the full
set of simulations sequentially.

Usage:
    python scripts/run_all.py --all
    python scripts/run_all.py --section 2
    python scripts/run_all.py --figure 1
    python scripts/run_all.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FIGURE_SCRIPTS = {
    1: ("Section 2", "fig1_exc_vs_osc.py", "Excitable vs Oscillatory single node"),
    2: ("Section 2", "fig2_ring_voltage.py", "Ring network voltage traces"),
    3: ("Section 2", "fig3_motivating_example.py", "Ring network raster and delays"),
    4: ("Section 2", "fig4_robustness.py", "Ring network robustness sweep"),
    6: ("Section 3", "fig6_edf_example.py", "Single-neuron eDF curve"),
    7: ("Section 3", "fig7_edf_metrics_sweep.py", "eDF parameter metrics sweep"),
    8: ("Section 3", "fig8_network_period.py", "Ring network period sweeps"),
    9: ("Section 3", "fig9_endogenous_tuning_hom_het.py", "Endogenous tuning in ring networks"),
    10: ("Section 3", "fig10_compare_fixed_period_vs_fixed_node.py", "Fixed period vs fixed node comparison"),
    11: ("Section 4", "fig11_eprc_example.py", "Single-neuron ePRC curves & traces"),
    12: ("Section 4", "fig12_entrainment.py", "Ring entrainment & Arnold Tongue"),
    13: ("Section 4", "fig13_eprc_tuning.py", "ePRC parameter sweeps"),
    14: ("Section 4", "fig14_eprc_comodulation.py", "ePRC co-modulation sweep"),
    15: ("Section 4", "fig15_exo_tuning_hom_het.py", "Exogenous tuning in ring networks"),
}


def run_script(fig_num: int) -> bool:
    section, filename, description = FIGURE_SCRIPTS[fig_num]
    script_path = SCRIPT_DIR / filename
    print(f"\n{'='*75}")
    print(f"Executing Figure {fig_num} ({section}): {description}")
    print(f"Script: scripts/{filename}")
    print(f"{'='*75}\n", flush=True)

    start_t = time.perf_counter()
    res = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - start_t

    if res.returncode == 0:
        print(f"\n[SUCCESS] Figure {fig_num} completed in {elapsed:.2f} s.\n")
        return True
    else:
        print(f"\n[FAILURE] Figure {fig_num} exited with code {res.returncode}.\n", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run paper reproduction simulation scripts for Neuromorphic Node Codebase."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Execute all paper figure simulations.")
    group.add_argument("--section", type=int, choices=[2, 3, 4], help="Execute all figures in a given paper section (2, 3, or 4).")
    group.add_argument("--figure", type=int, choices=list(FIGURE_SCRIPTS.keys()), help="Execute a single figure by number.")
    group.add_argument("--list", action="store_true", help="List all available figure scripts.")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Paper Figure Scripts:")
        print(f"{'Fig #':<8}{'Section':<12}{'Script Filename':<45}{'Description'}")
        print("-" * 90)
        for num, (sec, fname, desc) in sorted(FIGURE_SCRIPTS.items()):
            print(f"{num:<8}{sec:<12}{fname:<45}{desc}")
        print("\nNote: Figure 5 is a conceptual diagram in the paper without a Python script.\n")
        return

    if args.figure:
        targets = [args.figure]
    elif args.section:
        targets = [num for num, (sec, _, _) in FIGURE_SCRIPTS.items() if sec == f"Section {args.section}"]
    else:
        targets = sorted(FIGURE_SCRIPTS.keys())

    total_start = time.perf_counter()
    successes = []
    failures = []

    for num in targets:
        ok = run_script(num)
        if ok:
            successes.append(num)
        else:
            failures.append(num)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*75}")
    print(f"Summary: {len(successes)} succeeded, {len(failures)} failed.")
    print(f"Total Wall Clock Time: {total_elapsed:.2f} s ({total_elapsed/60.0:.2f} min).")
    print(f"{'='*75}\n")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
