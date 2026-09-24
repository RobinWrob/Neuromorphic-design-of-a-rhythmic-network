# Neuromorphic Node Dynamics and Event-Based Phase Regulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official simulation codebase and reproducible numerical experiments for the research paper submitted to *Neuromorphic Computing and Engineering* (NCE).

The framework models single-node biophysical dynamics (fast sodium, slow potassium, leak, and A-type rebound currents), dynamic synaptic transmission, event-driven ring networks, and tools for design and analysis:
- **Event Describing Functions (eDF)**: Characterizing steady-state onset delays $\delta(T)$.
- **Event Phase Response Curves (ePRC)**: Characterizing timing shifts $\Delta t(t_p)$ induced by periodic synaptic events.
- **Biomimetic Ring Networks**: Exploring event propagation, parametric heterogeneity, endogenous frequency modulation, and sensory entrainment (Arnold Tongues).

---

## Repository Structure

```
NCE_P2_FINAL/
├── configs/
│   ├── params_neuron.yaml              # Default biophysical parameters (Na, K, L, Na_reb, A)
│   ├── params_synapse.yaml             # Default dynamic synapse parameters (inh / exc)
│   └── sweep_configs.yaml              # Default numerical integration & sweep parameters
├── src/
│   ├── __init__.py                     # Package interface
│   ├── helper.py                       # Solvers (Euler/RK4 JIT), event detection, CSV exporters
│   ├── edf_engine.py                   # Event Describing Function simulation engine
│   └── eprc_engine.py                  # Event Phase Response Curve simulation engine
├── scripts/
│   ├── run_all.py                      # Master reproduction runner CLI
│   ├── fig1_exc_vs_osc.py              # Figure 1: Excitable vs. oscillatory single node
│   ├── fig2_ring_voltage.py            # Figure 2: Excitable ring network voltage traces
│   ├── fig3_motivating_example.py      # Figure 3: Ring network raster & delays under perturbation
│   ├── fig4_robustness.py              # Figure 4: Network robustness against parametric heterogeneity
│   ├── fig6_edf_example.py             # Figure 6: Single-neuron eDF curve & asymptotic delay
│   ├── fig7_edf_metrics_sweep.py       # Figure 7: eDF metrics sweeps over g_A and tau_decay
│   ├── fig8_network_period.py          # Figure 8: Ring network period sweeps (global vs local)
│   ├── fig9_endogenous_tuning_hom_het.py # Figure 9: Endogenous tuning in homogeneous/heterogeneous rings
│   ├── fig10_compare_fixed_period_vs_fixed_node.py # Figure 10: Fixed period vs fixed node modulation
│   ├── fig11_eprc_example.py           # Figure 11: Single-neuron ePRC curve & voltage traces
│   ├── fig12_entrainment.py            # Figure 12: Sensory entrainment & dynamic Arnold Tongue
│   ├── fig13_eprc_tuning.py            # Figure 13: ePRC parameter sweeps over g_A and tau_decay
│   ├── fig14_eprc_comodulation.py      # Figure 14: ePRC co-modulation sweep (shape preservation)
│   └── fig15_exo_tuning_hom_het.py     # Figure 15: Exogenous entrainment phase tuning
├── results/
│   ├── data/                           # Destination for exported simulation CSV files
│   └── figures/                        # Destination for generated validation plots (.png)
├── plotdata/                           # Destination for TikZ wide-format sweep datasets
├── environment.yml                     # Conda environment definition
├── requirements.txt                    # Standard pip requirements specification
├── pyproject.toml                      # Build system & package specification
├── CITATION.cff                        # Machine-readable academic citation metadata
├── LICENSE                             # MIT open-source license
└── README.md                           # Documentation & reproduction guide
```

---

## Installation & Setup

### Option A: Using Conda (Recommended)

Create and activate the dedicated Conda environment:

```bash
conda env create -f environment.yml
conda activate nce_neuromorphic
```

### Option B: Using Standard Python Virtual Environment (`venv` / `pip`)

Ensure Python 3.10 or higher is installed, then set up a virtual environment:

```bash
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### Editable Development Installation (Optional)

To install the core simulation package (`src`) in editable mode:

```bash
pip install -e .
```

---

## Reproducing Paper Figures

All simulation scripts can be executed individually from the repository root, or in batches via the master runner `scripts/run_all.py`.

### Master Reproduction Runner

The `run_all.py` utility allows running all experiments or targeting specific paper sections:

```bash
# List all available figure scripts
python scripts/run_all.py --list

# Run all paper simulations
python scripts/run_all.py --all

# Run only figures from Section 2 (Figures 1-4)
python scripts/run_all.py --section 2

# Run only figures from Section 3 (Figures 6-10)
python scripts/run_all.py --section 3

# Run only figures from Section 4 (Figures 11-15)
python scripts/run_all.py --section 4

# Run a single figure script (e.g. Figure 1)
python scripts/run_all.py --figure 1
```

---

## Detailed Figure Guide

### Section 2: Node Biophysics & Ring Wave Propagation

#### Figure 1: Excitable vs. Oscillatory Dynamics
- **Script**: `python scripts/fig1_exc_vs_osc.py`
- **Description**: Evaluates single-neuron membrane dynamics under constant tonic current (oscillatory regime) versus discrete synaptic input pulses (excitable regime) using Hodgkin-Huxley kinetics with A-type potassium current.
- **Outputs**:
  - `results/data/neuron_simulation_data.csv`
  - `results/data/input_pulse_times.csv`
  - Validation Plot: `results/figures/fig1_exc_vs_osc.png`

#### Figure 2: Autonomous Wave Propagation in an Excitable Ring
- **Script**: `python scripts/fig2_ring_voltage.py`
- **Description**: Simulates autonomous traveling-wave propagation across a 4-neuron ring network with inhibitory rebound coupling following a kickstart pulse.
- **Outputs**:
  - `results/data/plotdata_excitable_voltage.csv`
  - Validation Plot: `results/figures/fig2_ring_voltage.png`

#### Figure 3: Locality & Transient Perturbations (Ring vs. Oscillator Network)
- **Script**: `python scripts/fig3_motivating_example.py`
- **Description**: Compares transient delay shifts and locality of synaptic perturbations between an excitable ring network and an all-to-all coupled oscillator network under mid-run parameter modulation.
- **Outputs**:
  - `results/data/plotdata_excitable_raster.csv`, `results/data/plotdata_excitable_delay.csv`
  - `results/data/plotdata_oscillatory_raster.csv`, `results/data/plotdata_oscillatory_delay.csv`
  - Validation Plot: `results/figures/fig3_motivating_example.png`

#### Figure 4: Network Robustness to Parametric Heterogeneity
- **Script**: `python scripts/fig4_robustness.py`
- **Description**: Performs a Monte Carlo heterogeneity sweep (400 trials per step) varying ionic conductances ($g_{\mathrm{Na}}$, $g_{\mathrm{K}}$) and synaptic decay ($\tau_{\mathrm{decay}}$) from 0% to 100%, quantifying sequence survival (Valid %) and 95% confidence intervals.
- **Outputs**:
  - `results/data/plotdata_excitable_robustness.csv`
  - `results/data/plotdata_oscillatory_robustness.csv`
  - Validation Plot: `results/figures/fig4_robustness.png`

> **Note on Figure 5**: Figure 5 in the paper is an architectural schematic diagram of the ring topology and inhibitory rebound mechanism, rendered directly in LaTeX/TikZ without a standalone simulation script.

---

### Section 3: Event Describing Function (eDF) Analysis

#### Figure 6: Single-Neuron eDF Curve & Asymptotic Delay
- **Script**: `python scripts/fig6_edf_example.py`
- **Description**: Evaluates steady-state spike onset delay $\delta(T)$ across presynaptic input periods $T$, computing the minimum phase-locking boundary $T_{\min}$ and asymptotic delay $\delta_\infty$.
- **Outputs**:
  - `results/data/plotdata_edf_single_curve.csv`
  - Validation Plot: `results/figures/fig6_edf_example.png`

#### Figure 7: eDF Critical Metric Sweeps ($g_A$ and $\tau_{\mathrm{decay}}$)
- **Script**: `python scripts/fig7_edf_metrics_sweep.py`
- **Description**: Sweeps intrinsic potassium conductance $g_A$ and synaptic decay time constant $\tau_{\mathrm{decay}}$ using adaptive eDF sampling, measuring parametric shifts in $T_{\min}$ and $\delta_\infty$.
- **Outputs**:
  - `results/data/plotdata_metrics_sweep_edf_neuron_g_a.csv`
  - `results/data/plotdata_metrics_sweep_edf_synapse_tau_decay.csv`
  - Validation Plot: `results/figures/fig7_edf_metrics_sweep.png`

#### Figure 8: Global vs. Local Endogenous Period Modulation
- **Script**: `python scripts/fig8_network_period.py`
- **Description**: Compares global network frequency tuning (varying all nodes simultaneously) versus local tuning (varying Node 0 alone) across both $g_A$ and $\tau_{\mathrm{decay}}$.
- **Outputs**:
  - `results/data/ring_sweep_neuron_g_a_node0.csv`, `results/data/ring_sweep_neuron_g_a_all_nodes.csv`
  - `results/data/ring_sweep_synapse_tau_decay_node0.csv`, `results/data/ring_sweep_synapse_tau_decay_all_nodes.csv`
  - Validation Plot: `results/figures/fig8_network_period.png`

#### Figure 9: Dynamic Endogenous Frequency Tuning (Homogeneous vs. Heterogeneous)
- **Script**: `python scripts/fig9_endogenous_tuning_hom_het.py`
- **Description**: Simulates real-time endogenous frequency modulation triggered at $T_{\mathrm{end}} / 2$, comparing all-node modulation against single-node modulation.
- **Outputs**:
  - `results/data/ring_network_voltage.csv`, `results/data/ring_network_delay.csv`
  - `results/data/ring_network_voltage_n0.csv`, `results/data/ring_network_delay_n0.csv`
  - Validation Plot: `results/figures/fig9_endogenous_tuning_hom_het.png`

#### Figure 10: Phase Profile & Duty Cycle Modulation
- **Script**: `python scripts/fig10_compare_fixed_period_vs_fixed_node.py`
- **Description**: Demonstrates independent phase-profile shaping using two complementary strategies: Approach 1 (fixed total cycle period $T_{\mathrm{tot}}$) and Approach 2 (fixed unmodulated node delay $\delta_i$).
- **Outputs**:
  - `results/data/plotdata_duty_modulation_approach1_raster.csv`, `results/data/plotdata_duty_modulation_approach1_delay.csv`, `results/data/plotdata_duty_modulation_approach1_reldelay.csv`
  - `results/data/plotdata_duty_modulation_approach2_raster.csv`, `results/data/plotdata_duty_modulation_approach2_delay.csv`, `results/data/plotdata_duty_modulation_approach2_reldelay.csv`
  - `results/data/fig10_duty_modulation_data.csv`
  - Validation Plot: `results/figures/fig10_compare_fixed_period_vs_fixed_node.png`

---

### Section 4: Event Phase Response Curves (ePRC) & Entrainment

#### Figure 11: Single-Neuron ePRC Curves & Perturbation Traces
- **Script**: `python scripts/fig11_eprc_example.py`
- **Description**: Measures output spike timing shifts $\Delta t(t_p)$ induced by excitatory and inhibitory sensory synaptic perturbations at delay offset $t_p$.
- **Outputs**:
  - `results/data/plotdata_eprc_exc_curve.csv`, `results/data/plotdata_eprc_inh_curve.csv`
  - `results/data/plotdata_eprc_exc_trace_tp5.csv`, `results/data/plotdata_eprc_inh_trace_tp5.csv`
  - `results/data/plotdata_eprc_events.csv`, `results/data/fig11_eprc_data.csv`
  - Validation Plot: `results/figures/fig11_eprc_example.png`

#### Figure 12: Sensory Entrainment & Arnold Tongue
- **Script**: `python scripts/fig12_entrainment.py`
- **Description**: Simulates external periodic sensory pulse driving of the excitable ring, extracting the 1:1 phase-locking synchronization region (Arnold Tongue) across period mismatch $\Delta T$ and sensory conductance $g_{\mathrm{syn}}$.
- **Outputs**:
  - `results/data/entrain_ring_spike_phase_data.csv`
  - `results/data/fig12_arnold_tongue_data.csv`
  - `plotdata/plotdata_arnold_tongue.csv`
  - Validation Plot: `results/figures/fig12_entrainment.png`

#### Figure 13: ePRC Systematic Parameter Sweeps
- **Script**: `python scripts/fig13_eprc_tuning.py`
- **Description**: Evaluates parametric reshaping of the ePRC curve across independent sweeps of primary synaptic decay $\tau_{\mathrm{decay}}$ and intrinsic conductance $g_A$.
- **Outputs**:
  - `results/data/fig13_eprc_tuning_data.csv`
  - `plotdata/plotdata_eprc_sweep_exc_synapse_nom_tau_decay.csv` (wide-format for TikZ)
  - `plotdata/plotdata_eprc_sweep_exc_neuron_g_a.csv` (wide-format for TikZ)
  - Validation Plot: `results/figures/fig13_eprc_tuning.png`

#### Figure 14: ePRC Co-Modulation (Shape Invariance)
- **Script**: `python scripts/fig14_eprc_comodulation.py`
- **Description**: Simultaneously co-varies $g_A$ and $\tau_{\mathrm{decay}}$ in reciprocal directions, demonstrating preservation of the ePRC shape and perturbation sensitivity.
- **Outputs**:
  - `results/data/plotdata_eprc_covaried_exc_neuron_g_a_vs_synapse_nom_tau_decay.csv`
  - `results/data/fig14_eprc_comodulation_data.csv`
  - Validation Plot: `results/figures/fig14_eprc_comodulation.png`

#### Figure 15: Exogenous Entrainment Phase Tuning
- **Script**: `python scripts/fig15_exo_tuning_hom_het.py`
- **Description**: Simulates steady-state phase shifts between sensory input events and ring spikes during continuous entrainment, comparing global homogeneous modulation against localized single-node modulation.
- **Outputs**:
  - `results/data/entrain_ring_modulation_homo.csv`
  - `results/data/entrain_ring_modulation_hetero.csv`
  - `results/data/fig15_exo_tuning_hom_het_data.csv`
  - Validation Plot: `results/figures/fig15_exo_tuning_hom_het.png`

---

## LaTeX / TikZ Plotting Workflow

The CSV files generated in `results/data/` and `plotdata/` correspond to the exact data tables imported by PGFPlots in the LaTeX manuscript:

1. Execute the relevant Python simulation script (e.g., `python scripts/fig6_edf_example.py`).
2. The script outputs cleaned, formatted numerical tables into `results/data/` (or `plotdata/`).
3. Compile your LaTeX manuscript with `pdflatex` or `lualatex`: PGFPlots will read the updated CSV files and render vector publication graphics.

---

## Citation

If you use this codebase or simulation framework in your research, please cite the paper.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
