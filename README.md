# RA-CLHPS for Dynamic Seru Scheduling

This repository contains the disclosed simulator, policy-search code, and numerical results for **Regime-Adaptive Certified-Lazy Hierarchical Policy Search for Dynamic Seru Scheduling with Workforce Capability Development**.

RA-CLHPS combines two mechanisms:

1. **Regime-adaptive hierarchical policy search**, which changes dispatch and worker-assignment pressures as congestion and capability coverage evolve.
2. **Certified lazy score evaluation**, which skips expensive projected-work calculations only when a valid bound proves that they cannot change the selected action.

The numerical evidence is synthetic and is not calibrated to an identified factory. The repository therefore supports computational reproducibility, not a claim of immediate factory-level external validity.

## Repository structure

- `code/adaptive_lazy_seru_experiment.py`: simulator, cross-entropy policy search, baselines, ablations, paired tests, and main experiments.
- `code/extended_validation.py`: independent-training replications, out-of-distribution transfer, gain and objective-weight sensitivity, and scale experiments through 400 orders.
- `results/`: raw paired observations, summaries, hypothesis tests, learned parameters, metadata, and generated figures.
- `results/extended/`: independent-training, transfer, sensitivity, and extended-scale outputs.

## Environment

- Python 3.11 or later
- NumPy 2.2.2 (the reported runs used NumPy 2.2.2)

Install the disclosed dependency with:

```bash
python3 -m pip install -r requirements.txt
```

## Reproduce the main experiment

From the repository root, run:

```bash
python3 code/adaptive_lazy_seru_experiment.py --output results_reproduced
```

To evaluate the learned parameters stored with the reported run:

```bash
python3 code/adaptive_lazy_seru_experiment.py \
  --parameter-file results/metadata.json \
  --output results_reproduced
```

## Reproduce the extended validation

```bash
python3 code/extended_validation.py \
  --parameter-file results/metadata.json \
  --main-raw results/raw_results.csv \
  --output results_extended_reproduced
```

The scripts use fixed seeds and retain raw observations. For every evaluated instance, they verify that complete Adaptive-HPS and certified-lazy evaluation produce identical action traces and terminal quality metrics. Execution stops if an equality check fails.

For backward compatibility with the first experiment script, the CSV label `Adaptive-CLHPS` denotes the manuscript's RA-CLHPS policy.

## Interpretation boundary

Adaptation improves the prespecified composite objective in the main paired experiment, but the result varies across independently trained policies, initial-capability conditions, and managerial objective weights. The certified-lazy mechanism has a stronger guarantee: it returns exactly the same decisions as complete score evaluation while avoiding a substantial share of expensive projected-work calculations. Industrial calibration and common-budget comparisons with PPO, model-predictive control, and rolling-horizon optimization remain future work.

