# Approximate Gaussian Process Constructions for DeepRV Pretraining

This repository accompanies the MSc Statistics thesis *Approximate Gaussian Process Constructions for DeepRV Pretraining: Effects on Posterior Inference and Computational Cost*. It studies how Exact, Bilinear, Cubic, DTC and FITC Gaussian-process constructions used during DeepRV pretraining affect posterior inference and computational cost.

## Contents

- `notebooks/01_small_grid_foundations_8x8.ipynb`: 8 x 8 model foundations and the optional MCMC-budget comparison.
- `notebooks/02_support_and_spacing_16x16.ipynb`: 16 x 16 support and inducing-spacing comparisons.
- `notebooks/03_systematic_comparison_32x32.ipynb`: systematic DeepRV, Full GP and matched Direct GP comparisons at 32 x 32.
- `notebooks/04_scaled_comparison_64x64.ipynb`: the corresponding scaled comparison at 64 x 64.
- `notebooks/05_target128_scaling_frontier.ipynb`: Exact128 and Grid-64 DeepRV runs, plus the Full GP128 feasibility probe.
- `experiments/`: simulation, DeepRV pretraining, frozen-decoder inference, matched Direct GP inference, diagnostics, metrics and runtime accounting.
- `analysis/figures/` and `analysis/tables/`: scripts used to generate the thesis figures and tables from experiment outputs.

The public datasets are generated with Seed 0, Seed 1 and Seed 2. The 32 x 32 and 64 x 64 experiments use two NUTS chains, 1,000 warmup iterations per chain and 4,000 retained draws per chain. Target-128 uses one chain, 4,000 warmup iterations and 6,000 retained draws. The Full GP128 entry point is a feasibility and runtime probe only.

## Environment and installation

All five notebooks use the same Python 3.12 environment and are intended for an NVIDIA A100 GPU. From a fresh virtual environment:

```bash
python -m pip install -r requirements.txt
```

DeepRV is supplied by the pinned upstream `dl4bi` dependency; the upstream repository is not duplicated here. Spatial GP utilities are supplied by `dl4bi-sps`.

## Running experiments

Run one of the five notebooks from the repository root. Each notebook calls the corresponding scripts in `experiments/` and exposes controls for its experiment stages.

Experiment outputs are generated at runtime. Checkpoints, posterior samples and generated result files are intentionally not included. The larger experiments require substantial GPU memory and runtime; no training or NUTS run is needed to inspect the source.
