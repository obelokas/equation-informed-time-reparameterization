# Equation-Informed Time Reparameterization for Physics-Informed Neural Networks

Code and data accompanying the manuscript *Equation-Informed Time Reparameterization for Physics-Informed Neural
Networks* by Orestis Belokas and Christoforos Rekatsinas (INSANE Research Lab, Institute of Informatics and
Telecommunications, NCSR "Demokritos").

The method places the PINN's collocation times with a warp s(t) = ∫ rate(t′)^β dt′ built from the equation itself:
rate(t) is a growth rate of the linearised operator, estimated from a cheap pilot training (and, where used, refined
by bootstrap rounds), never from a reference solution. The full training then solves the equation in the warped time
σ = s(t)/s_max, with collocation times uniform in σ.

This repository contains one plain, commented script per stage and per configuration reported in the paper. It holds
no sweeps, competitors or ablations.

## Repository layout

```
allen_cahn/                      Allen-Cahn (pilot -> bootstrap round 1 -> bootstrap round 2 -> full training)
burgers/                         viscous Burgers (pilot -> full training)
kachanov_rabotnov/psi_1e-3/      Kachanov-Rabotnov creep damage, margin psi = 1e-3 (pilot -> full training)
kachanov_rabotnov/psi_1e-4/      margin psi = 1e-4 (pilot -> bootstrap round 1 -> bootstrap round 2 -> full training)
frank_kamenetskii/psi_3e-3/      Frank-Kamenetskii thermal explosion, margin psi = 3e-3 (pilot -> full training)
frank_kamenetskii/psi_1e-5/      margin psi = 1e-5 (pilot -> full training)
```

Each case folder holds:

| file | content |
|---|---|
| `*_common.py` | the equation, network, warp, training loop and rate formula shared by the case's scripts |
| `pilot_training.py` | the pilot training at its kept seed; writes the first rate file |
| `bootstrap_round{1,2}.py` | the bootstrap rounds at their kept seeds (Allen-Cahn, Kachanov-Rabotnov psi = 1e-4) |
| `full_training_*.py` | one script per configuration (N_x × N_t) at the best final seed |
| `rates/` | the rate files the paper's runs actually used (outputs of the pilot training and bootstrap rounds) |
| `data/` + `reference.py` | the reference solution used only for scoring, and the script that builds it |

Kachanov-Rabotnov has a closed-form solution (in `kr_common.py`), so it has no `data/` folder.

## Installation

Python 3.10 or newer, with PyTorch, NumPy and SciPy:

```
pip install -r requirements.txt
```

The paper's runs used PyTorch 2.8.0 (CUDA 12.8) on cloud machines and PyTorch 2.12.1 on an Apple M3 (Kachanov-Rabotnov
psi = 1e-4 full trainings). Every script runs on a CPU. The GPU runs (see below) also run on a CPU, with slightly
different numbers.

## Quickstart

Every script runs from its own folder with no arguments, for example:

```
cd kachanov_rabotnov/psi_1e-4
python full_training_Nt256.py
```

By default the full training reads the rate file stored in `rates/`, the one the paper used, so it reproduces the
paper's configuration directly. To run the whole pipeline yourself:

```
python pilot_training.py                                   # -> outputs/pilot_seed1008.npz
python bootstrap_round1.py --rate-file outputs/pilot_seed1008.npz
python bootstrap_round2.py --rate-file outputs/bootstrap1_seed2008.npz
python full_training_Nt256.py --rate-file outputs/bootstrap2_seed3001.npz
```

Outputs (rate files, predictions, errors) go to `outputs/` inside the folder. The Frank-Kamenetskii and Allen-Cahn
scripts take `--device cpu|cuda`.

## Pipeline and seeds

1. **Pilot training.** A small PINN in physical time on a coarse grid. The paper ran 10 seeds (1000-1009) and kept
   the run with the lowest final training loss (its own loss, no reference). The scripts use that kept seed.
2. **Bootstrap rounds** (Allen-Cahn; Kachanov-Rabotnov at psi = 1e-4). A warped training on the previous rate file
   whose trained solution re-derives the rate. Each round ran 10 seeds (2000-2009, then 3000-3009) on the kept input
   file, and the run with the lowest final training loss was kept, again without the reference.
3. **Full training.** The paper reports the median over ten seeds (90-99). Each script runs the best of those ten
   seeds by relative L2 error, a choice that uses the reference, so it serves only as a demonstration run. The
   docstring and the table below give both numbers.

The reference solution is used only to score a full training. It never enters a pilot training, a bootstrap round or
the rate.

## Results (rate method, relative L2 error)

| case | configuration | rate chain | seed | this seed | 10-seed median (paper) |
|---|---|---|---|---|---|
| Allen-Cahn | 256 × 24 | pilot → round 1 → round 2 | 91 | 0.15% | 0.245% |
| Allen-Cahn | 256 × 20 | pilot → round 1 → round 2 | 99 | 0.21% | 0.29% |
| Allen-Cahn | 256 × 16 | pilot → round 1 → round 2 | 94 | 0.13% | 0.30% |
| Burgers | 128 × 4 | pilot | 97 | 2.06% | 2.86% |
| Burgers | 128 × 6 | pilot | 91 | 1.79% | 3.02% |
| Burgers | 512 × 32 | pilot | 91 | 0.0425% | 0.0533% |
| Kachanov-Rabotnov, psi = 1e-3 | N_t = 256 | pilot | 93 | 0.0174% | 0.0395% |
| Kachanov-Rabotnov, psi = 1e-3 | N_t = 36 | pilot | 97 | 0.0246% | 0.206% |
| Kachanov-Rabotnov, psi = 1e-3 | N_t = 16 | pilot | 92 | 0.0538% | 0.151% |
| Kachanov-Rabotnov, psi = 1e-4 | N_t = 256 | pilot → round 1 → round 2 | 90 | 0.0126% | 0.0525% |
| Kachanov-Rabotnov, psi = 1e-4 | N_t = 36 | pilot → round 1 → round 2 | 92 | 0.00879% | 0.0879% |
| Kachanov-Rabotnov, psi = 1e-4 | N_t = 16 | pilot → round 1 → round 2 | 90 | 0.296% | 0.386% |
| Frank-Kamenetskii, psi = 3e-3 | 256 × 80 | pilot | 91 | 0.0420% (peak 0.221%) | 0.341% (peak 1.81%) |
| Frank-Kamenetskii, psi = 1e-5 | 256 × 80 | pilot | 90 | 6.89% (peak 43.6%) | 7.89% (peak 47.4%) |

Configuration = N_x × N_t collocation points (Kachanov-Rabotnov is an ODE: N_t collocation times). For
Frank-Kamenetskii the peak error is the relative error of max_x u at t_end.

## Reproducibility

Every run is seeded and, apart from the GPU runs, single-threaded. A run is bitwise reproducible on the same machine
type and library versions. Different CPUs or GPUs follow slightly different floating-point paths, and over thousands
of L-BFGS iterations those differences grow, so expect numbers close to the table, not identical. The paper's runs
used these machines:

| case | pilot / bootstrap rounds | full training |
|---|---|---|
| Allen-Cahn (float32) | pilot: AMD EPYC 7713 CPU; rounds: NVIDIA RTX A4000 GPU | N_t 24: EPYC 7763 CPU; N_t 20: EPYC 7713 CPU; N_t 16: NVIDIA RTX 3090 GPU |
| Burgers (float64) | AMD EPYC 9575F CPU | AMD EPYC 9575F CPU |
| Kachanov-Rabotnov psi = 1e-3 (float64) | AMD EPYC 7713 CPU | AMD EPYC 7713 CPU |
| Kachanov-Rabotnov psi = 1e-4 (float64) | AMD EPYC 9575F CPU | Apple M3 CPU |
| Frank-Kamenetskii (float64) | NVIDIA A100 GPU | NVIDIA A100 GPU |

On the GPU, Allen-Cahn initialises the network on the CPU, so CPU and GPU runs start from the same weights.
Frank-Kamenetskii initialises it on the GPU with deterministic kernels, so a CPU run starts from different weights.

## Citation

```
@article{belokas_rekatsinas_time_reparameterization,
  title  = {Equation-Informed Time Reparameterization for Physics-Informed Neural Networks},
  author = {Belokas, Orestis and Rekatsinas, Christoforos},
  note   = {Manuscript in preparation}
}
```
