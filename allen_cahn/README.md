# Allen-Cahn

u_t = 10⁻⁴ u_xx + 5 (u − u³), x ∈ [−1, 1] periodic, t ∈ [0, 1], u(x, 0) = x² cos(πx).

Rate: rate(t) = mean_x max(0, 5(1 − 3u²) − 10⁻⁴ u_x²), floored at 1% of its maximum. The final rate comes from two
bootstrap rounds after the pilot training.

| stage | script | network | grid N_x × N_t | optimiser | β | seed | input → output |
|---|---|---|---|---|---|---|---|
| pilot | `pilot_training.py` | 2 × 20, input (x, t) | 250 × 32 | Adam 1000 (lr 5e-3) + L-BFGS 600 | – | 1002 | → `rates/pilot_seed1002.npz` |
| round 1 | `bootstrap_round1.py` | 4 × 32, sign envelope, input (x, σ) | 60 × 8 | L-BFGS 1000 | 0.1 | 2000 | pilot → `rates/bootstrap1_seed2000.npz` |
| round 2 | `bootstrap_round2.py` | 4 × 32, sign envelope | 128 × 16 | L-BFGS 2500 | 0.1 | 3002 | round 1 → `rates/bootstrap2_seed3002.npz` |
| full | `full_training_Nt24.py` | 4 × 64 | 256 × 24 | L-BFGS 10000 | 0.758644 | 91 | round 2 |
| full | `full_training_Nt20.py` | 4 × 64 | 256 × 20 | L-BFGS 10000 | 1.011526 | 99 | round 2 |
| full | `full_training_Nt16.py` | 4 × 64 | 256 × 16 | L-BFGS 10000 | 0.758644 | 94 | round 2 |

Common settings: float32; tanh MLP on [cos(kπx), sin(kπx) for k = 1..10, time]; initial condition as a penalty with
weight 50; L-BFGS with strong Wolfe line search, history 50, tolerances 1e-9 / 1e-12. In the full training,
β = κ / ln R with R = 10.73 (max/min of the final rate) and κ = 1.8 (N_t 24, 16) or 2.4 (N_t 20). The warped residual
is the physical-time residual multiplied by rate^β / geomean(rate^β) over the columns.

Reference (`reference.py`, `data/reference_200x200.npz`): Fourier pseudo-spectral method with 512 modes, BDF time
stepping (rtol 1e-7, atol 1e-9), cubic interpolation onto the 200 × 200 scoring grid.

## Running

Each script is a complete, flat PyTorch script with no arguments and no shared modules. Run it from anywhere, e.g.

```
cd allen_cahn
python full_training_Nt24.py
```

To change a setting (seed, β, grid, budgets, network size, device, input file), edit the variables at the top of the
script, under `# hyperparams`. The bootstrap rounds and the full trainings read the stored paper files in `rates/`;
point `rate_file` at `outputs/pilot_seed1002.npz`, `outputs/bootstrap1_seed2000.npz` or
`outputs/bootstrap2_seed3002.npz` to chain your own runs. Results go to `outputs/`, figures to `plots/` (both inside
the script's folder). `python reference.py` rebuilds the reference in `data/`.
