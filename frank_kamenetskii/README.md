# Frank-Kamenetskii thermal explosion

u_t = u_xx + δ e^u, δ = 1.3177, x ∈ [−1, 1], u(x, 0) = 0, u(±1, t) = 0. The solution blows up at
T* = 1.4467001610; training stops at t_end = (1 − ψ) T*.

Rate (net growth coefficient): rate(t) = mean_x |δ e^u − u_x²|, from the pilot training; warp density
ρ = rate + 10⁻⁶ (the constant only keeps ρ > 0).

| ψ | stage | script | network | grid N_x × N_t | optimiser | β | seed | input → output |
|---|---|---|---|---|---|---|---|---|
| 3e-3 | pilot | `psi_3e-3/pilot_training.py` | 3 × 16, input (x, t/t_end) | 32 × 32 | Adam 1000 (lr 3e-3, constant) | – | 1001 | → `rates/pilot_seed1001.npz` |
| 3e-3 | full | `psi_3e-3/full_training_Nx256_Nt80.py` | 4 × 64, input (x, σ) | 256 × 80 | L-BFGS 10000 | 1.0 | 91 | pilot |
| 1e-5 | pilot | `psi_1e-5/pilot_training.py` | 3 × 16 | 32 × 32 | Adam 500 (lr 3e-3, constant) | – | 1001 | → `rates/pilot_seed1001.npz` |
| 1e-5 | full | `psi_1e-5/full_training_Nx256_Nt80.py` | 4 × 64 | 256 × 80 | L-BFGS 10000 | 1.0 | 90 | pilot |

Common settings: float64; network u = τ(1 − x²) NN(x, τ) (tanh MLP), which satisfies the initial and boundary
conditions; the two corner points (x = ±1, τ = 0) are removed; full-training collocation times uniform in σ, residual
u_σ − (s_max/ρ^β)(u_xx + δe^u); L-BFGS with strong Wolfe line search, history 50, tolerances effectively off. The
paper ran these on an NVIDIA A100 with deterministic kernels (the default device when CUDA is available).

Scoring: relative L2 error on 2000 × 200 uniform points; peak error = relative error of max_x u at t_end.

Reference (`reference.py`, `data/ref_eval_*.npz`, `data/ref_evalc_*.npz`): the solution is even in x, so it is solved
on [0, 1] with second-order finite differences (h = 2.5·10⁻⁴) and BDF (rtol 1e-11), stored on 1001 x-nodes by 1501
t-nodes clustered toward t_end, and read through a bicubic spline. The blow-up time comes from integrating until
max u = 14.

## Running

Each script is a complete, flat PyTorch script with no arguments and no shared modules. Run it from anywhere, e.g.

```
cd frank_kamenetskii/psi_3e-3
python full_training_Nx256_Nt80.py
```

To change a setting (seed, β, grid, budgets, network size, device, input file), edit the variables at the top of the
script, under `# hyperparams`. The full trainings read the stored paper file `rates/pilot_seed1001.npz`; point
`rate_file` at `outputs/pilot_seed1001.npz` to use your own pilot output. Results go to `outputs/`, figures to
`plots/` (both inside the script's folder). `python reference.py` (in `frank_kamenetskii/`) rebuilds the reference in
`data/`, caching the fine solve there.
