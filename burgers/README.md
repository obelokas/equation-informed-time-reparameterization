# Viscous Burgers

u_t + u u_x = (0.01/π) u_xx, x ∈ [−1, 1], t ∈ [0, 1], u(x, 0) = −sin(πx), u(±1, t) = 0.

Rate: rate(t) = median_x (|u_x| + ν u_x²), from the pilot training (no bootstrap rounds).

| stage | script | grid N_x × N_t | optimiser | β | seed | input → output |
|---|---|---|---|---|---|---|
| pilot | `pilot_training.py` | 128 × 16 (uniform t) | Adam 1000 (lr 5e-3, constant) + L-BFGS 600 | – | 1000 | → `rates/pilot_seed1000.npz` |
| full | `full_training_Nx128_Nt4.py` | 128 × 4 | Adam 4500 (lr 5e-3 → 5e-4) + L-BFGS 3000 | 0.85 | 97 | pilot |
| full | `full_training_Nx128_Nt6.py` | 128 × 6 | Adam 4500 + L-BFGS 3000 | 0.85 | 91 | pilot |
| full | `full_training_Nx512_Nt32.py` | 512 × 32 | Adam 4500 + L-BFGS 3000 | 0.85 | 91 | pilot |

Common settings: float64; network u = −sin(πx) + τ(1 − x²) NN(x, sin/cos(2^k πx) for k < 4, τ), 3 hidden tanh
layers of 20 units; L-BFGS with strong Wolfe line search, history 50, tolerances effectively off. In the full
training the collocation times are the midpoints of a uniform σ grid, the residual is the physical-time residual
(rate^β/s_max) u_σ + u u_x − ν u_xx, and its square at column t_j is weighted by rate(t_j)^β / mean_j rate(t_j)^β.

Reference (`reference.py`, `data/reference_nx3201.npz`): method of lines on 3201 points, BDF (rtol 1e-7,
atol 1e-9), cubic interpolation onto the 100 × 100 scoring grid.
