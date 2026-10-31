# Kachanov-Rabotnov creep damage

dω/dt = A (1 − ω)^(−φ), ω(0) = 0, A = M σ^χ (GTD-111 at 816 °C, 455 MPa: M = 64.127·10⁻¹¹, χ = 2.257, φ = 3.792).
Rupture at t_f = 1/((φ + 1)A); training stops at t_end = (1 − ψ) t_f. The exact solution is
(1 − ω)^(φ+1) = 1 − (φ + 1)At, so this case needs no numerical reference.

Rate: rate(t) = A φ (1 − ω)^(−(φ+1)), the growth rate of the linearised equation, evaluated on the trained network's ω.

| ψ | stage | script | network | N_t | optimiser | β | seed | input → output |
|---|---|---|---|---|---|---|---|---|
| 1e-3 | pilot | `psi_1e-3/pilot_training.py` | 2 × 8, input t/t_end | 64 | Adam 1000 (lr 1e-3 → 1e-4) + L-BFGS 2000 | – | 1002 | → `rates/pilot_seed1002.npz` |
| 1e-3 | full | `psi_1e-3/full_training_Nt256.py` | 2 × 16, input σ | 256 | Adam 3000 (lr 5e-3 → 5e-4) + L-BFGS 2000 | 1.5 | 93 | pilot |
| 1e-3 | full | `psi_1e-3/full_training_Nt36.py` | 2 × 16 | 36 | same | 1.1 | 97 | pilot |
| 1e-3 | full | `psi_1e-3/full_training_Nt16.py` | 2 × 16 | 16 | same | 1.3 | 92 | pilot |
| 1e-4 | pilot | `psi_1e-4/pilot_training.py` | 2 × 8 | 64 | Adam 1000 (lr 1e-3 → 1e-4) + L-BFGS 2000 | – | 1008 | → `rates/pilot_seed1008.npz` |
| 1e-4 | round 1 | `psi_1e-4/bootstrap_round1.py` | 2 × 16 | 64 | Adam 1000 (lr 5e-3, constant) + L-BFGS 600 | 1.75 | 2008 | pilot → `rates/bootstrap1_seed2008.npz` |
| 1e-4 | round 2 | `psi_1e-4/bootstrap_round2.py` | 2 × 16 | 256 | Adam 3000 (lr 5e-3, constant) + L-BFGS 2000 | 1.75 | 3001 | round 1 → `rates/bootstrap2_seed3001.npz` |
| 1e-4 | full | `psi_1e-4/full_training_Nt256.py` | 2 × 16 | 256 | Adam 3000 (lr 5e-3 → 5e-4) + L-BFGS 2000 | 1.3 | 90 | round 2 |
| 1e-4 | full | `psi_1e-4/full_training_Nt36.py` | 2 × 16 | 36 | same | 1.25 | 92 | round 2 |
| 1e-4 | full | `psi_1e-4/full_training_Nt16.py` | 2 × 16 | 16 | same | 1.0 | 90 | round 2 |

Common settings: float64, one CPU thread; network ω = τ NN(τ) (tanh MLP) with the initial condition built in;
collocation times uniform in σ (in t/t_end for the pilot); L-BFGS with strong Wolfe line search, history 50,
tolerances effectively off. Each rate file stores ω and the rate on 1000 points uniform in ω, read from the trained
network on a time grid clustered toward t_end.
