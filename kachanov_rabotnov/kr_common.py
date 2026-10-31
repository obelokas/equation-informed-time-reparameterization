"""Kachanov-Rabotnov creep damage: shared building blocks for the pilot training, the bootstrap rounds and the full
training scripts in psi_1e-3/ and psi_1e-4/.

Problem (single-axis creep damage, Stewart et al. 2011, GTD-111 at 816 C and 455 MPa):

    d omega / dt = A (1 - omega)^(-phi),   omega(0) = 0,   A = M sigma^chi,

which ruptures (omega -> 1) at t_f = 1 / ((phi + 1) A). Training stops before the singularity, at
t_end = (1 - psi) t_f, where psi is the margin to rupture (the subcase folders psi_1e-3 / psi_1e-4).

The network represents omega directly, with the initial condition built in: omega(tau) = tau * NN(tau), where tau is
the normalised time t / t_end (pilot training) or the warped time sigma in [0, 1] (bootstrap rounds, full training).

The rate is the growth rate of the equation's linearisation, d/d omega of the right-hand side:
    rate(omega) = A phi (1 - omega)^(-(phi + 1)).
The warp is s(t) = int_0^t rate(t')^beta dt', sigma = s / s_max. In sigma the equation reads
    d omega / d sigma = s_max A (1 - omega)^(-phi) / rate(t)^beta,
which is the residual the warped trainings minimise ('div' form).

Everything runs in float64 on one CPU thread with a fixed seed, so a run is bitwise reproducible on one machine type.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

# ---- material constants (Stewart et al. 2011, Table 3) ----
M = 64.127 * 10**(-11)      # MPa^-chi h^-1
SIGMA = 455.0               # applied stress, MPa
CHI = 2.257
PHI = 3.792

A = M * SIGMA**CHI                  # right-hand side prefactor [1/h]
T_F = 1.0 / ((PHI + 1) * A)         # rupture time [h]


class Problem:
    """Time interval and fixed grids for one margin psi (t_end = (1 - psi) t_f)."""

    def __init__(self, psi):
        self.psi = float(psi)
        self.t_end = (1 - self.psi) * T_F
        self.t_fine = np.linspace(0.0, self.t_end, 4001)    # grid for the warp's inverse
        self.t_eval = np.linspace(0.0, self.t_end, 1001)    # scoring grid (relative L2 against the exact solution)


# ---- exact solution (separable ODE: (1 - omega)^(phi + 1) = 1 - (phi + 1) A t) ----
def omega_exact(t):
    return 1.0 - (1.0 - (PHI + 1) * A * np.asarray(t, dtype=np.float64))**(1.0 / (PHI + 1))


def rate_of_omega(w):
    """Growth rate of the linearised equation, A phi (1 - omega)^(-(phi + 1))."""
    return A * PHI * (1.0 - w)**(-(PHI + 1))


def one_minus(u):
    """1 - omega, clamped only to keep a wild optimiser step from producing NaN."""
    return torch.clamp(1.0 - u, min=1e-6)


# ---- network ----
class Net(nn.Module):
    """omega(tau) = tau * NN(tau): tanh MLP, `depth` hidden layers of `width` units; omega(0) = 0 exactly."""

    def __init__(self, width, depth):
        super().__init__()
        layers, d = [], 1
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tau):
        return tau * self.net(tau)


def setup(seed):
    """One CPU thread, fixed seeds, float64 default (call before the network is built)."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)


# ---- warp ----
class Warp:
    """s(t) = int_0^t rate^beta dt (cubic spline of rate^beta and its exact antiderivative), sigma = s / s_max."""

    def __init__(self, problem, t_grid, rate_grid, beta):
        self.problem = problem
        r = np.asarray(rate_grid, dtype=np.float64) ** beta
        self.rate_fn = CubicSpline(t_grid, r)           # rate(t)^beta
        self.s_of_t = self.rate_fn.antiderivative()
        self.s_max = float(self.s_of_t(t_grid[-1]))
        self.t_end = float(t_grid[-1])
        if not np.all(self.rate_fn(problem.t_fine) > 0):
            raise ValueError("rate^beta spline is not strictly positive -> the warp is not monotone")
        self._s_fine = self.s_of_t(problem.t_fine)

    def sigma_of_t(self, t):
        return self.s_of_t(t) / self.s_max

    def t_of_sigma(self, sig):
        """Inverse warp: interpolation on the fine grid, then 3 Newton steps."""
        target = np.asarray(sig, dtype=np.float64) * self.s_max
        t = np.interp(target, self._s_fine, self.problem.t_fine)
        for _ in range(3):
            t = t - (self.s_of_t(t) - target) / self.rate_fn(t)
            t = np.clip(t, 0.0, self.t_end)
        return t


def load_rate(path):
    d = np.load(path)
    return d["t"], d["rate"]


# ---- training ----
def train(model, tau, residual_fn, adam_epochs, lbfgs_iters, lr, lr_decay, log_every=500):
    """Adam, then L-BFGS (strong Wolfe line search, tolerances effectively off, history 50), on mean(residual^2).
    lr_decay=True: the Adam learning rate decays exponentially to lr/10 over the Adam epochs."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1**(1.0 / max(adam_epochs, 1))) if lr_decay else None
    for ep in range(adam_epochs):
        opt.zero_grad()
        loss = torch.mean(residual_fn(model, tau)**2)
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        if log_every and ep % log_every == 0:
            print(f"  Adam {ep:5d}  loss {loss.item():.4e}", flush=True)
    n_eval = [0]
    if lbfgs_iters > 0:
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters, tolerance_grad=1e-15,
                               tolerance_change=1e-16, history_size=50, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            l = torch.mean(residual_fn(model, tau)**2)
            l.backward()
            n_eval[0] += 1
            return l
        lb.step(closure)
    final_loss = float(torch.mean(residual_fn(model, tau)**2).item())
    print(f"  final loss {final_loss:.4e}  (L-BFGS loss evaluations: {n_eval[0]})", flush=True)
    return final_loss


# ---- rate writer (pilot training and bootstrap rounds; never sees the exact solution) ----
def clustered_t(problem, n_log=3000, n_uni=2000):
    """Evaluation times: 2000 uniform points plus t_end - t log-spaced down to 1e-7 t_end (dense near rupture)."""
    t = np.concatenate([np.linspace(0.0, problem.t_end, n_uni),
                        problem.t_end - problem.t_end * np.logspace(-7, 0, n_log)])
    return np.unique(np.clip(t, 0.0, problem.t_end))


def write_rate(problem, omega_fn):
    """Rate file from a trained network: omega_fn(t) -> omega at physical times t.
    1. evaluate omega on the clustered grid; it must be finite, non-decreasing and < 1 (else the pass is invalid);
    2. resample onto 1000 points uniform in omega (t from monotone interpolation of t(omega));
    3. rate = rate_of_omega(omega) on those points.
    Returns the arrays to save, or raises ValueError."""
    tc = clustered_t(problem)
    om = omega_fn(tc)
    if not np.all(np.isfinite(om)):
        raise ValueError("omega is not finite")
    if np.any(np.diff(om) < 0):
        raise ValueError("omega is not monotone")
    if np.any(om >= 1):
        raise ValueError("omega >= 1")
    om_u = np.linspace(om[0], om[-1], 1000)
    keep = np.concatenate([[True], np.diff(om) > 0])        # plateaus collapse to their first t
    t_u = np.interp(om_u, om[keep], tc[keep])
    t_u[0], t_u[-1] = 0.0, problem.t_end
    if np.any(np.diff(t_u) <= 0):
        raise ValueError("t(omega) is not strictly increasing")
    return dict(t=t_u, omega=om_u, rate=rate_of_omega(om_u))


def relative_l2(pred, exact):
    return float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))
