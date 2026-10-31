"""Viscous Burgers: shared building blocks for pilot_training.py and the full_training_*.py scripts.

Problem:
    u_t + u u_x = nu u_xx,   nu = 0.01 / pi,   x in [-1, 1],   t in [0, 1],   u(x, 0) = -sin(pi x),   u(+-1, t) = 0.
A shock forms at t ~ 1/pi.

Network (initial and boundary conditions built in):
    u = -sin(pi x) + tau (1 - x^2) NN([x, sin(2^k pi x), cos(2^k pi x) for k < L, tau]),
with tau = t in the pilot training and tau = sigma (the warped time) in the full training.

Rate: the local field r(x, t) = |u_x| + nu u_x^2, reduced over x by the median, rate(t) = median_x r(x, t).
Warp: s(t) = int_0^t rate^beta dt, sigma = s / s_max. In sigma the equation reads
    (rate^beta / s_max) u_sigma + u u_x - nu u_xx = 0,
the physical-time residual at the warped points ('phys' form). Its squared value at column t_j is weighted by
w_j = rate(t_j)^beta / mean_j rate(t_j)^beta.

Everything runs in float64 on one CPU thread with a fixed seed, so a run is bitwise reproducible on one machine type.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
NU = 0.01 / np.pi
X_EVAL = np.linspace(-1.0, 1.0, 100)          # scoring grid (100 x 100 uniform)
T_EVAL = np.linspace(0.0, 1.0, 100)
T_RATE = np.linspace(0.0, 1.0, 401)           # times at which rate(t) is stored
X_RATE = np.linspace(-1.0, 1.0, 4001)         # dense x on which the median over x is taken
T_FINE = np.linspace(0.0, 1.0, 4001)          # grid for the warp's inverse


def setup(seed):
    """One CPU thread, fixed seeds, float64 default (call before the network is built)."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)


# ---- network ----
class Net(nn.Module):
    def __init__(self, width, depth, L):
        super().__init__()
        d = 1 + 2 * L + 1
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("freqs", (2.0 ** torch.arange(L)) * np.pi)

    def forward(self, x, tau):
        p = x * self.freqs.view(1, -1)
        raw = self.net(torch.cat([x, torch.sin(p), torch.cos(p), tau], dim=1))
        return -torch.sin(np.pi * x) + tau * (1 - x**2) * raw


def grid(xs, taus):
    """Tensor-product collocation grid, x-major: row i * len(taus) + j is (xs[i], taus[j])."""
    X, T = np.meshgrid(np.asarray(xs, np.float64), np.asarray(taus, np.float64), indexing="ij")
    x = torch.tensor(X.reshape(-1, 1), dtype=torch.float64).requires_grad_(True)
    t = torch.tensor(T.reshape(-1, 1), dtype=torch.float64).requires_grad_(True)
    return x, t


def grads(u, x, tau):
    """u_x, u_xx and u_tau by autograd."""
    u_x, u_tau = torch.autograd.grad(u, (x, tau), torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_x, u_xx, u_tau


def predict(model, taus, xs=X_EVAL, chunk=20000):
    """u on xs by taus (the model's own time variable) -> numpy array (len(xs), len(taus))."""
    X, T = np.meshgrid(xs, np.asarray(taus, np.float64), indexing="ij")
    xf, tf = X.reshape(-1, 1), T.reshape(-1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(xf), chunk):
            out.append(model(torch.tensor(xf[i:i + chunk]), torch.tensor(tf[i:i + chunk])).numpy())
    return np.concatenate(out).reshape(X.shape)


# ---- training ----
def train(model, loss_fn, adam_epochs, lbfgs_iters, lr, lr_decay, log_every=500):
    """Adam (lr_decay=True: exponential decay to lr/10 over the Adam epochs), then L-BFGS (strong Wolfe line search,
    tolerances effectively off, history 50)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1**(1.0 / max(adam_epochs, 1))) if lr_decay else None
    for ep in range(adam_epochs):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        if log_every and ep % log_every == 0:
            print(f"  Adam {ep:5d}  loss {loss.item():.4e}", flush=True)
    n = [0]
    if lbfgs_iters > 0:
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters, tolerance_grad=1e-15,
                               tolerance_change=1e-16, history_size=50, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            l = loss_fn()
            l.backward()
            n[0] += 1
            return l
        lb.step(closure)
    print(f"  final loss {loss_fn().item():.4e}  (L-BFGS loss evaluations: {n[0]})", flush=True)


# ---- rate (pilot training; never uses the reference) ----
def median_rate(model, chunk_t=10):
    """rate(t) = median over x of |u_x| + nu u_x^2 on X_RATE, at the times T_RATE (u_x by autograd)."""
    rate = np.zeros(len(T_RATE))
    for j0 in range(0, len(T_RATE), chunk_t):
        ts = T_RATE[j0:j0 + chunk_t]
        x, t = grid(X_RATE, ts)
        u = model(x, t)
        u_x, _ = torch.autograd.grad(u, (x, t), torch.ones_like(u), create_graph=True)
        ux = np.abs(u_x.detach().numpy().reshape(len(X_RATE), len(ts)))
        rate[j0:j0 + len(ts)] = np.percentile(ux + NU * ux**2, 50, axis=0)
    return rate


# ---- warp ----
class Warp:
    """s(t) = int rate^beta dt (cubic spline of rate^beta and its exact antiderivative), sigma = s / s_max."""

    def __init__(self, t_grid, rate_grid, beta):
        r = np.asarray(rate_grid, dtype=np.float64) ** beta
        self.rate_fn = CubicSpline(t_grid, r)                # rate(t)^beta
        self.s_of_t = self.rate_fn.antiderivative()
        self.s_max = float(self.s_of_t(t_grid[-1]))
        self.t_end = float(t_grid[-1])
        if not np.all(self.rate_fn(T_FINE) > 0):
            raise ValueError("rate^beta spline is not strictly positive -> the warp is not monotone")
        self._s_fine = self.s_of_t(T_FINE)

    def sigma_of_t(self, t):
        return self.s_of_t(t) / self.s_max

    def t_of_sigma(self, sig):
        """Inverse warp: interpolation on the fine grid, then 3 Newton steps."""
        target = np.asarray(sig, dtype=np.float64) * self.s_max
        t = np.interp(target, self._s_fine, T_FINE)
        for _ in range(3):
            t = t - (self.s_of_t(t) - target) / self.rate_fn(t)
            t = np.clip(t, 0.0, self.t_end)
        return t


# ---- scoring ----
def reference():
    """Reference solution on the 100 x 100 scoring grid (data/reference_nx3201.npz, written by reference.py)."""
    return np.load(os.path.join(HERE, "data", "reference_nx3201.npz"))["U"]


def relative_l2(U, U_ref):
    return float(np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref))
