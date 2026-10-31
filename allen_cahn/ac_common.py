"""Allen-Cahn: shared building blocks for the pilot training, the two bootstrap rounds and the full training scripts.

Problem (periodic in x):
    u_t = gamma1 u_xx + gamma2 (u - u^3),   gamma1 = 1e-4,   gamma2 = 5,   x in [-1, 1],   t in [0, 1],
    u(x, 0) = x^2 cos(pi x).

Network: u = NN([cos(k pi x), sin(k pi x) for k = 1..10, tau]). The Fourier embedding makes u periodic in x exactly;
the initial condition is a penalty (weight 50) on the line tau = 0. tau = t in the pilot training and tau = sigma
(the warped time) in the bootstrap rounds and the full training. The bootstrap rounds also fix the sign pattern of
the solution with an output envelope: u = tanh(12 cos(pi x)) softplus(NN).

Rate: the linearised growth rate lambda(x, t) = max(0, gamma2 (1 - 3u^2) - gamma1 u_x^2), averaged over x,
rate(t) = mean_x lambda(x, t), computed from the trained network on a 200 x 200 grid. Late in time lambda is 0
almost everywhere, so the rate is floored at 1% of its maximum (rate >= 0.01 max rate) to keep the warp
strictly increasing.

Warp: s(t) = int_0^t rate^beta dt, sigma = s / s_max. With u_s = u_sigma / s_max, the physical-time residual at the
warped points is rate^beta u_s - gamma1 u_xx - gamma2 (u - u^3); the trainings minimise it multiplied by
rate^beta / geomean_j(rate(t_j)^beta), i.e. the high-rate (early) columns get more weight ('mul' form).

All runs are float32. The network is always initialised on the CPU (so a GPU run starts from the same weights).
"""
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
GAMMA1, GAMMA2 = 0.0001, 5.0
LAMBDA_IC = 50.0                       # weight of the initial-condition penalty
FOURIER_M = 10                         # Fourier modes of the periodic x-embedding
REL_FLOOR = 0.01                       # rate floor, as a fraction of max rate
X_PLOT = np.linspace(-1, 1, 200)       # 200 x 200 grid: rate construction and scoring
T_PLOT = np.linspace(0, 1, 200)


def setup(seed, device):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.device(device)


def pick_device(paper_device):
    """The device the paper used for this run, or the CPU when no CUDA GPU is present."""
    return paper_device if paper_device == "cpu" or torch.cuda.is_available() else "cpu"


def dF(u):
    return u - u**3


def periodic_embedding(x, m=FOURIER_M):
    freqs = torch.arange(1, m + 1, device=x.device, dtype=x.dtype).reshape(1, -1)
    angle = torch.pi * x * freqs
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)


class PINN(nn.Module):
    """Tanh MLP on [Fourier embedding of x, tau]. hard_sign=True multiplies softplus(NN) by tanh(k_sign cos(pi x)),
    which fixes the sign of u to the sign of cos(pi x) (the interfaces at x = +-0.5 do not move for t in [0, 1])."""

    def __init__(self, hidden_sizes, hard_sign=False, k_sign=12.0):
        super().__init__()
        self.hard_sign, self.k_sign = hard_sign, k_sign
        sizes = [2 * FOURIER_M + 1] + list(hidden_sizes)
        layers = []
        for in_f, out_f in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(in_f, out_f), nn.Tanh()]
        layers.append(nn.Linear(sizes[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        x, tau = xt[:, 0:1], xt[:, 1:2]
        raw = self.net(torch.cat([periodic_embedding(x), tau], dim=1))
        if self.hard_sign:
            return torch.tanh(self.k_sign * torch.cos(np.pi * x)) * nn.functional.softplus(raw)
        return raw


def initial_condition_points(x):
    """Points (x, 0) on the collocation x grid and the target x^2 cos(pi x)."""
    x_ic = x.reshape(-1, 1)
    return torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1), x_ic**2 * torch.cos(np.pi * x_ic)


def lbfgs(model, closure, max_iter):
    """L-BFGS with strong Wolfe line search, history 50 (tolerances 1e-9 / 1e-12)."""
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter, tolerance_grad=1e-9,
                            tolerance_change=1e-12, history_size=50, line_search_fn="strong_wolfe")

    def step():
        opt.zero_grad()
        l = closure()
        l.backward()
        return l
    opt.step(step)
    st = opt.state[opt._params[0]]
    print(f"  L-BFGS: {st.get('n_iter')} iterations, {st.get('func_evals')} loss evaluations", flush=True)


# ---- rate (pilot training and bootstrap rounds; never uses the reference) ----
def rate_from_field(U, x=X_PLOT, t=T_PLOT):
    """rate(t) = mean_x max(0, gamma2 (1 - 3U^2) - gamma1 U_x^2) from U (nx, nt), U_x by cubic splines in x,
    floored at REL_FLOOR * max. U is the network's float32 output (kept in float32, as in the paper)."""
    ux = np.array([CubicSpline(x, U[:, j])(x, 1) for j in range(U.shape[1])]).T
    lam = np.maximum(0.0, GAMMA2 * (1 - 3 * U**2) - GAMMA1 * ux**2)
    r = lam.mean(axis=0)
    return np.maximum(r, REL_FLOOR * r.max())


def save_rate(path, U):
    r = rate_from_field(U)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, t=T_PLOT, rate=r, x_plot=X_PLOT, t_plot=T_PLOT, U_pred=U)
    print(f"rate file written: {path}  (R = max/min of rate = {r.max() / r.min():.4g})")


# ---- warp ----
class Warp:
    """s(t) = int rate^beta dt (cubic spline of rate^beta and its antiderivative), sigma = s / s_max;
    the inverse t(s) is a cubic spline through (s(t_k), t_k)."""

    def __init__(self, t_grid, rate, beta):
        self.rate_fn = CubicSpline(t_grid, rate**beta)       # rate(t)^beta
        self.s_of_t = self.rate_fn.antiderivative()
        s = self.s_of_t(t_grid)
        self.t_of_s = CubicSpline(s, t_grid)
        self.s_max = float(s[-1])


def warped_problem(rate_file, beta, n_x, n_t, device):
    """Warp from the rate file, and the n_x x n_t collocation grid: x uniform, sigma uniform (both ends included).
    Returns warp, collocation points, initial-condition points/target, RATE (rate^beta per point) and its geomean."""
    d = np.load(rate_file)
    warp = Warp(d["t"], d["rate"], beta)
    x = torch.linspace(-1, 1, n_x, device=device)
    ic_pts, ic_target = initial_condition_points(x)
    sigma_np = np.linspace(0, 1, n_t)
    t_i = warp.t_of_s(sigma_np * warp.s_max)              # physical times of the columns
    rate_i = torch.tensor(warp.rate_fn(t_i), dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    X, S = torch.meshgrid(x, sigma, indexing="ij")
    col = torch.stack([X.flatten(), S.flatten()], dim=1).requires_grad_(True)
    RATE = rate_i.view(1, n_t).expand(n_x, n_t).reshape(-1, 1)
    RATE_W = RATE / torch.exp(torch.log(rate_i).mean())      # rate^beta / geomean over the columns
    return warp, col, ic_pts, ic_target, RATE, RATE_W, t_i


def warped_loss(model, col, ic_pts, ic_target, RATE, RATE_W, s_max):
    """mean( (RATE_W (RATE u_s - gamma1 u_xx - gamma2 (u - u^3)))^2 ) + LAMBDA_IC * mean( (u(x, 0) - u0)^2 )."""
    u = model(col)
    u_ic = model(ic_pts)
    g = torch.autograd.grad(u, col, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx, du_ds = g[:, 0:1], g[:, 1:2] / s_max
    du2_dx2 = torch.autograd.grad(du_dx, col, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
    res = RATE_W * (RATE * du_ds - GAMMA1 * du2_dx2 - GAMMA2 * dF(u))
    return torch.mean(res**2) + LAMBDA_IC * torch.mean((u_ic - ic_target)**2)


def predict_warped(model, warp, device):
    """Network on the 200 x 200 (x, t) grid, through this run's own sigma(t) -> float32 array (x, t)."""
    s_plot = warp.s_of_t(T_PLOT) / warp.s_max
    Xp, Sp = np.meshgrid(X_PLOT, s_plot, indexing="ij")
    pts = torch.tensor(np.stack([Xp.flatten(), Sp.flatten()], axis=1), dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(pts).cpu().numpy().reshape(200, 200)


# ---- scoring ----
def reference():
    """Reference on the 200 x 200 grid (data/reference_200x200.npz, written by reference.py)."""
    return np.load(os.path.join(HERE, "data", "reference_200x200.npz"))["U_ref"]


def relative_l2(U, U_ref):
    return float(np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref))
