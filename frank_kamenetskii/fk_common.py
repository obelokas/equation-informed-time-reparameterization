"""Frank-Kamenetskii thermal explosion: shared building blocks for the pilot training and full training scripts in
psi_3e-3/ and psi_1e-5/.

Problem:
    u_t = u_xx + delta e^u,   x in [-1, 1],   u(x, 0) = 0,   u(+-1, t) = 0,   delta = 1.3177,
which blows up at T* = 1.4467001610 (reference.py). Training stops at t_end = (1 - psi) T*, psi = margin to blow-up.

Network: u = tau (1 - x^2) NN(x, tau), which satisfies the initial and boundary conditions exactly. tau is the
normalised time t / t_end in the pilot training and the warped time sigma in [0, 1] in the full training.

Rate (the 'net growth coefficient'): rate(t) = mean_x | delta e^u - u_x^2 |, the x-average of the magnitude of the
linearised reaction growth (delta e^u) minus the diffusion closure scale (u_x^2). The warp density is
rho(t) = rate(t) + 1e-6 (the 1e-6 only keeps rho strictly positive), s(t) = int_0^t rho^beta dt, sigma = s / s_max.
In sigma the equation reads u_sigma = (s_max / rho^beta) (u_xx + delta e^u), the residual of the full training.

All runs are float64. The paper's Frank-Kamenetskii runs used one NVIDIA A100 GPU with deterministic kernels
(setup(device='cuda')). A CPU run is valid too, but the network starts from a different random initialisation, so its
numbers differ from the paper's.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (must be set before CUDA starts)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

DELTA = 1.3177
T_STAR = 1.4467001610282852        # blow-up time, reference.blowup_time(1.3177)
EPS = 1e-6                         # positivity constant of the warp density rho = rate + EPS
HERE = os.path.dirname(os.path.abspath(__file__))


class Problem:
    """Time interval and fixed grids for one margin psi (t_end = (1 - psi) T*)."""

    def __init__(self, psi):
        self.psi = float(psi)
        self.t_end = (1 - self.psi) * T_STAR
        self.x_eval = np.linspace(-1, 1, 2000)                                     # scoring grid in x
        self.t_eval = np.linspace(0, self.t_end, 200)                              # scoring grid in t (uniform)
        self.t_eval_c = self.t_end * (1 - (1 - np.linspace(0, 1, 200)) ** 3)       # clustered toward t_end (peak)
        self.x_rate = np.linspace(-1, 1, 1001)                                     # grid on which rate(t) is built
        self.t_rate = self.t_end * (1 - (1 - np.linspace(0, 1, 801)) ** 3)         # (time clustered toward t_end)
        self.tag = f"psi{self.psi:g}"

    def reference(self):
        """Reference solution on the scoring grids (data/ files written by reference.py)."""
        d = os.path.join(HERE, "data")
        return np.load(os.path.join(d, f"ref_eval_{self.tag}.npz"))["U"], np.load(os.path.join(d, f"ref_evalc_{self.tag}.npz"))["U"]


def setup(device):
    """One CPU thread; on CUDA: tensors created on the GPU, deterministic algorithms, no TF32 (as in the paper)."""
    torch.set_num_threads(1)
    if device == "cuda":
        torch.set_default_device("cuda")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---- network ----
class Net(nn.Module):
    """u(x, tau) = tau (1 - x^2) NN(x, tau): tanh MLP with `depth` hidden layers of `width` units.
    The layers are built in float32 and then cast to float64 (this is how the paper's networks were initialised)."""

    def __init__(self, width, depth):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers).to(torch.float64)

    def forward(self, xs):
        x, tau = xs[:, 0:1], xs[:, 1:2]
        return tau * (1 - x ** 2) * self.net(xs)


def drop_corners(pts):
    """Remove the two corner points (x = +-1, tau = 0), where the ansatz makes every term vanish."""
    corner = (pts[:, 1] == 0) & (pts[:, 0].abs() == 1)
    return pts[~corner]


def derivs(model, pts):
    """u, u_tau and u_xx at the points (autograd)."""
    u = model(pts)
    g = torch.autograd.grad(u, pts, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0:1], pts, torch.ones_like(u), create_graph=True)[0][:, 0:1]
    return u, g[:, 1:2], uxx


def eval_u(model, x, tau, chunk=200000):
    """Network on the tensor grid x (nx) by tau (nt) -> numpy array (nx, nt)."""
    X, S = np.meshgrid(x, tau, indexing="ij")
    pts = np.stack([X.ravel(), S.ravel()], axis=1)
    out = np.empty(len(pts))
    with torch.no_grad():
        for a in range(0, len(pts), chunk):
            out[a:a + chunk] = model(torch.tensor(pts[a:a + chunk], dtype=torch.float64)).cpu().numpy().ravel()
    return out.reshape(len(x), len(tau))


# ---- training ----
def train(model, loss_fn, adam_epochs, lr, lbfgs_iters, log_every=100):
    """Adam at a constant learning rate, then L-BFGS (strong Wolfe line search, tolerances effectively off,
    history 50). Stops early if the loss becomes non-finite."""
    params = list(model.parameters())
    if adam_epochs > 0:
        opt = torch.optim.Adam(params, lr=lr)
        for ep in range(adam_epochs):
            loss = loss_fn()
            if not torch.isfinite(loss):
                print("  non-finite loss, Adam stopped", flush=True)
                return
            opt.zero_grad()
            loss.backward()
            opt.step()
            if log_every and ep % log_every == 0:
                print(f"  Adam {ep:5d}  loss {loss.item():.4e}", flush=True)
    if lbfgs_iters > 0:
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters, history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-15, tolerance_change=1e-16)
        calls = [0]

        def closure():
            opt.zero_grad()
            l = loss_fn()
            if not torch.isfinite(l):                      # guard: a huge loss makes the line search back off
                l = torch.zeros((), dtype=l.dtype, requires_grad=True) + 1e30
            l.backward()
            calls[0] += 1
            if log_every and calls[0] % (10 * log_every) == 0:
                print(f"  L-BFGS evaluation {calls[0]:6d}  loss {l.item():.4e}", flush=True)
            return l
        opt.step(closure)
    print(f"  final loss {loss_fn().item():.4e}", flush=True)


# ---- rate (pilot training; never uses the reference) ----
def net_mean_rate(U, x):
    """rate(t) = mean_x | delta e^U - U_x^2 | from a field U(x, t) of shape (nx, nt)."""
    ux = np.gradient(U, x, axis=0, edge_order=2)
    return np.abs(DELTA * np.exp(U) - ux ** 2).mean(axis=0)


# ---- warp ----
class Warp:
    """s(t) = int rho^beta dt on the rate grid (PCHIP keeps rho^beta positive), sigma = s / s_max."""

    def __init__(self, t_grid, rho, beta):
        self.t_grid = np.asarray(t_grid)
        r = np.asarray(rho, float) ** float(beta)
        self.rate_fn = PchipInterpolator(self.t_grid, r)        # rho(t)^beta
        self.s_fn = self.rate_fn.antiderivative()
        s = self.s_fn(self.t_grid)
        assert np.all(np.diff(s) > 0), "warp not strictly increasing"
        self.s_max = float(s[-1])
        self._t_of_s = PchipInterpolator(s, self.t_grid)

    def sigma(self, t):
        return np.clip(self.s_fn(np.asarray(t)) / self.s_max, 0.0, 1.0)

    def t_of_sigma(self, sig):
        return self._t_of_s(np.clip(np.asarray(sig), 0.0, 1.0) * self.s_max)


# ---- scoring ----
def scores(U_pred, U_pred_c, U_ref, U_ref_c):
    """Relative L2 error on the uniform scoring grid, and the peak error: relative error of max_x u at t_end."""
    rel_l2 = float(np.sqrt(((U_pred - U_ref) ** 2).sum() / (U_ref ** 2).sum()))
    peak_pred, peak_ref = float(np.nanmax(U_pred_c[:, -1])), float(U_ref_c[:, -1].max())
    return rel_l2, abs(peak_pred - peak_ref) / peak_ref
