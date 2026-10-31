# Full training -- Frank-Kamenetskii thermal explosion, psi = 1e-5, N_x x N_t = 256 x 80 collocation points.
#
# Rate-scaled PINN in the warped time sigma = s(t) / s_max, s(t) = int_0^t (rate + 1e-6)^beta dt, with the rate from
# the pilot training (rates/pilot_seed1001.npz); the reference solution is only used at the end, to score.
#
# Seed 90 is the best of the paper's ten final seeds 90-99 (by relative L2 error); the paper's tables report the
# ten-seed median. Paper values for this cell (NVIDIA A100 GPU, float64): seed 90 relative L2 6.89%, peak error
# 43.6%; ten-seed median relative L2 7.89%, peak error 47.4%.
#
#     python full_training_Nx256_Nt80.py    # change the settings by editing the variables below
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS; must be set before CUDA starts

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# float64 tensors (the layers are built in float32 and cast, see the model); one CPU thread. On CUDA, as in the
# paper: tensors are created on the GPU, deterministic kernels, no TF32. A CPU run is valid too, but the network
# starts from a different random initialisation, so its numbers differ from the paper's.
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)
if device == "cuda":
    torch.set_default_device("cuda")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))

# hyperparams
N_x, N_t = 256, 80           # N_x uniform x by N_t times uniform in sigma (corners removed)
lbfgs_max_iter = 10000       # L-BFGS iterations from the start (no Adam)
hidden_size = 64             # 4 hidden tanh layers of 64 units
n_hidden = 4
seed = 90
beta = 1.0                   # warp exponent

# the equation: u_t = u_xx + delta e^u, x in [-1, 1], u(x, 0) = 0, u(+-1, t) = 0; blow-up at T_star (reference.py)
delta = 1.3177
T_star = 1.4467001610282852
psi = 1e-5                   # margin to blow-up
t_end = (1 - psi) * T_star   # training stops here

# the stored paper file; point rate_file at outputs/pilot_seed1001.npz to use your own pilot output
rate_file = os.path.join(HERE, "rates/pilot_seed1001.npz")
out_file = os.path.join(HERE, "outputs", f"full_Nx{N_x}_Nt{N_t}_seed{seed}.npz")

# rate(t) and the warp s(t) = int_0^t rho^beta dt with rho = rate + 1e-6 (the 1e-6 only keeps rho > 0);
# PCHIP keeps rho^beta positive
t_rate = t_end * (1 - (1 - np.linspace(0, 1, 801)) ** 3)        # the pilot's rate grid (clustered toward t_end)
data = np.load(rate_file)
assert np.allclose(data["t"], t_rate), "rate file is not on this margin's rate grid"
rho = data["net_mean"] + 1e-6
rate_fn = PchipInterpolator(t_rate, rho**beta)   # rho(t)^beta
s_t = rate_fn.antiderivative()                   # s(t)
s_rough = s_t(t_rate)
assert np.all(np.diff(s_rough) > 0), "warp not strictly increasing"
s_max = float(s_rough[-1])
t_of_s = PchipInterpolator(s_rough, t_rate)      # inverse warp t(s)

# collocation points: N_x uniform x by N_t times uniform in sigma; the two corners (x = +-1, sigma = 0), where the
# ansatz makes every term vanish, are removed
sig = np.linspace(0, 1, N_t)
t_i = t_of_s(sig * s_max)                        # their physical times (dense where the rate is large)
rate_i = rate_fn(t_i)                            # rho(t_i)^beta
x = torch.linspace(-1, 1, N_x, dtype=torch.float64)
X, S = torch.meshgrid(x, torch.tensor(sig, dtype=torch.float64), indexing="ij")
pts_all = torch.stack([X.flatten(), S.flatten()], dim=1)
RATE_all = torch.tensor(rate_i, dtype=torch.float64).view(1, -1).expand(N_x, -1).reshape(-1, 1)
corner = (pts_all[:, 1] == 0) & (pts_all[:, 0].abs() == 1)
pts = pts_all[~corner].clone().requires_grad_(True)
RATE = RATE_all[~corner]


# model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        # built in float32, then cast to float64 (this is how the paper's networks were initialised)
        self.net = nn.Sequential(*layers).to(torch.float64)

    def forward(self, xs):
        x, tau = xs[:, 0:1], xs[:, 1:2]
        # hard ansatz: u = sigma (1 - x^2) NN(x, sigma) satisfies u(x, 0) = 0 and u(+-1, t) = 0 exactly
        return tau * (1 - x ** 2) * self.net(xs)


def loss_fn(model):
    # u_sigma - (s_max / rho^beta) (u_xx + delta e^u)  (chain rule of ds = rho^beta dt)
    u = model(pts)
    g = torch.autograd.grad(u, pts, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0:1], pts, torch.ones_like(u), create_graph=True)[0][:, 0:1]
    us = g[:, 1:2]
    return torch.mean((us - (s_max / RATE) * (uxx + delta * torch.exp(u))) ** 2)


def eval_u(model, x, tau, chunk=200000):
    # the network on the tensor grid x (nx) by tau (nt), in chunks -> numpy array (nx, nt)
    X, S = np.meshgrid(x, tau, indexing="ij")
    p = np.stack([X.ravel(), S.ravel()], axis=1)
    out = np.empty(len(p))
    with torch.no_grad():
        for a in range(0, len(p), chunk):
            out[a:a + chunk] = model(torch.tensor(p[a:a + chunk], dtype=torch.float64)).cpu().numpy().ravel()
    return out.reshape(len(x), len(tau))


# L-BFGS from the start
torch.manual_seed(seed)
model = PINN().to(device)
loss_history = []
print(f"Full training, psi = {psi:g}, {N_x} x {N_t}, beta {beta}, seed {seed}, device {device}")
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_max_iter, history_size=50,
                                    line_search_fn="strong_wolfe", tolerance_grad=1e-15, tolerance_change=1e-16)


def closure():
    optimizer_lbfgs.zero_grad()
    loss = loss_fn(model)
    if not torch.isfinite(loss):                 # guard: a huge loss makes the line search back off
        loss = torch.zeros((), dtype=loss.dtype, requires_grad=True) + 1e30
    loss.backward()
    loss_history.append(loss.item())
    if len(loss_history) % 1000 == 0:
        print(f"  L-BFGS evaluation {len(loss_history):6d}  loss {loss.item():.4e}", flush=True)
    return loss


optimizer_lbfgs.step(closure)
print(f"  final loss {loss_fn(model).item():.4e}", flush=True)

# ____L2 vs the reference: relative L2 on 2000 x 200 uniform points; peak error = relative error of max_x u at t_end
# (on a second time grid clustered toward t_end)
x_eval = np.linspace(-1, 1, 2000)
t_eval = np.linspace(0, t_end, 200)
t_eval_c = t_end * (1 - (1 - np.linspace(0, 1, 200)) ** 3)
U_ref = np.load(os.path.join(HERE, "..", "data", f"ref_eval_psi{psi:g}.npz"))["U"]       # written by reference.py
U_ref_c = np.load(os.path.join(HERE, "..", "data", f"ref_evalc_psi{psi:g}.npz"))["U"]
U = eval_u(model, x_eval, np.clip(s_t(t_eval) / s_max, 0.0, 1.0))         # the network sees this run's sigma(t)
U_c = eval_u(model, x_eval, np.clip(s_t(t_eval_c) / s_max, 0.0, 1.0))
rel_l2 = float(np.sqrt(((U - U_ref) ** 2).sum() / (U_ref ** 2).sum()))
peak_pred, peak_ref = float(np.nanmax(U_c[:, -1])), float(U_ref_c[:, -1].max())
peak = abs(peak_pred - peak_ref) / peak_ref
print(f"relative L2 error: {100 * rel_l2:.4f}%   peak error: {100 * peak:.3f}%   "
      f"(paper, this seed: 6.89% / 43.6%)")
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, x=x_eval, t=t_eval, U_pred=U, U_ref=U_ref, t_collocation=t_i, rel_l2=rel_l2, peak_relerr=peak)

# plot: solution vs reference, abs error, loss curve
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for a, field, title in [(ax[0], U_ref, "reference"), (ax[1], U, "PINN")]:
    im = a.pcolormesh(t_eval, x_eval, field, cmap="inferno", shading="auto")
    fig.colorbar(im, ax=a)
    a.set_title(title)
ax[1].plot(t_i, np.full_like(t_i, -1.0), "c|", ms=20, label="collocation times")
ax[1].legend(loc="upper left")
im = ax[2].pcolormesh(t_eval, x_eval, np.abs(U - U_ref), cmap="viridis", shading="auto")
fig.colorbar(im, ax=ax[2])
ax[2].set_title(f"|error|, relative L2 {100 * rel_l2:.4f}%, peak {100 * peak:.3f}%")
for a in ax[:3]:
    a.set_xlabel("t")
    a.set_ylabel("x")
ax[3].semilogy(loss_history)
ax[3].set_xlabel("L-BFGS evaluation")
ax[3].set_title("loss")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", f"full_Nx{N_x}_Nt{N_t}_seed{seed}.png"), dpi=150)
print(f"written {out_file} and plots/full_Nx{N_x}_Nt{N_t}_seed{seed}.png")
