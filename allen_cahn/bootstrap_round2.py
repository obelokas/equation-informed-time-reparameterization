# Bootstrap round 2 -- Allen-Cahn.
#
# A warped training run on the rate from the previous stage (bootstrap round 1). Its trained solution re-derives the
# rate with the same formula as the pilot training; the reference solution is never used. The round uses a fixed
# small warp exponent (beta = 0.1) and the sign envelope u = tanh(12 cos(pi x)) softplus(NN).
#
# Seed 3002 is the kept run of this round: the paper ran seeds 3000-3009 on the kept input file and kept the one with
# the lowest final training loss (reference-free). Its stored output is rates/bootstrap2_seed3002.npz
# (NVIDIA RTX A4000 GPU).
#
#     python bootstrap_round2.py          # reads rates/bootstrap1_seed2000.npz, writes outputs/bootstrap2_seed3002.npz
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# float32 (the torch default); one CPU thread. The network is built on the CPU and then moved to the device, so a
# CPU and a GPU run start from the same weights.
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # the paper ran this on a GPU

HERE = os.path.dirname(os.path.abspath(__file__))

# hyperparams
N_x, N_t = 128, 16           # x uniform, sigma uniform (both ends included)
lbfgs_max_iter = 2500        # L-BFGS iterations from the start (no Adam)
hidden_size = 32             # 4 hidden tanh layers of 32 units
n_hidden = 4
seed = 3002
beta = 0.1                   # warp exponent of the round (rate^beta)
k_sign = 12.0                # steepness of the sign envelope tanh(k_sign cos(pi x))

# the equation: u_t = gamma1 u_xx + gamma2 (u - u^3), x in [-1, 1] periodic, t in [0, 1], u(x, 0) = x^2 cos(pi x)
gamma1, gamma2 = 0.0001, 5.0
lambda_ic = 50.0             # weight of the initial-condition penalty
fourier_m = 10               # Fourier modes of the periodic x-embedding
REL_FLOOR = 0.01             # rate floor, as a fraction of max rate

# the stored paper file; point rate_file at outputs/bootstrap1_seed2000.npz to use your own round-1 output
rate_file = os.path.join(HERE, "rates/bootstrap1_seed2000.npz")
out_file = os.path.join(HERE, "outputs", f"bootstrap2_seed{seed}.npz")

# rate(t) and the warp s(t) = int_0^t rate^beta dt
data = np.load(rate_file)
t_rough, rate = data["t"], data["rate"]
rate_fn = CubicSpline(t_rough, rate**beta)       # rate(t)^beta
s_t = rate_fn.antiderivative()                   # s(t), the exact antiderivative of the spline
s_rough = s_t(t_rough)
s_max = float(s_rough[-1])
t_of_s = CubicSpline(s_rough, t_rough)           # inverse warp t(s): cubic spline through (s(t_k), t_k)

# collocation points: x uniform, sigma uniform; initial condition on the x grid
x = torch.linspace(-1, 1, N_x, device=device)
x_ic = x.reshape(-1, 1)
ic_pts = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)        # points (x, 0)
ic_target = x_ic**2 * torch.cos(np.pi * x_ic)                    # x^2 cos(pi x)
sigma_np = np.linspace(0, 1, N_t)
t_i = t_of_s(sigma_np * s_max)                                   # physical times of the columns
rate_i = torch.tensor(rate_fn(t_i), dtype=torch.float32, device=device)   # rate(t_i)^beta
sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
X, S = torch.meshgrid(x, sigma, indexing="ij")
col = torch.stack([X.flatten(), S.flatten()], dim=1).requires_grad_(True)
RATE = rate_i.view(1, N_t).expand(N_x, N_t).reshape(-1, 1)        # rate^beta per point
RATE_W = RATE / torch.exp(torch.log(rate_i).mean())               # rate^beta / geomean over the columns


# model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        layers = [nn.Linear(2 * fourier_m + 1, hidden_size), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        x, tau = xt[:, 0:1], xt[:, 1:2]
        # periodic in x exactly: the network sees [cos(k pi x), sin(k pi x) for k = 1..10, sigma]
        freqs = torch.arange(1, fourier_m + 1, device=x.device, dtype=x.dtype).reshape(1, -1)
        angle = torch.pi * x * freqs
        raw = self.net(torch.cat([torch.cat([torch.cos(angle), torch.sin(angle)], dim=1), tau], dim=1))
        # sign envelope: u has the sign of cos(pi x) (the interfaces at x = +-0.5 do not move for t in [0, 1])
        return torch.tanh(k_sign * torch.cos(np.pi * x)) * nn.functional.softplus(raw)


def loss_fn(model):
    # mean( (RATE_W (RATE u_s - gamma1 u_xx - gamma2 (u - u^3)))^2 ) + lambda_ic mean( (u(x, 0) - u0)^2 ):
    # the physical-time residual at the warped points (u_t = rate^beta u_s, u_s = u_sigma / s_max), multiplied by
    # rate^beta / geomean, so the high-rate (early) columns get more weight
    u = model(col)
    u_ic = model(ic_pts)
    g = torch.autograd.grad(u, col, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx, du_ds = g[:, 0:1], g[:, 1:2] / s_max
    du2_dx2 = torch.autograd.grad(du_dx, col, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
    res = RATE_W * (RATE * du_ds - gamma1 * du2_dx2 - gamma2 * (u - u**3))
    return torch.mean(res**2) + lambda_ic * torch.mean((u_ic - ic_target)**2)


# L-BFGS from the start
torch.manual_seed(seed)
np.random.seed(seed)
model = PINN().to(device)
loss_history = []
print(f"Bootstrap round 2, seed {seed}, {N_x} x {N_t}, beta {beta}, device {device}")
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_max_iter, tolerance_grad=1e-9,
                                    tolerance_change=1e-12, history_size=50, line_search_fn="strong_wolfe")


def closure():
    optimizer_lbfgs.zero_grad()
    loss = loss_fn(model)
    loss.backward()
    loss_history.append(loss.item())
    if len(loss_history) % 500 == 0:
        print(f"  L-BFGS evaluation {len(loss_history):6d}  loss {loss.item():.4e}", flush=True)
    return loss


optimizer_lbfgs.step(closure)
print(f"  final loss {loss_fn(model).item():.4e}", flush=True)

# ____rate file (never uses the reference)
# the trained network on the 200 x 200 (x, t) grid through this round's own sigma(t), kept in float32 as in the paper
x_plot = np.linspace(-1, 1, 200)
t_plot = np.linspace(0, 1, 200)
Xp, Sp = np.meshgrid(x_plot, s_t(t_plot) / s_max, indexing="ij")
pts = torch.tensor(np.stack([Xp.flatten(), Sp.flatten()], axis=1), dtype=torch.float32, device=device)
with torch.no_grad():
    U = model(pts).cpu().numpy().reshape(200, 200)
# rate(t) = mean_x max(0, gamma2 (1 - 3U^2) - gamma1 U_x^2), U_x by cubic splines in x. Late in time the growth
# rate is 0 almost everywhere, so the rate is floored at REL_FLOOR * max to keep the warp strictly increasing.
U_x = np.array([CubicSpline(x_plot, U[:, j])(x_plot, 1) for j in range(U.shape[1])]).T
lam = np.maximum(0.0, gamma2 * (1 - 3 * U**2) - gamma1 * U_x**2)
rate_new = lam.mean(axis=0)
rate_new = np.maximum(rate_new, REL_FLOOR * rate_new.max())
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, t=t_plot, rate=rate_new, x_plot=x_plot, t_plot=t_plot, U_pred=U)
print(f"rate file written: {out_file}  (R = max/min of rate = {rate_new.max() / rate_new.min():.4g})")

# plot: this round's solution, input and output rate, loss curve
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
im = ax[0].pcolormesh(t_plot, x_plot, U, cmap="RdBu_r", shading="auto")
fig.colorbar(im, ax=ax[0])
ax[0].plot(t_i, np.full_like(t_i, -1.0), "k|", ms=20, label="collocation times")
ax[0].legend(loc="upper right")
ax[0].set_xlabel("t")
ax[0].set_ylabel("x")
ax[0].set_title("round solution u(x, t)")
ax[1].plot(t_rough, rate, "k-", label="input rate")
ax[1].plot(t_plot, rate_new, "r--", label="this round's rate")
ax[1].set_xlabel("t")
ax[1].set_title("rate(t)")
ax[1].legend()
ax[2].semilogy(loss_history)
ax[2].set_xlabel("L-BFGS evaluation")
ax[2].set_title("loss")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", f"bootstrap2_seed{seed}.png"), dpi=150)
