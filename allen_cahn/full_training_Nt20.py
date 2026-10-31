# Full training -- Allen-Cahn, N_x x N_t = 256 x 20 collocation points.
#
# Rate-scaled PINN in the warped time sigma = s(t) / s_max, s(t) = int_0^t rate^beta dt. The rate is the final rate:
# pilot training -> bootstrap round 1 -> bootstrap round 2 (rates/bootstrap2_seed3002.npz). The reference solution is
# only used at the end, to score. beta = kappa / ln R (R = max/min of the final rate = 10.73), kappa = 2.4.
#
# Seed 99 is the best of the paper's ten final seeds 90-99 (by relative L2 error); the paper's tables report the
# ten-seed median. Paper values for this cell (AMD EPYC 7713 CPU): seed 99 0.21%, ten-seed median 0.29%.
#
#     python full_training_Nt20.py        # change the settings by editing the variables below
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
device = torch.device("cpu")   # the paper ran this on a CPU ("cuda" starts from the same weights)

HERE = os.path.dirname(os.path.abspath(__file__))

# hyperparams
N_x, N_t = 256, 20           # x uniform, sigma uniform (both ends included)
lbfgs_max_iter = 10000       # L-BFGS iterations from the start (no Adam)
hidden_size = 64             # 4 hidden tanh layers of 64 units
n_hidden = 4
seed = 99
beta = 1.011526              # warp exponent, kappa / ln R with kappa = 2.4

# the equation: u_t = gamma1 u_xx + gamma2 (u - u^3), x in [-1, 1] periodic, t in [0, 1], u(x, 0) = x^2 cos(pi x)
gamma1, gamma2 = 0.0001, 5.0
lambda_ic = 50.0             # weight of the initial-condition penalty
fourier_m = 10               # Fourier modes of the periodic x-embedding

# the stored paper file; point rate_file at outputs/bootstrap2_seed3002.npz to use your own round-2 output
rate_file = os.path.join(HERE, "rates/bootstrap2_seed3002.npz")
out_file = os.path.join(HERE, "outputs", f"full_Nt{N_t}_seed{seed}.npz")

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
        return self.net(torch.cat([torch.cat([torch.cos(angle), torch.sin(angle)], dim=1), tau], dim=1))


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
print(f"Full training, {N_x} x {N_t}, beta {beta}, seed {seed}, device {device}")
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

# ____L2 vs the reference, on the 200 x 200 (x, t) grid, through this run's own sigma(t)
x_plot = np.linspace(-1, 1, 200)
t_plot = np.linspace(0, 1, 200)
Xp, Sp = np.meshgrid(x_plot, s_t(t_plot) / s_max, indexing="ij")
pts = torch.tensor(np.stack([Xp.flatten(), Sp.flatten()], axis=1), dtype=torch.float32, device=device)
with torch.no_grad():
    U = model(pts).cpu().numpy().reshape(200, 200)
U_ref = np.load(os.path.join(HERE, "data", "reference_200x200.npz"))["U_ref"]     # written by reference.py
rel_l2 = float(np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref))
print(f"relative L2 error: {100 * rel_l2:.3f}%   (paper, this seed: 0.21%)")
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, x=x_plot, t=t_plot, U_pred=U, U_ref=U_ref, t_collocation=t_i, rel_l2=rel_l2)

# plot: solution vs reference, abs error, loss curve
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for a, field, title in [(ax[0], U_ref, "reference"), (ax[1], U, "PINN")]:
    im = a.pcolormesh(t_plot, x_plot, field, cmap="RdBu_r", shading="auto")
    fig.colorbar(im, ax=a)
    a.set_title(title)
ax[1].plot(t_i, np.full_like(t_i, -1.0), "k|", ms=20, label="collocation times")
ax[1].legend(loc="upper right")
im = ax[2].pcolormesh(t_plot, x_plot, np.abs(U - U_ref), cmap="viridis", shading="auto")
fig.colorbar(im, ax=ax[2])
ax[2].set_title(f"|error|, relative L2 {100 * rel_l2:.3f}%")
for a in ax[:3]:
    a.set_xlabel("t")
    a.set_ylabel("x")
ax[3].semilogy(loss_history)
ax[3].set_xlabel("L-BFGS evaluation")
ax[3].set_title("loss")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", f"full_Nt{N_t}_seed{seed}.png"), dpi=150)
print(f"written {out_file} and plots/full_Nt{N_t}_seed{seed}.png")
