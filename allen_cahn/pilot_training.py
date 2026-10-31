# Pilot training -- Allen-Cahn.
#
# A small plain PINN in the physical time t on a 250 x 32 grid. Its only job is to give a first estimate of
# rate(t) = mean_x max(0, gamma2 (1 - 3u^2) - gamma1 u_x^2); the reference solution is never used.
#
# Seed 1002 is the kept pilot run: the paper ran seeds 1000-1009 and kept the one with the lowest final training loss
# (its own loss, reference-free). Its stored output is rates/pilot_seed1002.npz (AMD EPYC 7713 CPU).
#
#     python pilot_training.py              # writes outputs/pilot_seed1002.npz
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
N_x, N_t = 250, 32           # uniform collocation grid in (x, t)
learning_rate = 0.005        # constant Adam learning rate (no decay in the pilot)
max_itters = 1000            # Adam epochs
lbfgs_max_iter = 600         # L-BFGS iterations after Adam
hidden_size = 20             # 2 hidden tanh layers of 20 units
n_hidden = 2
seed = 1002

# the equation: u_t = gamma1 u_xx + gamma2 (u - u^3), x in [-1, 1] periodic, t in [0, 1], u(x, 0) = x^2 cos(pi x)
gamma1, gamma2 = 0.0001, 5.0
lambda_ic = 50.0             # weight of the initial-condition penalty
fourier_m = 10               # Fourier modes of the periodic x-embedding
REL_FLOOR = 0.01             # rate floor, as a fraction of max rate

out_file = os.path.join(HERE, "outputs", f"pilot_seed{seed}.npz")

# collocation points: uniform in (x, t); initial condition on the x grid
x = torch.linspace(-1, 1, N_x, device=device)
t = torch.linspace(0, 1, N_t, device=device)
X, T = torch.meshgrid(x, t, indexing="ij")
col = torch.stack([X.flatten(), T.flatten()], dim=1).requires_grad_(True)
x_ic = torch.linspace(-1, 1, N_x, device=device).reshape(-1, 1)
ic_pts = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)        # points (x, 0)
ic_target = x_ic**2 * torch.cos(np.pi * x_ic)                    # x^2 cos(pi x)


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
        # periodic in x exactly: the network sees [cos(k pi x), sin(k pi x) for k = 1..10, t]
        freqs = torch.arange(1, fourier_m + 1, device=x.device, dtype=x.dtype).reshape(1, -1)
        angle = torch.pi * x * freqs
        return self.net(torch.cat([torch.cat([torch.cos(angle), torch.sin(angle)], dim=1), tau], dim=1))


def loss_fn(model):
    # mean( (u_t - gamma1 u_xx - gamma2 (u - u^3))^2 ) + lambda_ic mean( (u(x, 0) - x^2 cos(pi x))^2 )
    u = model(col)
    u_ic = model(ic_pts)
    g = torch.autograd.grad(u, col, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx, du_dt = g[:, 0:1], g[:, 1:2]
    du2_dx2 = torch.autograd.grad(du_dx, col, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
    return (torch.mean((du_dt - gamma1 * du2_dx2 - gamma2 * (u - u**3))**2)
            + lambda_ic * torch.mean((u_ic - ic_target)**2))


# Adam
torch.manual_seed(seed)
np.random.seed(seed)
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_history = []
print(f"Pilot training, seed {seed}, device {device}")
for epoch in range(max_itters):
    loss = loss_fn(model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 500 == 0:
        print(f"  Adam {epoch:5d}  loss {loss.item():.4e}", flush=True)

# L-BFGS
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_max_iter, tolerance_grad=1e-9,
                                    tolerance_change=1e-12, history_size=50, line_search_fn="strong_wolfe")


def closure():
    optimizer_lbfgs.zero_grad()
    loss = loss_fn(model)
    loss.backward()
    loss_history.append(loss.item())
    return loss


optimizer_lbfgs.step(closure)
print(f"  final loss {loss_fn(model).item():.4e}", flush=True)

# ____rate file (never uses the reference)
# the trained network on the 200 x 200 (x, t) grid, kept in float32 as in the paper
x_plot = np.linspace(-1, 1, 200)
t_plot = np.linspace(0, 1, 200)
Xp, Tp = np.meshgrid(x_plot, t_plot, indexing="ij")
with torch.no_grad():
    pts = torch.tensor(np.stack([Xp.flatten(), Tp.flatten()], axis=1), dtype=torch.float32, device=device)
    U = model(pts).cpu().numpy().reshape(200, 200)
# rate(t) = mean_x max(0, gamma2 (1 - 3U^2) - gamma1 U_x^2), U_x by cubic splines in x. Late in time the growth
# rate is 0 almost everywhere, so the rate is floored at REL_FLOOR * max to keep the warp strictly increasing.
U_x = np.array([CubicSpline(x_plot, U[:, j])(x_plot, 1) for j in range(U.shape[1])]).T
lam = np.maximum(0.0, gamma2 * (1 - 3 * U**2) - gamma1 * U_x**2)
rate = lam.mean(axis=0)
rate = np.maximum(rate, REL_FLOOR * rate.max())
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, t=t_plot, rate=rate, x_plot=x_plot, t_plot=t_plot, U_pred=U)
print(f"rate file written: {out_file}  (R = max/min of rate = {rate.max() / rate.min():.4g})")

# plot: the pilot's solution, rate(t), loss curve
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
im = ax[0].pcolormesh(t_plot, x_plot, U, cmap="RdBu_r", shading="auto")
fig.colorbar(im, ax=ax[0])
ax[0].set_xlabel("t")
ax[0].set_ylabel("x")
ax[0].set_title("pilot solution u(x, t)")
ax[1].plot(t_plot, rate)
ax[1].set_xlabel("t")
ax[1].set_title("rate(t)")
ax[2].semilogy(loss_history)
ax[2].set_xlabel("Adam epoch / L-BFGS evaluation")
ax[2].set_title("loss")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", f"pilot_seed{seed}.png"), dpi=150)
