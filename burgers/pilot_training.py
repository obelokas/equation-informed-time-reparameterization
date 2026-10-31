# Pilot training -- viscous Burgers.
#
# A small hard-constrained PINN in the physical time t, on a coarse 128 x 16 grid. Its only job is to estimate
# rate(t) = median_x (|u_x| + nu u_x^2); the reference solution is never used.
#
# Seed 1000 is the kept pilot run: the paper ran seeds 1000-1009 and kept the one with the lowest final training loss
# (its own loss, reference-free). Its stored output is rates/pilot_seed1000.npz (AMD EPYC 9575F CPU).
#
#     python pilot_training.py              # writes outputs/pilot_seed1000.npz
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)   # float64 everywhere
torch.set_num_threads(1)                 # one CPU thread: a run is bitwise reproducible on one machine type
device = torch.device("cpu")             # the paper's Burgers runs were all on CPUs

HERE = os.path.dirname(os.path.abspath(__file__))

# hyperparams
N_x, N_t = 128, 16           # uniform collocation grid in (x, t)
learning_rate = 0.005        # constant Adam learning rate (no decay in the pilot)
max_itters = 1000            # Adam epochs
lbfgs_max_iter = 600         # L-BFGS iterations after Adam
hidden_size = 20             # 3 hidden tanh layers of 20 units
n_hidden = 3
L = 4                        # Fourier features sin/cos(2^k pi x), k < L
seed = 1000

# the equation: u_t + u u_x = nu u_xx, x in [-1, 1], t in [0, 1], u(x, 0) = -sin(pi x), u(+-1, t) = 0
nu = 0.01 / np.pi

out_file = os.path.join(HERE, "outputs", f"pilot_seed{seed}.npz")


def grid(xs, ts):
    # tensor-product grid, x-major: row i * len(ts) + j is (xs[i], ts[j])
    X, T = np.meshgrid(xs, ts, indexing="ij")
    x = torch.tensor(X.reshape(-1, 1), dtype=torch.float64, device=device).requires_grad_(True)
    t = torch.tensor(T.reshape(-1, 1), dtype=torch.float64, device=device).requires_grad_(True)
    return x, t


# collocation points: uniform in (x, t)
x, t = grid(np.linspace(-1, 1, N_x), np.linspace(0, 1, N_t))


# model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        layers = [nn.Linear(1 + 2 * L + 1, hidden_size), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("freqs", (2.0 ** torch.arange(L)) * np.pi)

    def forward(self, x, tau):
        p = x * self.freqs.view(1, -1)
        raw = self.net(torch.cat([x, torch.sin(p), torch.cos(p), tau], dim=1))
        # hard ansatz: u = -sin(pi x) + t (1 - x^2) NN gives the initial and boundary conditions exactly
        return -torch.sin(np.pi * x) + tau * (1 - x**2) * raw


def loss_fn(model):
    # mean( (u_t + u u_x - nu u_xx)^2 )
    u = model(x, t)
    u_x, u_t = torch.autograd.grad(u, (x, t), torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return torch.mean((u_t + u * u_x - nu * u_xx) ** 2)


# Adam
torch.manual_seed(seed)
np.random.seed(seed)
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_history = []
print(f"Pilot training, seed {seed}")
for epoch in range(max_itters):
    optimizer.zero_grad()
    loss = loss_fn(model)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 500 == 0:
        print(f"  Adam {epoch:5d}  loss {loss.item():.4e}", flush=True)

# L-BFGS
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_max_iter, tolerance_grad=1e-15,
                                    tolerance_change=1e-16, history_size=50, line_search_fn="strong_wolfe")


def closure():
    optimizer_lbfgs.zero_grad()
    loss = loss_fn(model)
    loss.backward()
    loss_history.append(loss.item())
    return loss


optimizer_lbfgs.step(closure)
print(f"  final loss {loss_fn(model).item():.4e}", flush=True)

# ____rate file: rate(t) = median over x of |u_x| + nu u_x^2 (u_x by autograd), never uses the reference
# 4001 uniform x by 401 uniform t, in chunks of 10 times
t_rate = np.linspace(0.0, 1.0, 401)
x_rate = np.linspace(-1.0, 1.0, 4001)
r_q50 = np.zeros(len(t_rate))
for j0 in range(0, len(t_rate), 10):
    ts = t_rate[j0:j0 + 10]
    xr, tr = grid(x_rate, ts)
    u = model(xr, tr)
    u_x, _ = torch.autograd.grad(u, (xr, tr), torch.ones_like(u), create_graph=True)
    ux = np.abs(u_x.detach().cpu().numpy().reshape(len(x_rate), len(ts)))
    r_q50[j0:j0 + len(ts)] = np.percentile(ux + nu * ux**2, 50, axis=0)
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, t=t_rate, r_q50=r_q50)
print(f"rate file written: {out_file}  (peak rate {r_q50.max():.4g} at t = {t_rate[r_q50.argmax()]:.3f})")

# plot: the pilot's solution, rate(t), loss curve
x_plot, t_plot = np.linspace(-1, 1, 200), np.linspace(0, 1, 200)
Xp, Tp = np.meshgrid(x_plot, t_plot, indexing="ij")
with torch.no_grad():
    U = model(torch.tensor(Xp.reshape(-1, 1), device=device),
              torch.tensor(Tp.reshape(-1, 1), device=device)).cpu().numpy().reshape(Xp.shape)
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
im = ax[0].pcolormesh(t_plot, x_plot, U, cmap="RdBu_r", shading="auto")
fig.colorbar(im, ax=ax[0])
ax[0].set_xlabel("t")
ax[0].set_ylabel("x")
ax[0].set_title("pilot solution u(x, t)")
ax[1].semilogy(t_rate, r_q50)
ax[1].set_xlabel("t")
ax[1].set_title("rate(t) = median_x (|u_x| + nu u_x^2)")
ax[2].semilogy(loss_history)
ax[2].set_xlabel("Adam epoch / L-BFGS evaluation")
ax[2].set_title("loss")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", f"pilot_seed{seed}.png"), dpi=150)
