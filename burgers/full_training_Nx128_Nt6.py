# Full training -- viscous Burgers, N_x x N_t = 128 x 6 collocation points.
#
# Rate-scaled PINN in the warped time sigma = s(t) / s_max, s(t) = int_0^t rate^beta dt, with the rate from the pilot
# training (rates/pilot_seed1000.npz); the reference solution is only used at the end, to score.
#
# Seed 91 is the best of the paper's ten final seeds 90-99 (by relative L2 error); the paper's tables report the
# ten-seed median. Paper values for this cell (AMD EPYC 9575F CPU): seed 91 1.79%, ten-seed median 3.02%.
#
#     python full_training_Nx128_Nt6.py    # change the settings by editing the variables below
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

torch.set_default_dtype(torch.float64)   # float64 everywhere
torch.set_num_threads(1)                 # one CPU thread: a run is bitwise reproducible on one machine type
device = torch.device("cpu")             # the paper's Burgers runs were all on CPUs

HERE = os.path.dirname(os.path.abspath(__file__))

# hyperparams
N_x, N_t = 128, 6            # N_x uniform x by N_t times at the midpoints of a uniform sigma grid
learning_rate = 0.005        # exp-decayed to lr/10 over the Adam budget
max_itters = 4500            # Adam epochs
lbfgs_max_iter = 3000        # L-BFGS iterations after Adam
hidden_size = 20             # 3 hidden tanh layers of 20 units
n_hidden = 3
L = 4                        # Fourier features sin/cos(2^k pi x), k < L
seed = 91
beta = 0.85                  # warp exponent

# the equation: u_t + u u_x = nu u_xx, x in [-1, 1], t in [0, 1], u(x, 0) = -sin(pi x), u(+-1, t) = 0
nu = 0.01 / np.pi

# the stored paper file; point rate_file at outputs/pilot_seed1000.npz to use your own pilot output
rate_file = os.path.join(HERE, "rates/pilot_seed1000.npz")
out_file = os.path.join(HERE, "outputs", f"full_Nx{N_x}_Nt{N_t}_seed{seed}.npz")

# rate(t) and the warp s(t) = int_0^t rate^beta dt
data = np.load(rate_file)
t_rough, rate = data["t"], data["r_q50"]         # rate(t) = median_x (|u_x| + nu u_x^2) of the pilot
rate_fn = CubicSpline(t_rough, rate**beta)       # rate(t)^beta
s_t = rate_fn.antiderivative()                   # s(t), the exact antiderivative of the spline
s_max = float(s_t(t_rough[-1]))
t_dense = np.linspace(0.0, 1.0, 4001)            # dense grid for the inverse warp
if not np.all(rate_fn(t_dense) > 0):
    raise ValueError("rate^beta spline is not strictly positive -> the warp is not monotone")
s_dense = s_t(t_dense)


def t_of_sigma(sig):
    # inverse warp: interpolation on the dense grid, then 3 Newton steps
    target = sig * s_max
    t = np.interp(target, s_dense, t_dense)
    for _ in range(3):
        t = t - (s_t(t) - target) / rate_fn(t)
        t = np.clip(t, 0.0, t_rough[-1])
    return t


# collocation points: N_x uniform x by N_t times at the midpoints of a uniform sigma grid (x-major)
sig = (np.arange(N_t) + 0.5) / N_t
t_j = t_of_sigma(sig)                            # their physical times (dense where the rate is large)
r_j = rate_fn(t_j)                               # rate(t_j)^beta
X, S = np.meshgrid(np.linspace(-1, 1, N_x), sig, indexing="ij")
x = torch.tensor(X.reshape(-1, 1), dtype=torch.float64, device=device).requires_grad_(True)
s = torch.tensor(S.reshape(-1, 1), dtype=torch.float64, device=device).requires_grad_(True)
coef = torch.tensor(np.tile(r_j / s_max, N_x).reshape(-1, 1), dtype=torch.float64, device=device)
w = torch.tensor(r_j / r_j.mean(), dtype=torch.float64, device=device).reshape(1, -1)   # column weight rate^beta / mean


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

    def raw(self, x, tau):
        p = x * self.freqs.view(1, -1)
        return self.net(torch.cat([x, torch.sin(p), torch.cos(p), tau], dim=1))

    def forward(self, x, tau):
        # hard ansatz: u = -sin(pi x) + tau (1 - x^2) NN gives the initial and boundary conditions exactly
        # (keep this evaluation order: it fixes the order of autograd's floating-point sums, so runs are bitwise
        # reproducible against the paper's code)
        return -torch.sin(np.pi * x) + tau * (1 - x**2) * self.raw(x, tau)


def loss_fn(model):
    # (rate^beta / s_max) u_sigma + u u_x - nu u_xx: the physical-time residual at the warped points;
    # its square at column t_j is weighted by rate(t_j)^beta / mean_j rate(t_j)^beta
    u = model(x, s)
    u_x, u_s = torch.autograd.grad(u, (x, s), torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    r = coef * u_s + (u * u_x - nu * u_xx)
    return torch.mean(w * (r**2).reshape(N_x, N_t))


# Adam
torch.manual_seed(seed)
np.random.seed(seed)
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1**(1 / max_itters))
loss_history = []
print(f"Full training, {N_x} x {N_t}, beta {beta}, seed {seed}")
for epoch in range(max_itters):
    optimizer.zero_grad()
    loss = loss_fn(model)
    loss.backward()
    optimizer.step()
    scheduler.step()
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

# ____L2 vs the reference, on the 100 x 100 uniform (x, t) grid
x_eval = np.linspace(-1.0, 1.0, 100)
t_eval = np.linspace(0.0, 1.0, 100)
Xe, Se = np.meshgrid(x_eval, s_t(t_eval) / s_max, indexing="ij")    # the network sees this run's sigma(t)
with torch.no_grad():
    U = model(torch.tensor(Xe.reshape(-1, 1), device=device),
              torch.tensor(Se.reshape(-1, 1), device=device)).cpu().numpy().reshape(Xe.shape)
U_ref = np.load(os.path.join(HERE, "data", "reference_nx3201.npz"))["U"]       # written by reference.py
rel_l2 = float(np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref))
print(f"relative L2 error: {100 * rel_l2:.4f}%   (paper, this seed: 1.79%)")
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, x=x_eval, t=t_eval, U_pred=U, U_ref=U_ref, t_collocation=t_j, rel_l2=rel_l2)

# plot: solution vs reference, abs error, loss curve
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for a, field, title in [(ax[0], U_ref, "reference"), (ax[1], U, "PINN")]:
    im = a.pcolormesh(t_eval, x_eval, field, cmap="RdBu_r", shading="auto")
    fig.colorbar(im, ax=a)
    a.set_title(title)
ax[1].plot(t_j, np.zeros_like(t_j), "k|", ms=20, label="collocation times")
ax[1].legend(loc="upper right")
im = ax[2].pcolormesh(t_eval, x_eval, np.abs(U - U_ref), cmap="viridis", shading="auto")
fig.colorbar(im, ax=ax[2])
ax[2].set_title(f"|error|, relative L2 {100 * rel_l2:.4f}%")
for a in ax[:3]:
    a.set_xlabel("t")
    a.set_ylabel("x")
ax[3].semilogy(loss_history)
ax[3].set_xlabel("Adam epoch / L-BFGS evaluation")
ax[3].set_title("loss")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", f"full_Nx{N_x}_Nt{N_t}_seed{seed}.png"), dpi=150)
print(f"written {out_file} and plots/full_Nx{N_x}_Nt{N_t}_seed{seed}.png")
