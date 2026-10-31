# Bootstrap round 2 -- Kachanov-Rabotnov creep damage, psi = 1e-4.
#
# A warped training run on the rate from the previous stage (bootstrap round 1). Its trained omega(t) re-derives
# the rate (same writer as the pilot training), which is sharper near rupture than the pilot's. The exact solution
# is never used.
#
# Seed 3001 is the kept run of this round: the paper ran seeds 3000-3009 on the kept input file and kept the one with
# the lowest final training loss (reference-free). Its stored output is rates/bootstrap2_seed3001.npz
# (AMD EPYC 9575F CPU).
#
#     python bootstrap_round2.py          # reads rates/bootstrap1_seed2008.npz, writes outputs/bootstrap2_seed3001.npz
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
device = torch.device("cpu")             # the paper's Kachanov-Rabotnov runs were all on CPUs

HERE = os.path.dirname(os.path.abspath(__file__))

# hyperparams
N_t = 256                    # collocation times, uniform in sigma
learning_rate = 0.005        # flat Adam learning rate (no decay in the bootstrap rounds)
max_itters = 3000            # Adam epochs
lbfgs_max_iter = 2000        # L-BFGS iterations after Adam
hidden_size = 16             # 2 hidden tanh layers of 16 units
n_hidden = 2
seed = 3001
beta = 1.75                  # warp exponent of the round (rate^beta)

# the equation: d omega / dt = A (1 - omega)^(-fi), omega(0) = 0
# (GTD-111 at 816 C and 455 MPa, Stewart et al. 2011, Table 3)
M = 64.127 * 10**(-11)       # MPa^-xi h^-1
sigma = 455.0                # applied stress, MPa
xi = 2.257
fi = 3.792
A = M * sigma**xi            # right-hand side prefactor [1/h]
t_f = 1.0 / ((fi + 1) * A)   # rupture time (omega -> 1) [h]
psi = 1e-4                   # margin to rupture
t_end = (1 - psi) * t_f      # training stops here

# the stored paper file; point rate_file at outputs/bootstrap1_seed2008.npz to use your own round-1 output
rate_file = os.path.join(HERE, "rates/bootstrap1_seed2008.npz")
out_file = os.path.join(HERE, "outputs", f"bootstrap2_seed{seed}.npz")

# rate(t) and the warp s(t) = int_0^t rate^beta dt
data = np.load(rate_file)
t_rough, rate = data["t"], data["rate"]
rate_fn = CubicSpline(t_rough, rate**beta)       # rate(t)^beta
s_t = rate_fn.antiderivative()                   # s(t), the exact antiderivative of the spline
s_max = float(s_t(t_rough[-1]))
t_dense = np.linspace(0.0, t_end, 4001)          # dense grid for the inverse warp
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


# collocation points: uniform in the warped time sigma
sig = np.linspace(0, 1, N_t)
t_i = t_of_sigma(sig)                            # their physical times
r_i = rate_fn(t_i)                               # rate(t_i)^beta
tau = torch.tensor(sig, dtype=torch.float64, device=device).reshape(-1, 1).requires_grad_(True)
coef = torch.tensor(s_max * A / r_i, dtype=torch.float64, device=device).reshape(-1, 1)


# model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tau):
        # hard ansatz: omega = tau * NN(tau), so omega(0) = 0 exactly
        return tau * self.net(tau)


def residual(model):
    # d omega / d sigma - s_max A (1 - omega)^(-fi) / rate(t)^beta  (chain rule of ds = rate^beta dt);
    # 1 - omega is clamped only to keep a wild optimiser step from producing NaN
    u = model(tau)
    du = torch.autograd.grad(u, tau, torch.ones_like(u), create_graph=True)[0]
    return du - coef * torch.clamp(1.0 - u, min=1e-6)**(-fi)


# Adam
torch.manual_seed(seed)
np.random.seed(seed)
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_history = []
print(f"Bootstrap round 2, psi = {psi:g}, seed {seed}, beta {beta}, {N_t} columns")
for epoch in range(max_itters):
    optimizer.zero_grad()
    loss = torch.mean(residual(model)**2)
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
    loss = torch.mean(residual(model)**2)
    loss.backward()
    loss_history.append(loss.item())
    return loss


optimizer_lbfgs.step(closure)
print(f"  final loss {torch.mean(residual(model)**2).item():.4e}", flush=True)

# ____rate file (never uses the exact solution)
# 1. omega on 2000 uniform times plus t_end - t log-spaced down to 1e-7 t_end (dense near rupture),
#    read from the network at this round's own sigma(t)
t_c = np.concatenate([np.linspace(0.0, t_end, 2000), t_end - t_end * np.logspace(-7, 0, 3000)])
t_c = np.unique(np.clip(t_c, 0.0, t_end))
with torch.no_grad():
    s_c = torch.tensor(s_t(t_c) / s_max, dtype=torch.float64, device=device).reshape(-1, 1)
    om = model(s_c).cpu().numpy().ravel()
# 2. the trained omega must be finite, non-decreasing and < 1, otherwise the run gives no rate
if not np.all(np.isfinite(om)):
    raise ValueError("omega is not finite")
if np.any(np.diff(om) < 0):
    raise ValueError("omega is not monotone")
if np.any(om >= 1):
    raise ValueError("omega >= 1")
# 3. resample onto 1000 points uniform in omega (t from monotone interpolation of t(omega))
om_u = np.linspace(om[0], om[-1], 1000)
keep = np.concatenate([[True], np.diff(om) > 0])        # plateaus collapse to their first t
t_u = np.interp(om_u, om[keep], t_c[keep])
t_u[0], t_u[-1] = 0.0, t_end
if np.any(np.diff(t_u) <= 0):
    raise ValueError("t(omega) is not strictly increasing")
# 4. rate = A fi (1 - omega)^(-(fi + 1)) on those points
rate_new = A * fi * (1.0 - om_u)**(-(fi + 1))
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, t=t_u, omega=om_u, rate=rate_new)
print(f"rate file written: {out_file}  (omega(t_end) = {om_u[-1]:.6f})")

# plot: omega(t), input and output rate, loss curve
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(t_u, om_u)
ax[0].plot(t_i, np.zeros_like(t_i), "b|", label="collocation times")
ax[0].set_xlabel("t [h]")
ax[0].set_ylabel("omega (trained network)")
ax[0].legend()
ax[1].semilogy(t_rough, rate, "k-", label="input rate")
ax[1].semilogy(t_u, rate_new, "r--", label="this round's rate")
ax[1].set_xlabel("t [h]")
ax[1].set_title("rate(t) = A fi (1 - omega)^-(fi + 1)")
ax[1].legend()
ax[2].semilogy(loss_history)
ax[2].set_xlabel("Adam epoch / L-BFGS evaluation")
ax[2].set_title("loss")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", f"bootstrap2_seed{seed}.png"), dpi=150)
