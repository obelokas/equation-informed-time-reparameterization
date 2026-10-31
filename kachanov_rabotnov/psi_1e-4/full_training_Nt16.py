# Full training -- Kachanov-Rabotnov creep damage, psi = 1e-4, N_t = 16 collocation times.
#
# Rate-scaled PINN in the warped time sigma = s(t) / s_max, s(t) = int_0^t rate^beta dt. The rate is the estimated
# one (pilot training -> bootstrap round 1 -> bootstrap round 2); it never comes from the exact solution. The exact
# solution is only used at the end, to score.
#
# Seed 90 is the best of the paper's ten final seeds 90-99 (by relative L2 error); the paper's tables report the
# ten-seed median. Paper values for this cell (Apple M3 CPU, PyTorch 2.12.1; rate files built on an AMD EPYC
# 9575F CPU): seed 90 0.296%, ten-seed median 0.386%.
#
#     python full_training_Nt16.py       # change the settings by editing the variables below
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
N_t = 16                     # collocation times, uniform in sigma
learning_rate = 0.005        # exp-decayed to lr/10 over the Adam budget
max_itters = 3000            # Adam epochs
lbfgs_max_iter = 2000        # L-BFGS iterations after Adam
hidden_size = 16             # 2 hidden tanh layers of 16 units
n_hidden = 2
seed = 90
beta = 1.0                # warp exponent

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

# the stored paper file; point rate_file at outputs/bootstrap2_seed3001.npz to use your own round-2 output
rate_file = os.path.join(HERE, "rates/bootstrap2_seed3001.npz")
out_file = os.path.join(HERE, "outputs", f"full_Nt{N_t}_seed{seed}.npz")

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
t_i = t_of_sigma(sig)                            # their physical times (dense where the rate is large)
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
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1**(1 / max_itters))
loss_history = []
print(f"Full training, psi = {psi:g}, N_t = {N_t}, beta {beta}, seed {seed}")
for epoch in range(max_itters):
    optimizer.zero_grad()
    loss = torch.mean(residual(model)**2)
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
    loss = torch.mean(residual(model)**2)
    loss.backward()
    loss_history.append(loss.item())
    return loss


optimizer_lbfgs.step(closure)
print(f"  final loss {torch.mean(residual(model)**2).item():.4e}", flush=True)

# ____L2 vs the exact solution (1 - omega)^(fi + 1) = 1 - (fi + 1) A t, on 1001 uniform physical times
t_eval = np.linspace(0.0, t_end, 1001)
with torch.no_grad():
    s_eval = torch.tensor(s_t(t_eval) / s_max, dtype=torch.float64, device=device).reshape(-1, 1)
    omega_pred = model(s_eval).cpu().numpy().ravel()
omega_exact = 1.0 - (1.0 - (fi + 1) * A * t_eval)**(1.0 / (fi + 1))
rel_l2 = float(np.linalg.norm(omega_pred - omega_exact) / np.linalg.norm(omega_exact))
print(f"relative L2 error: {100 * rel_l2:.4f}%   (paper, this seed: 0.296%)")
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, t=t_eval, omega_pred=omega_pred, omega_exact=omega_exact, t_collocation=t_i, rel_l2=rel_l2)

# plot: solution vs exact, abs error, loss curve
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(t_eval, omega_exact, "k-", label="exact")
ax[0].plot(t_eval, omega_pred, "r--", label="PINN")
ax[0].plot(t_i, np.zeros_like(t_i), "b|", label="collocation times")
ax[0].set_xlabel("t [h]")
ax[0].set_ylabel("omega")
ax[0].legend()
ax[1].semilogy(t_eval, np.abs(omega_pred - omega_exact))
ax[1].set_xlabel("t [h]")
ax[1].set_title(f"|error|, relative L2 {100 * rel_l2:.4f}%")
ax[2].semilogy(loss_history)
ax[2].set_xlabel("Adam epoch / L-BFGS evaluation")
ax[2].set_title("loss")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", f"full_Nt{N_t}_seed{seed}.png"), dpi=150)
print(f"written {out_file} and plots/full_Nt{N_t}_seed{seed}.png")
