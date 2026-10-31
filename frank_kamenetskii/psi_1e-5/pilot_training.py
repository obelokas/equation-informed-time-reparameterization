# Pilot training -- Frank-Kamenetskii thermal explosion, psi = 1e-5.
#
# A small plain PINN on the normalised time tau = t / t_end, trained with Adam only. Its only job is to estimate
# rate(t) = mean_x | delta e^u - u_x^2 | (the net growth coefficient) on a fine (x, t) grid; the reference solution
# is never used.
#
# Seed 1001 is the kept pilot run: the paper ran seeds 1000-1009 and kept the one with the lowest final training loss
# (its own loss, reference-free). Its stored output is rates/pilot_seed1001.npz (NVIDIA A100 GPU, float64).
#
#     python pilot_training.py              # writes outputs/pilot_seed1001.npz
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS; must be set before CUDA starts

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
N_x, N_t = 32, 32            # uniform collocation grid in (x, tau), corners (x = +-1, tau = 0) removed
learning_rate = 0.003        # constant Adam learning rate
max_itters = 500             # Adam epochs, no L-BFGS
hidden_size = 16             # 3 hidden tanh layers of 16 units
n_hidden = 3
seed = 1001

# the equation: u_t = u_xx + delta e^u, x in [-1, 1], u(x, 0) = 0, u(+-1, t) = 0; blow-up at T_star (reference.py)
delta = 1.3177
T_star = 1.4467001610282852
psi = 1e-5                   # margin to blow-up
t_end = (1 - psi) * T_star   # training stops here

out_file = os.path.join(HERE, "outputs", f"pilot_seed{seed}.npz")

# collocation points: uniform in (x, tau); the two corners (x = +-1, tau = 0), where the ansatz makes every term
# vanish, are removed
x = torch.linspace(-1, 1, N_x, dtype=torch.float64)
t = torch.linspace(0, 1, N_t, dtype=torch.float64)
X, T = torch.meshgrid(x, t, indexing="ij")
pts = torch.stack([X.flatten(), T.flatten()], dim=1)
corner = (pts[:, 1] == 0) & (pts[:, 0].abs() == 1)
pts = pts[~corner].clone().requires_grad_(True)


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
        # hard ansatz: u = tau (1 - x^2) NN(x, tau) satisfies u(x, 0) = 0 and u(+-1, t) = 0 exactly
        return tau * (1 - x ** 2) * self.net(xs)


def loss_fn(model):
    # u_tau - t_end (u_xx + delta e^u): the equation in normalised time
    u = model(pts)
    g = torch.autograd.grad(u, pts, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0:1], pts, torch.ones_like(u), create_graph=True)[0][:, 0:1]
    ut = g[:, 1:2]
    return torch.mean((ut - t_end * (uxx + delta * torch.exp(u))) ** 2)


def eval_u(model, x, tau, chunk=200000):
    # the network on the tensor grid x (nx) by tau (nt), in chunks -> numpy array (nx, nt)
    X, S = np.meshgrid(x, tau, indexing="ij")
    p = np.stack([X.ravel(), S.ravel()], axis=1)
    out = np.empty(len(p))
    with torch.no_grad():
        for a in range(0, len(p), chunk):
            out[a:a + chunk] = model(torch.tensor(p[a:a + chunk], dtype=torch.float64)).cpu().numpy().ravel()
    return out.reshape(len(x), len(tau))


# Adam, constant learning rate; stops if the loss becomes non-finite
torch.manual_seed(seed)
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_history = []
print(f"Pilot training, psi = {psi:g}, seed {seed}, device {device}")
for epoch in range(max_itters):
    loss = loss_fn(model)
    if not torch.isfinite(loss):
        print("  non-finite loss, Adam stopped", flush=True)
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 500 == 0:
        print(f"  Adam {epoch:5d}  loss {loss.item():.4e}", flush=True)
print(f"  final loss {loss_fn(model).item():.4e}", flush=True)

# ____rate file (never uses the reference): rate(t) = mean_x | delta e^U - U_x^2 | from the trained network on
# 1001 x 801 points, time clustered toward t_end
x_rate = np.linspace(-1, 1, 1001)
t_rate = t_end * (1 - (1 - np.linspace(0, 1, 801)) ** 3)
U = eval_u(model, x_rate, t_rate / t_end)
U_x = np.gradient(U, x_rate, axis=0, edge_order=2)
rate = np.abs(delta * np.exp(U) - U_x ** 2).mean(axis=0)
os.makedirs(os.path.dirname(out_file), exist_ok=True)
np.savez(out_file, t=t_rate, net_mean=rate)
print(f"rate file written: {out_file}  (max/min of rate: {rate.max() / rate.min():.3g})")

# plot: the pilot's solution, rate(t), loss curve
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
im = ax[0].pcolormesh(t_rate, x_rate, U, cmap="inferno", shading="auto")
fig.colorbar(im, ax=ax[0])
ax[0].set_xlabel("t")
ax[0].set_ylabel("x")
ax[0].set_title("pilot solution u(x, t)")
ax[1].semilogy(t_rate, rate)
ax[1].set_xlabel("t")
ax[1].set_title("rate(t) = mean_x |delta e^u - u_x^2|")
ax[2].semilogy(loss_history)
ax[2].set_xlabel("Adam epoch")
ax[2].set_title("loss")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", f"pilot_seed{seed}.png"), dpi=150)
