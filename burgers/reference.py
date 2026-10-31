# Reference solution -- viscous Burgers (scoring only).
#
# Method of lines on 3201 uniform x points (second-order central differences, u = 0 at x = +-1), integrated with
# SciPy's BDF (rtol 1e-7, atol 1e-9), then cubic interpolation onto the 100 x 100 uniform scoring grid. This is
# within 0.01-0.04% (relative L2) of the converged solution.
#
#     python reference.py      # writes data/reference_nx3201.npz (U, shape (100, 100) = (x, t))
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))

# the equation: u_t + u u_x = nu u_xx, x in [-1, 1], t in [0, 1], u(x, 0) = -sin(pi x), u(+-1, t) = 0
nu = 0.01 / np.pi
nx_ref = 3201                          # finite-difference points in x
x_eval = np.linspace(-1.0, 1.0, 100)   # scoring grid (100 x 100 uniform)
t_eval = np.linspace(0.0, 1.0, 100)

x_ref = np.linspace(-1, 1, nx_ref)
dx = x_ref[1] - x_ref[0]


def rhs(t, u_interior):
    # -u u_x + nu u_xx with central differences; u = 0 at both ends
    u = np.concatenate(([0.0], u_interior, [0.0]))
    dudx = (u[2:] - u[:-2]) / (2 * dx)
    d2udx2 = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return -u[1:-1] * dudx + nu * d2udx2


sol = solve_ivp(rhs, [t_eval.min(), t_eval.max()], -np.sin(np.pi * x_ref[1:-1]), t_eval=t_eval, method="BDF",
                rtol=1e-7, atol=1e-9)
U_full = np.zeros((nx_ref, len(t_eval)))
U_full[1:-1, :] = sol.y
# cubic interpolation in x onto the scoring grid, one time at a time
U = np.stack([interp1d(x_ref, U_full[:, j], kind="cubic")(x_eval) for j in range(len(t_eval))], axis=1)

os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
np.savez(os.path.join(HERE, "data", "reference_nx3201.npz"), U=U)
print("written data/reference_nx3201.npz", U.shape)

# plot
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.pcolormesh(t_eval, x_eval, U, cmap="RdBu_r", shading="auto")
fig.colorbar(im, ax=ax)
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_title("Burgers reference u(x, t)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", "reference.png"), dpi=150)
