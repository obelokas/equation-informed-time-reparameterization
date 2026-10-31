# Reference solution -- Frank-Kamenetskii thermal explosion (scoring only).
#
# u_t = u_xx + delta e^u, x in [-1, 1], u(x, 0) = 0, u(+-1, t) = 0. The solution is even in x, so it is solved on
# [0, 1] only (u_x(0) = 0, u(1) = 0) with second-order finite differences on a fine uniform grid (h = 2.5e-4) and
# implicit BDF in time with a sparse analytic Jacobian, rtol 1e-11. Blow-up time: integrate until max u reaches 14,
# then add the remaining ODE-like time e^-14 / delta (the source dominates there). For each margin psi the solve is
# stored on 1001 x-nodes by 1501 t-nodes clustered toward t_end, and read through a bicubic spline.
#
#     python reference.py      # prints T*, then writes data/ref_eval_psi*.npz and data/ref_evalc_psi*.npz
#                              # (the uniform and the t_end-clustered scoring grids of the full trainings)
import os

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

delta = 1.3177
T_star = 1.4467001610282852    # the blow-up time used by the training scripts (printed again below)
n_solve = 4000                 # grid intervals on [0, 1] (h = 2.5e-4)
rtol, atol = 1e-11, 1e-13
level_stop = 14.0              # max-u level at which the blow-up solve stops
n_store_x = 1001               # stored x nodes on [0, 1]
n_store_t = 1501               # stored t nodes, clustered toward t_end

# the semi-discrete system on x_k = k h, k = 0..n-1 (u(1) = 0 is the boundary node)
h = 1.0 / n_solve
up = np.ones(n_solve - 1)
up[0] = 2.0                                        # row 0: Neumann condition via a mirror ghost node
lap = sp.diags([np.ones(n_solve - 1), -2 * np.ones(n_solve), up], [-1, 0, 1], format="csc") / h**2


def rhs(t, u):
    return lap @ u + delta * np.exp(u)


def jac(t, u):
    return (lap + sp.diags(delta * np.exp(u), 0)).tocsc()


# ____blow-up time
def hit(t, u):
    return u[0] - level_stop


hit.terminal = True
sol = solve_ivp(rhs, [0, 10.0], np.zeros(n_solve), method="BDF", jac=jac, events=hit, rtol=rtol, atol=atol)
T_blowup = sol.t_events[0][0] + np.exp(-level_stop) / delta
print(f"T* = {T_blowup:.10f}   (T_star in the training scripts = {T_star:.10f})")

# ____reference on the scoring grids, for each margin
os.makedirs(DATA, exist_ok=True)
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
for psi in (3e-3, 1e-5):
    t_end = (1 - psi) * T_star
    tag = f"psi{psi:g}"
    # the fine solve, stored on 1001 x 1501 nodes (cached in data/)
    path = os.path.join(DATA, f"fk_reference_cache_d{delta:.6f}_te{t_end:.6f}_n{n_solve}.npz")
    if os.path.exists(path):
        z = np.load(path)
        x_store, t_store, U_store = z["x"], z["t"], z["U"]
    else:
        x_store = np.linspace(0, 1, n_store_x)
        t_store = t_end * (1 - (1 - np.linspace(0, 1, n_store_t)) ** 3)
        sol = solve_ivp(rhs, [0, t_end], np.zeros(n_solve), method="BDF", jac=jac, t_eval=t_store, rtol=rtol,
                        atol=atol)
        assert sol.success and sol.y.shape[1] == n_store_t, sol.message
        x_fine = np.arange(n_solve) / n_solve
        U_store = np.stack([np.interp(x_store, x_fine, sol.y[:, j]) for j in range(n_store_t)])   # (t, x)
        np.savez(path, x=x_store, t=t_store, U=U_store)
    U_store[:, -1] = 0.0                           # u(1, t) = 0; the solve grid stops at x = 1 - h
    spline = RectBivariateSpline(t_store, x_store, U_store, kx=3, ky=3)

    def u_ref(x, t):
        # u_ref on the tensor grid x by t, shape (len(x), len(t)); the solution is even in x
        X, T = np.broadcast_arrays(np.asarray(x, float)[:, None], np.asarray(t, float)[None, :])
        return spline.ev(np.clip(T, 0, t_end).ravel(), np.abs(X).ravel()).reshape(X.shape)

    # the scoring grids of the full trainings: 2000 x 200 uniform, and 2000 x 200 clustered toward t_end (peak)
    x_eval = np.linspace(-1, 1, 2000)
    t_eval = np.linspace(0, t_end, 200)
    t_eval_c = t_end * (1 - (1 - np.linspace(0, 1, 200)) ** 3)
    U = u_ref(x_eval, t_eval)
    np.savez_compressed(os.path.join(DATA, f"ref_eval_{tag}.npz"), U=U)
    np.savez_compressed(os.path.join(DATA, f"ref_evalc_{tag}.npz"), U=u_ref(x_eval, t_eval_c))
    print(f"psi = {psi:g}: t_end = {t_end:.6f}, max u(t_end) = {u_ref([0.0], [t_end])[0, 0]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.pcolormesh(t_eval, x_eval, U, cmap="inferno", shading="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(f"Frank-Kamenetskii reference, psi = {psi:g}")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plots", f"reference_{tag}.png"), dpi=150)
    plt.close(fig)
