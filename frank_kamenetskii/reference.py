"""High-accuracy reference for u_t = u_xx + delta e^u, x in [-1, 1], u(x, 0) = 0, u(+-1, t) = 0 (scoring only).

Method: the solution is even in x, so solve on [0, 1] only (u_x(0) = 0, u(1) = 0) with second-order finite differences
on a fine uniform grid (h = 2.5e-4) and implicit BDF in time with a sparse analytic Jacobian, rtol 1e-11. Blow-up time:
integrate until max u reaches 14, then add the remaining ODE-like time e^-14 / delta (the source dominates there).
The solve is stored on 1001 x-nodes by 1501 t-nodes clustered toward t_end, and read through a bicubic spline.

    python reference.py      # prints T*, then writes data/ref_eval_psi*.npz and data/ref_evalc_psi*.npz
                             # (the uniform and the t_end-clustered scoring grids of fk_common.Problem)
"""
import os

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline

import fk_common as C

N_SOLVE = 4000         # grid intervals on [0, 1] (h = 2.5e-4)
RTOL, ATOL = 1e-11, 1e-13
L_STOP = 14.0          # max-u level at which the blow-up solve stops
N_STORE_X = 1001       # stored x nodes on [0, 1]
N_STORE_T = 1501       # stored t nodes, clustered toward t_end
DATA = os.path.join(C.HERE, "data")


def _operators(delta, n):
    h = 1.0 / n
    up = np.ones(n - 1)
    up[0] = 2.0                                            # row 0: Neumann condition via a mirror ghost node
    lap = sp.diags([np.ones(n - 1), -2 * np.ones(n), up], [-1, 0, 1], format="csc") / h**2
    rhs = lambda t, u: lap @ u + delta * np.exp(u)
    jac = lambda t, u: (lap + sp.diags(delta * np.exp(u), 0)).tocsc()
    return rhs, jac


def blowup_time(delta, n=N_SOLVE, level=L_STOP):
    rhs, jac = _operators(delta, n)
    hit = lambda t, u: u[0] - level
    hit.terminal = True
    sol = solve_ivp(rhs, [0, 10.0], np.zeros(n), method="BDF", jac=jac, events=hit, rtol=RTOL, atol=ATOL)
    return sol.t_events[0][0] + np.exp(-level) / delta


class FKReference:
    """u_ref(x, t) for t in [0, t_end]; bicubic spline of the fine solve (cached in data/)."""

    def __init__(self, delta, t_end, n=N_SOLVE):
        path = os.path.join(DATA, f"fk_reference_cache_d{delta:.6f}_te{t_end:.6f}_n{n}.npz")
        if os.path.exists(path):
            z = np.load(path)
            self.x_store, self.t_store, self.U = z["x"], z["t"], z["U"]
        else:
            self.x_store = np.linspace(0, 1, N_STORE_X)
            s = np.linspace(0, 1, N_STORE_T)
            self.t_store = t_end * (1 - (1 - s) ** 3)
            rhs, jac = _operators(delta, n)
            sol = solve_ivp(rhs, [0, t_end], np.zeros(n), method="BDF", jac=jac, t_eval=self.t_store, rtol=RTOL, atol=ATOL)
            assert sol.success and sol.y.shape[1] == N_STORE_T, sol.message
            x_fine = np.arange(n) / n
            self.U = np.stack([np.interp(self.x_store, x_fine, sol.y[:, j]) for j in range(N_STORE_T)])   # (t, x)
            os.makedirs(DATA, exist_ok=True)
            np.savez(path, x=self.x_store, t=self.t_store, U=self.U)
        self.t_end = float(t_end)
        self.U[:, -1] = 0.0                                # u(1, t) = 0; the solve grid stops at x = 1 - h
        self._spl = RectBivariateSpline(self.t_store, self.x_store, self.U, kx=3, ky=3)

    def grid(self, x, t):
        """u_ref on the tensor grid x by t, shape (len(x), len(t))."""
        X, T = np.broadcast_arrays(np.asarray(x, float)[:, None], np.asarray(t, float)[None, :])
        return self._spl.ev(np.clip(T, 0, self.t_end).ravel(), np.abs(X).ravel()).reshape(X.shape)


if __name__ == "__main__":
    print(f"T* = {blowup_time(C.DELTA):.10f}   (fk_common.T_STAR = {C.T_STAR:.10f})")
    for psi in (3e-3, 1e-5):
        p = C.Problem(psi)
        ref = FKReference(C.DELTA, p.t_end)
        np.savez_compressed(os.path.join(DATA, f"ref_eval_{p.tag}.npz"), U=ref.grid(p.x_eval, p.t_eval))
        np.savez_compressed(os.path.join(DATA, f"ref_evalc_{p.tag}.npz"), U=ref.grid(p.x_eval, p.t_eval_c))
        print(f"psi = {psi:g}: t_end = {p.t_end:.6f}, max u(t_end) = {ref.grid([0.0], [p.t_end])[0, 0]:.4f}")
