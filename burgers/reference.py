"""Reference solution for viscous Burgers (scoring only).

Method of lines on 3201 uniform x points (second-order central differences, u = 0 at x = +-1), integrated with
SciPy's BDF (rtol 1e-7, atol 1e-9), then cubic interpolation onto the 100 x 100 uniform scoring grid. This is within
0.01-0.04% (relative L2) of the converged solution.

    python reference.py      # writes data/reference_nx3201.npz (U, shape (100, 100) = (x, t))
"""
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import burgers_common as B


def _rhs(t, u_interior, dx, nu):
    u = np.concatenate(([0.0], u_interior, [0.0]))
    dudx = (u[2:] - u[:-2]) / (2 * dx)
    d2udx2 = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return -u[1:-1] * dudx + nu * d2udx2


def reference_solution(x_eval, t_eval, nu, nx_ref=3201):
    x_ref = np.linspace(-1, 1, nx_ref)
    dx = x_ref[1] - x_ref[0]
    sol = solve_ivp(_rhs, [t_eval.min(), t_eval.max()], -np.sin(np.pi * x_ref[1:-1]), t_eval=t_eval, args=(dx, nu),
                    method="BDF", rtol=1e-7, atol=1e-9)
    U_ref = np.zeros((nx_ref, len(t_eval)))
    U_ref[1:-1, :] = sol.y
    return np.stack([interp1d(x_ref, U_ref[:, j], kind="cubic")(x_eval) for j in range(len(t_eval))], axis=1)


if __name__ == "__main__":
    U = reference_solution(B.X_EVAL, B.T_EVAL, B.NU)
    os.makedirs(os.path.join(B.HERE, "data"), exist_ok=True)
    np.savez(os.path.join(B.HERE, "data", "reference_nx3201.npz"), U=U)
    print("written data/reference_nx3201.npz", U.shape)
