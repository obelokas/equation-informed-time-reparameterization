"""Reference solution for Allen-Cahn (scoring only).

Fourier pseudo-spectral method in x (512 modes on the periodic interval [-1, 1)), SciPy BDF in time (rtol 1e-7,
atol 1e-9), then periodic cubic interpolation onto the 200 x 200 uniform (x, t) grid.

    python reference.py      # writes data/reference_200x200.npz (U_ref, shape (200, 200) = (x, t))
"""
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import ac_common as A


def _rhs(t, u, k2, g1, g2):
    u_xx = np.real(np.fft.ifft(-k2 * np.fft.fft(u)))
    return g1 * u_xx + g2 * (u - u**3)


def reference_solution(x_eval, t_eval, g1, g2, nx_ref=512):
    x_ref = np.linspace(-1, 1, nx_ref, endpoint=False)
    k = 2 * np.pi * np.fft.fftfreq(nx_ref, d=x_ref[1] - x_ref[0])
    sol = solve_ivp(_rhs, [t_eval.min(), t_eval.max()], x_ref**2 * np.cos(np.pi * x_ref), t_eval=t_eval,
                    args=(k**2, g1, g2), method="BDF", rtol=1e-7, atol=1e-9)
    x_wrap = np.concatenate([x_ref, [1.0]])
    U = np.zeros((len(x_eval), len(t_eval)))
    for j in range(len(t_eval)):
        U[:, j] = interp1d(x_wrap, np.concatenate([sol.y[:, j], [sol.y[0, j]]]), kind="cubic")(x_eval)
    return U


if __name__ == "__main__":
    U = reference_solution(A.X_PLOT, A.T_PLOT, A.GAMMA1, A.GAMMA2)
    os.makedirs(os.path.join(A.HERE, "data"), exist_ok=True)
    np.savez(os.path.join(A.HERE, "data", "reference_200x200.npz"), U_ref=U, x_plot=A.X_PLOT, t_plot=A.T_PLOT)
    print("written data/reference_200x200.npz", U.shape)
