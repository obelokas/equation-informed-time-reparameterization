# Reference solution -- Allen-Cahn (scoring only).
#
# Fourier pseudo-spectral method in x (512 modes on the periodic interval [-1, 1)), SciPy BDF in time (rtol 1e-7,
# atol 1e-9), then periodic cubic interpolation onto the 200 x 200 uniform (x, t) grid.
#
#     python reference.py      # writes data/reference_200x200.npz (U_ref, shape (200, 200) = (x, t))
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))

# the equation: u_t = gamma1 u_xx + gamma2 (u - u^3), x in [-1, 1] periodic, t in [0, 1], u(x, 0) = x^2 cos(pi x)
gamma1, gamma2 = 0.0001, 5.0
nx_ref = 512                           # Fourier modes
x_plot = np.linspace(-1, 1, 200)       # 200 x 200 scoring grid
t_plot = np.linspace(0, 1, 200)

x_ref = np.linspace(-1, 1, nx_ref, endpoint=False)
k = 2 * np.pi * np.fft.fftfreq(nx_ref, d=x_ref[1] - x_ref[0])
k2 = k**2


def rhs(t, u):
    # u_xx spectrally, the reaction term pointwise
    u_xx = np.real(np.fft.ifft(-k2 * np.fft.fft(u)))
    return gamma1 * u_xx + gamma2 * (u - u**3)


sol = solve_ivp(rhs, [t_plot.min(), t_plot.max()], x_ref**2 * np.cos(np.pi * x_ref), t_eval=t_plot, method="BDF",
                rtol=1e-7, atol=1e-9)
# periodic cubic interpolation in x onto the scoring grid, one time at a time
x_wrap = np.concatenate([x_ref, [1.0]])
U_ref = np.zeros((len(x_plot), len(t_plot)))
for j in range(len(t_plot)):
    U_ref[:, j] = interp1d(x_wrap, np.concatenate([sol.y[:, j], [sol.y[0, j]]]), kind="cubic")(x_plot)

os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
np.savez(os.path.join(HERE, "data", "reference_200x200.npz"), U_ref=U_ref, x_plot=x_plot, t_plot=t_plot)
print("written data/reference_200x200.npz", U_ref.shape)

# plot
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.pcolormesh(t_plot, x_plot, U_ref, cmap="RdBu_r", shading="auto")
fig.colorbar(im, ax=ax)
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_title("Allen-Cahn reference u(x, t)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", "reference.png"), dpi=150)
