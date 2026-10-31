"""Pilot training -- Frank-Kamenetskii, psi = 1e-5.

A small plain PINN on the normalised time tau = t / t_end, trained with Adam only. Its only job is to estimate
rate(t) = mean_x | delta e^u - u_x^2 | on a fine (x, t) grid; the reference solution is never used.

Seed 1001 is the kept pilot run: the paper ran seeds 1000-1009 and kept the one with the lowest final training loss
(its own loss, reference-free). Its stored output is rates/pilot_seed1001.npz (NVIDIA A100, float64).

    python pilot_training.py              # writes outputs/pilot_seed1001.npz
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fk_common as C  # noqa: E402
import torch  # noqa: E402  (after fk_common, which sets the cuBLAS determinism variable)

PSI = 1e-5
SEED = 1001
WIDTH, DEPTH = 16, 3         # 3 hidden tanh layers of 16 units
N_X, N_T = 32, 32            # uniform collocation grid in (x, tau), corners (x = +-1, tau = 0) removed
ADAM = 500                  # Adam epochs at a constant learning rate, no L-BFGS
LR = 3e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=C.default_device(), help="cuda (as in the paper) or cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", f"pilot_seed{SEED}.npz"))
    args = ap.parse_args()
    prob = C.Problem(PSI)
    C.setup(args.device)
    torch.manual_seed(SEED)
    model = C.Net(WIDTH, DEPTH)
    x = torch.linspace(-1, 1, N_X, dtype=torch.float64)
    t = torch.linspace(0, 1, N_T, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    pts = C.drop_corners(torch.stack([X.flatten(), T.flatten()], dim=1)).clone().requires_grad_(True)

    def loss_fn():
        # u_tau - t_end (u_xx + delta e^u): the equation in normalised time
        u, ut, uxx = C.derivs(model, pts)
        return torch.mean((ut - prob.t_end * (uxx + C.DELTA * torch.exp(u))) ** 2)

    print(f"Pilot training, psi = {PSI:g}, seed {SEED}, device {args.device}")
    C.train(model, loss_fn, ADAM, LR, 0)

    # the rate, from the trained network on 1001 x 801 points (time clustered toward t_end)
    U = C.eval_u(model, prob.x_rate, prob.t_rate / prob.t_end)
    rate = C.net_mean_rate(U, prob.x_rate)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, t=prob.t_rate, net_mean=rate)
    print(f"rate file written: {args.out}  (max/min of rate: {rate.max() / rate.min():.3g})")


if __name__ == "__main__":
    main()
