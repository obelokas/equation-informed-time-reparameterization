"""Pilot training -- Burgers.

A small hard-constrained PINN in the physical time t, on a coarse 128 x 16 grid. Its only job is to estimate
rate(t) = median_x (|u_x| + nu u_x^2); the reference solution is never used.

Seed 1000 is the kept pilot run: the paper ran seeds 1000-1009 and kept the one with the lowest final training loss
(its own loss, reference-free). Its stored output is rates/pilot_seed1000.npz (AMD EPYC 9575F CPU).

    python pilot_training.py              # writes outputs/pilot_seed1000.npz
"""
import argparse
import os

import numpy as np
import torch

import burgers_common as B

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 1000
WIDTH, DEPTH, L = 20, 3, 4   # 3 hidden tanh layers of 20 units; Fourier features sin/cos(2^k pi x), k < 4
N_X, N_T = 128, 16           # uniform collocation grid in (x, t)
ADAM, LBFGS = 1000, 600
LR = 5e-3                    # constant Adam learning rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", f"pilot_seed{SEED}.npz"))
    args = ap.parse_args()
    B.setup(SEED)
    model = B.Net(WIDTH, DEPTH, L)
    x, t = B.grid(np.linspace(-1, 1, N_X), np.linspace(0, 1, N_T))

    def loss_fn():
        u = model(x, t)
        u_x, u_xx, u_t = B.grads(u, x, t)
        return torch.mean((u_t + u * u_x - B.NU * u_xx) ** 2)

    print(f"Pilot training, seed {SEED}")
    B.train(model, loss_fn, ADAM, LBFGS, LR, lr_decay=False)
    rate = B.median_rate(model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, t=B.T_RATE, r_q50=rate)
    print(f"rate file written: {args.out}  (peak rate {rate.max():.4g} at t = {B.T_RATE[rate.argmax()]:.3f})")


if __name__ == "__main__":
    main()
