"""Pilot training -- Kachanov-Rabotnov, psi = 1e-3.

A small, cheap PINN on the normalised physical time tau = t / t_end. Its only job is to estimate rate(t); the
exact solution is never used. The trained network's omega(t) is turned into a rate file by K.write_rate.

Seed 1002 is the kept pilot run: the paper ran seeds 1000-1009 and kept the one with the lowest final training
loss (its own loss, reference-free). The stored output of that run is rates/pilot_seed1002.npz (AMD EPYC 7713 CPU).

    python pilot_training.py            # writes outputs/pilot_seed1002.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import kr_common as K  # noqa: E402

PSI = 1e-3
SEED = 1002
WIDTH, DEPTH = 8, 2          # 2 hidden tanh layers of 8 units
N_POINTS = 64                # collocation times, uniform in tau = t / t_end
ADAM, LBFGS = 1000, 2000
LR = 1e-3                    # Adam learning rate, decays exponentially to LR / 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", f"pilot_seed{SEED}.npz"))
    args = ap.parse_args()
    prob = K.Problem(PSI)
    K.setup(SEED)
    model = K.Net(WIDTH, DEPTH)
    tau = torch.linspace(0, 1, N_POINTS).reshape(-1, 1).requires_grad_(True)

    def residual(m, tt):
        # d omega / d tau - t_end * A (1 - omega)^(-phi): the equation in normalised time
        u = m(tt)
        du = torch.autograd.grad(u, tt, torch.ones_like(u), create_graph=True)[0]
        return du - prob.t_end * K.A * K.one_minus(u)**(-K.PHI)

    print(f"Pilot training, psi = {PSI:g}, seed {SEED}")
    K.train(model, tau, residual, ADAM, LBFGS, LR, lr_decay=True)

    def omega_fn(t):
        with torch.no_grad():
            return model(torch.tensor(t / prob.t_end, dtype=torch.float64).reshape(-1, 1)).numpy().ravel()

    out = K.write_rate(prob, omega_fn)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print(f"rate file written: {args.out}  (omega(t_end) = {out['omega'][-1]:.6f})")


if __name__ == "__main__":
    main()
