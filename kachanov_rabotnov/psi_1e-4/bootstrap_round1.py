"""Bootstrap round 1 -- Kachanov-Rabotnov, psi = 1e-4.

A warped training run on the rate from the previous stage (the pilot training). Its trained omega(t) re-derives the rate
(same writer as the pilot training), which is sharper near rupture than the pilot's. The exact solution is never used.

Seed 2008 is the kept run of this round: the paper ran seeds 2000-2009 on the kept input file and kept the one with
the lowest final training loss (reference-free). Its stored output is rates/bootstrap1_seed2008.npz (AMD EPYC 9575F).

    python bootstrap_round1.py          # reads rates/pilot_seed1008.npz, writes outputs/bootstrap1_seed2008.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import kr_common as K  # noqa: E402

PSI = 1e-4
SEED = 2008
BETA = 1.75                  # warp exponent of the round (rate^beta)
N_T = 64                     # collocation times, uniform in sigma
WIDTH, DEPTH = 16, 2
ADAM, LBFGS = 1000, 600
LR = 5e-3                    # flat Adam learning rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-file", default=os.path.join(HERE, "rates/pilot_seed1008.npz"),
                    help="input rate file (default: the stored file from the paper)")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", f"bootstrap1_seed{SEED}.npz"))
    args = ap.parse_args()
    prob = K.Problem(PSI)
    warp = K.Warp(prob, *K.load_rate(args.rate_file), BETA)
    K.setup(SEED)
    sig = np.linspace(0, 1, N_T)                 # collocation points, uniform in the warped time sigma
    t_i = warp.t_of_sigma(sig)                   # their physical times
    r_i = warp.rate_fn(t_i)                      # rate(t_i)^beta
    tau = torch.tensor(sig, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    coef = torch.tensor(warp.s_max * K.A / r_i, dtype=torch.float64).reshape(-1, 1)

    def residual(m, s):
        # d omega / d sigma - s_max A (1 - omega)^(-phi) / rate(t)^beta  (chain rule of ds = rate^beta dt)
        u = m(s)
        du = torch.autograd.grad(u, s, torch.ones_like(u), create_graph=True)[0]
        return du - coef * K.one_minus(u)**(-K.PHI)

    model = K.Net(WIDTH, DEPTH)
    print(f"Bootstrap round 1, psi = {PSI:g}, seed {SEED}, beta {BETA}, {N_T} columns")
    K.train(model, tau, residual, ADAM, LBFGS, LR, lr_decay=False)

    def omega_fn(t):
        # omega at physical time t = network evaluated at this round's own sigma(t)
        with torch.no_grad():
            s = torch.tensor(warp.sigma_of_t(t), dtype=torch.float64).reshape(-1, 1)
            return model(s).numpy().ravel()

    out = K.write_rate(prob, omega_fn)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print(f"rate file written: {args.out}  (omega(t_end) = {out['omega'][-1]:.6f})")


if __name__ == "__main__":
    main()
