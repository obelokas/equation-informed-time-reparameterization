"""Bootstrap round 1 -- Allen-Cahn.

A warped training run on the rate from the previous stage (the pilot training). Its trained solution re-derives the rate with the
same formula as the pilot training; the reference solution is never used. The round uses a fixed small warp exponent
(beta = 0.1) and the sign envelope u = tanh(12 cos(pi x)) softplus(NN).

Seed 2000 is the kept run of this round: the paper ran seeds 2000-2009 on the kept input file and kept the one with
the lowest final training loss (reference-free). Its stored output is rates/bootstrap1_seed2000.npz
(NVIDIA RTX A4000 GPU).

    python bootstrap_round1.py          # reads rates/pilot_seed1002.npz, writes outputs/bootstrap1_seed2000.npz
"""
import argparse
import os

import numpy as np
import torch

import ac_common as A

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 2000
BETA = 0.1                   # warp exponent of the round (rate^beta)
HIDDEN = [32, 32, 32, 32]    # 4 hidden tanh layers of 32 units
N_X, N_T = 60, 8             # x uniform, sigma uniform (both ends included)
LBFGS = 1000                 # L-BFGS iterations from the start (no Adam)
PAPER_DEVICE = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=A.pick_device(PAPER_DEVICE))
    ap.add_argument("--rate-file", default=os.path.join(HERE, "rates/pilot_seed1002.npz"),
                    help="input rate file (default: the stored file from the paper)")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", f"bootstrap1_seed{SEED}.npz"))
    args = ap.parse_args()
    dev = A.setup(SEED, args.device)
    warp, col, ic_pts, ic_target, RATE, RATE_W, _ = A.warped_problem(args.rate_file, BETA, N_X, N_T, dev)
    model = A.PINN(HIDDEN, hard_sign=True, k_sign=12.0).to(dev)
    print(f"Bootstrap round 1, seed {SEED}, {N_X} x {N_T}, beta {BETA}, device {dev}")
    A.lbfgs(model, lambda: A.warped_loss(model, col, ic_pts, ic_target, RATE, RATE_W, warp.s_max), LBFGS)
    print(f"  final loss {A.warped_loss(model, col, ic_pts, ic_target, RATE, RATE_W, warp.s_max).item():.4e}")
    A.save_rate(args.out, A.predict_warped(model, warp, dev))


if __name__ == "__main__":
    main()
