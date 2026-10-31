"""Full training -- Allen-Cahn, N_x x N_t = 256 x 16 collocation points.

Rate-scaled PINN in the warped time sigma = s(t) / s_max, s(t) = int_0^t rate^beta dt. The rate is the final rate:
pilot training -> bootstrap round 1 -> bootstrap round 2 (rates/bootstrap2_seed3002.npz). The reference solution is
only used at the end, to score. beta = kappa / ln R (R = max/min of the final rate = 10.73), kappa = 1.8.

Seed 94 is the best of the paper's ten final seeds 90-99 (by relative L2 error); the paper's tables report the
ten-seed median. Paper values for this cell (NVIDIA RTX 3090 GPU): seed 94 0.13%, ten-seed median 0.3%.

    python full_training_Nt16.py        # uses the stored rate file rates/bootstrap2_seed3002.npz
"""
import argparse
import os

import numpy as np
import torch

import ac_common as A

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 94
BETA = 0.758644               # warp exponent, kappa / ln R with kappa = 1.8
HIDDEN = [64, 64, 64, 64]    # 4 hidden tanh layers of 64 units
N_X, N_T = 256, 16           # x uniform, sigma uniform (both ends included)
LBFGS = 10000                # L-BFGS iterations from the start (no Adam)
PAPER_DEVICE = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=A.pick_device(PAPER_DEVICE))
    ap.add_argument("--rate-file", default=os.path.join(HERE, "rates/bootstrap2_seed3002.npz"),
                    help="rate file (default: the stored file from the paper)")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", f"full_Nt{N_T}_seed{SEED}.npz"))
    args = ap.parse_args()
    dev = A.setup(SEED, args.device)
    warp, col, ic_pts, ic_target, RATE, RATE_W, t_i = A.warped_problem(args.rate_file, BETA, N_X, N_T, dev)
    model = A.PINN(HIDDEN).to(dev)
    print(f"Full training, 256 x {N_T}, beta {BETA}, seed {SEED}, device {dev}")
    A.lbfgs(model, lambda: A.warped_loss(model, col, ic_pts, ic_target, RATE, RATE_W, warp.s_max), LBFGS)
    print(f"  final loss {A.warped_loss(model, col, ic_pts, ic_target, RATE, RATE_W, warp.s_max).item():.4e}")

    # score on the 200 x 200 (x, t) grid against the reference
    U = A.predict_warped(model, warp, dev)
    U_ref = A.reference()
    l2 = A.relative_l2(U, U_ref)
    print(f"relative L2 error: {100 * l2:.3f}%   (paper, this seed: 0.13%)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, x=A.X_PLOT, t=A.T_PLOT, U_pred=U, U_ref=U_ref, t_collocation=t_i, rel_l2=l2)


if __name__ == "__main__":
    main()
