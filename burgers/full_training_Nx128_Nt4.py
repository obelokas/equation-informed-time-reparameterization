"""Full training -- Burgers, N_x x N_t = 128 x 4 collocation points.

Rate-scaled PINN in the warped time sigma = s(t) / s_max, s(t) = int_0^t rate^beta dt, with the rate from the pilot
training (rates/pilot_seed1000.npz); the reference solution is only used at the end, to score.

Seed 97 is the best of the paper's ten final seeds 90-99 (by relative L2 error); the paper's tables report the
ten-seed median. Paper values for this cell (AMD EPYC 9575F CPU): seed 97 2.06%, ten-seed median 2.86%.

    python full_training_Nx128_Nt4.py     # uses the stored rate file rates/pilot_seed1000.npz
"""
import argparse
import os

import numpy as np
import torch

import burgers_common as B

HERE = os.path.dirname(os.path.abspath(__file__))

N_X, N_T = 128, 4            # N_X uniform x by N_T times at the midpoints of a uniform sigma grid
BETA = 0.85                  # warp exponent
SEED = 97
WIDTH, DEPTH, L = 20, 3, 4   # 3 hidden tanh layers of 20 units; Fourier features sin/cos(2^k pi x), k < 4
ADAM, LBFGS = 4500, 3000
LR = 5e-3                    # Adam learning rate, decays exponentially to LR / 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-file", default=os.path.join(HERE, "rates/pilot_seed1000.npz"),
                    help="rate file (default: the stored file from the paper)")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", f"full_Nx{N_X}_Nt{N_T}_seed{SEED}.npz"))
    args = ap.parse_args()
    B.setup(SEED)
    d = np.load(args.rate_file)
    warp = B.Warp(d["t"], d["r_q50"], BETA)
    sig = (np.arange(N_T) + 0.5) / N_T          # collocation times: midpoints of a uniform sigma grid
    t_j = warp.t_of_sigma(sig)                   # their physical times (dense where the rate is large)
    r_j = warp.rate_fn(t_j)                      # rate(t_j)^beta
    x, s = B.grid(np.linspace(-1, 1, N_X), sig)
    coef = torch.tensor(np.tile(r_j / warp.s_max, N_X).reshape(-1, 1), dtype=torch.float64)
    w = torch.tensor(r_j / r_j.mean(), dtype=torch.float64).reshape(1, -1)   # column weight rate^beta / mean
    model = B.Net(WIDTH, DEPTH, L)

    def loss_fn():
        # (rate^beta / s_max) u_sigma + u u_x - nu u_xx: the physical-time residual at the warped points
        u = model(x, s)
        u_x, u_xx, u_s = B.grads(u, x, s)
        r = coef * u_s + (u * u_x - B.NU * u_xx)
        return torch.mean(w * (r ** 2).reshape(N_X, N_T))

    print(f"Full training, {N_X} x {N_T}, beta {BETA}, seed {SEED}")
    B.train(model, loss_fn, ADAM, LBFGS, LR, lr_decay=True)

    # score on the 100 x 100 uniform grid against the reference
    U = B.predict(model, warp.sigma_of_t(B.T_EVAL))
    U_ref = B.reference()
    l2 = B.relative_l2(U, U_ref)
    print(f"relative L2 error: {100 * l2:.4f}%   (paper, this seed: 2.06%)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, x=B.X_EVAL, t=B.T_EVAL, U_pred=U, U_ref=U_ref, t_collocation=t_j, rel_l2=l2)


if __name__ == "__main__":
    main()
