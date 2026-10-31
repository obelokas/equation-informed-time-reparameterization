"""Full training -- Frank-Kamenetskii, psi = 1e-5, N_x x N_t = 256 x 80 collocation points.

Rate-scaled PINN in the warped time sigma = s(t) / s_max, s(t) = int_0^t (rate + 1e-6)^beta dt, with the rate from
the pilot training (rates/pilot_seed1001.npz); the reference solution is only used at the end, to score.

Seed 90 is the best of the paper's ten final seeds 90-99 (by relative L2 error); the paper's tables report the
ten-seed median. Paper values for this cell (NVIDIA A100, float64): seed 90 relative L2 6.89%, peak error
43.6%; ten-seed median relative L2 7.89%, peak error 47.4%.

    python full_training_Nx256_Nt80.py    # uses the stored rate file rates/pilot_seed1001.npz
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
N_X, N_T = 256, 80           # collocation grid: N_X uniform x by N_T times uniform in sigma (corners removed)
BETA = 1.0                   # warp exponent
SEED = 90
WIDTH, DEPTH = 64, 4         # 4 hidden tanh layers of 64 units
LBFGS = 10000                # L-BFGS iterations from the start (no Adam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=C.default_device(), help="cuda (as in the paper) or cpu")
    ap.add_argument("--rate-file", default=os.path.join(HERE, "rates/pilot_seed1001.npz"),
                    help="rate file (default: the stored file from the paper)")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", f"full_Nx{N_X}_Nt{N_T}_seed{SEED}.npz"))
    args = ap.parse_args()
    prob = C.Problem(PSI)
    C.setup(args.device)
    torch.manual_seed(SEED)
    d = np.load(args.rate_file)
    assert np.allclose(d["t"], prob.t_rate), "rate file is not on this margin's rate grid"
    warp = C.Warp(prob.t_rate, d["net_mean"] + C.EPS, BETA)
    sig = np.linspace(0, 1, N_T)                 # collocation times, uniform in the warped time sigma
    t_i = warp.t_of_sigma(sig)                   # their physical times (dense where the rate is large)
    rate_i = warp.rate_fn(t_i)                   # rho(t_i)^beta
    x = torch.linspace(-1, 1, N_X, dtype=torch.float64)
    X, S = torch.meshgrid(x, torch.tensor(sig, dtype=torch.float64), indexing="ij")
    pts_all = torch.stack([X.flatten(), S.flatten()], dim=1)
    RATE_all = torch.tensor(rate_i, dtype=torch.float64).view(1, -1).expand(N_X, -1).reshape(-1, 1)
    corner = (pts_all[:, 1] == 0) & (pts_all[:, 0].abs() == 1)
    pts = pts_all[~corner].clone().requires_grad_(True)
    RATE = RATE_all[~corner]
    model = C.Net(WIDTH, DEPTH)

    def loss_fn():
        # u_sigma - (s_max / rho^beta) (u_xx + delta e^u)  (chain rule of ds = rho^beta dt)
        u, us, uxx = C.derivs(model, pts)
        return torch.mean((us - (warp.s_max / RATE) * (uxx + C.DELTA * torch.exp(u))) ** 2)

    print(f"Full training, psi = {PSI:g}, {N_X} x {N_T}, beta {BETA}, seed {SEED}, device {args.device}")
    C.train(model, loss_fn, 0, 0.0, LBFGS)

    # score against the reference: relative L2 on 2000 x 200 uniform points, peak error at t_end
    U_ref, U_ref_c = prob.reference()
    U = C.eval_u(model, prob.x_eval, warp.sigma(prob.t_eval))
    U_c = C.eval_u(model, prob.x_eval, warp.sigma(prob.t_eval_c))
    rel_l2, peak = C.scores(U, U_c, U_ref, U_ref_c)
    print(f"relative L2 error: {100 * rel_l2:.4f}%   peak error: {100 * peak:.3f}%   "
          f"(paper, this seed: 6.89% / 43.6%)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, x=prob.x_eval, t=prob.t_eval, U_pred=U, U_ref=U_ref, t_collocation=t_i, rel_l2=rel_l2, peak_relerr=peak)


if __name__ == "__main__":
    main()
