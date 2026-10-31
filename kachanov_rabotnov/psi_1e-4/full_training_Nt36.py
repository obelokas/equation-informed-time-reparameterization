"""Full training -- Kachanov-Rabotnov, psi = 1e-4, N_t = 36 collocation times.

Rate-scaled PINN in the warped time sigma = s(t) / s_max, s(t) = int_0^t rate^beta dt. The rate is the estimated one
(pilot training -> bootstrap round 1 -> bootstrap round 2); it never comes from the exact solution. The exact solution is only used at the end, to score.

Seed 92 is the best of the paper's ten final seeds 90-99 (by relative L2 error); the paper's tables report the
ten-seed median. Paper values for this cell: seed 92 0.00879%, ten-seed median 0.0879%
(run on Apple M3 CPU, PyTorch 2.12.1; rate files built on AMD EPYC 9575F).

    python full_training_Nt36.py       # uses the stored rate file rates/bootstrap2_seed3001.npz
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
N_T = 36                     # collocation times, uniform in sigma
BETA = 1.25                  # warp exponent
SEED = 92
WIDTH, DEPTH = 16, 2         # 2 hidden tanh layers of 16 units
ADAM, LBFGS = 3000, 2000
LR = 5e-3                    # Adam learning rate, decays exponentially to LR / 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-file", default=os.path.join(HERE, "rates/bootstrap2_seed3001.npz"),
                    help="rate file (default: the stored file from the paper)")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", f"full_Nt{N_T}_seed{SEED}.npz"))
    args = ap.parse_args()
    prob = K.Problem(PSI)
    warp = K.Warp(prob, *K.load_rate(args.rate_file), BETA)
    K.setup(SEED)
    sig = np.linspace(0, 1, N_T)                 # collocation points, uniform in the warped time sigma
    t_i = warp.t_of_sigma(sig)                   # their physical times (dense where the rate is large)
    r_i = warp.rate_fn(t_i)                      # rate(t_i)^beta
    tau = torch.tensor(sig, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    coef = torch.tensor(warp.s_max * K.A / r_i, dtype=torch.float64).reshape(-1, 1)

    def residual(m, s):
        # d omega / d sigma - s_max A (1 - omega)^(-phi) / rate(t)^beta  (chain rule of ds = rate^beta dt)
        u = m(s)
        du = torch.autograd.grad(u, s, torch.ones_like(u), create_graph=True)[0]
        return du - coef * K.one_minus(u)**(-K.PHI)

    model = K.Net(WIDTH, DEPTH)
    print(f"Full training, psi = {PSI:g}, N_t = {N_T}, beta {BETA}, seed {SEED}")
    K.train(model, tau, residual, ADAM, LBFGS, LR, lr_decay=True)

    # score on 1001 uniform physical times against the exact solution
    with torch.no_grad():
        s_eval = torch.tensor(warp.sigma_of_t(prob.t_eval), dtype=torch.float64).reshape(-1, 1)
        pred = model(s_eval).numpy().ravel()
    exact = K.omega_exact(prob.t_eval)
    l2 = K.relative_l2(pred, exact)
    print(f"relative L2 error: {100 * l2:.4f}%   (paper, this seed: 0.00879%)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, t=prob.t_eval, omega_pred=pred, omega_exact=exact, t_collocation=t_i, rel_l2=l2)


if __name__ == "__main__":
    main()
