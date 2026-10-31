"""Pilot training -- Allen-Cahn.

A small plain PINN in the physical time t on a 250 x 32 grid. Its only job is to give a first estimate of
rate(t) = mean_x max(0, gamma2 (1 - 3u^2) - gamma1 u_x^2); the reference solution is never used.

Seed 1002 is the kept pilot run: the paper ran seeds 1000-1009 and kept the one with the lowest final training loss
(its own loss, reference-free). Its stored output is rates/pilot_seed1002.npz (AMD EPYC 7713 CPU).

    python pilot_training.py              # writes outputs/pilot_seed1002.npz
"""
import argparse
import os

import numpy as np
import torch

import ac_common as A

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 1002
HIDDEN = [20, 20]            # 2 hidden tanh layers of 20 units
N_X, N_T = 250, 32           # uniform collocation grid in (x, t)
ADAM, LBFGS = 1000, 600
LR = 5e-3                    # constant Adam learning rate
PAPER_DEVICE = "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=PAPER_DEVICE)
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", f"pilot_seed{SEED}.npz"))
    args = ap.parse_args()
    dev = A.setup(SEED, args.device)
    x = torch.linspace(-1, 1, N_X, device=dev)
    t = torch.linspace(0, 1, N_T, device=dev)
    X, T = torch.meshgrid(x, t, indexing="ij")
    col = torch.stack([X.flatten(), T.flatten()], dim=1).requires_grad_(True)
    model = A.PINN(HIDDEN).to(dev)
    ic_pts, ic_target = A.initial_condition_points(torch.linspace(-1, 1, N_X, device=dev))

    def loss_fn():
        # mean( (u_t - gamma1 u_xx - gamma2 (u - u^3))^2 ) + 50 mean( (u(x, 0) - x^2 cos(pi x))^2 )
        u = model(col)
        u_ic = model(ic_pts)
        g = torch.autograd.grad(u, col, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dx, du_dt = g[:, 0:1], g[:, 1:2]
        du2_dx2 = torch.autograd.grad(du_dx, col, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
        return (torch.mean((du_dt - A.GAMMA1 * du2_dx2 - A.GAMMA2 * A.dF(u))**2)
                + A.LAMBDA_IC * torch.mean((u_ic - ic_target)**2))

    print(f"Pilot training, seed {SEED}, device {dev}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(ADAM):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 250 == 0:
            print(f"  Adam {ep:5d}  loss {loss.item():.4e}", flush=True)
    A.lbfgs(model, loss_fn, LBFGS)
    print(f"  final loss {loss_fn().item():.4e}")

    # the trained network on the 200 x 200 (x, t) grid -> rate(t)
    Xp, Tp = np.meshgrid(A.X_PLOT, A.T_PLOT, indexing="ij")
    with torch.no_grad():
        pts = torch.tensor(np.stack([Xp.flatten(), Tp.flatten()], axis=1), dtype=torch.float32, device=dev)
        U = model(pts).cpu().numpy().reshape(200, 200)
    A.save_rate(args.out, U)


if __name__ == "__main__":
    main()
