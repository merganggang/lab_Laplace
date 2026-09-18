"""Check an analytic gradient with finite differences: NMO misfit E(t0, v).

Run:  python3 fd_gradient.py
Out:  results/fd_gradient.txt
"""

from pathlib import Path

import numpy as np

# ---- variables to change ---------------------------------------------------
M = (0.8, 2500.0)                          # model (t0 in s, v in m/s) where g is checked
H_LIST = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]    # relative step: h_j = H * |m_j|
BUG = False                                # True: plant a sign error in the analytic gradient
# -----------------------------------------------------------------------------

offsets = np.arange(0.0, 3001.0, 250.0)
rng = np.random.default_rng(3)
t_obs = np.sqrt(1.0 + (offsets / 2000.0) ** 2) + rng.normal(0.0, 0.005, offsets.size)
calls = 0


def misfit(m):
    global calls
    calls += 1
    t = np.sqrt(m[0] ** 2 + (offsets / m[1]) ** 2)
    r = t - t_obs
    return 0.5 * r @ r


def analytic_gradient(m):
    t = np.sqrt(m[0] ** 2 + (offsets / m[1]) ** 2)
    r = t - t_obs
    J = np.stack([m[0] / t, -offsets ** 2 / (m[1] ** 3 * t)], axis=1)
    g = J.T @ r                                   # g = J^T r
    if BUG:
        g[1] = -g[1]
    return g


m = np.array(M, dtype=float)
g = analytic_gradient(m)
lines = [f"model m = {M}, analytic g = J^T r = {g}", "",
         f"{'H':>8} {'fd dE/dt0':>14} {'fd dE/dv':>14} {'err t0':>9} {'err v':>9} {'E calls':>8}"]
for H in H_LIST:
    calls = 0
    E0 = misfit(m)
    g_fd = np.zeros(m.size)
    for j in range(m.size):                       # one extra misfit per parameter
        e = np.zeros(m.size)
        h = H * abs(m[j])
        e[j] = h
        g_fd[j] = (misfit(m + e) - E0) / h
    rel = np.abs(g_fd - g) / np.abs(g)            # relative error of each component
    lines.append(f"{H:8.0e} {g_fd[0]:14.6e} {g_fd[1]:14.6e} {rel[0]:9.1e} {rel[1]:9.1e} {calls:8d}")
lines += ["", "forward differences: n + 1 misfit calls for n parameters",
          "a correct gradient: errors shrink with H, then round-off stops them near 1e-7",
          "a wrong gradient: one component keeps an error of order 1 for every H"]
text = "\n".join(lines)
print(text)
out = Path("results")
out.mkdir(parents=True, exist_ok=True)
(out / "fd_gradient.txt").write_text(text + "\n", encoding="utf-8")
print(f"\nSaved: {out / 'fd_gradient.txt'}")
