"""L-BFGS with the two-loop recursion on the Rosenbrock function.

Run:  python3 lbfgs2d.py
Out:  results/lbfgs2d.png, results/lbfgs2d.txt
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import line_search, minimize, rosen, rosen_der

# ---- variables to change ---------------------------------------------------
ELL = 5               # number of stored (s, y) pairs ; try 1, 3, 10
X0 = (-1.2, 1.0)      # starting model
C1, C2 = 1e-4, 0.9    # Wolfe constants ; try C2 = 0.1
TOL = 1e-8            # stop when |g| < TOL
MAX_ITER = 1000
# -----------------------------------------------------------------------------

E, grad = rosen, rosen_der


def lbfgs_direction(g, S, Y):
    """S, Y: recent s_i, y_i (oldest first). Returns p = -B^{-1} g."""
    q = g.copy()
    a = []
    for s, y in reversed(list(zip(S, Y))):      # newest -> oldest
        rho = 1.0 / (y @ s)
        a.append(rho * (s @ q))
        q -= a[-1] * y
    gamma = (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])   # B0^{-1} = gamma I
    z = gamma * q
    for (s, y), ai in zip(zip(S, Y), reversed(a)):  # oldest -> newest
        rho = 1.0 / (y @ s)
        b = rho * (y @ z)
        z += (ai - b) * s
    return -z


m = np.array(X0, dtype=float)
g = grad(m)
S, Y = [], []
history = [E(m)]
n_E = n_g = 1
lines = [f"Rosenbrock, m0 = {X0}, ELL = {ELL}, C1 = {C1}, C2 = {C2}",
         f"{'k':>4} {'E(m_k)':>12} {'|g_k|':>10} {'alpha_k':>9}"]
for k in range(MAX_ITER):
    if np.linalg.norm(g) < TOL:
        break
    p = lbfgs_direction(g, S, Y) if S else -g / np.linalg.norm(g)
    alpha, fc, gc = line_search(E, grad, m, p, gfk=g, c1=C1, c2=C2)[:3]
    n_E += fc; n_g += gc
    if alpha is None:
        lines.append(f"{k:4d} line search failed: try C2 closer to 0.9")
        break
    m_new = m + alpha * p
    g_new = grad(m_new)
    S.append(m_new - m); Y.append(g_new - g)      # s_k, y_k
    if len(S) > ELL:
        S.pop(0); Y.pop(0)                        # forget the oldest pair
    m, g = m_new, g_new
    history.append(E(m))
    if k < 10 or k % 10 == 9:
        lines.append(f"{k + 1:4d} {E(m):12.4e} {np.linalg.norm(g):10.2e} {alpha:9.3f}")

ref = minimize(E, X0, jac=grad, method="L-BFGS-B", options={"maxcor": ELL, "gtol": TOL})
lines += ["", f"this code : {len(history) - 1} iterations, {n_E} E and {n_g} gradient evaluations, m = {m.round(6)}",
          f"scipy L-BFGS-B (maxcor = {ELL}): {ref.nit} iterations, m = {ref.x.round(6)}",
          "slide values: Newton + line search 21, gradient descent + line search 6380"]
text = "\n".join(lines)
print(text)

out = Path("results")
out.mkdir(parents=True, exist_ok=True)
(out / "lbfgs2d.txt").write_text(text + "\n", encoding="utf-8")
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.semilogy(np.maximum(history, 1e-25), "o-", ms=3)
ax.set_xlabel("iteration k"); ax.set_ylabel("E(m_k)"); ax.set_title(f"L-BFGS, ELL = {ELL}")
ax.grid(True, which="both", lw=0.3)
fig.savefig(out / "lbfgs2d.png", dpi=150)
print(f"Saved: {out / 'lbfgs2d.txt'}, {out / 'lbfgs2d.png'}")
