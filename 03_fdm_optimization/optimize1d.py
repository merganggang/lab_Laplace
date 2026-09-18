"""One-variable minimisation of E(x) = x^4 - 3x^2 + x: Newton versus gradient descent.

Run:  python3 optimize1d.py
Out:  results/optimize1d.txt
"""

from pathlib import Path

import numpy as np

# ---- variables to change ---------------------------------------------------
X0 = 2.0          # starting point ; try 0.5 (Newton walks to the maximum)
ALPHA = 0.05      # gradient descent step ; try 0.1 (other minimum) and 0.2 (diverges)
N_NEWTON = 7      # Newton iterations to print
N_GD = 31         # gradient descent iterations to print
# -----------------------------------------------------------------------------


def E(x):
    return x ** 4 - 3 * x ** 2 + x


def dE(x):
    return 4 * x ** 3 - 6 * x + 1


def d2E(x):
    return 12 * x ** 2 - 6


X_STAR = 1.1309011226301735   # minimum on the right; the other minimum is -1.3008
lines = [f"E(x) = x^4 - 3x^2 + x, x0 = {X0}, x* = {X_STAR:.7f}", ""]

lines.append("Newton: x_{k+1} = x_k - E'(x_k) / E''(x_k)")
h2 = "E''(x_k)"
lines.append(f"{'k':>3} {'x_k':>16} {'E(x_k)':>12} {h2:>10} {'|x_k - x*|':>12}")
x = X0
for k in range(N_NEWTON):
    lines.append(f"{k:3d} {x:16.10f} {E(x):12.6f} {d2E(x):10.3f} {abs(x - X_STAR):12.2e}")
    x = x - dE(x) / d2E(x)
lines.append("")

lines.append(f"Gradient descent: x_{{k+1}} = x_k - alpha E'(x_k), alpha = {ALPHA}")
lines.append(f"{'k':>3} {'x_k':>16} {'E(x_k)':>12} {'|x_k - x*|':>12}")
x = X0
for k in range(N_GD):
    if k <= 6 or k % 5 == 0:
        Ex = E(x) if abs(x) < 1e6 else float("inf")
        lines.append(f"{k:3d} {x:16.6f} {Ex:12.4e} {abs(x - X_STAR):12.2e}")
    x = x - ALPHA * dE(x)
    if abs(x) > 1e6:
        lines.append(f"{k + 1:3d} diverged")
        break

text = "\n".join(lines)
print(text)
out = Path("results")
out.mkdir(parents=True, exist_ok=True)
(out / "optimize1d.txt").write_text(text + "\n", encoding="utf-8")
print(f"\nSaved: {out / 'optimize1d.txt'}")
