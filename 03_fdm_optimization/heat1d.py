"""1-D heat equation with the explicit FTCS scheme: stability depends on r.

Run:  python3 heat1d.py
Out:  results/heat1d.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- variables to change ---------------------------------------------------
N = 41                     # grid points
R = 0.4                    # r = alpha * dt / h**2 ; stable when r <= 0.5, try 0.6
STEPS = 40                 # time steps to run
SNAPSHOTS = [0, 10, 20, 40]  # steps to draw (must be <= STEPS)
# -----------------------------------------------------------------------------

u = np.zeros(N)
u[N // 2] = 1.0
u[N // 2 - 1] = u[N // 2 + 1] = 0.5    # a small hot spot in the middle
history = {0: u.copy()}
for n in range(1, STEPS + 1):
    u[1:-1] = u[1:-1] + R * (u[2:] - 2 * u[1:-1] + u[:-2])   # interior update
    history[n] = u.copy()

print(f"N = {N}, r = {R}, steps = {STEPS}")
print(f"{'step':>5} {'max|u|':>12} {'sum u':>10}")
for n in SNAPSHOTS:
    print(f"{n:5d} {np.abs(history[n]).max():12.4e} {history[n].sum():10.4f}")
print("stable run: max|u| decreases and sum u stays constant; unstable: max|u| grows")

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for n in SNAPSHOTS:
    ax.plot(history[n], label=f"step {n}")
ax.set_xlabel("grid index i"); ax.set_ylabel("u"); ax.set_title(f"FTCS, r = {R}")
ax.legend(); ax.grid(True, lw=0.3)
out = Path("results")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "heat1d.png", dpi=150)
print(f"Saved: {out / 'heat1d.png'}")
