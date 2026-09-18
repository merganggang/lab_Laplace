"""2-D heat equation with explicit FTCS: wall time and peak memory versus grid size.

Run:  python3 heat2d.py
Out:  results/heat2d_<N>.json, results/heat2d_<N>.png
"""

import json
from pathlib import Path
import platform
import resource
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- variables to change ---------------------------------------------------
N = 501             # grid is N x N ; try 501, 1001, 2001
STEPS = 200         # time steps
R = 0.2             # r = alpha * dt / h**2 ; 2-D stability needs r <= 0.25
METHOD = "slicing"  # "slicing" or "loop" ; use "loop" only with N <= 201
# -----------------------------------------------------------------------------

# u[i, j] = u(x_i, z_j): i is the x (distance) index, j is the z (depth) index
u = np.zeros((N, N))   # shape (nx, nz)
c = N // 2
u[c - N // 10: c + N // 10, c - N // 10: c + N // 10] = 1.0   # hot square
bytes_per_array = u.nbytes

t0 = time.perf_counter()
if METHOD == "slicing":
    for _ in range(STEPS):
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + R * (
            u[2:, 1:-1] + u[:-2, 1:-1]      # x neighbours: i+1, i-1
            + u[1:-1, 2:] + u[1:-1, :-2]    # z neighbours: j+1, j-1
            - 4 * u[1:-1, 1:-1])
elif METHOD == "loop":
    un = u.copy()
    for _ in range(STEPS):
        for i in range(1, N - 1):          # x
            for j in range(1, N - 1):      # z
                un[i, j] = u[i, j] + R * (
                    u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j])
        u, un = un, u
else:
    raise SystemExit(f"unknown METHOD {METHOD!r}")
wall = time.perf_counter() - t0

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mib = rss / 2 ** 20 if platform.system() == "Darwin" else rss / 2 ** 10   # bytes vs KiB

result = {
    "N": N, "steps": STEPS, "r": R, "method": METHOD,
    "array_MiB": round(bytes_per_array / 2 ** 20, 2),
    "wall_s": round(wall, 3),
    "peak_rss_MiB": round(peak_mib, 1),
    "max_u": float(u.max()),
}
print(json.dumps(result, indent=2))
print(f"cells x steps = {N * N * STEPS:.2e}, {N * N * STEPS / wall:.2e} cell updates per second")

out = Path("results")
out.mkdir(parents=True, exist_ok=True)
(out / f"heat2d_{N}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
fig, ax = plt.subplots(figsize=(5, 4.4), constrained_layout=True)
im = ax.imshow(u.T, cmap="Blues", vmin=0, vmax=1)   # transpose: x across, depth down
fig.colorbar(im, ax=ax, label="u"); ax.set_title(f"N = {N}, steps = {STEPS}, r = {R}")
ax.set_xlabel("i (x)"); ax.set_ylabel("j (z)")
fig.savefig(out / f"heat2d_{N}.png", dpi=150)
print(f"Saved: {out / f'heat2d_{N}.json'}, {out / f'heat2d_{N}.png'}")
