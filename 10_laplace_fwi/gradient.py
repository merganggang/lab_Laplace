"""First gradient of the Laplace-domain FWI: one damping constant at a time, their sum, the l2 objective and the
preconditioned sum.

Run:  python3 gradient.py                                  defaults below ; python3 gradient.py -h lists the options
      python3 gradient.py --objective l2 --out results/grad_l2
      python3 gradient.py --precond --out results/grad_p
Out:  <out>/gradient.png (v_true - v0 and -g for each damping constant and their sum), gradient_pair.png (one shot and
      one receiver: the sensitivity of a single datum), gradient.json

-g is the steepest-descent direction: where it is positive, raising v lowers E. Each panel is scaled to its own
maximum so the shapes can be compared: a large damping constant keeps the early, shallow part of the wavefield,
a small one reaches deeper. With the l2 objective the near offsets of every shot dominate the sum of squares and
the gradient gathers around the sources; the log objective weights every offset alike.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))     # geoinv when PYTHONPATH is not set (the container sets it)
import fwiproblem as F

# ---- variables to change ---------------------------------------------------
H = 40.0                  # salt model grid spacing (m)
INIT = "linear"           # "linear", "smooth"
SVALS = [2.0, 6.0, 12.0]  # damping constants (1/s), each shown alone and summed
OBJECTIVE = "log"         # "log" or "l2"
PRECOND = False
NSHOT = 48
TMAX = 10.0
F0 = 6.0
MIN_OFFSET = None         # receivers closer than this to the source are not used (m); None: 0 (log), 400 (l2)
OFFSET = 3000.0           # gradient_pair.png: receiver offset from the middle shot (m)
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="First gradient per damping constant; options override the variables at the top.")
ap.add_argument("--h", type=float, choices=[40.0, 20.0], default=H)
ap.add_argument("--init", choices=["linear", "smooth"], default=INIT)
ap.add_argument("--s", type=float, nargs="+", default=SVALS, help="damping constants (1/s)")
ap.add_argument("--objective", choices=["log", "l2"], default=OBJECTIVE)
ap.add_argument("--precond", action="store_true", default=PRECOND)
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--tmax", type=float, default=TMAX)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--min-offset", type=float, default=MIN_OFFSET, help="default 0 (log objective), 400 (l2)")
ap.add_argument("--offset", type=float, default=OFFSET, help="gradient_pair.png: receiver offset (m)")
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
if args.min_offset is None:
    args.min_offset = 0.0 if args.objective == "log" else 400.0
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

v_true, h = F.salt(args.h), args.h
n_fixed = F.water_rows(v_true)
v0 = F.initial(v_true, h, args.init, known_rows=n_fixed)
nx, nz = v_true.shape
src, rec = F.acquisition(nx, args.nshot, h)
svals = list(args.s)
tr, dtr, path, sec = F.observed_records("salt", v_true, h, src, rec, "ricker", args.f0, args.tmax, args.workers)
d = F.laplace_data(tr, dtr, svals)
use = np.abs(src[:, None, 0] - rec[None, :, 0]) * h >= args.min_offset
use = np.broadcast_to(use, d.shape).copy()                                  # (n_s, ns, nr)
if args.objective == "log":                                                 # as in fwi.py: no log of a datum of the wrong sign
    for k in range(len(svals)):
        sign = 1.0 if np.sum(d[k][use[k]] > 0) >= np.sum(d[k][use[k]] < 0) else -1.0
        use[k] &= d[k] * sign > 0
dt, nt = F.time_axis(h, args.tmax)
wlap = F.wavelet_lap(F.wavelet("ricker", np.arange(nt) * dt, args.f0), dt, svals)
prob = F.LaplaceFWI(nx, nz, h, svals, src, rec, wlap, fixed_rows=n_fixed, objective=args.objective, srcest=True,
                    d_obs=d, mask=use, workers=args.workers)
free = ~prob.fixed_mask.reshape(nx, nz)
m0 = v0.ravel() / 1000.0
r = (prob.forward(m0) - prob.data_vector(d)).reshape(d.shape)
r = np.where(use, r, 0.0)

# per damping constant: its residual alone, scaled so that each damping constant counts equally
grads = []
for k in range(len(svals)):
    rk = np.zeros_like(r)
    rk[k] = r[k] / (1.0 if args.objective == "log" else np.linalg.norm(d[k][use[k]]) ** 2)
    grads.append(-prob.gradient(m0, rk.ravel()).reshape(nx, nz))
total = sum(g / np.abs(g[free]).max() for g in grads)
if args.precond:
    H_P = prob.illumination(m0).reshape(nx, nz)
    Hs = H_P + prob.illum_damp * H_P[free].max()           # log: no damping, H_P > 0 on the free nodes
    total = np.divide(total, Hs, out=np.zeros_like(total), where=Hs > 0)

# one shot, one receiver, residual 1: the sensitivity of one datum to every node
mid = len(src) // 2
ir = int(np.argmin(np.abs((rec[:, 0] - src[mid, 0]) * h - args.offset)))
pair = []
for k in range(len(svals)):
    rk = np.zeros_like(r)
    rk[k, mid, ir] = 1.0
    pair.append(-prob.gradient(m0, rk.ravel()).reshape(nx, nz))
prob.close()


def corr(a):
    """Correlation with the true perturbation v_true - v0 over the free nodes."""
    return float(np.corrcoef(a[free], (v_true - v0)[free])[0, 1])


def grid(n):
    """n sections in two columns (x across, depth down); an odd n leaves the last cell empty."""
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 0.5 + rows * (5.6 * nz / nx + 0.6)), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    for k, ax in enumerate(axes):
        if k % 2 == 0:
            ax.set_ylabel("z (km)")
        if k >= len(axes) - 2 - n % 2:
            ax.set_xlabel("x (km)")
            ax.xaxis.set_tick_params(labelbottom=True)
    if n % 2:
        axes[-1].set_axis_off()
    return fig, axes[:n], (axes[-1] if n % 2 else None)


x, z = np.arange(nx) * h / 1000, np.arange(nz) * h / 1000
ext = [x[0], x[-1], z[-1], z[0]]
panels = [(v_true - v0, "v_true - v0 (m/s)", np.abs(v_true - v0)[free].max())]
for s, g in zip(svals, grads):
    panels.append((g, f"-g, s = {s:g} 1/s ({args.objective}; corr {corr(g):+.2f})", np.abs(g[free]).max()))
panels.append((total, f"-g, sum{' / H_P' if args.precond else ''} (corr {corr(total):+.2f})",
               np.abs(total[free]).max()))
fig, axes, _ = grid(len(panels))
for ax, (a, ttl, c) in zip(axes, panels):
    a = np.where(free, a, 0.0) if ttl.startswith("-g") else a
    ax.imshow(a.T, extent=ext, cmap="RdBu_r", vmin=-c, vmax=c, aspect="equal")
    ax.set_title(ttl, fontsize=10)
fig.savefig(out / "gradient.png", dpi=150)

fig, axes, _ = grid(len(svals))
for ax, s, g in zip(axes, svals, pair):
    c = np.abs(g[free]).max()
    ax.imshow(np.where(free, g, 0.0).T, extent=ext, cmap="RdBu_r", vmin=-c, vmax=c, aspect="equal")
    ax.plot([src[mid, 0] * h / 1000, rec[ir, 0] * h / 1000], [src[mid, 1] * h / 1000] * 2, "k*", ms=7)
    ax.set_title(f"one shot, one receiver (offset {(rec[ir, 0] - src[mid, 0]) * h / 1000:.1f} km), s = {s:g} 1/s",
                 fontsize=10)
fig.savefig(out / "gradient_pair.png", dpi=150)


def depth_center(g):
    """Depth (km) of the centre of |g| over the free nodes: where the gradient is."""
    a = np.abs(np.where(free, g, 0.0)).sum(0)
    return float((a * z).sum() / a.sum())


summary = {"h_m": h, "init": args.init, "s": svals, "objective": args.objective, "precond": args.precond,
           "corr_per_s": {f"{s:g}": corr(g) for s, g in zip(svals, grads)}, "corr_sum": corr(total),
           "depth_center_km": {f"{s:g}": depth_center(g) for s, g in zip(svals, grads)},
           "pair_depth_center_km": {f"{s:g}": depth_center(g) for s, g in zip(svals, pair)},
           "pair_offset_m": float((rec[ir, 0] - src[mid, 0]) * h)}
(out / "gradient.json").write_text(json.dumps(summary, indent=2) + "\n")
print("correlation with v_true - v0: " + ", ".join(f"s = {s:g} {corr(g):+.2f}" for s, g in zip(svals, grads))
      + f", sum {corr(total):+.2f}")
print("depth of the gradient centre (km): " + ", ".join(f"s = {s:g} {depth_center(g):.2f}" for s, g in zip(svals, grads)))
print(f"saved {out}/gradient.png, gradient_pair.png, gradient.json")
