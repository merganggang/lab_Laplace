"""First gradient of the frequency-domain FWI: one frequency at a time, their sum, and the preconditioned sum.

Run:  python3 gradient.py                                  defaults below ; python3 gradient.py -h lists the options
      python3 gradient.py --precond --out results/grad_p
      python3 gradient.py --model anomaly --init background --freqs 2 3 4 --out results/grad_anom
Out:  <out>/gradient.png (v_true - v0 and -g for each frequency and their sum), gradient_pair.png (one shot and one
      receiver: the sensitivity of a single datum, the wavepath), gradient.json

-g is the steepest-descent direction: where it is positive, raising v lowers E. Each frequency is scaled to its
own maximum so the shapes can be compared: low frequencies give wide smooth updates, high frequencies narrow
oscillating ones. --precond divides the sum by the pseudo-Hessian diagonal (illumination).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))     # geoinv when PYTHONPATH is not set (the container sets it)
import freqdomain as fd
import fwiproblem as F

# ---- variables to change ---------------------------------------------------
MODEL = "marmousi"        # "marmousi", "anomaly"
H = 24.0                  # Marmousi grid spacing (m)
INIT = "linear"           # "linear", "smooth", "background" (anomaly)
FREQS = [2.0, 4.0, 8.0]   # Hz, each shown alone and summed
PRECOND = False
NSHOT = 48
TMAX = 6.0
F0 = 6.0
MIN_OFFSET = 400.0
OFFSET = 3000.0           # gradient_pair.png: receiver offset from the middle shot (m)
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="First gradient per frequency; options override the variables at the top.")
ap.add_argument("--model", choices=["anomaly", "marmousi"], default=MODEL)
ap.add_argument("--h", type=float, choices=[24.0, 48.0], default=H)
ap.add_argument("--init", choices=["background", "linear", "smooth"], default=INIT)
ap.add_argument("--freqs", type=float, nargs="+", default=FREQS)
ap.add_argument("--precond", action="store_true", default=PRECOND)
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--tmax", type=float, default=TMAX)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--min-offset", type=float, default=MIN_OFFSET)
ap.add_argument("--offset", type=float, default=OFFSET, help="gradient_pair.png: receiver offset (m)")
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

if args.model == "marmousi":
    v_true, h = F.marmousi(args.h), args.h
else:
    v_true, h = F.anomaly(), 48.0
n_fixed = max(F.water_rows(v_true), int(round(F.SRC_DEPTH / h)) + 2)
v0 = F.initial(v_true, h, args.init, known_rows=n_fixed)
nx, nz = v_true.shape
src, rec = F.acquisition(nx, args.nshot, h)
freqs = list(args.freqs)
dt, nt = F.time_axis(h, args.tmax)
damp = math.log(100.0) / args.tmax
D, path, sec = F.observed_spectra(args.model, v_true, h, src, rec, "ricker", args.f0, args.tmax, args.workers)
d = D[[int(np.argmin(np.abs(F.FGRID - f))) for f in freqs]]
use = np.abs(src[:, None, 0] - rec[None, :, 0]) * h >= args.min_offset
wspec = fd.source_spectrum(F.wavelet("ricker", np.arange(nt) * dt, args.f0), dt, 2 * np.pi * np.array(freqs) - 1j * damp)
prob = F.FreqFWI(nx, nz, h, freqs, src, rec, wspec, damp=damp, fixed_rows=n_fixed, workers=args.workers)
free = ~prob.fixed_mask.reshape(nx, nz)
m0 = v0.ravel() / 1000.0
u = prob.solve(m0)
r = np.where(use, u - d, 0.0)

# per frequency: residual of that frequency only, scaled by its data norm (each frequency counts equally)
grads = []
for k in range(len(freqs)):
    rk = np.zeros_like(r)
    rk[k] = r[k] / np.linalg.norm(d[k][use]) ** 2
    grads.append(-prob.gradient(m0, prob.to_real(rk)).reshape(nx, nz))
total = sum(grads)
H_P = prob.illumination(m0).reshape(nx, nz) if args.precond else None
if args.precond:
    total = total / (H_P + 0.01 * H_P[free].max())

# one shot, one receiver, residual 1: the sensitivity of one datum to every node (wavepath) per frequency
mid = len(src) // 2
ir = int(np.argmin(np.abs((rec[:, 0] - src[mid, 0]) * h - args.offset)))
pair = []
for k in range(len(freqs)):
    rk = np.zeros_like(r)
    rk[k, mid, ir] = 1.0
    pair.append(-prob.gradient(m0, prob.to_real(rk)).reshape(nx, nz))
prob.close()


def corr(a):
    b = (v_true - v0)[free]
    return float(np.corrcoef(a[free], b)[0, 1])


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
for f, g in zip(freqs, grads):
    panels.append((g, f"-g, {f:g} Hz (corr with v_true - v0 {corr(g):+.2f})", np.abs(g[free]).max()))
panels.append((total, f"-g, sum{' / H_P' if args.precond else ''} (corr {corr(total):+.2f})", np.abs(total[free]).max()))
fig, axes, _ = grid(len(panels))
for ax, (a, ttl, c) in zip(axes, panels):
    a = np.where(free, a, 0.0) if ttl.startswith("-g") else a
    ax.imshow(a.T, extent=ext, cmap="RdBu_r", vmin=-c, vmax=c, aspect="equal")
    ax.set_title(ttl, fontsize=10)
fig.savefig(out / "gradient.png", dpi=150)

fig, axes, _ = grid(len(freqs))
for ax, f, g in zip(axes, freqs, pair):
    c = np.abs(g[free]).max()
    ax.imshow(np.where(free, g, 0.0).T, extent=ext, cmap="RdBu_r", vmin=-c, vmax=c, aspect="equal")
    ax.plot([src[mid, 0] * h / 1000, rec[ir, 0] * h / 1000], [src[mid, 1] * h / 1000] * 2, "k*", ms=7)
    ax.set_title(f"one shot, one receiver (offset {(rec[ir, 0] - src[mid, 0]) * h / 1000:.1f} km), {f:g} Hz", fontsize=10)
fig.savefig(out / "gradient_pair.png", dpi=150)

rms = {f"{f:g}": [float(np.sqrt(np.mean(g[:, j] ** 2))) for j in range(nz)] for f, g in zip(freqs, grads)}
summary = {"model": args.model, "h_m": h, "init": args.init, "freqs_Hz": freqs, "precond": args.precond,
           "corr_per_freq": {f"{f:g}": corr(g) for f, g in zip(freqs, grads)}, "corr_sum": corr(total),
           "rms_per_depth": rms, "pair_offset_m": float((rec[ir, 0] - src[mid, 0]) * h)}
(out / "gradient.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"correlation with v_true - v0: " + ", ".join(f"{f:g} Hz {corr(g):+.2f}" for f, g in zip(freqs, grads))
      + f", sum {corr(total):+.2f}")
print(f"saved {out}/gradient.png, gradient_pair.png, gradient.json")
