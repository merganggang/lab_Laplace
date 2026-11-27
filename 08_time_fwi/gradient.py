"""First gradient of the waveform misfit against the model error, with and without pseudo-Hessian preconditioning.

Run:  python3 gradient.py                            defaults below ; python3 gradient.py -h lists the options
      python3 gradient.py --precond --out results/grad_precond
      python3 gradient.py --init smooth --out results/grad_smooth
Out:  <out>/gradient.png, gradient_depth.png, gradient.json

-g is the steepest-descent direction. Where the starting model is too slow, v_true - v0 > 0 and -g should be
positive too. The raw gradient is large near the surface (sources, receivers, strong wavefields) and small at
depth (geometrical spreading); dividing by the pseudo-Hessian diagonal H_P (Shin et al. 2001) evens it out.
"""

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
import geoinv
import fwiproblem as F
import propagator as P

# ---- variables to change ---------------------------------------------------
MODEL = "marmousi"        # "marmousi", "anomaly"
INIT = "linear"           # "linear", "smooth", "background"
PRECOND = False           # also show the preconditioned gradient
NSHOT = 32
F0 = 3.0
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="First FWI gradient against v_true - v0; options override the variables at the top.")
ap.add_argument("--model", choices=["marmousi", "anomaly"], default=MODEL)
ap.add_argument("--init", choices=["linear", "smooth", "background"], default=INIT)
ap.add_argument("--precond", action="store_true", default=PRECOND)
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--batch", type=int, default=None, help="shots stepped together (default: fit the GPU memory)")
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
dev = P.device(args.device)

if args.model == "marmousi":
    v_true, h = F.marmousi(48.0), 48.0
else:
    v_true, h = F.anomaly(), 48.0
n_fixed = max(F.water_rows(v_true), 3)
v0 = F.initial(v_true, h, args.init, known_rows=n_fixed)
nx, nz = v_true.shape
dt, nt = F.time_axis(h, 4.0)
t = np.arange(nt) * dt
src, rec = F.acquisition(nx, args.nshot)
wavelet = P.ricker(t, args.f0, 0.5)

obs = F.TimeFWI(nx, nz, h, dt, nt, src, rec, wavelet, order=8, nb=20, f0=args.f0, batch=args.batch, cache_gib=0, dev=dev)
d = obs.forward(v_true.ravel() / 1000.0)
prob = F.TimeFWI(nx, nz, h, dt, nt, src, rec, wavelet, f0=args.f0, batch=args.batch, fixed_depth=n_fixed - 1, dev=dev)
obj = geoinv.Objective(prob, d)
E, g = obj.value_and_gradient(v0.ravel() / 1000.0)
free = ~prob.fixed_mask
g[~free] = 0.0
panels = [("v_true - v0 (m/s)", (v_true - v0).ravel()), ("-g (raw)", -g)]
H = None
if args.precond:
    H = prob.illumination(v0.ravel() / 1000.0)
    gp = -g / (H + 0.01 * H.max())
    gp[~free] = 0.0
    panels.append(("-g / (H_P + 0.01 max H_P)", gp))

dv = (v_true - v0).ravel()
stats = {}
for name, a in panels[1:]:
    c = float(np.corrcoef(a[free], dv[free])[0, 1])
    stats[name] = {"correlation_with_dv": c}
    print(f"{name}: correlation with v_true - v0 over the free nodes {c:+.2f}")

x, z = np.arange(nx) * h / 1000, np.arange(nz) * h / 1000
ext = [x[0], x[-1], z[-1], z[0]]
fig, axes = plt.subplots(len(panels), 1, figsize=(9, 1.0 + 2.6 * len(panels)), constrained_layout=True)
for ax, (name, a) in zip(axes, panels):
    a = a.reshape(nx, nz)
    cl = np.percentile(np.abs(a[free.reshape(nx, nz)]), 99)
    im = ax.imshow(a.T, extent=ext, cmap="RdBu_r", vmin=-cl, vmax=cl, aspect="equal")
    ax.set_title(name, fontsize=10)
    ax.set_ylabel("z (km)")
    fig.colorbar(im, ax=ax, shrink=0.9)
axes[0].plot(src[:, 0] * h / 1000, src[:, 1] * h / 1000, "*", color="gold", mec="k", ms=5, clip_on=False)
axes[-1].set_xlabel("x (km)")
fig.savefig(out / "gradient.png", dpi=150)

# rms over each depth row, normalised: how the gradient falls with depth
fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
rows = slice(n_fixed, nz - 1)
for name, a in panels:
    prof = np.sqrt(np.mean(a.reshape(nx, nz)[1:-1, rows] ** 2, axis=0))
    ax.semilogx(prof / prof.max(), z[rows], label=name)
    stats.setdefault(name, {})["rms_bottom_over_top"] = float(prof[-1] / prof[0])
ax.invert_yaxis()
ax.set_xlabel("rms over x / max"); ax.set_ylabel("z (km)")
ax.legend(fontsize=7)
fig.savefig(out / "gradient_depth.png", dpi=150)

summary = {"model": args.model, "init": args.init, "nshot": len(src), "precond": args.precond, "E": E,
           "device": str(dev), "panels": stats}
(out / "gradient.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"saved {out}/gradient*.png, gradient.json")
