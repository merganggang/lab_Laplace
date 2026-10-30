"""First-arrival traveltime from the damped wavefield (SWEET, Shin et al. 2002) against the eikonal FSM.

For each damping constant s:  S G = q  (impulse source, q = 1 / h^2)  and  S dG/ds = -(dS/ds) G
with the same LU, then t(x) = -(dG/ds) / G at every node. One arrival A exp(-s t_f) gives t_f; the
2-D spreading of K0 makes it later (t = (r / v) K1 / K0 in a homogeneous medium), and a source
wavelet adds its own weighted mean time (--wavelet negricker). Both shrink as s grows, so traveltimes
use an impulse source and a large s. The grid sets the upper limit: on the 5-point grid t comes out
early by about (s h / v)^2 / 8 of itself, so s = 50 is near the limit for h = 24 m and v = 1500 m/s.

Both sides and the top absorb (no free surface): the free-surface ghost would change the bias near the top.

Run:  python3 sweet.py                          defaults below ; python3 sweet.py -h lists the options
      python3 sweet.py --wavelet negricker --out results/sweet_ricker
      python3 sweet.py --model marmousi --out results/sweet_marmousi
Out:  <out>/sweet_line.png (traveltime along the source depth for each s), sweet_maps.png (FSM, SWEET
      at the largest s, their difference), sweet.json
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # before NumPy: SuperLU gains nothing from BLAS threads on these grids, and
                                          # idle threads spinning under the container CPU quota slow the other steps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import k0, k1

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import eikonal as E
import laplacedomain as L
import timedomain as T

# ---- variables to change ---------------------------------------------------
MODEL = "homogeneous"            # "homogeneous", "marmousi"
SVALS = [10, 20, 30, 40, 50]     # damping constants (1/s)
WAVELET = "impulse"              # "impulse" (Green's function), "negricker" (adds the wavelet's mean time)
F0 = 5.0                         # Ricker peak frequency (Hz) for --wavelet negricker
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="SWEET traveltime against the FSM; options override the variables at the top.")
ap.add_argument("--model", choices=["homogeneous", "marmousi"], default=MODEL)
ap.add_argument("--s", type=float, nargs="+", default=SVALS, help="damping constants (1/s)")
ap.add_argument("--wavelet", choices=["impulse", "negricker"], default=WAVELET)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
svals = np.asarray(args.s, float)

if args.model == "homogeneous":
    h, v = 10.0, np.full((301, 301), 2000.0)
    src_xz = (1500.0, 1500.0)
else:
    model_path = HERE / "data" / "marmousi_24m.npy"
    if not model_path.is_file():
        sys.exit(f"{model_path} not found: copy the complete 07_laplace_modeling folder, including data/")
    v = np.load(model_path).astype(float)
    h = 24.0
    src_xz = (4608.0, 96.0)
nx, nz = v.shape
x, z = np.arange(nx) * h, np.arange(nz) * h
isrc = np.array([[int(round(src_xz[0] / h)), int(round(src_xz[1] / h))]])
irec = np.array([[i, isrc[0, 1]] for i in range(nx)])          # the line through the source

# u = w~(s) G, so -d ln u / ds = -d ln G / ds - d ln w~ / ds: the wavelet adds its own damped mean time
if args.wavelet == "impulse":
    wlap, t_wav = np.ones(len(svals)), np.zeros(len(svals))
else:
    dt_w = 1e-4
    tw = np.arange(int(round(6.0 / args.f0 / dt_w)) + 1) * dt_w
    w = -T.ricker(tw, args.f0)
    wlap = L.laplace_transform(w, dt_w, svals)
    t_wav = L.laplace_transform(tw * w, dt_w, svals) / wlap      # -(dw~/ds) / w~ = sum t w e^-st / sum w e^-st

t0 = time.perf_counter()
res = L.solve(v, h, svals, isrc, irec, np.ones(len(svals)), "pml", "absorb", fields=True, derivative=True)
wall = time.perf_counter() - t0
t_sweet = L.sweet_time(res["field"], res["dfield"]) + t_wav[:, None, None]   # (n_s, nx, nz)

t1 = time.perf_counter()
TT = E.fsm(1.0 / v, h, ((isrc[0, 0] + 0.5) * h, (isrc[0, 1] + 0.5) * h))   # velocity nodes = FSM cells
I, J = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
t_fsm = E.sample(TT, h, np.column_stack([(I.ravel() + 0.5) * h, (J.ravel() + 0.5) * h])).reshape(nx, nz)
wall_fsm = time.perf_counter() - t1
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mib = rss / 2 ** 20 if platform.system() == "Darwin" else rss / 2 ** 10

r = np.hypot((I - isrc[0, 0]) * h, (J - isrc[0, 1]) * h)
far = r >= 500.0                                                # away from the source node
summary = {"model": args.model, "wavelet": args.wavelet, "h_m": h, "nx": nx, "nz": nz,
           "laplace_s": round(wall, 2), "fsm_s": round(wall_fsm, 2), "LU_MiB": res["lu_bytes"] / 2 ** 20,
           "peak_rss_MiB": round(peak_mib, 1), "s": {}}
print(f"{args.model}: {nx} x {nz} nodes, h = {h:g} m, source {args.wavelet}; Laplace {len(svals)} s values "
      f"(LU and two solves each) {wall:.1f} s, FSM {wall_fsm:.1f} s")
print("nodes 500 m or more from the source: t_SWEET - t_FSM (s)")
for k, s in enumerate(svals):
    d = (t_sweet[k] - t_fsm)[far]
    row = {"mean": float(d.mean()), "median": float(np.median(d)), "min": float(d.min()), "max": float(d.max()),
           "abs_median": float(np.median(np.abs(d))), "w_laplace": float(wlap[k]),
           "wavelet_time": float(t_wav[k])}
    summary["s"][f"{s:g}"] = row
    print(f"  s = {s:4g} 1/s: mean {row['mean']:+.3f}, median {row['median']:+.3f}, range {row['min']:+.3f} ~ "
          f"{row['max']:+.3f}; |difference| median {row['abs_median']:.3f}"
          + (f"; wavelet term {t_wav[k]:+.3f}" if args.wavelet != "impulse" else ""))

# ---- along the line through the source
right = (irec[:, 0] - isrc[0, 0]) * h
cmap = plt.get_cmap("viridis")
cols = [cmap(0.9 * k / max(len(svals) - 1, 1)) for k in range(len(svals))]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
m = right > 0
jl = isrc[0, 1]
axes[0].plot(right[m], t_fsm[m, jl], "k", lw=2.4, label="FSM")
if args.model == "homogeneous":
    rr = right[m]
    for k, s in enumerate(svals):
        xx = s * rr / v[0, 0]
        axes[0].plot(rr, rr / v[0, 0] * k1(xx) / k0(xx), color=cols[k], lw=0.8, ls=":")
for k, s in enumerate(svals):
    axes[0].plot(right[m], t_sweet[k][m, jl], color=cols[k], lw=1.4, label=f"SWEET, s = {s:g}")
axes[0].set(xlabel="offset (m)", ylabel="t (s)", title="traveltime along the source depth")
axes[0].legend(fontsize=8)
means = [summary["s"][f"{s:g}"]["mean"] for s in svals]
axes[1].plot(svals, means, "o-", color="tab:blue", label="mean of t_SWEET - t_FSM")
axes[1].axhline(0, color="0.6", lw=0.8)
axes[1].set(xlabel="s (1/s)", ylabel="bias (s)", title=f"nodes 500 m or more from the source, {args.wavelet}")
axes[1].legend(fontsize=9)
fig.savefig(out / "sweet_line.png", dpi=150)

# ---- maps at the largest s
k = len(svals) - 1
fig, axes = plt.subplots(3, 1, figsize=(9, 2.0 + 9.5 * nz / nx * (1 if nz < nx else 0.6)), constrained_layout=True)
ext = [x[0], x[-1], z[-1], z[0]]
tmax_map = float(np.nanmax(t_fsm))
for ax, f, name in ((axes[0], t_fsm, "FSM t (s)"), (axes[1], t_sweet[k], f"SWEET t (s), s = {svals[k]:g}")):
    im = ax.imshow(f.T, extent=ext, cmap="viridis", vmin=0, vmax=tmax_map, aspect="equal")
    ax.contour(x, z, f.T, np.arange(0.5, tmax_map, 0.5), colors="w", linewidths=0.6)
    fig.colorbar(im, ax=ax, shrink=0.9, label=name)
diff = t_sweet[k] - t_fsm
im = axes[2].imshow(diff.T, extent=ext, cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="equal")
fig.colorbar(im, ax=axes[2], shrink=0.9, label="t_SWEET - t_FSM (s)")
for ax in axes:
    ax.plot(*src_xz, "*", color="gold", mec="k", ms=10)
    ax.set_ylabel("z (m)")
axes[-1].set_xlabel("x (m)")
fig.savefig(out / "sweet_maps.png", dpi=150)

(out / "sweet.json").write_text(json.dumps(summary, indent=2))
print(f"saved {out}/sweet_*.png, sweet.json")
