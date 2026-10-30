"""2-D Laplace-domain modeling at one damping constant: wavefield in log scale, analytic check, boundaries, cost.

Run:  python3 laplace2d.py                       defaults below ; python3 laplace2d.py -h lists the options
      python3 laplace2d.py --s 2 --out results/s2
      python3 laplace2d.py --wavelet ricker --out results/ricker
      python3 laplace2d.py --s 2 --bc dirichlet --out results/dirichlet
      python3 laplace2d.py --model marmousi --nshot 50 --out results/marmousi
      python3 laplace2d.py --model marmousi --coarsen 4 --out results/coarse4
Out:  <out>/laplace2d_field.png, laplace2d.json
      homogeneous: laplace2d_analytic.png (finite difference against K0, the analytic solution)
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

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import laplacedomain as L
import timedomain as T

# ---- variables to change ---------------------------------------------------
MODEL = "homogeneous"     # "homogeneous", "marmousi"
S_DAMP = 8.0              # damping constant s (1/s)
WAVELET = "negricker"     # "negricker" (-Ricker, positive field), "ricker" (negative field), "impulse"
F0 = 5.0                  # Ricker peak frequency (Hz)
BC = "pml"                # left, right, bottom: "dirichlet", "ce" (Clayton-Engquist A2), "pml"
NB = 20                   # PML thickness (nodes)
COARSEN = 1               # Marmousi: keep every COARSEN-th node (24 m x COARSEN)
NSHOT = 1                 # sources solved with the same LU
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="2-D Laplace-domain modeling; options override the variables at the top.")
ap.add_argument("--model", choices=["homogeneous", "marmousi"], default=MODEL)
ap.add_argument("--s", type=float, default=S_DAMP, help="damping constant s (1/s)")
ap.add_argument("--wavelet", choices=["negricker", "ricker", "impulse"], default=WAVELET)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--bc", choices=["dirichlet", "ce", "pml"], default=BC)
ap.add_argument("--nb", type=int, default=NB, help="PML thickness (nodes)")
ap.add_argument("--coarsen", type=int, choices=[1, 2, 4], default=COARSEN, help="Marmousi grid h = 24 m x COARSEN")
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
s = args.s

# model defaults: grid spacing h (m), source and receivers in metres
if args.model == "homogeneous":       # source in the middle, all four sides absorbing: compare with K0
    h, v = 10.0, np.full((301, 301), 2000.0)
    src_xz, rec_z, top = (1500.0, 1500.0), 1500.0, "absorb"
else:                                 # Marmousi, 24 m (Versteeg 1994), bundled with this lab
    model_path = HERE / "data" / "marmousi_24m.npy"
    if not model_path.is_file():
        sys.exit(f"{model_path} not found: copy the complete 07_laplace_modeling folder, including data/")
    v24 = np.load(model_path).astype(float)
    v = v24[::args.coarsen, ::args.coarsen]
    h = 24.0 * args.coarsen
    src_xz, rec_z, top = (4608.0, 96.0), 96.0, "free"    # 96 m: a node of every coarsened grid
nx, nz = v.shape
x, z = np.arange(nx) * h, np.arange(nz) * h
vmin, vmax = float(v.min()), float(v.max())
Gd = vmin / s / h                               # grid points per decay length v_min / s

# Laplace transform of the source wavelet: the wavefield has its sign
dt_w = 1e-4
t_w = np.arange(int(round(6.0 / args.f0 / dt_w)) + 1) * dt_w    # the Ricker is 0 after 2.4 / f0
if args.wavelet == "impulse":
    wlap = 1.0
else:
    w = T.ricker(t_w, args.f0) * (-1.0 if args.wavelet == "negricker" else 1.0)
    wlap = float(L.laplace_transform(w, dt_w, s)[0])

isrc = np.array([[int(round(src_xz[0] / h)), int(round(src_xz[1] / h))]])
if args.nshot > 1:                              # extra sources along the receiver line, same LU
    xs = np.linspace(0.1, 0.9, args.nshot) * (nx - 1)
    isrc = np.vstack([isrc, np.column_stack([np.round(xs[1:]).astype(int), np.full(args.nshot - 1, isrc[0, 1])])])
jr = int(round(rec_z / h))
irec = np.array([[i, jr] for i in range(nx)])

print(f"{args.model}: {nx} x {nz} nodes, h = {h:g} m, 5-point, bc {args.bc}, top {top}")
print(f"s = {s:g} 1/s, v = {vmin:.0f} - {vmax:.0f} m/s, decay length v_min / s = {vmin / s:.0f} m, "
      f"G_d = {Gd:.1f} nodes per decay length")
print(f"source: {args.wavelet}, w~(s) = {wlap:+.4e}")

t0 = time.perf_counter()
res = L.solve(v, h, [s], isrc, irec, np.full(len(isrc), wlap), args.bc, top, args.nb, fields=True)
wall = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mib = rss / 2 ** 20 if platform.system() == "Darwin" else rss / 2 ** 10
field = res["field"][0]
lay = res["layout"]
nnz, nun = res["nnz_S"], res["n_unknowns"]
lu_mib = res["lu_bytes"] / 2 ** 20
summary = {"model": args.model, "s": s, "wavelet": args.wavelet, "f0_Hz": args.f0, "w_laplace": wlap,
           "bc": args.bc, "nb": args.nb, "top": top, "coarsen": args.coarsen, "nx": nx, "nz": nz, "NX": lay.NX,
           "NZ": lay.NZ, "h_m": h, "decay_length_m": vmin / s, "G_d": Gd, "unknowns": int(nun), "nnz_S": int(nnz),
           "S_MiB": nnz * 12 / 2 ** 20, "LU_MiB": lu_mib, "fill_ratio": res["lu_bytes"] / (nnz * 12),
           "factor_s": round(res["time_factor_s"], 3), "nshot": len(isrc), "solve_s": round(res["time_solve_s"], 3),
           "solve_per_shot_s": res["time_solve_s"] / len(isrc), "residual": res["residual"],
           "field_max": float(field.max()), "field_min": float(field[:, 1 if top == "free" else 0:].min()),
           "negative_fraction": float(np.mean(field < 0)), "wall_s": round(wall, 2), "peak_rss_MiB": round(peak_mib, 1)}
print(f"S: {nun} unknowns, {nnz} nonzeros, {summary['S_MiB']:.1f} MiB (float64); "
      f"LU {lu_mib:.0f} MiB = {summary['fill_ratio']:.0f} x S")
print(f"factor {res['time_factor_s']:.2f} s, solve {res['time_solve_s'] / len(isrc) * 1e3:.1f} ms per source x "
      f"{len(isrc)}, residual |S u - q| / |q| = {res['residual']:.1e}")
print(f"field: max {field.max():.3e}, min {summary['field_min']:.3e}"
      + (" (below the free surface)" if top == "free" else "") + f", negative nodes {summary['negative_fraction']:.0%}")
if field.max() <= 0:
    print("every value is negative: ln u is undefined. The Ricker wavelet has w~(s) < 0; use -Ricker (negricker).")

# ---- coarse Marmousi grid: compare ln u with the 24 m grid at the common receivers
if args.model == "marmousi" and args.coarsen > 1:
    fine = L.solve(v24, 24.0, [s], isrc[:1] * args.coarsen, irec * args.coarsen, [wlap], args.bc, top, args.nb)
    a, b = res["data"][0, 0], fine["data"][0, 0]
    far = np.abs(irec[:, 0] - isrc[0, 0]) * h >= 500.0        # away from the source node
    d = np.abs(np.log(a[far] / b[far]))
    summary["ln_difference_vs_24m"] = {"max": float(d.max()), "median": float(np.median(d))}
    summary["unknowns_24m"], summary["LU_MiB_24m"] = int(fine["n_unknowns"]), fine["lu_bytes"] / 2 ** 20
    print(f"24 m grid: {fine['n_unknowns']} unknowns, LU {fine['lu_bytes'] / 2 ** 20:.0f} MiB; "
          f"|ln u - ln u_24m| at receivers 500 m or more from the source: median {np.median(d):.3f}, max {d.max():.3f}")
print(f"wall {wall:.1f} s, peak RSS {peak_mib:.0f} MiB")


def show(ax, f, **kw):
    """(nx, nz) field as a section: x across, depth down."""
    return ax.imshow(f.T, extent=[x[0], x[-1], z[-1], z[0]], origin="upper", aspect="equal", **kw)


# ---- wavefield: natural log of the amplitude (the field spans tens of orders of magnitude)
amp = np.abs(field)
lg = np.log(np.maximum(amp, amp.max() * 1e-30))
lo = np.percentile(lg[:, 1:] if top == "free" else lg, 0.5)   # skip the u = 0 row of the free surface
fig, ax = plt.subplots(figsize=(8 if args.model == "homogeneous" else 11, 1.6 + (6.2 if args.model == "homogeneous" else 9.0) * nz / nx),
                       constrained_layout=True)
im = show(ax, lg, cmap="viridis", vmin=lo, vmax=lg.max())
ax.contour(x, z, lg.T, np.arange(np.ceil(lo), lg.max(), 1.0), colors="w", linewidths=0.4)   # every factor e
fig.colorbar(im, ax=ax, shrink=0.8, label="ln |u|" if field.max() <= 0 else "ln u")
if args.model != "homogeneous":
    ax.contour(x, z, v.T, [2000, 3000, 4000], colors="k", linewidths=0.4, alpha=0.4)
ax.plot(*src_xz, "*", color="gold", mec="k", ms=10)
ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
sign = "all negative, |u| shown" if field.max() <= 0 else "positive"
ax.set_title(f"{args.model}, s = {s:g} 1/s, {args.wavelet} ({sign}), bc {args.bc}, h = {h:g} m")
fig.savefig(out / "laplace2d_field.png", dpi=150)

# ---- homogeneous: along the receiver line against w~(s) K0(s r / v) / (2 pi)
if args.model == "homogeneous":
    i0 = isrc[0, 0]
    r = (np.arange(nx) - i0) * h
    line = res["data"][0, 0]
    right = r > 0
    a = wlap * L.analytic_2d_laplace(r[right], s, vmax)
    num = line[right]
    edge = x[-1] - src_xz[0]
    summary["ln_errors"] = {}
    for lo, hi in ((100.0, 500.0), (500.0, 1000.0)):
        sel = (r[right] >= lo) & (r[right] <= hi)
        err = float(np.abs(np.log(num[sel] / a[sel])).max())
        summary["ln_errors"][f"{lo:g}-{hi:g} m"] = err
        print(f"offset {lo:g} - {hi:g} m: max |ln u - ln u_exact| = {err:.4f}")
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), constrained_layout=True, sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(r[right], np.log(np.abs(a)), "k", lw=2.2, label="analytic  w~ K0(s r / v) / (2 pi)")
    axes[0].plot(r[right], np.log(np.abs(num)), color="tab:red", lw=1.0, ls="--", label=f"5-point, bc {args.bc}")
    axes[0].set_ylabel("ln |u|")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"receiver line z = {rec_z:g} m, s = {s:g} 1/s; edge of the model at {edge:g} m")
    axes[1].plot(r[right], np.log(num / a), color="tab:blue", lw=1.0)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("ln u - ln u_exact")
    axes[1].set_xlabel("offset r (m)")
    fig.savefig(out / "laplace2d_analytic.png", dpi=150)

(out / "laplace2d.json").write_text(json.dumps(summary, indent=2))
print(f"saved {out}/laplace2d_*.png, laplace2d.json")
