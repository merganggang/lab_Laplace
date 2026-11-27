"""2-D frequency-domain modeling at one frequency: wavefield, analytic check, boundaries, matrix and LU cost.

Run:  python3 freq2d.py                       defaults below ; python3 freq2d.py -h lists the options
      python3 freq2d.py --stencil 9 --out results/freq_9pt
      python3 freq2d.py --bc dirichlet --out results/freq_dirichlet
      python3 freq2d.py --bc ce --out results/freq_ce
      python3 freq2d.py --model marmousi --nshot 50 --out results/freq_marmousi
      python3 freq2d.py --model marmousi --ordering natural --out results/freq_natural
Out:  <out>/freq2d_field.png, freq2d.json
      homogeneous: freq2d_analytic.png (finite difference against the Hankel-function solution)
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
import freqdomain as F

# ---- variables to change ---------------------------------------------------
MODEL = "homogeneous"     # "homogeneous", "marmousi"
FREQ = 5.0                # frequency (Hz)
STENCIL = 5               # 5-point or 9-point (Jo et al. 1996) operator
BC = "pml"                # left, right, bottom: "dirichlet", "ce" (Clayton-Engquist A2), "pml"
NB = 20                   # PML thickness (nodes)
DAMP = 0.0                # alpha_d (1/s): solve at the complex frequency omega - i alpha_d
ORDERING = "mindegree"    # LU elimination order: "mindegree", "colamd", "natural"
NSHOT = 1                 # sources solved with the same LU (cost of reusing the factorization)
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="2-D frequency-domain modeling; options override the variables at the top.")
ap.add_argument("--model", choices=["homogeneous", "marmousi"], default=MODEL)
ap.add_argument("--freq", type=float, default=FREQ)
ap.add_argument("--stencil", type=int, choices=[5, 9], default=STENCIL)
ap.add_argument("--bc", choices=["dirichlet", "ce", "pml"], default=BC)
ap.add_argument("--nb", type=int, default=NB, help="PML thickness (nodes)")
ap.add_argument("--damp", type=float, default=DAMP, help="alpha_d (1/s) of the complex frequency")
ap.add_argument("--ordering", choices=list(F.ORDERINGS), default=ORDERING)
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# model defaults: grid spacing h (m), source and receivers in metres
if args.model == "homogeneous":       # source in the middle, all four sides absorbing: compare with analytic
    h, v = 10.0, np.full((301, 301), 2000.0)
    src_xz, rec_z, top = (1500.0, 1500.0), 1500.0, "absorb"
else:                                 # Marmousi, 24 m (Versteeg 1994), bundled with this lab
    model_path = HERE / "data" / "marmousi_24m.npy"
    if not model_path.is_file():
        sys.exit(f"{model_path} not found: copy the complete 06_frequency_modeling folder, including data/")
    v = np.load(model_path).astype(float)
    h = 24.0
    src_xz, rec_z, top = (4608.0, 48.0), 48.0, "free"
nx, nz = v.shape
x, z = np.arange(nx) * h, np.arange(nz) * h
vmin, vmax = float(v.min()), float(v.max())
G = vmin / args.freq / h                        # grid points per wavelength at this frequency
omega = 2 * np.pi * args.freq

isrc = np.array([[int(round(src_xz[0] / h)), int(round(src_xz[1] / h))]])
if args.nshot > 1:                              # extra sources along the receiver line, same LU
    xs = np.linspace(0.1, 0.9, args.nshot) * (nx - 1)
    isrc = np.vstack([isrc, np.column_stack([np.round(xs[1:]).astype(int), np.full(args.nshot - 1, isrc[0, 1])])])
jr = int(round(rec_z / h))
irec = np.array([[i, jr] for i in range(nx)])

print(f"{args.model}: {nx} x {nz} nodes, h = {h:g} m, {args.stencil}-point, bc {args.bc}, top {top}")
print(f"f = {args.freq:g} Hz, v = {vmin:.0f} - {vmax:.0f} m/s, G = v_min / (f h) = {G:.1f}"
      + (f", omega - i {args.damp:g}" if args.damp else ""))

S, lay, keep = F.build_matrix(v, h, omega, args.stencil, args.bc, top, args.nb, args.damp)
print(f"S: {S.shape[0]} unknowns, {S.nnz} nonzeros ({S.nnz / S.shape[0]:.1f} per row), "
      f"{S.nnz * 20 / 2 ** 20:.1f} MiB; symmetric: {abs(S - S.T).max() == 0}")
t0 = time.perf_counter()
res = F.solve(v, h, [args.freq], isrc, irec, np.ones(len(isrc)), args.stencil, args.bc, top, args.nb,
              args.damp, args.ordering, fields=True)
wall = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mib = rss / 2 ** 20 if platform.system() == "Darwin" else rss / 2 ** 10
field = res["field"][0]                         # impulse response (w~ = 1) of the first source
lu_mib = res["lu_bytes"] / 2 ** 20
summary = {"model": args.model, "freq_Hz": args.freq, "stencil": args.stencil, "bc": args.bc, "nb": args.nb,
           "top": top, "damp": args.damp, "ordering": args.ordering, "nx": nx, "nz": nz, "NX": lay.NX,
           "NZ": lay.NZ, "h_m": h, "G": G, "unknowns": int(S.shape[0]), "nnz_S": int(S.nnz),
           "nnz_per_row": S.nnz / S.shape[0], "S_MiB": S.nnz * 20 / 2 ** 20, "LU_MiB": lu_mib,
           "fill_ratio": res["lu_bytes"] / (S.nnz * 20), "factor_s": round(res["time_factor_s"], 3),
           "nshot": len(isrc), "solve_s": round(res["time_solve_s"], 3),
           "solve_per_shot_s": res["time_solve_s"] / len(isrc), "residual": res["residual"], "wall_s": round(wall, 2),
           "peak_rss_MiB": round(peak_mib, 1)}
print(f"LU ({args.ordering}): {lu_mib:.0f} MiB = {summary['fill_ratio']:.0f} x S ; factor {res['time_factor_s']:.2f} s, "
      f"solve {res['time_solve_s'] / len(isrc) * 1e3:.1f} ms per source x {len(isrc)}, "
      f"residual |S u - q| / |q| = {res['residual']:.1e}")
print(f"wall {wall:.1f} s, peak RSS {peak_mib:.0f} MiB")


def show(ax, f, **kw):
    """(nx, nz) field as a section: x across, depth down."""
    return ax.imshow(f.T, extent=[x[0], x[-1], z[-1], z[0]], origin="upper", aspect="equal", **kw)


# ---- wavefield: real part (one instant of the oscillation) and amplitude
fig, axes = plt.subplots(1, 2, figsize=(11, 1.6 + 5.0 * nz / nx), constrained_layout=True)
re = field.real
clip = np.percentile(np.abs(re), 99) or 1.0
show(axes[0], np.clip(re, -clip, clip), cmap="RdBu_r", vmin=-clip, vmax=clip)
amp = np.abs(field)
db = 20 * np.log10(np.maximum(amp / np.percentile(amp, 99.5), 1e-6))
im = show(axes[1], db, cmap="viridis", vmin=-60, vmax=0)
fig.colorbar(im, ax=axes[1], shrink=0.8, label="|u| (dB)")
for ax, name in zip(axes, ["Re u", "|u|"]):
    if args.model != "homogeneous":
        ax.contour(x, z, v.T, 6, colors="k", linewidths=0.4, alpha=0.5)
    ax.plot(*src_xz, "*", color="gold", mec="k", ms=10)
    ax.set_title(name)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
fig.suptitle(f"{args.model}, f = {args.freq:g} Hz, {args.stencil}-point, bc {args.bc}, G = {G:.1f}")
fig.savefig(out / "freq2d_field.png", dpi=150)

# ---- homogeneous: along the receiver line against -(i/4) H0^(2)(omega r / v)
if args.model == "homogeneous":
    i0 = isrc[0, 0]
    r = (np.arange(nx) - i0) * h
    line = res["data"][0, 0]
    right = r > 0
    a = F.analytic_2d_freq(np.abs(r[right]), omega - 1j * args.damp, vmax)
    num = line[right]
    edge = x[-1] - src_xz[0]
    summary["errors"] = {}
    for lo, hi in ((100.0, 500.0), (500.0, 1000.0)):
        sel = (r[right] >= lo) & (r[right] <= hi)
        err = float(np.abs(num - a)[sel].max() / np.abs(a[sel]).max())
        summary["errors"][f"{lo:g}-{hi:g} m"] = err
        print(f"offset {lo:g} - {hi:g} m: |u - u_exact| / |u_exact| max {err:.1%}")
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), constrained_layout=True, sharex=True)
    for ax, part, name in zip(axes, (np.real, np.imag), ("Re u", "Im u")):
        ax.plot(r[right], part(a), "k", lw=2.2, label="analytic  -(i/4) H0(2)(omega r / v)")
        ax.plot(r[right], part(num), color="tab:red", lw=1.0, label=f"{args.stencil}-point, bc {args.bc}")
        ax.plot(r[right], 5 * part(num - a), color="tab:blue", lw=0.8, label="difference x 5")
        ax.set_ylabel(name)
        ax.set_ylim(-0.15, 0.15)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"receiver line z = {rec_z:g} m, f = {args.freq:g} Hz; edge of the model at {edge:g} m")
    axes[-1].set_xlabel("offset r (m)")
    fig.savefig(out / "freq2d_analytic.png", dpi=150)

(out / "freq2d.json").write_text(json.dumps(summary, indent=2))
print(f"saved {out}/freq2d_*.png, freq2d.json")
