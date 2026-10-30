"""First-arrival traveltime by the log method (Shin et al. 2002): frequency-domain modeling at a complex frequency.

At the complex frequency omega - i s (laplacedomain with the complex constant p = s + i omega) one
arrival A exp(-s t_f) exp(-i omega t_f) gives ln G = ln A - s t_f - i omega t_f, so

    t_f = -Im[ln G] / omega        (one solve, no derivative)

At a low frequency this is the SWEET time -d ln G / ds of sweet.py: both are the damping-weighted mean
time. The phase Im[ln G] = -omega t_f is known only in (-pi, pi], so omega t_f must stay below pi: the
textbook uses 0.01 Hz or less. At higher frequencies the phase wraps and t_f jumps by 1 / f.

Impulse source, both sides and the top absorb (as sweet.py).

Run:  python3 logtime.py                                defaults below ; python3 logtime.py -h lists the options
      python3 logtime.py --s 50 --freq 0.01 0.5 1 --out results/logtime_freq
      python3 logtime.py --model marmousi --out results/logtime_marmousi
Out:  <out>/logtime_line.png (traveltime along the source depth), logtime_maps.png (FSM, log method at the
      largest s and first frequency, their difference), logtime.json
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
import eikonal as E
import laplacedomain as L

# ---- variables to change ---------------------------------------------------
MODEL = "homogeneous"            # "homogeneous", "marmousi"
SVALS = [10, 20, 30, 40, 50]     # damping constants (1/s)
FREQS = [0.01]                   # frequencies (Hz) of the complex frequency omega - i s
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Log-method traveltime against the FSM; options override the variables at the top.")
ap.add_argument("--model", choices=["homogeneous", "marmousi"], default=MODEL)
ap.add_argument("--s", type=float, nargs="+", default=SVALS, help="damping constants (1/s)")
ap.add_argument("--freq", type=float, nargs="+", default=FREQS, help="frequencies (Hz)")
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
svals, freqs = np.asarray(args.s, float), np.asarray(args.freq, float)

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

# every (s, f) pair: complex constant p = s + i omega, one complex LU and one solve each
pairs = [(s, f) for f in freqs for s in svals]
p = np.array([s + 2j * np.pi * f for s, f in pairs])
t0 = time.perf_counter()
res = L.solve(v, h, p, isrc, irec, np.ones(len(p)), "pml", "absorb", fields=True)
wall = time.perf_counter() - t0
t_log = np.array([L.log_time(res["field"][k], 2 * np.pi * f) for k, (s, f) in enumerate(pairs)])

t1 = time.perf_counter()
TT = E.fsm(1.0 / v, h, ((isrc[0, 0] + 0.5) * h, (isrc[0, 1] + 0.5) * h))   # velocity nodes = FSM cells
I, J = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
t_fsm = E.sample(TT, h, np.column_stack([(I.ravel() + 0.5) * h, (J.ravel() + 0.5) * h])).reshape(nx, nz)
wall_fsm = time.perf_counter() - t1
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mib = rss / 2 ** 20 if platform.system() == "Darwin" else rss / 2 ** 10

r = np.hypot((I - isrc[0, 0]) * h, (J - isrc[0, 1]) * h)
far = r >= 500.0                                                # away from the source node
summary = {"model": args.model, "h_m": h, "nx": nx, "nz": nz, "complex_s": round(wall, 2), "fsm_s": round(wall_fsm, 2),
           "LU_MiB": res["lu_bytes"] / 2 ** 20, "peak_rss_MiB": round(peak_mib, 1), "runs": []}
print(f"{args.model}: {nx} x {nz} nodes, h = {h:g} m, impulse source; {len(p)} complex LU and solves {wall:.1f} s, "
      f"FSM {wall_fsm:.1f} s")
print("nodes 500 m or more from the source: t_log - t_FSM (s); omega t_f > pi wraps the phase")
for k, (s, f) in enumerate(pairs):
    d = (t_log[k] - t_fsm)[far]
    wrapped = float(np.mean(2 * np.pi * f * t_fsm[far] > np.pi))
    row = {"s": s, "f_Hz": f, "mean": float(d.mean()), "median": float(np.median(d)), "min": float(d.min()),
           "max": float(d.max()), "abs_median": float(np.median(np.abs(d))),
           "nodes_beyond_pi": wrapped}
    summary["runs"].append(row)
    print(f"  s = {s:4g} 1/s, f = {f:g} Hz: mean {row['mean']:+.3f}, median {row['median']:+.3f}, range {row['min']:+.3f} ~ "
          f"{row['max']:+.3f}; |difference| median {row['abs_median']:.3f}"
          + (f"; omega t_FSM > pi at {wrapped:.0%} of the nodes" if wrapped else ""))

# ---- along the line through the source
right = (irec[:, 0] - isrc[0, 0]) * h
m = right > 0
jl = isrc[0, 1]
cmap = plt.get_cmap("viridis")
cols = [cmap(0.9 * k / max(len(pairs) - 1, 1)) for k in range(len(pairs))]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
axes[0].plot(right[m], t_fsm[m, jl], "k", lw=2.4, label="FSM")
for k, (s, f) in enumerate(pairs):
    axes[0].plot(right[m], t_log[k][m, jl], color=cols[k], lw=1.4, label=f"log, s = {s:g}, f = {f:g} Hz")
axes[0].set(xlabel="offset (m)", ylabel="t (s)", title="traveltime along the source depth")
axes[0].legend(fontsize=8)
for f in freqs:
    rows = [rw for rw in summary["runs"] if rw["f_Hz"] == f]
    axes[1].plot([rw["s"] for rw in rows], [rw["mean"] for rw in rows], "o-", label=f"mean of t_log - t_FSM, f = {f:g} Hz")
axes[1].axhline(0, color="0.6", lw=0.8)
axes[1].set(xlabel="s (1/s)", ylabel="bias (s)", title="nodes 500 m or more from the source")
axes[1].legend(fontsize=9)
fig.savefig(out / "logtime_line.png", dpi=150)

# ---- maps at the largest s and the first frequency
k = pairs.index((svals[-1], freqs[0]))
fig, axes = plt.subplots(3, 1, figsize=(9, 2.0 + 9.5 * nz / nx * (1 if nz < nx else 0.6)), constrained_layout=True)
ext = [x[0], x[-1], z[-1], z[0]]
tmax_map = float(np.nanmax(t_fsm))
for ax, f_, name in ((axes[0], t_fsm, "FSM t (s)"), (axes[1], t_log[k], f"log t (s), s = {svals[-1]:g}, f = {freqs[0]:g} Hz")):
    im = ax.imshow(f_.T, extent=ext, cmap="viridis", vmin=0, vmax=tmax_map, aspect="equal")
    ax.contour(x, z, f_.T, np.arange(0.5, tmax_map, 0.5), colors="w", linewidths=0.6)
    fig.colorbar(im, ax=ax, shrink=0.9, label=name)
diff = t_log[k] - t_fsm
im = axes[2].imshow(diff.T, extent=ext, cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="equal")
fig.colorbar(im, ax=axes[2], shrink=0.9, label="t_log - t_FSM (s)")
for ax in axes:
    ax.plot(*src_xz, "*", color="gold", mec="k", ms=10)
    ax.set_ylabel("z (m)")
axes[-1].set_xlabel("x (m)")
fig.savefig(out / "logtime_maps.png", dpi=150)

(out / "logtime.json").write_text(json.dumps(summary, indent=2))
print(f"saved {out}/logtime_*.png, logtime.json")
