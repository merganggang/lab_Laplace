"""Laplace transform of time-domain traces against the Laplace-domain solution of the same shot.

Time domain : timedomain.propagate with the -Ricker source, PML (CPML), then
              u~(s) = sum_n u(t_n) exp(-s t_n) dt for every trace (the "observed" data)
Laplace     : laplacedomain.solve for each s with w~(s), the same sum for the source wavelet, PML

Both use the same grid and the same 5-point spatial stencil. The time stepping also accumulates the
Laplace transform of the whole field, so the two results are compared at every node, not only at the
receivers. Optional noise on the traces shows why the part before the first arrival must be muted:
exp(-s t) weights early times most.

Run:  python3 compare_laplace.py                      defaults below ; python3 compare_laplace.py -h lists the options
      python3 compare_laplace.py --tmax 1.0 --out results/short
      python3 compare_laplace.py --noise 1e-4 --out results/noise
      python3 compare_laplace.py --noise 1e-4 --mute --out results/noise_mute
      python3 compare_laplace.py --model marmousi --out results/compare_marmousi
Out:  <out>/compare_field.png (ln u~ of both and their ln difference over the model, smallest and
      largest s), compare_laplace.png (ln u~ along the receivers), compare_trace.png (a trace, its damped
      versions and their spectra), compare.json
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
import timedomain as T

# ---- variables to change ---------------------------------------------------
MODEL = "homogeneous"            # "homogeneous", "marmousi"
SVALS = [2, 4, 6, 8, 10, 12]     # damping constants (1/s)
F0 = 5.0                         # Ricker peak frequency (Hz); the source is -Ricker
COURANT = 0.3                    # time domain: C = v_max dt / h
T_MAX = None                     # recording time (s) ; None: model default below
NOISE = 0.0                      # random noise on the traces, standard deviation / largest |u| of the gather
MUTE = False                     # set the traces to 0 before the first arrival (FSM traveltime)
SEED = 7                         # random seed of the noise
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Laplace transform of time traces against the Laplace-domain solution; "
                                         "options override the variables at the top.")
ap.add_argument("--model", choices=["homogeneous", "marmousi"], default=MODEL)
ap.add_argument("--s", type=float, nargs="+", default=SVALS, help="damping constants (1/s)")
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--courant", type=float, default=COURANT)
ap.add_argument("--tmax", type=float, default=T_MAX)
ap.add_argument("--noise", type=float, default=NOISE, help="noise standard deviation / largest |u| of the gather")
ap.add_argument("--mute", action=argparse.BooleanOptionalAction, default=MUTE)
ap.add_argument("--seed", type=int, default=SEED)
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
svals = np.asarray(args.s, float)

# model defaults: grid spacing h (m), source and receivers in metres, recording time (s), trace to plot
if args.model == "homogeneous":       # source in the middle, all four sides absorbing
    h, v = 10.0, np.full((301, 301), 2000.0)
    src_xz, rec_z, top, tmax, off_plot = (1500.0, 1500.0), 1500.0, "absorb", 3.0, 1000.0
else:                                 # Marmousi, 24 m (Versteeg 1994), bundled with this lab
    model_path = HERE / "data" / "marmousi_24m.npy"
    if not model_path.is_file():
        sys.exit(f"{model_path} not found: copy the complete 07_laplace_modeling folder, including data/")
    v = np.load(model_path).astype(float)
    h = 24.0
    src_xz, rec_z, top, tmax, off_plot = (4608.0, 96.0), 96.0, "free", 6.0, 3000.0
tmax = args.tmax or tmax
f0 = args.f0
nx, nz = v.shape
x = np.arange(nx) * h
vmin, vmax = float(v.min()), float(v.max())
isrc = np.array([[int(round(src_xz[0] / h)), int(round(src_xz[1] / h))]])
jr = int(round(rec_z / h))
irec = np.array([[i, jr] for i in range(nx)])
offset = (irec[:, 0] - isrc[0, 0]) * h

# ---- time domain: the "observed" traces
dt = args.courant * h / vmax
nt = int(round(tmax / dt)) + 1
t = np.arange(nt) * dt
wav = -T.ricker(t, f0)                                    # -Ricker: w~(s) > 0, so u~(s) > 0
t0 = time.perf_counter()
td = T.propagate(v, h, dt, nt, isrc, wav[None, :], irec, order=2, bc="pml", top=top, f0=f0, dtype=np.float64,
                 laplace_s=svals)                         # also sum u exp(-s t) dt over the whole field
wall_td = time.perf_counter() - t0
gather = td["traces"].astype(float)
clean = gather.copy()
gmax = np.abs(gather).max()
if args.noise > 0:
    gather = gather + args.noise * gmax * np.random.default_rng(args.seed).standard_normal(gather.shape)
tfirst = E.sample(E.fsm(1.0 / v, h, ((isrc[0, 0] + 0.5) * h, (isrc[0, 1] + 0.5) * h)), h,
                  np.column_stack([(irec[:, 0] + 0.5) * h, (irec[:, 1] + 0.5) * h]))   # velocity nodes = FSM cells
if args.mute:
    gather = np.where(t[None, :] < tfirst[:, None], 0.0, gather)    # the -Ricker starts at the first arrival

# ---- Laplace transform of the traces and of the source wavelet (the same sum)
obs = L.laplace_transform(gather, dt, svals)              # (n_s, nr)
wlap = L.laplace_transform(wav, dt, svals)                # (n_s,)

# ---- Laplace domain: one LU per damping constant
t0 = time.perf_counter()
ld = L.solve(v, h, svals, isrc, irec, wlap, "pml", top, fields=True)
wall_ld = time.perf_counter() - t0
calc = ld["data"][:, 0, :]
field_td, field_ld = td["laplace"], ld["field"]            # (n_s, nx, nz): time domain transformed, Laplace domain
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mib = rss / 2 ** 20 if platform.system() == "Darwin" else rss / 2 ** 10

# nodes compared over the model: 100 m or more from the source, below the free-surface row u = 0
I, J = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
far = np.hypot((I - isrc[0, 0]) * h, (J - isrc[0, 1]) * h) >= 100.0
if top == "free":
    far &= J >= 1

lo = 100.0 if args.model == "homogeneous" else 500.0     # away from the source node
hi = 1000.0 if args.model == "homogeneous" else np.inf   # inside the absorbing layer's reach
sel = (np.abs(offset) >= lo) & (np.abs(offset) <= hi)
summary = {"model": args.model, "f0_Hz": f0, "h_m": h, "nx": nx, "nz": nz, "tmax_s": tmax, "dt_s": dt, "nt": nt,
           "noise": args.noise, "mute": args.mute, "offset_range_m": [lo, None if np.isinf(hi) else hi],
           "time_domain_s": round(wall_td, 2), "laplace_domain_s": round(wall_ld, 3),
           "LU_MiB": ld["lu_bytes"] / 2 ** 20, "residual": ld["residual"], "peak_rss_MiB": round(peak_mib, 1),
           "s": {}, "field": {}}
print(f"{args.model}: {nx} x {nz} nodes, h = {h:g} m, f0 = {f0:g} Hz, -Ricker source, record {tmax:g} s "
      f"({nt} steps of {dt * 1e3:.2f} ms)" + (f", noise {args.noise:g} x max" if args.noise else "")
      + (", muted before the first arrival" if args.mute else ""))
print(f"time domain {wall_td:.1f} s; Laplace domain {len(svals)} damping constants {wall_ld:.2f} s "
      f"(LU {ld['lu_bytes'] / 2 ** 20:.0f} MiB)")
print(f"receivers {lo:g} m{'' if np.isinf(hi) else f' - {hi:g} m'} from the source: |ln u~_time - ln u~_Laplace|")
for k, s in enumerate(svals):
    a, b = obs[k, sel], calc[k, sel]
    bad = a <= 0
    d = np.abs(np.log(a[~bad] / b[~bad]))
    row = {"w_laplace": float(wlap[k]), "max": float(d.max()) if d.size else None,
           "median": float(np.median(d)) if d.size else None, "nonpositive": int(bad.sum()),
           "truncation": float(np.exp(-s * tmax))}
    summary["s"][f"{s:g}"] = row
    print(f"  s = {s:4g} 1/s: median {row['median']:.4f}, max {row['max']:.4f}"
          + (f", {row['nonpositive']} receivers with u~ <= 0 (ln undefined)" if bad.any() else "")
          + f"; exp(-s T) = {row['truncation']:.1e}")
print("whole model, nodes 100 m or more from the source: |ln u~_time - ln u~_Laplace| (noise-free time-domain field)")
for k, s in enumerate(svals):
    a, b = field_td[k][far], field_ld[k][far]
    bad = a <= 0                                          # a short record can leave u~ <= 0 (ln undefined)
    d = np.abs(np.log(a[~bad] / b[~bad]))
    row = {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max()),
           "nonpositive": int(bad.sum())}
    summary["field"][f"{s:g}"] = row
    print(f"  s = {s:4g} 1/s: median {row['median']:.4f}, 95th percentile {row['p95']:.4f}, max {row['max']:.4f}"
          + (f", {row['nonpositive']} nodes with u~ <= 0 (ln undefined)" if bad.any() else ""))
print(f"peak RSS {peak_mib:.0f} MiB")

# ---- whole field: time-domain field transformed vs Laplace-domain field, smallest and largest s
xz = [x[0], x[-1], (nz - 1) * h, 0]
pick = [0, len(svals) - 1]
fig, axes = plt.subplots(len(pick), 3, figsize=(14, 1.2 + len(pick) * (4.2 if args.model == "homogeneous" else 2.2)),
                         constrained_layout=True, squeeze=False)
for row_ax, k in zip(axes, pick):
    lg_ld = np.log(np.maximum(field_ld[k], 1e-300))
    lo_c = np.percentile(lg_ld[:, 1:] if top == "free" else lg_ld, 0.5)
    for ax, f, name in ((row_ax[0], field_td[k], "time domain, transformed"), (row_ax[1], field_ld[k], "Laplace domain")):
        with np.errstate(invalid="ignore", divide="ignore"):
            im = ax.imshow(np.log(f).T, extent=xz, cmap="viridis", vmin=lo_c, vmax=lg_ld.max(), aspect="equal")
        ax.set_title(f"{name}, s = {svals[k]:g}")
    fig.colorbar(im, ax=row_ax[1], shrink=0.85, label="ln u~")
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.log(field_td[k] / field_ld[k])
    im = row_ax[2].imshow(diff.T, extent=xz, cmap="RdBu_r", vmin=-0.05, vmax=0.05, aspect="equal")
    fig.colorbar(im, ax=row_ax[2], shrink=0.85, label="ln difference")
    row_ax[2].set_title(f"difference, median {summary['field'][f'{svals[k]:g}']['median']:.4f}")
    for ax in row_ax:
        ax.plot(src_xz[0], src_xz[1], "*", color="gold", mec="k", ms=9)
axes[-1, 0].set_xlabel("x (m)")
fig.savefig(out / "compare_field.png", dpi=150)

# ---- ln u~ along the receivers: transform of the traces (lines) and the Laplace-domain solution (dots)
cmap = plt.get_cmap("viridis")
cols = [cmap(0.9 * k / max(len(svals) - 1, 1)) for k in range(len(svals))]
fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), constrained_layout=True, sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
right = offset > 0
for k, s in enumerate(svals):
    with np.errstate(invalid="ignore", divide="ignore"):
        axes[0].plot(offset[right], np.log(obs[k, right]), color=cols[k], lw=1.4, label=f"s = {s:g}")
        axes[0].plot(offset[right][::6], np.log(calc[k, right][::6]), "o", ms=3, color=cols[k])
        axes[1].plot(offset[right], np.log(obs[k, right] / calc[k, right]), color=cols[k], lw=1.0)
axes[0].set_ylabel("ln u~")
axes[0].legend(ncol=3, fontsize=9, title="lines: transformed traces, dots: Laplace domain", title_fontsize=9)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_ylim(-0.1, 0.1)
axes[1].set_ylabel("difference of ln u~")
axes[1].set_xlabel("offset (m)")
axes[0].set_title(f"{args.model}, record {tmax:g} s" + (f", noise {args.noise:g}" if args.noise else "")
                  + (", muted" if args.mute else ""))
fig.savefig(out / "compare_laplace.png", dpi=150)

# ---- one trace, its damped versions exp(-s t) u(t), and their spectra: the 0 Hz value is u~(s)
kr = int(np.argmin(np.abs(offset - off_plot)))
show_s = [svals[0], svals[len(svals) // 2], svals[-1]]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
axes[0].plot(t, gather[kr] / np.abs(clean[kr]).max(), color="0.3", lw=1.0, label="u(t)")
for s, c in zip(show_s, (cols[0], cols[len(svals) // 2], cols[-1])):
    damped = gather[kr] * np.exp(-s * t)
    axes[0].plot(t, damped / np.abs(clean[kr]).max(), color=c, lw=1.2, label=f"u(t) exp(-{s:g} t)")
    f_ax = np.fft.rfftfreq(nt, dt)
    spec = np.abs(np.fft.rfft(damped)) * dt
    axes[1].semilogy(f_ax, spec, color=c, lw=1.2, label=f"s = {s:g}: 0 Hz value = u~(s)")
spec0 = np.abs(np.fft.rfft(gather[kr])) * dt
axes[1].semilogy(f_ax, np.maximum(spec0, 1e-30), color="0.3", lw=1.0, label="u(t)")
if args.mute:
    axes[0].axvline(tfirst[kr], color="tab:red", ls=":", lw=1)
axes[0].set(xlabel="t (s)", ylabel="u / max |u|", title=f"trace at offset {offset[kr]:g} m")
axes[0].legend(fontsize=9)
axes[1].set(xlabel="f (Hz)", ylabel="|spectrum|", xlim=(0, 4 * f0), title="spectra of the damped traces")
axes[1].set_ylim(spec0.max() * 1e-4, spec0.max() * 3)
axes[1].legend(fontsize=9)
fig.savefig(out / "compare_trace.png", dpi=150)

(out / "compare.json").write_text(json.dumps(summary, indent=2))
print(f"saved {out}/compare_*.png, compare.json")
