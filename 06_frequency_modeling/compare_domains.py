"""Time-domain and frequency-domain modeling of the same shot: traces, wrap-around, and cost.

Time domain   : timedomain.propagate, leapfrog in time, PML (CPML with memory variables)
Frequency     : freqdomain.solve at f_k = k df up to f_max, df = 1 / T, at omega - i alpha_d, PML,
                then freqdomain.to_time (inverse FFT and exp(alpha_d t))

Run:  python3 compare_domains.py                     defaults below ; python3 compare_domains.py -h lists the options
      python3 compare_domains.py --period 0.6 --damp 0 --out results/wrap
      python3 compare_domains.py --period 0.6 --out results/wrap_damped
      python3 compare_domains.py --model marmousi --out results/compare_marmousi
      python3 compare_domains.py --model marmousi --nshot 50 --out results/compare_50
Out:  <out>/compare_traces.png (homogeneous, with the analytic solution) or compare_gather.png (marmousi),
      compare_spectrum.png, compare.json
"""

import argparse
import json
import math
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
import timedomain as T

# ---- variables to change ---------------------------------------------------
MODEL = "homogeneous"     # "homogeneous", "marmousi"
F0 = 5.0                  # Ricker peak frequency (Hz)
FMAX = None               # highest frequency solved (Hz) ; None: 3 f0, where the Ricker spectrum is 0.3 %
STENCIL = 5               # frequency domain: 5-point or 9-point operator
ORDER = 2                 # time domain: 2nd- or 4th-order stencil
COURANT = 0.3             # time domain: C = v_max dt / h
T_MAX = None              # recording time (s) ; None: model default below
PERIOD = None             # frequency domain: period T = 1 / df (s) ; None: T_MAX
DAMP = None               # alpha_d (1/s) ; None: ln(100) / T ; 0: real frequencies
NSHOT = 1                 # sources for the cost comparison (only the first is plotted)
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Time and frequency-domain modeling compared; options override the variables at the top.")
ap.add_argument("--model", choices=["homogeneous", "marmousi"], default=MODEL)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--fmax", type=float, default=FMAX)
ap.add_argument("--stencil", type=int, choices=[5, 9], default=STENCIL)
ap.add_argument("--order", type=int, choices=[2, 4], default=ORDER)
ap.add_argument("--courant", type=float, default=COURANT)
ap.add_argument("--tmax", type=float, default=T_MAX)
ap.add_argument("--period", type=float, default=PERIOD, help="frequency-domain period T = 1 / df (s)")
ap.add_argument("--damp", type=float, default=DAMP, help="alpha_d (1/s); default ln(100) / T")
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# model defaults: grid spacing h (m), source and receivers in metres, recording time (s)
if args.model == "homogeneous":       # source in the middle, all four sides absorbing: compare with analytic
    h, v = 10.0, np.full((301, 301), 2000.0)
    src_xz, rec_z, top, tmax = (1500.0, 1500.0), 1500.0, "absorb", 1.5
else:                                 # Marmousi, 24 m (Versteeg 1994), bundled with this lab
    model_path = HERE / "data" / "marmousi_24m.npy"
    if not model_path.is_file():
        sys.exit(f"{model_path} not found: copy the complete 06_frequency_modeling folder, including data/")
    v = np.load(model_path).astype(float)
    h = 24.0
    src_xz, rec_z, top, tmax = (4608.0, 48.0), 48.0, "free", 4.0
f0 = args.f0
fmax = args.fmax or 3.0 * f0
tmax = args.tmax or tmax
period = args.period or tmax
nx, nz = v.shape
x = np.arange(nx) * h
vmin, vmax = float(v.min()), float(v.max())

isrc = np.array([[int(round(src_xz[0] / h)), int(round(src_xz[1] / h))]])
if args.nshot > 1:
    xs = np.linspace(0.1, 0.9, args.nshot) * (nx - 1)
    isrc = np.vstack([isrc, np.column_stack([np.round(xs[1:]).astype(int), np.full(args.nshot - 1, isrc[0, 1])])])
jr = int(round(rec_z / h))
irec = np.array([[i, jr] for i in range(nx)])

# ---- time domain: one shot (all shots would take nshot times as long)
dt = args.courant * h / vmax
nt_td = int(round(tmax / dt)) + 1
t_td = np.arange(nt_td) * dt
wav = T.ricker(t_td, f0)
t0 = time.perf_counter()
td = T.propagate(v, h, dt, nt_td, isrc[:1], wav[None, :], irec, order=args.order, bc="pml", top=top, f0=f0)
wall_td = time.perf_counter() - t0
gather_td = td["traces"].astype(float)

# ---- frequency domain: all frequencies, all shots with one LU per frequency
nt_fd, df, freqs = F.frequency_axis(period, dt, fmax)
t_fd = np.arange(nt_fd) * dt
damp = math.log(100.0) / (nt_fd * dt) if args.damp is None else args.damp
if damp == 0.0:
    freqs = freqs[1:]                 # the PML stretch 1 + d / (i omega) is undefined at omega = 0
omega_c = 2 * np.pi * freqs - 1j * damp
wspec = F.source_spectrum(T.ricker(t_fd, f0), dt, omega_c)
t0 = time.perf_counter()
fd = F.solve(v, h, freqs, isrc, irec, np.repeat(wspec[:, None], len(isrc), 1), args.stencil, "pml", top,
             damp=damp)
wall_fd = time.perf_counter() - t0
spec = fd["data"][:, 0, :].T                     # (nr, nf) of the first shot
if damp == 0.0:
    spec = np.concatenate([np.zeros((len(irec), 1)), spec], axis=1)
gather_fd = F.to_time(spec, nt_fd, dt, damp)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mib = rss / 2 ** 20 if platform.system() == "Darwin" else rss / 2 ** 10

nshot = len(isrc)
G = vmin / fmax / h
summary = {"model": args.model, "f0_Hz": f0, "fmax_Hz": fmax, "G_at_fmax": G, "h_m": h, "nx": nx, "nz": nz,
           "time_domain": {"order": args.order, "courant": args.courant, "dt_s": dt, "nt": nt_td,
                           "wall_per_shot_s": round(wall_td, 3), "wall_all_shots_s": round(wall_td * nshot, 2)},
           "frequency_domain": {"stencil": args.stencil, "period_s": nt_fd * dt, "df_Hz": df, "nf": len(freqs),
                                "damp_per_s": damp, "build_s": round(fd["time_build_s"], 3),
                                "factor_s": round(fd["time_factor_s"], 3),
                                "solve_s": round(fd["time_solve_s"], 3),
                                "solve_per_shot_s": fd["time_solve_s"] / nshot, "wall_s": round(wall_fd, 2),
                                "LU_MiB": fd["lu_bytes"] / 2 ** 20, "residual": fd["residual"]},
           "nshot": nshot, "peak_rss_MiB": round(peak_mib, 1)}
print(f"{args.model}: {nx} x {nz} nodes, h = {h:g} m, f0 = {f0:g} Hz, f_max = {fmax:g} Hz, G at f_max = {G:.1f}")
print(f"time domain  (order {args.order}, C = {args.courant}): {nt_td} steps of {dt * 1e3:.2f} ms, "
      f"{wall_td:.1f} s per shot -> {wall_td * nshot:.1f} s for {nshot} shot(s)")
print(f"frequency    ({args.stencil}-point, PML): T = {nt_fd * dt:.2f} s, df = {df:.3f} Hz, {len(freqs)} frequencies, "
      f"alpha_d = {damp:.2f} 1/s")
print(f"             assembly {fd['time_build_s']:.1f} s + LU {fd['time_factor_s']:.1f} s + solves {fd['time_solve_s']:.1f} s "
      f"({fd['time_solve_s'] / nshot / len(freqs) * 1e3:.1f} ms per shot per frequency) + other = {wall_fd:.1f} s "
      f"for {nshot} shot(s); largest LU {fd['lu_bytes'] / 2 ** 20:.0f} MiB")
print(f"peak RSS {peak_mib:.0f} MiB")

# ---- traces: homogeneous with the analytic solution, Marmousi as gathers
n_cmp = min(nt_td, nt_fd)
if args.model == "homogeneous":
    offsets = [400.0, 1000.0]
    fig, axes = plt.subplots(len(offsets), 1, figsize=(9, 5.5), constrained_layout=True, sharex=True)
    summary["traces"] = {}
    for ax, off in zip(axes, offsets):
        k = int(round((src_xz[0] + off) / h))
        a = T.analytic_2d(off, t_td, vmax, f0)
        amp = np.abs(a).max()
        e_td = float(np.abs(gather_td[k] - a).max() / amp)
        e_fd = float(np.abs(gather_fd[k, :n_cmp] - a[:n_cmp]).max() / amp)
        e_x = float(np.abs(gather_fd[k, :n_cmp] - gather_td[k, :n_cmp]).max() / amp)
        summary["traces"][f"{off:g} m"] = {"time_vs_analytic": e_td, "freq_vs_analytic": e_fd, "time_vs_freq": e_x}
        ax.plot(t_td, a, "k", lw=2.4, label="analytic")
        ax.plot(t_td, gather_td[k], color="tab:blue", lw=1.2, label=f"time domain, order {args.order}")
        ax.plot(t_fd, gather_fd[k], color="tab:red", lw=1.0, ls="--", label=f"frequency domain, {args.stencil}-point")
        ax.axvline(nt_fd * dt, color="0.6", ls=":", lw=1)
        ax.set_ylim(-1.6 * amp, 1.6 * amp)
        ax.set_title(f"offset {off:g} m: error time {e_td:.1%}, frequency {e_fd:.1%}, between them {e_x:.1%}")
        print(f"offset {off:g} m: max error against analytic, time {e_td:.1%}, frequency {e_fd:.1%}; "
              f"time vs frequency {e_x:.1%}")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("t (s)")
    fig.savefig(out / "compare_traces.png", dpi=150)
else:
    g = np.percentile(np.abs(gather_td), 99) or 1.0
    diff = gather_fd[:, :n_cmp] - gather_td[:, :n_cmp]
    summary["gather_difference_rms"] = float(np.sqrt(np.mean(diff ** 2)) / np.sqrt(np.mean(gather_td ** 2)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), constrained_layout=True, sharey=True)
    for ax, d, name in zip(axes, (gather_td, gather_fd, 10 * diff),     # difference magnified 10 times
                           ("time domain", "frequency domain", "(frequency - time) x 10")):
        tt = np.arange(d.shape[1]) * dt
        ax.imshow(d.T, extent=[x[0], x[-1], tt[-1], tt[0]], aspect="auto", cmap="gray_r", vmin=-g, vmax=g)
        ax.set_title(name); ax.set_xlabel("receiver x (m)")
    axes[0].set_ylabel("t (s)")
    fig.suptitle(f"Marmousi shot gather at z = {rec_z:g} m, f0 = {f0:g} Hz; rms difference "
                 f"{summary['gather_difference_rms']:.1%} of the time-domain rms")
    fig.savefig(out / "compare_gather.png", dpi=150)
    print(f"rms difference of the gathers: {summary['gather_difference_rms']:.1%} of the time-domain rms")

# ---- spectrum of the source and of one trace: the frequencies that were solved
# The solved values are at omega - i alpha_d, the transform of u(t) exp(-alpha_d t). The source and the
# time-domain trace get the same exp(-alpha_d t), so all three curves are the same quantity; without it
# the low frequencies (below a few alpha_d / 2 pi) and the notches of a trace with many arrivals differ.
k = int(round((src_xz[0] + (400.0 if args.model == "homogeneous" else 2000.0)) / h))
f_all = np.fft.rfftfreq(nt_td, dt)
taper = np.exp(-damp * t_td)
w_all = np.abs(np.fft.rfft(wav * taper))
u_all = np.abs(np.fft.rfft(gather_td[k] * taper)) * dt     # * dt: same scale as the solved u~
fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
ax.plot(f_all, w_all / w_all.max(), "k", lw=1.5, label="Ricker |w~|")
ax.plot(f_all, u_all / u_all.max(), color="tab:blue", lw=1.0, label="time-domain trace")
ax.plot(freqs, np.abs(spec[k, -len(freqs):]) / u_all.max(), "o", ms=3,           # same scale as the trace
        color="tab:red", label=f"solved frequencies (df = {df:.2f} Hz)")
ax.axvline(fmax, color="0.6", ls=":", lw=1)
ax.set_xlim(0, 1.5 * fmax)
ax.set_xlabel("f (Hz)"); ax.set_ylabel("normalized amplitude")
ax.set_title(f"at omega - i alpha_d, alpha_d = {damp:.2f} 1/s: spectra of the signals times exp(-alpha_d t)"
             if damp else "at real frequencies", fontsize=10)
ax.legend(fontsize=9)
fig.savefig(out / "compare_spectrum.png", dpi=150)

(out / "compare.json").write_text(json.dumps(summary, indent=2))
print(f"saved {out}/compare_*.png, compare.json")
