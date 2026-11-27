"""Laplace-domain or frequency-domain full waveform inversion of the SEG/EAGE salt model (A-A' section) with L-BFGS; the Laplace-domain result can
be the starting model of the frequency-domain inversion.

Run:  python3 fwi.py                                          Laplace domain from the linear model ; -h lists the options
      python3 fwi.py --domain frequency --init results/lap/fwi_models.npz --out results/lap_freq
      python3 fwi.py --domain frequency --init linear --out results/freq_linear
      python3 fwi.py --lowcut 3 --out results/lap_lc3 ; python3 fwi.py --lowcut 3 --s 2 4 6 8 10 --out results/lap_lc3_s2
      python3 fwi.py --domain frequency --lowcut 3 --freqs 3 4 5 --init results/lap_lc3/fwi_models.npz --out results/lc3_freq
      python3 fwi.py --noise 1e-6 --out results/noise ; python3 fwi.py --noise 1e-6 --mute --out results/noise_mute
Out:  <out>/fwi_models.png, fwi_history.png, fwi_data.png, fwi_wavelet.png (--srcest), fwi.json, fwi_models.npz,
      history_stage<k>.json (geoinv.History per stage)

Observed data: the true model with the time-domain code (4th-order space, CPML 20 nodes, record 10 s), kept in
results/obs/ and reused by later runs. Noise, the low-cut filter and the mute are applied to these traces, then
the traces are Laplace transformed at the damping constants (--domain laplace) or transformed at omega - i alpha_d,
alpha_d = ln(100) / T (--domain frequency). Inversion: the 5-point Laplace-domain operator or the 9-point
frequency-domain operator, both with a PML. Receivers closer than --min-offset to the source are left out: none by
default with the log objective, 400 m with the l2 objective (the frequency domain always uses l2). The observed data
come from a different scheme than the inversion, so the values next to the source differ by a few percent (log
difference up to 0.13 at zero offset); the log objective weighs that like any other relative difference, but the l2
objective is dominated by the large near-offset values and fits the difference with false shallow structure.
The wavelet is estimated at every forward by default (Laplace: ln|w~| is the mean log difference).
With the log objective, data whose sign differs from most data of the same damping constant (noise) are left out.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # one BLAS thread per process: damping constants or frequencies in parallel

import argparse
import json
import math
import sys
import time
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

# ---- variables to change ---------------------------------------------------
DOMAIN = "laplace"        # "laplace" or "frequency"
H = 40.0                  # salt model grid spacing (m): 40 or 20
INIT = "linear"           # "linear" (in depth), "smooth" (true model smoothed) or a file: fwi_models.npz of a run
SVALS = [4, 6, 8, 10]                 # damping constants (1/s), Laplace domain
FREQS = [2, 3, 4, 5]                  # frequencies (Hz), frequency domain
GROUPS = []               # damping constants or frequencies inverted one group after the other, e.g. ["10,8", "6,4"]
NITER = None              # iterations per group; None: 40 (Laplace), 45 (frequency)
OBJECTIVE = "log"         # Laplace domain: "log" or "l2"
PRECOND = None            # pseudo-Hessian diagonal preconditioning; None: on (Laplace), off (frequency)
WAVELET = "ricker"        # true source wavelet of the observed data: "ricker", "other"
SRCEST = True             # estimate the wavelet at every forward from the data and the current model
LOWCUT = 0.0              # low-cut filter (Hz) on the observed traces, 0: none
NOISE = 0.0               # noise on the traces: standard deviation / largest |u| of each shot gather
MUTE = False              # set the traces to 0 before the first arrival (FSM traveltime in the true model)
NSHOT = 48
TMAX = 10.0               # record length (s)
F0 = 6.0                  # Ricker peak frequency (Hz)
MIN_OFFSET = None         # receivers closer than this to the source are not used (m); None: 0 (log), 400 (l2)
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Laplace- or frequency-domain FWI; options override the variables at the top.")
ap.add_argument("--domain", choices=["laplace", "frequency"], default=DOMAIN)
ap.add_argument("--h", type=float, choices=[40.0, 20.0], default=H, help="salt model grid spacing (m)")
ap.add_argument("--init", default=INIT, help="linear, smooth or the fwi_models.npz of an earlier run")
ap.add_argument("--s", type=float, nargs="+", default=SVALS, help="damping constants (1/s), Laplace domain")
ap.add_argument("--freqs", type=float, nargs="+", default=FREQS, help="frequencies (Hz), frequency domain (40 m grid: up to 5)")
ap.add_argument("--groups", nargs="*", default=GROUPS, help="groups inverted one after the other, e.g. 10,8 6,4")
ap.add_argument("--niter", type=int, default=NITER, help="iterations per group")
ap.add_argument("--objective", choices=["log", "l2"], default=OBJECTIVE, help="Laplace domain objective")
ap.add_argument("--precond", action=argparse.BooleanOptionalAction, default=PRECOND)
ap.add_argument("--wavelet", choices=["ricker", "other"], default=WAVELET, help="true wavelet of the observed data")
ap.add_argument("--srcest", action=argparse.BooleanOptionalAction, default=SRCEST,
                help="estimate the wavelet at every forward (--no-srcest: use the true wavelet)")
ap.add_argument("--lowcut", type=float, default=LOWCUT, help="low-cut filter corner (Hz) on the observed traces")
ap.add_argument("--noise", type=float, default=NOISE)
ap.add_argument("--mute", action=argparse.BooleanOptionalAction, default=MUTE)
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--tmax", type=float, default=TMAX)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--min-offset", type=float, default=MIN_OFFSET, help="receivers nearer the source are left out (m); default 0 (log objective), 400 (l2, also the frequency domain)")
ap.add_argument("--workers", type=int, default=None, help="processes (default: LAB_CPUS, at most 8)")
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
lap = args.domain == "laplace"
if args.min_offset is None:
    args.min_offset = 0.0 if lap and args.objective == "log" else 400.0
niter = args.niter if args.niter is not None else (40 if lap else 45)
precond = lap if args.precond is None else args.precond

# ---- model, acquisition, damping constants or frequencies
v_true, h = F.salt(args.h), args.h
n_fixed = F.water_rows(v_true)
nx, nz = v_true.shape
if args.init in ("linear", "smooth"):
    init = args.init
    v0 = F.initial(v_true, h, init, known_rows=n_fixed)
else:
    z0 = np.load(args.init)
    v0 = np.asarray(z0["v_final"] if hasattr(z0, "files") else z0, float)
    if v0.shape != v_true.shape:
        sys.exit(f"{args.init}: model shape {v0.shape}, expected {v_true.shape} (same --h)")
    init = str(Path(args.init).parent.name or args.init)
src, rec = F.acquisition(nx, args.nshot, h)
if args.groups:
    stages = [[float(f) for f in g.split(",")] for g in args.groups]
else:
    stages = [list(map(float, args.s if lap else args.freqs))]
pars = sorted({f for g in stages for f in g})
unit = "1/s" if lap else "Hz"
damp = math.log(100.0) / args.tmax
print(f"salt: {nx} x {nz} nodes, h = {h:g} m, {args.nshot} shots, {len(rec)} receivers, {args.domain} domain, "
      f"{' / '.join(','.join(f'{f:g}' for f in g) for g in stages)} {unit}"
      + (f", alpha_d = {damp:.2f} 1/s" if not lap else f", {args.objective} objective"))

# ---- observed data: time-domain traces (kept in results/obs/), then noise, low-cut filter, mute, transform
tr, dtr, path, sec = F.observed_records("salt", v_true, h, src, rec, args.wavelet, args.f0, args.tmax, args.workers)
print(f"observed traces (time domain, 4th order, CPML {F.OBS_NB}, {args.tmax:g} s): {sec:.1f} s, {path}"
      if sec else f"observed traces from {path}")
mute = None
if args.mute:
    tpath = path.parent / f"firstarrivals_h{h:g}_s{len(src)}.npy"            # FSM in the true model, kept
    if tpath.is_file():
        mute = np.load(tpath)
    else:
        mute = F.first_arrivals(v_true, h, src, rec)
        np.save(tpath, mute)
tr = F.prepare(tr, dtr, noise=args.noise, mute=mute, lowcut=args.lowcut)
dt, nt = F.time_axis(h, args.tmax)
t = np.arange(nt) * dt
w_true = F.wavelet(args.wavelet, t, args.f0)
use = np.abs(src[:, None, 0] - rec[None, :, 0]) * h >= args.min_offset      # (ns, nr)
if lap:
    d_all = F.laplace_data(tr, dtr, pars)                                    # (n_s, ns, nr)
    w_par = F.wavelet_lap(w_true, dt, pars, args.lowcut)
else:
    d_all = F.freq_data(tr, dtr, pars, damp)                                 # (nf, ns, nr)
    w_par = F.wavelet_spec(w_true, dt, 2 * np.pi * np.array(pars) - 1j * damp, args.lowcut)
mask = np.broadcast_to(use, d_all.shape).copy()
n_wrong = 0
if lap and args.objective == "log":
    for k in range(len(pars)):
        sign = 1.0 if np.sum(d_all[k][use] > 0) >= np.sum(d_all[k][use] < 0) else -1.0
        bad = use & ~(d_all[k] * sign > 0)
        mask[k] &= ~bad
        n_wrong += int(bad.sum())
    if n_wrong:
        print(f"data of the wrong sign left out: {n_wrong} of {int(use.sum()) * len(pars)}")

lower, upper = 1.4, 6.0                              # km/s
free = None


def model_error(m):
    """rms difference from the true model over the free nodes (m/s)."""
    sel = free.reshape(nx, nz)
    return float(np.sqrt(np.mean((m.reshape(nx, nz) * 1000.0 - v_true)[sel] ** 2)))


v_ref = F.initial(v_true, h, "linear", known_rows=n_fixed)     # the correlation is measured from the linear model


def model_corr(m):
    """Correlation of the update v - v_lin with the true perturbation v_true - v_lin over the free nodes, v_lin the
    linear starting model, so that runs from different starting models compare."""
    sel = free.reshape(nx, nz)
    dv = (m.reshape(nx, nz) * 1000.0 - v_ref)[sel]
    if np.abs(dv).max() < 1e-6:                      # the linear model itself (km/s round trip leaves ~1e-13)
        return 0.0
    return float(np.corrcoef(dv, (v_true - v_ref)[sel])[0, 1])


class Scaled(geoinv.Problem):
    """m = m_ref + D x: L-BFGS on x with D = H_P^(-1/2) (diagonal preconditioning by change of variables)."""

    def __init__(self, base, m_ref, D):
        self.base, self.m_ref, self.D = base, m_ref, D
        self.n_model, self.fixed_mask = base.n_model, base.fixed_mask

    def forward(self, x):
        return self.base.forward(self.m_ref + self.D * x)

    def gradient(self, x, r):
        return self.D * self.base.gradient(self.m_ref + self.D * x, r)


def make_problem(group, d_obs, msk):
    gi = [pars.index(f) for f in group]
    if lap:
        return F.LaplaceFWI(nx, nz, h, group, src, rec, w_par[gi], fixed_rows=n_fixed, objective=args.objective,
                            srcest=args.srcest, d_obs=d_obs, mask=msk, workers=args.workers)
    return F.FreqFWI(nx, nz, h, group, src, rec, w_par[gi], stencil=9, nb=20, damp=damp, fixed_rows=n_fixed,
                     srcest=args.srcest, d_obs=d_obs, mask=msk[0], workers=args.workers)


def residual(prob, mm, d, msk):
    """Data vector, residual and its weight for the stage: Laplace log: ln|R u| - ln|d|; else R u - d."""
    if lap:
        dv = prob.data_vector(d)
        r = prob.forward(mm) - dv
        return dv, r.reshape(d.shape)
    dv = prob.to_real(np.where(msk[0], d, 0.0))
    return dv, prob.to_complex(prob.forward(mm)) - d


m = v0.ravel() / 1000.0
models, stage_rows, history, w_est = [], [], [], {}
solves_before = 0
wall0 = time.perf_counter()
for k, group in enumerate(stages):
    gi = [pars.index(f) for f in group]
    d_k, m_k = d_all[gi], mask[gi]
    prob = make_problem(group, d_k, m_k)
    free = ~prob.fixed_mask
    if k == 0:
        err0, corr0 = model_error(m), model_corr(m)
        print(f"initial model ({init}): rms model error {err0:.0f} m/s, correlation {corr0:+.2f}, {free.sum()} free nodes")
    dvec, r0 = residual(prob, m, d_k, m_k)
    if lap and args.objective == "log":
        sig_f = np.ones(len(group))                                          # log residuals: every value alike
    else:
        sig_f = np.sqrt(np.sum(np.abs(np.where(m_k, d_k, 0.0)) ** 2, axis=(1, 2)))    # each s or f counts equally
    E0 = 0.5 * float(np.sum((np.abs(np.where(m_k, r0, 0.0)) / sig_f[:, None, None]) ** 2))
    sig = np.where(m_k, np.broadcast_to((sig_f * math.sqrt(E0))[:, None, None], d_k.shape), np.inf)
    sigma = sig.ravel() if lap else np.concatenate([sig.ravel(), sig.ravel()])
    label = ",".join(f"{f:g}" for f in group) + f" {unit}"
    D = None
    if precond:
        H_P = prob.illumination(m)
        Hs = H_P + prob.illum_damp * H_P[free].max()        # log: 0, the pseudo-Hessian is never 0 on the free nodes
        D = np.divide(1.0, Hs, out=np.zeros_like(Hs), where=Hs > 0)
        D /= D[free].max()
    m_ref, Dh = m.copy(), (np.sqrt(D) if D is not None else None)
    to_m = (lambda xk: m_ref + Dh * xk) if D is not None else (lambda xk: xk)

    def keep(xk, row, k=k, label=label, prob=prob):
        mk = to_m(xk)
        models.append(mk.reshape(nx, nz) * 1000.0)
        row["stage"], row["model_error"], row["model_corr"] = k, model_error(mk), model_corr(mk)
        row["n_solve"] = solves_before + prob.n_solve
        print(f"  group {k} ({label}) iter {row['iter']:2d}: E = {row['E']:.4f}, model error {row['model_error']:.0f} m/s, "
              f"corr {row['model_corr']:+.2f}, solves {row['n_solve']}, {row['wall_s']:.0f} s", flush=True)

    opts = {"maxfun": 2 * niter + 5, "ftol": 1e-12, "gtol": 1e-12}
    if D is None:
        obj = geoinv.Objective(prob, dvec, sigma=sigma)
        m, hist = geoinv.lbfgs(obj, m, max_iter=niter, lower=lower, upper=upper, callback=keep, options=opts)
    else:
        obj = geoinv.Objective(Scaled(prob, m_ref, Dh), dvec, sigma=sigma)
        bound = lambda b: np.divide(b - m_ref, Dh, out=np.zeros_like(m), where=Dh > 0)   # fixed nodes (Dh = 0): x = 0
        x, hist = geoinv.lbfgs(obj, np.zeros_like(m), max_iter=niter, lower=bound(lower), upper=bound(upper),
                               callback=keep, options=opts)
        m = to_m(x)
    if args.srcest:
        prob.forward(m)
        for f, w in zip(group, prob.w_hat):
            w_est[f] = w
    solves_before += prob.n_solve
    prob.close()
    hist.save(out / f"history_stage{k}.json", group=label)
    history.append(hist)
    stage_rows.append({"stage": k, "group": group, "iterations": len(hist.rows) - 1, "stop": hist.stop,
                       "E_end": hist.rows[-1]["E"], "model_error_m_s": model_error(m), "model_corr": model_corr(m)})
wall = time.perf_counter() - wall0
err, corr = model_error(m), model_corr(m)
print(f"final rms model error {err:.0f} m/s (initial {err0:.0f}), correlation {corr:+.2f} (initial {corr0:+.2f}); "
      f"{solves_before} solves, {wall:.0f} s")

# ---- data fit of every damping constant or frequency, initial and final model
ev = make_problem(pars, d_all, mask)
res, pred, misfit = {}, {}, {}
for name, mm in (("initial", v0.ravel() / 1000.0), ("final", m)):
    _, r = residual(ev, mm, d_all, mask)
    if lap and args.objective == "log":
        res[name] = [float(np.sqrt(np.mean(r[i][mask[i]] ** 2))) for i in range(len(pars))]   # rms log residual
    else:
        res[name] = [float(np.linalg.norm(r[i][mask[i]]) / np.linalg.norm(d_all[i][mask[i]])) for i in range(len(pars))]
    # the objective of all damping constants or frequencies together, weighted as in the inversion
    wt = [1.0 if lap and args.objective == "log" else 1.0 / float(np.sum(np.abs(d_all[i][mask[i]]) ** 2))
          for i in range(len(pars))]
    misfit[name] = 0.5 * sum(wt[i] * float(np.sum(np.abs(r[i][mask[i]]) ** 2)) for i in range(len(pars)))
    pred[name] = ev.pred.copy() if lap else ev.to_complex(ev.forward(mm))
misfit["ratio"] = misfit["final"] / misfit["initial"]
print(f"objective of all {'damping constants' if lap else 'frequencies'}: {misfit['initial']:.4g} -> "
      f"{misfit['final']:.4g} (final / initial {misfit['ratio']:.2e})")
ev.close()
mid = len(src) // 2


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
vlim = (float(v_true.min()), float(v_true.max()))
fig, axes, spare = grid(3)
glab = " / ".join(",".join(f"{f:g}" for f in g) for g in stages) + f" {unit}"
for ax, a, ttl in zip(axes, (v_true, v0, m.reshape(nx, nz) * 1000.0),
                      ("true", f"initial ({init})\nrms error {err0:.0f} m/s, correlation {corr0:+.2f}",
                       f"inverted ({args.domain}, {glab})\nrms error {err:.0f} m/s, correlation {corr:+.2f}")):
    im = ax.imshow(a.T, extent=ext, cmap="turbo", vmin=vlim[0], vmax=vlim[1], aspect="equal")
    ax.set_title(ttl, fontsize=10)
axes[0].plot(src[:, 0] * h / 1000, src[:, 1] * h / 1000, "*", color="w", mec="k", ms=5, clip_on=False)
fig.colorbar(im, cax=spare.inset_axes([0.1, 0.5, 0.8, 0.1]), orientation="horizontal", label="v (m/s)")
fig.savefig(out / "fwi_models.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
it0 = 0
for k, hst in enumerate(history):
    E = hst.column("E")
    its = it0 + np.arange(len(E))
    ax[0].semilogy(its, E, "o-", ms=3, color=f"C{k}", label=",".join(f"{f:g}" for f in stages[k]) + f" {unit}")
    ax[1].plot(its, [r["model_corr"] for r in hst.rows], "o-", ms=3, color=f"C{k}")
    it0 = its[-1]
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("E / E(group start)"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("corr(v - v_lin, v_true - v_lin)")
fig.savefig(out / "fwi_history.png", dpi=150)

xr = rec[:, 0] * h / 1000
ok = mask[0, mid]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
axes[0].plot(pars, res["initial"], "o-", label="initial")
axes[0].plot(pars, res["final"], "o-", label="final")
axes[0].set_xlabel(f"{'damping constant s' if lap else 'frequency'} ({unit})")
axes[0].set_ylabel("rms of ln|R u| - ln|d|" if lap and args.objective == "log" else "|R u - d| / |d|")
axes[0].legend(); axes[0].grid(alpha=0.3)
if lap:
    for i, s in enumerate(pars):
        c = f"C{i}"
        axes[1].plot(xr[ok], np.log(np.abs(d_all[i, mid, ok])), "-", color=c, lw=2, alpha=0.5, label=f"s = {s:g}")
        axes[1].plot(xr[ok], np.log(np.abs(pred["initial"][i, mid, ok])), ":", color=c, lw=1)
        axes[1].plot(xr[ok], np.log(np.abs(pred["final"][i, mid, ok])), "--", color=c, lw=1)
    axes[1].set_ylabel("ln|d|"); axes[1].legend(fontsize=7, ncol=2)
    axes[1].set_title("middle shot: observed (thick), initial (dotted), final (dashed)", fontsize=9)
else:
    ip = int(np.argmin(np.abs(np.array(pars) - 4.0)))
    axes[1].plot(xr[ok], np.angle(d_all[ip, mid, ok]), ".", ms=3, label="observed")
    axes[1].plot(xr[ok], np.angle(pred["initial"][ip, mid, ok]), ".", ms=3, label="initial")
    axes[1].plot(xr[ok], np.angle(pred["final"][ip, mid, ok]), ".", ms=3, label="final")
    axes[1].set_ylabel("phase (rad)"); axes[1].legend(fontsize=8)
    axes[1].set_title(f"middle shot, {pars[ip]:g} Hz", fontsize=9)
axes[1].set_xlabel("receiver x (km)")
fig.savefig(out / "fwi_data.png", dpi=150)

if args.srcest:
    fw = np.array(sorted(w_est))
    we = np.array([w_est[f] for f in fw])
    wt = w_par[[pars.index(f) for f in fw]]
    fig, ax = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)
    if lap:
        ax.plot(fw, np.log(np.abs(wt)), "o-", label="true (with the low-cut filter)" if args.lowcut else "true")
        ax.plot(fw, np.log(np.abs(we)), "s--", label="estimated")
        ax.set_xlabel("damping constant s (1/s)"); ax.set_ylabel("ln|w~|")
    else:
        ax.plot(fw, np.abs(wt), "o-", label="true")
        ax.plot(fw, np.abs(we), "s--", label="estimated")
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("|w~|")
    ax.legend()
    fig.savefig(out / "fwi_wavelet.png", dpi=150)

np.savez_compressed(out / "fwi_models.npz", v_true=v_true, v0=v0, v_final=m.reshape(nx, nz) * 1000.0,
                    models=np.array(models, dtype=np.float32), h=h)
summary = {"domain": args.domain, "h_m": h, "nx": nx, "nz": nz, "nshot": len(src), "init": init,
           "groups": stages, "objective": args.objective if lap else "l2", "precond": precond, "niter": niter,
           "wavelet": args.wavelet, "srcest": args.srcest, "lowcut_Hz": args.lowcut, "noise": args.noise,
           "mute": args.mute, "tmax_s": args.tmax, "alpha_d": None if lap else damp, "f0_Hz": args.f0,
           "min_offset_m": args.min_offset, "wrong_sign_left_out": n_wrong,
           "model_error_initial_m_s": err0, "model_error_final_m_s": err,
           "model_corr_initial": corr0, "model_corr_final": corr, "misfit": misfit,
           "residual": {"values": pars, **res}, "stages": stage_rows, "solves": solves_before, "wall_s": round(wall, 1),
           "wavelet_estimate": ({f"{f:g}": (float(w) if lap else [float(w.real), float(w.imag)]) for f, w in w_est.items()}
                                if w_est else None)}
(out / "fwi.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"saved {out}/fwi_*.png, fwi.json, fwi_models.npz")
