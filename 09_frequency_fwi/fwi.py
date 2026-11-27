"""Frequency-domain full waveform inversion: L-BFGS, Gauss-Newton or steepest descent, frequency groups,
assumed or estimated source wavelet.

Run:  python3 fwi.py                                          defaults below ; python3 fwi.py -h lists the options
      python3 fwi.py --solver gn --out results/gn             Gauss-Newton (LSQR with J p and J^T y)
      python3 fwi.py --model marmousi --init linear --out results/A
      python3 fwi.py --model marmousi --init linear --groups 2,3 4,5 6,7,8 --out results/B
      python3 fwi.py --model marmousi --init smooth --out results/C
      python3 fwi.py --model marmousi --init linear --groups 2,3 4,5 6,7,8 --wavelet other --no-srcest --assume ricker --out results/wrong
      python3 fwi.py --model marmousi --init linear --groups 2,3 4,5 6,7,8 --wavelet other --out results/other
      python3 fwi.py --no-srcest --out results/known            the true wavelet, not estimated
Out:  <out>/fwi_models.png, fwi_history.png, fwi_residual.png, fwi_wavelet.png (--srcest), fwi.json, fwi_models.npz,
      history_stage<k>.json (geoinv.History per stage)

Observed data: the true model with the time-domain code (4th-order space, CPML 20 nodes), transformed at
omega - i alpha_d, alpha_d = ln(100) / T. Inversion: the 9-point frequency-domain operator with a PML, at the
same omega - i alpha_d. Receivers closer than --min-offset to the source are left out (the point-source value
next to the source depends on the discretization). The wavelet is estimated at every forward by default: even
the true wavelet does not fit exactly, because the two codes radiate a point source with a few percent
different strength (1 % at 2 Hz, 6 % at 4 Hz on the 48 m grid), and the inversion would otherwise change the
velocity next to the sources to make up for it. The spectra of the observed data are kept in
results/obs/ and reused by later runs with the same model, grid, wavelet and record length.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # one BLAS thread per process: the frequencies run in parallel

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
import freqdomain as fd
import fwiproblem as F

# ---- variables to change ---------------------------------------------------
MODEL = "anomaly"         # "anomaly" (two discs in a background growing with depth), "marmousi"
H = 24.0                  # Marmousi grid spacing (m): 24 or 48 ; the anomaly model is on 48 m
INIT = "background"       # "background" (anomaly), "linear" (in depth), "smooth" (true model smoothed)
FREQS = None              # frequencies (Hz) inverted together; None: 2 3 4 (anomaly), 2 3 4 5 6 7 8 (marmousi)
GROUPS = []               # frequency groups inverted one after the other, e.g. ["2,3", "4,5", "6,7,8"]
NITER = 20                # iterations per group
SOLVER = "lbfgs"          # "lbfgs", "gn" (Gauss-Newton), "sd" (steepest descent)
LSQR_ITER = 10            # Gauss-Newton: LSQR iterations per step
PRECOND = False           # pseudo-Hessian diagonal preconditioning
WAVELET = "ricker"        # true source wavelet of the observed data: "ricker", "other"
SRCEST = True             # estimate the wavelet at every forward from the data and the current model
ASSUME = "true"           # with --no-srcest: the wavelet the inversion uses, "true" (the observed one) or "ricker"
NSHOT = 48
NOISE = 0.0               # complex Gaussian noise, fraction of the rms of each frequency
TMAX = 6.0                # record length (s)
F0 = 6.0                  # Ricker peak frequency (Hz)
MIN_OFFSET = 400.0        # receivers closer than this to the source are not used (m)
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Frequency-domain FWI; options override the variables at the top.")
ap.add_argument("--model", choices=["anomaly", "marmousi"], default=MODEL)
ap.add_argument("--h", type=float, choices=[24.0, 48.0], default=H, help="Marmousi grid spacing (m)")
ap.add_argument("--init", choices=["background", "linear", "smooth"], default=INIT)
ap.add_argument("--freqs", type=float, nargs="+", default=FREQS, help="frequencies (Hz) inverted together")
ap.add_argument("--groups", nargs="*", default=GROUPS, help="frequency groups, low to high, e.g. 2,3 4,5 6,8")
ap.add_argument("--niter", type=int, default=NITER, help="iterations per group")
ap.add_argument("--solver", choices=["lbfgs", "gn", "sd"], default=SOLVER)
ap.add_argument("--lsqr-iter", type=int, default=LSQR_ITER, help="Gauss-Newton: LSQR iterations per step")
ap.add_argument("--precond", action="store_true", default=PRECOND)
ap.add_argument("--wavelet", choices=["ricker", "other"], default=WAVELET, help="true wavelet of the observed data")
ap.add_argument("--srcest", action=argparse.BooleanOptionalAction, default=SRCEST,
                help="estimate the wavelet at every forward (--no-srcest: use the --assume wavelet)")
ap.add_argument("--assume", choices=["true", "ricker"], default=ASSUME, help="with --no-srcest: wavelet used")
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--noise", type=float, default=NOISE)
ap.add_argument("--tmax", type=float, default=TMAX)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--min-offset", type=float, default=MIN_OFFSET)
ap.add_argument("--workers", type=int, default=None, help="processes (default: LAB_CPUS, at most 8)")
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# ---- model, acquisition, frequencies
if args.model == "marmousi":
    v_true, h = F.marmousi(args.h), args.h
else:
    v_true, h = F.anomaly(), 48.0
init = args.init if not (args.model == "anomaly" and args.init == "linear") else "background"
n_fixed = max(F.water_rows(v_true), int(round(F.SRC_DEPTH / h)) + 2)
v0 = F.initial(v_true, h, init, known_rows=n_fixed)
nx, nz = v_true.shape
src, rec = F.acquisition(nx, args.nshot, h)
if args.groups:
    stages = [[float(f) for f in g.split(",")] for g in args.groups]
else:
    stages = [args.freqs or ([2, 3, 4] if args.model == "anomaly" else [2, 3, 4, 5, 6, 7, 8])]
freqs = sorted({f for g in stages for f in g})
dt, nt = F.time_axis(h, args.tmax)
t = np.arange(nt) * dt
damp = math.log(100.0) / args.tmax
w_true = F.wavelet(args.wavelet, t, args.f0)
w_assume = w_true if args.assume == "true" else F.wavelet("ricker", t, args.f0)
print(f"{args.model}: {nx} x {nz} nodes, h = {h:g} m, {args.nshot} shots, {len(rec)} receivers, "
      f"frequencies {' / '.join(','.join(f'{f:g}' for f in g) for g in stages)} Hz, alpha_d = {damp:.2f} 1/s")

# ---- observed data: time-domain traces -> spectra on a 0.25 Hz grid, kept in results/obs/
FGRID = F.FGRID
D, path, sec = F.observed_spectra(args.model, v_true, h, src, rec, args.wavelet, args.f0, args.tmax, args.workers)
print(f"observed data (time domain, 4th order, CPML 20, {nt} steps): {sec:.1f} s, {path}"
      if sec else f"observed data from {path}")
fi = [int(np.argmin(np.abs(FGRID - f))) for f in freqs]
if any(abs(FGRID[i] - f) > 1e-9 for i, f in zip(fi, freqs)):
    sys.exit("frequencies must be multiples of 0.25 Hz between 0.25 and 15 Hz")
d_all = D[fi]                                         # (nf, ns, nr)
if args.noise > 0:
    rng = np.random.default_rng(0)
    rms = np.sqrt(np.mean(np.abs(d_all) ** 2, axis=(1, 2), keepdims=True))
    d_all = d_all + args.noise * rms * (rng.standard_normal(d_all.shape) + 1j * rng.standard_normal(d_all.shape)) / math.sqrt(2)
use = np.abs(src[:, None, 0] - rec[None, :, 0]) * h >= args.min_offset      # (ns, nr)
omega_c = 2 * np.pi * np.array(freqs) - 1j * damp
wsp_assume = fd.source_spectrum(w_assume, dt, omega_c)
wsp_true = fd.source_spectrum(w_true, dt, omega_c)

lower, upper = 1.4, 6.0                              # km/s
free = None


def model_error(m):
    v = m.reshape(nx, nz) * 1000.0
    return float(np.sqrt(np.mean((v - v_true)[free.reshape(nx, nz)] ** 2)))


SHALLOW = 1500.0                                     # depth limit of the shallow metrics (m)


def model_corr(m, zmax=None):
    """Correlation of the update v - v0 with the true perturbation v_true - v0 over the free nodes (down to zmax).
    Over the whole model it is dominated by the deep fast layers that the data hardly reach; the shallow
    value (z <= 1.5 km) shows the difference between runs better."""
    sel = free.reshape(nx, nz).copy()
    if zmax is not None:
        sel[:, np.arange(nz) * h > zmax] = False
    dv = (m.reshape(nx, nz) * 1000.0 - v0)[sel]
    if not np.any(dv):
        return 0.0
    return float(np.corrcoef(dv, (v_true - v0)[sel])[0, 1])


def shallow_error(m):
    sel = free.reshape(nx, nz).copy()
    sel[:, np.arange(nz) * h > SHALLOW] = False
    return float(np.sqrt(np.mean((m.reshape(nx, nz) * 1000.0 - v_true)[sel] ** 2)))


class Scaled(geoinv.Problem):
    """m = m_ref + D x: L-BFGS on x with D = H_P^(-1/2) (diagonal preconditioning by change of variables)."""

    def __init__(self, base, m_ref, D):
        self.base, self.m_ref, self.D = base, m_ref, D
        self.n_model, self.fixed_mask = base.n_model, base.fixed_mask

    def forward(self, x):
        return self.base.forward(self.m_ref + self.D * x)

    def gradient(self, x, r):
        return self.D * self.base.gradient(self.m_ref + self.D * x, r)


m = v0.ravel() / 1000.0
models, stage_rows, history, w_est = [], [], [], {}
solves_before = 0
wall0 = time.perf_counter()
for k, group in enumerate(stages):
    gi = [freqs.index(f) for f in group]
    d_k = d_all[gi]
    prob = F.FreqFWI(nx, nz, h, group, src, rec, wsp_assume[gi], stencil=9, nb=20, damp=damp, fixed_rows=n_fixed,
                     srcest=args.srcest, d_obs=d_k, mask=use, workers=args.workers)
    free = ~prob.fixed_mask
    if k == 0:
        err0 = model_error(m)
        err0_s = shallow_error(m)
        print(f"initial model ({init}): rms model error {err0:.0f} m/s (z <= 1.5 km {err0_s:.0f} m/s), {free.sum()} free nodes")
    # weights: each frequency counts equally (1 / |d_f|), muted near offsets, E = 1 at the start of the group
    r0 = prob.to_complex(prob.forward(m)) - d_k
    sig_f = np.sqrt(np.sum(np.abs(np.where(use, d_k, 0.0)) ** 2, axis=(1, 2)))
    E0 = 0.5 * float(np.sum((np.abs(np.where(use, r0, 0.0)) / sig_f[:, None, None]) ** 2))
    sig = np.broadcast_to((sig_f * math.sqrt(E0))[:, None, None], d_k.shape)
    sig = np.where(use, sig, np.inf)
    sigma = np.concatenate([sig.ravel(), sig.ravel()])
    label = ",".join(f"{f:g}" for f in group) + " Hz"
    D = None
    if args.precond:
        H_P = prob.illumination(m)
        D = 1.0 / (H_P + 0.01 * H_P[free].max())
        D /= D[free].max()
    scaled = args.solver == "lbfgs" and D is not None
    m_ref, Dh = m.copy(), (np.sqrt(D) if scaled else None)
    to_m = (lambda xk: m_ref + Dh * xk) if scaled else (lambda xk: xk)

    def keep(xk, row, k=k, label=label, prob=prob):
        mk = to_m(xk)
        models.append(mk.reshape(nx, nz) * 1000.0)
        row["stage"], row["model_error"], row["model_corr"] = k, model_error(mk), model_corr(mk)
        row["model_corr_shallow"] = model_corr(mk, SHALLOW)
        row["n_solve"] = solves_before + prob.n_solve
        extra = f", LSQR {row['lsqr_iter']}" if "lsqr_iter" in row else ""
        print(f"  group {k} ({label}) iter {row['iter']:2d}: E = {row['E']:.4f}, model error {row['model_error']:.0f} m/s, "
              f"corr {row['model_corr']:+.2f} (z <= 1.5 km {row['model_corr_shallow']:+.2f}), solves {row['n_solve']}{extra}, {row['wall_s']:.0f} s", flush=True)

    d_vec = prob.to_real(np.where(use, d_k, 0.0))
    if args.solver == "sd":
        obj = geoinv.Objective(prob, d_vec, sigma=sigma)
        m, hist = geoinv.steepest_descent(obj, m, n_iter=args.niter, precond=D, lower=lower, upper=upper, callback=keep)
    elif args.solver == "gn":
        obj = geoinv.Objective(prob, d_vec, sigma=sigma)
        m, hist = geoinv.gauss_newton(obj, m, n_iter=args.niter, tol=1e-6, lsqr_iter=args.lsqr_iter,
                                      lower=lower, upper=upper, callback=keep)
    elif not scaled:
        opts = {"maxfun": 2 * args.niter + 5, "ftol": 1e-12, "gtol": 1e-12}
        obj = geoinv.Objective(prob, d_vec, sigma=sigma)
        m, hist = geoinv.lbfgs(obj, m, max_iter=args.niter, lower=lower, upper=upper, callback=keep, options=opts)
    else:
        opts = {"maxfun": 2 * args.niter + 5, "ftol": 1e-12, "gtol": 1e-12}
        obj = geoinv.Objective(Scaled(prob, m_ref, Dh), d_vec, sigma=sigma)
        x, hist = geoinv.lbfgs(obj, np.zeros_like(m), max_iter=args.niter, lower=(lower - m_ref) / Dh,
                               upper=(upper - m_ref) / Dh, callback=keep, options=opts)
        m = to_m(x)
    if args.srcest:
        prob.forward(m)
        for f, w in zip(group, prob.w_hat):
            w_est[f] = w
    solves_before += prob.n_solve
    prob.close()
    hist.save(out / f"history_stage{k}.json", band=label)
    history.append(hist)
    stage_rows.append({"stage": k, "freqs_Hz": group, "iterations": len(hist.rows) - 1, "stop": hist.stop,
                       "E_end": hist.rows[-1]["E"], "model_error_m_s": model_error(m), "model_corr": model_corr(m),
                       "model_corr_shallow": model_corr(m, SHALLOW)})
wall = time.perf_counter() - wall0
err, corr = model_error(m), model_corr(m)
err_s, corr_s = shallow_error(m), model_corr(m, SHALLOW)
print(f"final rms model error {err:.0f} m/s (initial {err0:.0f}), update correlation {corr:+.2f}; "
      f"z <= 1.5 km: rms error {err_s:.0f} m/s (initial {err0_s:.0f}), correlation {corr_s:+.2f}; "
      f"{solves_before} solves, {wall:.0f} s")

# ---- residual of every frequency, initial and final model, with the true wavelet (or the estimate)
ev = F.FreqFWI(nx, nz, h, freqs, src, rec, wsp_assume, stencil=9, nb=20, damp=damp, fixed_rows=n_fixed,
               srcest=args.srcest, d_obs=d_all, mask=use, workers=args.workers)
res = {}
for name, mm in (("initial", v0.ravel() / 1000.0), ("final", m)):
    u = ev.to_complex(ev.forward(mm))
    res[name] = [float(np.linalg.norm((u[i] - d_all[i])[use]) / np.linalg.norm(d_all[i][use])) for i in range(len(freqs))]
    if name == "final":
        u_fin = u
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
glab = " / ".join(",".join(f"{f:g}" for f in g) for g in stages) + " Hz"
for ax, a, ttl in zip(axes, (v_true, v0, m.reshape(nx, nz) * 1000.0),
                      ("true", f"initial ({init}), rms error {err0:.0f} m/s",
                       f"inverted ({glab}, {args.solver})\ncorrelation {corr:+.2f}, z <= 1.5 km {corr_s:+.2f}")):
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
    ax[0].semilogy(its, E, "o-", ms=3, color=f"C{k}", label=",".join(f"{f:g}" for f in stages[k]) + " Hz")
    ax[1].plot([r["n_solve"] for r in hst.rows], [r["model_corr"] for r in hst.rows], "o-", ms=3, color=f"C{k}")
    ax[1].plot([r["n_solve"] for r in hst.rows], [r["model_corr_shallow"] for r in hst.rows], "s--", ms=3, color=f"C{k}")
    it0 = its[-1]
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("E / E(group start)"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("solves (one shot, one frequency)"); ax[1].set_ylabel("corr(v - v0, v_true - v0)")
ax[1].set_title("solid: whole model, dashed: z <= 1.5 km", fontsize=9)
fig.savefig(out / "fwi_history.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
axes[0].plot(freqs, res["initial"], "o-", label="initial")
axes[0].plot(freqs, res["final"], "o-", label="final")
axes[0].set_xlabel("frequency (Hz)"); axes[0].set_ylabel("|R u - d| / |d|"); axes[0].legend(); axes[0].grid(alpha=0.3)
ip = int(np.argmin(np.abs(np.array(freqs) - 4.0)))
xr = rec[:, 0] * h / 1000
dm, um = d_all[ip, mid], u_fin[ip, mid]
ok = use[mid]
axes[1].plot(xr[ok], np.abs(dm[ok]), ".", ms=3, label="observed")
axes[1].plot(xr[ok], np.abs(um[ok]), ".", ms=3, label="final")
axes[1].set_title(f"amplitude, {freqs[ip]:g} Hz, middle shot", fontsize=10); axes[1].set_xlabel("receiver x (km)")
axes[1].legend(fontsize=8)
axes[2].plot(xr[ok], np.angle(dm[ok]), ".", ms=3)
axes[2].plot(xr[ok], np.angle(um[ok]), ".", ms=3)
axes[2].set_title("phase (rad)", fontsize=10); axes[2].set_xlabel("receiver x (km)")
fig.savefig(out / "fwi_residual.png", dpi=150)

if args.srcest:
    fw = np.array(sorted(w_est))
    we = np.array([w_est[f] for f in fw])
    wt = wsp_true[[freqs.index(f) for f in fw]]
    wa = fd.source_spectrum(F.wavelet("ricker", t, args.f0), dt, 2 * np.pi * fw - 1j * damp)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for a, lab, st in ((wt, "true", "o-"), (we, "estimated", "s--"), (wa, "Ricker", "x:")):
        ax[0].plot(fw, np.abs(a), st, label=lab)
        ax[1].plot(fw, np.angle(a), st)
    ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("|w|"); ax[0].legend()
    ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("phase (rad)")
    fig.savefig(out / "fwi_wavelet.png", dpi=150)

np.savez_compressed(out / "fwi_models.npz", v_true=v_true, v0=v0, v_final=m.reshape(nx, nz) * 1000.0,
                    models=np.array(models, dtype=np.float32), h=h)
summary = {"model": args.model, "h_m": h, "nx": nx, "nz": nz, "nshot": len(src), "init": init,
           "groups_Hz": stages, "solver": args.solver, "lsqr_iter": args.lsqr_iter if args.solver == "gn" else None,
           "precond": args.precond, "niter": args.niter, "wavelet": args.wavelet, "assume": args.assume,
           "srcest": args.srcest, "noise": args.noise, "tmax_s": args.tmax, "alpha_d": damp, "f0_Hz": args.f0,
           "min_offset_m": args.min_offset, "model_error_initial_m_s": err0, "model_error_final_m_s": err,
           "model_corr_final": corr, "model_error_shallow_initial_m_s": err0_s, "model_error_shallow_final_m_s": err_s,
           "model_corr_shallow_final": corr_s, "residual_per_freq": {"freqs_Hz": freqs, **res}, "stages": stage_rows,
           "solves": solves_before, "wall_s": round(wall, 1),
           "wavelet_estimate": ({f"{f:g}": [float(w.real), float(w.imag)] for f, w in w_est.items()} if w_est else None)}
(out / "fwi.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"saved {out}/fwi_*.png, fwi.json")
