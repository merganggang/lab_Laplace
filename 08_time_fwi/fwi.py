"""Time-domain full waveform inversion: L-BFGS or steepest descent, optional multiscale low-pass stages.

Run:  python3 fwi.py                                          defaults below ; python3 fwi.py -h lists the options
      python3 fwi.py --model marmousi --init linear --out results/A
      python3 fwi.py --model marmousi --init linear --bands 2 4 6 --out results/B
      python3 fwi.py --model marmousi --init smooth --out results/C
      python3 fwi.py --model marmousi --solver sd --precond --out results/sd
Out:  <out>/fwi_models.png, fwi_history.png, fwi_gather.png, fwi.json, fwi_models.npz,
      history_stage<k>.json (geoinv.History per stage)

Observed data: the true model with the 8th-order stencil and a 20-node CPML. Inversion: 4th order, 10-node CPML.
Same grid and the same known source wavelet: an inverse crime only partly avoided (see the lecture).
"""

import argparse
import json
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
import propagator as P

# ---- variables to change ---------------------------------------------------
MODEL = "anomaly"         # "anomaly" (two discs in a background growing with depth), "marmousi"
H = 48.0                  # grid spacing (m): Marmousi 48 or 24
INIT = "background"       # "background" (anomaly), "linear" (in depth), "smooth" (true model smoothed)
BANDS = []                # low-pass corners (Hz) of the multiscale stages, e.g. [2, 4, 6]; [] = full band
NITER = 20                # iterations per stage
SOLVER = "lbfgs"          # "lbfgs", "sd" (steepest descent: Armijo + parabolic step)
PRECOND = False           # pseudo-Hessian diagonal preconditioning
GRAD = 0.5                # anomaly: background gradient (1/s); 0 = homogeneous
NSHOT = 32
NOISE = 0.0               # Gaussian noise, fraction of the rms of the observed data
TMAX = 4.0                # record length (s)
F0 = 3.0                  # Ricker peak frequency (Hz)
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Time-domain FWI; options override the variables at the top.")
ap.add_argument("--model", choices=["anomaly", "marmousi"], default=MODEL)
ap.add_argument("--h", type=float, choices=[24.0, 48.0], default=H, help="Marmousi grid spacing (m)")
ap.add_argument("--init", choices=["background", "linear", "smooth"], default=INIT)
ap.add_argument("--grad", type=float, default=GRAD, help="anomaly: background gradient (1/s), 0 = homogeneous")
ap.add_argument("--bands", type=float, nargs="*", default=BANDS, help="low-pass corners (Hz), low to high")
ap.add_argument("--niter", type=int, default=NITER, help="iterations per stage")
ap.add_argument("--solver", choices=["lbfgs", "sd"], default=SOLVER)
ap.add_argument("--precond", action="store_true", default=PRECOND)
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--noise", type=float, default=NOISE)
ap.add_argument("--tmax", type=float, default=TMAX)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--batch", type=int, default=None, help="shots stepped together (default: fit the GPU memory)")
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
dev = P.device(args.device)

# ---- model, acquisition, wavelet
if args.model == "marmousi":
    v_true, h = F.marmousi(args.h), args.h
else:
    v_true, h = F.anomaly(grad=args.grad), 48.0
init = args.init if not (args.model == "anomaly" and args.init == "linear") else "background"
# fixed rows: the water (Marmousi) or the rows down to one node below the sources (anomaly); known in v0
n_fixed = max(F.water_rows(v_true), 3)
v0 = F.initial(v_true, h, init, known_rows=n_fixed)
nx, nz = v_true.shape
dt, nt = F.time_axis(h, args.tmax)
t = np.arange(nt) * dt
src, rec = F.acquisition(nx, args.nshot)
wavelet = P.ricker(t, args.f0, 0.5)
print(f"{args.model}: {nx} x {nz} nodes, h = {h:g} m, {nt} steps of {dt * 1e3:.2f} ms, {args.nshot} shots, "
      f"{len(rec)} receivers, device {dev}")

# ---- observed data: 8th order, 20-node CPML
t0 = time.perf_counter()
obs = F.TimeFWI(nx, nz, h, dt, nt, src, rec, wavelet, order=8, nb=20, f0=args.f0, batch=args.batch, cache_gib=0, dev=dev)
d_full = obs.forward(v_true.ravel() / 1000.0).reshape(len(src), len(rec), nt)
if args.noise > 0:
    rng = np.random.default_rng(0)
    d_full = d_full + args.noise * np.sqrt(np.mean(d_full ** 2)) * rng.standard_normal(d_full.shape)
print(f"observed data (8th order, CPML 20): {time.perf_counter() - t0:.1f} s")

prob = F.TimeFWI(nx, nz, h, dt, nt, src, rec, wavelet, order=4, nb=10, f0=args.f0, batch=args.batch,
                 fixed_depth=n_fixed - 1, dev=dev)
free = ~prob.fixed_mask
lower, upper = 1.4, 6.0                         # km/s
print(f"forward fields {prob.field_gib():.2f} GiB, kept in memory: {prob.field_gib() <= prob.cache_gib}")


class Scaled(geoinv.Problem):
    """m = m_ref + D x: L-BFGS on x with D = H_P^(-1/2) (diagonal preconditioning by change of variables)."""

    def __init__(self, base, m_ref, D):
        self.base, self.m_ref, self.D = base, m_ref, D
        self.n_model, self.fixed_mask = base.n_model, base.fixed_mask

    def forward(self, x):
        return self.base.forward(self.m_ref + self.D * x)

    def gradient(self, x, r):
        return self.D * self.base.gradient(self.m_ref + self.D * x, r)


def model_error(m):
    v = m.reshape(nx, nz) * 1000.0
    return float(np.sqrt(np.mean((v - v_true)[free.reshape(nx, nz)] ** 2)))


def model_corr(m):
    """Correlation of the update v - v0 with the true perturbation v_true - v0 over the free nodes: 1 is the
    true pattern. The rms error is dominated by the large deep contrasts the data hardly reach; this is not."""
    dv = (m.reshape(nx, nz) * 1000.0 - v0)[free.reshape(nx, nz)]
    if not np.any(dv):
        return 0.0
    return float(np.corrcoef(dv, (v_true - v0)[free.reshape(nx, nz)])[0, 1])


m = v0.ravel() / 1000.0
stages = args.bands or [None]
models, stage_rows, history = [], [], []
err0 = model_error(m)
print(f"initial model ({init}): rms model error {err0:.0f} m/s")
wall0 = time.perf_counter()
for k, fc in enumerate(stages):
    wav_k = wavelet if fc is None else P.lowpass(wavelet, fc, dt)
    d_k = d_full if fc is None else P.lowpass(d_full, fc, dt)
    prob.wavelet = wav_k
    r0 = prob.forward(m) - d_k.ravel()
    sigma = float(np.sqrt(np.sum(r0 ** 2) / 2.0))              # E(m at stage start) = 1
    label = "full band" if fc is None else f"0 - {fc:g} Hz"
    D = None
    if args.precond:
        H_P = prob.illumination(m)
        D = 1.0 / (H_P + 0.01 * H_P.max())
        D /= D[free].max()
    # L-BFGS with preconditioning works on x, m = m_ref + sqrt(D) x; the other cases work on m directly
    scaled = args.solver == "lbfgs" and D is not None
    m_ref, Dh = m.copy(), (np.sqrt(D) if scaled else None)
    to_m = (lambda xk: m_ref + Dh * xk) if scaled else (lambda xk: xk)

    def keep(xk, row, k=k, label=label):
        mk = to_m(xk)
        models.append(mk.reshape(nx, nz) * 1000.0)
        row["stage"], row["model_error"], row["model_corr"] = k, model_error(mk), model_corr(mk)
        print(f"  stage {k} ({label}) iter {row['iter']:2d}: E = {row['E']:.4f}, model error {row['model_error']:.0f} m/s, "
              f"corr {row['model_corr']:+.2f}, "
              f"forward {row['n_forward']}, {row['wall_s']:.0f} s", flush=True)

    opts = {"maxfun": 2 * args.niter + 5, "ftol": 1e-12, "gtol": 1e-12}
    if args.solver == "sd":
        obj = geoinv.Objective(prob, d_k.ravel(), sigma=sigma)
        m, hist = geoinv.steepest_descent(obj, m, n_iter=args.niter, precond=D, lower=lower, upper=upper, callback=keep)
    elif not scaled:
        obj = geoinv.Objective(prob, d_k.ravel(), sigma=sigma)
        m, hist = geoinv.lbfgs(obj, m, max_iter=args.niter, lower=lower, upper=upper, callback=keep, options=opts)
    else:
        obj = geoinv.Objective(Scaled(prob, m_ref, Dh), d_k.ravel(), sigma=sigma)
        x, hist = geoinv.lbfgs(obj, np.zeros_like(m), max_iter=args.niter, lower=(lower - m_ref) / Dh,
                               upper=(upper - m_ref) / Dh, callback=keep, options=opts)
        m = to_m(x)
    hist.save(out / f"history_stage{k}.json", band=label, sigma=sigma)
    history.append(hist)
    stage_rows.append({"stage": k, "band": label, "iterations": len(hist.rows) - 1, "stop": hist.stop,
                       "E_end": hist.rows[-1]["E"], "model_error_m_s": model_error(m), "model_corr": model_corr(m)})
wall = time.perf_counter() - wall0
err = model_error(m)
corr = model_corr(m)
print(f"final rms model error {err:.0f} m/s (initial {err0:.0f}), update correlation {corr:+.2f}, "
      f"{prob.n_prop} shot propagations, {wall:.0f} s")

# ---- residual gather of the middle shot, full band, initial and final model
mid = len(src) // 2
prob.wavelet = wavelet
u_init = prob.forward(v0.ravel() / 1000.0).reshape(len(src), len(rec), nt)[mid]
u_fin = prob.forward(m).reshape(len(src), len(rec), nt)[mid]
d_mid = d_full[mid]
res_rel = {"initial": float(np.linalg.norm(u_init - d_mid) / np.linalg.norm(d_mid)),
           "final": float(np.linalg.norm(u_fin - d_mid) / np.linalg.norm(d_mid))}

x, z = np.arange(nx) * h / 1000, np.arange(nz) * h / 1000
ext = [x[0], x[-1], z[-1], z[0]]
vlim = (float(v_true.min()), float(v_true.max()))
fig, axes = plt.subplots(3, 1, figsize=(8, 1.0 + 3 * (6.4 * nz / nx + 0.45)), constrained_layout=True)
for ax, a, ttl in zip(axes, (v_true, v0, m.reshape(nx, nz) * 1000.0),
                      ("true", f"initial ({init}), rms error {err0:.0f} m/s",
                       f"inverted ({', '.join(s['band'] for s in stage_rows)}), rms error {err:.0f} m/s, "
                       f"update correlation {corr:+.2f}")):
    im = ax.imshow(a.T, extent=ext, cmap="turbo", vmin=vlim[0], vmax=vlim[1], aspect="equal")
    ax.set_title(ttl, fontsize=10)
    ax.set_ylabel("z (km)")
axes[0].plot(src[:, 0] * h / 1000, src[:, 1] * h / 1000, "*", color="w", mec="k", ms=6, clip_on=False)
axes[-1].set_xlabel("x (km)")
fig.colorbar(im, ax=axes, shrink=0.8, label="v (m/s)")
fig.savefig(out / "fwi_models.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
it0 = 0
for k, hst in enumerate(history):
    E = hst.column("E")
    errs = [r.get("model_corr", np.nan) for r in hst.rows]
    its = it0 + np.arange(len(E))
    ax[0].semilogy(its, E, "o-", ms=3, label=stage_rows[k]["band"])
    ax[1].plot(its, errs, "o-", ms=3)
    it0 = its[-1]
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("E / E(stage start)"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("corr(v - v0, v_true - v0)")
fig.savefig(out / "fwi_history.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True, sharey=True)
clip = np.percentile(np.abs(d_mid), 99)
for ax, a, ttl in zip(axes, (d_mid, u_init - d_mid, u_fin - d_mid),
                      ("observed", f"residual, initial ({res_rel['initial']:.2f})",
                       f"residual, final ({res_rel['final']:.2f})")):
    ax.imshow(a.T, extent=[x[0], x[-1], t[-1], 0], cmap="gray_r", vmin=-clip, vmax=clip, aspect="auto")
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("receiver x (km)")
axes[0].set_ylabel("t (s)")
fig.savefig(out / "fwi_gather.png", dpi=150)

np.savez_compressed(out / "fwi_models.npz", v_true=v_true, v0=v0, v_final=m.reshape(nx, nz) * 1000.0,
                    models=np.array(models, dtype=np.float32), h=h)
summary = {"model": args.model, "h_m": h, "nx": nx, "nz": nz, "nt": nt, "dt_s": dt, "nshot": len(src),
           "init": init, "bands": args.bands, "solver": args.solver, "precond": args.precond, "niter": args.niter,
           "noise": args.noise, "anomaly_grad": args.grad if args.model == "anomaly" else None, "f0_Hz": args.f0, "device": str(dev), "model_error_initial_m_s": err0,
           "model_error_final_m_s": err, "model_corr_final": corr, "residual_mid_shot": res_rel, "stages": stage_rows,
           "shot_propagations": prob.n_prop, "wall_s": round(wall, 1)}
(out / "fwi.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"saved {out}/fwi_*.png, fwi.json")
