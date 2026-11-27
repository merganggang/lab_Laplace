"""Source wavelet estimation: one complex number per frequency from the data and the current model.

Run:  python3 wavelet.py                                   defaults below ; python3 wavelet.py -h lists the options
      python3 wavelet.py --wavelet ricker --out results/wav_ricker
Out:  <out>/wavelet.png (amplitude and phase spectra, time wavelets), wavelet.json

For each frequency the modeled data are w~ g~ (g~: response to a unit source), so the best w~ is the
one-unknown least-squares solution w~ = sum conj(g~) d~ / sum |g~|^2 over shots and receivers. It is estimated
twice: in the true model (it recovers the true wavelet up to the modeling difference) and in the starting model
(the model error goes into the wavelet, mostly as a phase that grows with frequency, i.e. a time shift).
The time wavelets come back from the spectra at 0.25 .. FMAX Hz with the inverse transform of the
frequency-domain material (complex frequency undone with exp(alpha_d t)).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))     # geoinv when PYTHONPATH is not set (the container sets it)
import freqdomain as fd
import fwiproblem as F

# ---- variables to change ---------------------------------------------------
WAVELET = "other"         # true wavelet of the observed data: "other", "ricker"
INIT = "linear"           # starting model of the second estimate: "linear", "smooth"
H = 24.0
FMAX = 12.0               # highest estimated frequency (Hz)
NSHOT = 48
TMAX = 6.0
F0 = 6.0
MIN_OFFSET = 400.0
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Wavelet estimation; options override the variables at the top.")
ap.add_argument("--wavelet", choices=["other", "ricker"], default=WAVELET)
ap.add_argument("--init", choices=["linear", "smooth"], default=INIT)
ap.add_argument("--h", type=float, choices=[24.0, 48.0], default=H)
ap.add_argument("--fmax", type=float, default=FMAX)
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--tmax", type=float, default=TMAX)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--min-offset", type=float, default=MIN_OFFSET)
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

v_true, h = F.marmousi(args.h), args.h
n_fixed = F.water_rows(v_true)
v0 = F.initial(v_true, h, args.init, known_rows=n_fixed)
nx, nz = v_true.shape
src, rec = F.acquisition(nx, args.nshot, h)
dt, nt = F.time_axis(h, args.tmax)
t = np.arange(nt) * dt
damp = math.log(100.0) / args.tmax
D, path, sec = F.observed_spectra("marmousi", v_true, h, src, rec, args.wavelet, args.f0, args.tmax, args.workers)
sel = F.FGRID <= args.fmax + 1e-9
freqs = F.FGRID[sel]
d = D[sel]
use = np.abs(src[:, None, 0] - rec[None, :, 0]) * h >= args.min_offset
omega_c = 2 * np.pi * freqs - 1j * damp
w_true = fd.source_spectrum(F.wavelet(args.wavelet, t, args.f0), dt, omega_c)
w_ricker = fd.source_spectrum(F.wavelet("ricker", t, args.f0), dt, omega_c)

prob = F.FreqFWI(nx, nz, h, freqs, src, rec, np.ones(len(freqs)), damp=damp, fixed_rows=n_fixed, mask=use,
                 workers=args.workers)
est, fit = {}, {}
for name, v in (("true model", v_true), (f"initial ({args.init})", v0)):
    G = prob.solve(v.ravel() / 1000.0)
    w = prob.estimate_wavelet(G, d)
    est[name] = w
    fit[name] = [float(np.linalg.norm((w[k] * G[k] - d[k])[use]) / np.linalg.norm(d[k][use])) for k in range(len(freqs))]
    print(f"{name}: data misfit after the estimate {np.mean(fit[name]):.3f} (mean over frequencies), "
          f"|w_est - w_true| / |w_true| = {np.linalg.norm(w - w_true) / np.linalg.norm(w_true):.3f}")
prob.close()

# time wavelets: the spectra on 0 .. Nyquist at df = 0.25 Hz (record 4 s), zero above FMAX
nt4 = int(round(4.0 / dt))
df = 1.0 / (nt4 * dt)


def to_t(w):
    U = np.zeros(nt4 // 2 + 1, complex)
    U[np.round(freqs / df).astype(int)] = w
    return fd.to_time(U, nt4, dt, damp)


tt = np.arange(nt4) * dt
fig = plt.figure(figsize=(11, 6.2), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax_a, ax_p, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
curves = [("true", w_true, "k-"), ("Ricker (assumed)", w_ricker, ":")] + \
         [(f"estimated, {n}", w, s) for (n, w), s in zip(est.items(), ("o", "s"))]
for lab, w, st in curves:
    ms = 3 if st in ("o", "s") else None
    ax_a.plot(freqs, np.abs(w), st, ms=ms, label=lab)
    ax_p.plot(freqs, np.unwrap(np.angle(w)), st, ms=ms)
    ax_t.plot(tt, to_t(w), st if st not in ("o", "s") else "-", lw=1.2, label=lab)
ax_a.set_xlabel("frequency (Hz)"); ax_a.set_ylabel("|w~|"); ax_a.legend(fontsize=8)
ax_p.set_xlabel("frequency (Hz)"); ax_p.set_ylabel("phase, unwrapped (rad)")
ax_t.set_xlim(0, 1.0); ax_t.set_xlabel("t (s)"); ax_t.set_ylabel("w(t)"); ax_t.grid(alpha=0.3)
ax_t.set_title(f"time wavelets from 0.25 - {args.fmax:g} Hz", fontsize=10)
fig.savefig(out / "wavelet.png", dpi=150)

summary = {"wavelet": args.wavelet, "init": args.init, "h_m": h, "freqs_Hz": list(map(float, freqs)),
           "w_true": [[float(c.real), float(c.imag)] for c in w_true],
           "estimates": {n: [[float(c.real), float(c.imag)] for c in w] for n, w in est.items()},
           "misfit_after_estimate": fit}
(out / "wavelet.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"saved {out}/wavelet.png, wavelet.json")
