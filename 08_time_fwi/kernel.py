"""Sensitivity kernel of one source-receiver pair, with forward and adjoint wavefield snapshots.

Run:  python3 kernel.py                              defaults below ; python3 kernel.py -h lists the options
      python3 kernel.py --window reflection --out results/kernel_refl
      python3 kernel.py --f0 6 --out results/kernel_f6
Out:  <out>/kernel.png, kernel_snapshots.png, kernel_trace.png, kernel.json

The kernel is the gradient dE/dv for E = 1/2 integral (W(t) u(x_r, t))^2 dt: the energy of one arrival, picked by
the time window W, at one receiver. Its adjoint source is the windowed trace itself, injected backward in time.
direct    : source and receiver at 1.2 km depth in a homogeneous 2000 m/s medium (the banana of the first
            Fresnel zone, away from the free surface)
reflection: source and receiver below the free surface, an interface at 1.2 km (2000 above, 2600 m/s below);
            the window keeps the reflection (sensitivity along the isochron ellipse and the two ray paths)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import fwiproblem as F
import propagator as P

# ---- variables to change ---------------------------------------------------
WINDOW = "direct"         # "direct", "reflection"
F0 = 3.0                  # Ricker peak frequency (Hz)
OFFSET = 3000.0           # source-receiver distance (m)
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="One source-receiver sensitivity kernel; options override the variables at the top.")
ap.add_argument("--window", choices=["direct", "reflection"], default=WINDOW)
ap.add_argument("--f0", type=float, default=F0)
ap.add_argument("--offset", type=float, default=OFFSET)
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
dev = P.device(args.device)

h = 24.0
nx, nz = 256, 100                                  # 6.1 km x 2.4 km
v = np.full((nx, nz), 2000.0)
zr = 1200.0                                        # interface depth (reflection)
if args.window == "reflection":
    v[:, int(round(zr / h)):] = 2600.0
    depth = 2
else:
    depth = int(round(1200.0 / h))
i_s = int(round((nx * h / 2 - args.offset / 2) / h))
i_r = int(round((nx * h / 2 + args.offset / 2) / h))
src = np.array([[i_s, depth]])
rec = np.array([[i_r, depth]])
x_s, x_r, z_sr = i_s * h, i_r * h, depth * h
dt, nt = F.time_axis(h, 4.0)
t = np.arange(nt) * dt
t0 = 1.2 / args.f0
wavelet = P.ricker(t, args.f0, t0)

# arrival time and window (one period either side of the peak)
if args.window == "direct":
    t_arr = args.offset / 2000.0
else:
    t_arr = 2.0 * np.hypot(args.offset / 2, zr - z_sr) / 2000.0
t_pk = t_arr + t0
half = 1.0 / args.f0
win = np.clip(1.0 - np.abs(t - t_pk) / half, 0.0, None) ** 0.5 * (np.abs(t - t_pk) < half)

snap_n = [int(round(f * nt)) for f in (0.15, 0.3, 0.45)]
fwd_snaps = {}


def keep_fwd(n, u):
    if n in snap_n:
        fwd_snaps[n] = P.Layout(nx, nz, 10).inner(u)[0].double().cpu().numpy()


res = P.propagate(v, h, dt, nt, src[None], wavelet[None, None], rec, f0=args.f0, vmax=F.VMAX, store=True,
                  on_step=keep_fwd, dev=dev)
trace = res["traces"][0, 0].double().cpu().numpy()
r = (win * win * trace)[None, None]                # dE/du for E = 1/2 int (W u)^2 dt
adj_snaps = {}


def keep_adj(k, u):
    n = nt - 1 - k
    if n in snap_n:
        adj_snaps[n] = P.Layout(nx, nz, 10).inner(u)[0].double().cpu().numpy()


# the gradient run with snapshots of the adjoint field (same as propagator.gradient, plus the callback)
lay = P.Layout(nx, nz, 10)
field = res["field"]
acc = torch.zeros((nx, nz), dtype=field.dtype, device=field.device)


def xcorr(k, ua):
    keep_adj(k, ua)
    n = nt - 1 - k
    if n <= nt - 2:
        d2 = field[n + 1] - 2.0 * field[n] + (field[n - 1] if n > 0 else 0.0)
        acc.add_(lay.inner(ua)[0] * d2[0])


P.propagate(v, h, dt, nt, rec[None], h ** 2 * r[:, :, ::-1], rec, f0=args.f0, vmax=F.VMAX, on_step=xcorr, dev=dev)
vt = torch.as_tensor(v, dtype=acc.dtype, device=acc.device)
K = (2.0 / (vt ** 3 * dt ** 2) * acc).double().cpu().numpy()

x, z = np.arange(nx) * h / 1000, np.arange(nz) * h / 1000
ext = [x[0], x[-1], z[-1], z[0]]
lam = 2000.0 / args.f0
fresnel = np.sqrt((args.offset + lam / 2) ** 2 - args.offset ** 2) / 2   # first Fresnel zone half width at mid-offset (m)
clip = np.percentile(np.abs(K), 99.5)
fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
im = ax.imshow(K.T, extent=ext, cmap="RdBu_r", vmin=-clip, vmax=clip, aspect="equal")
ax.plot(x_s / 1000, z_sr / 1000, "*", color="gold", mec="k", ms=12)
ax.plot(x_r / 1000, z_sr / 1000, "v", color="w", mec="k", ms=9)
if args.window == "reflection":
    ax.axhline(zr / 1000, color="k", lw=0.8, ls="--")
    # isochron: points with |x - x_s| + |x - x_r| = v t_arr (ellipse with the source and receiver as foci)
    a = 2000.0 * t_arr / 2
    b = np.sqrt(max(a ** 2 - (args.offset / 2) ** 2, 0.0))
    th = np.linspace(0, np.pi, 200)
    ax.plot((nx * h / 2 + a * np.cos(th)) / 1000, (z_sr + b * np.sin(th)) / 1000, color="k", lw=0.8, ls=":")
else:
    # first Fresnel zone: path through the point longer than the offset by half a wavelength (ellipse, foci s and r)
    a = (args.offset + lam / 2) / 2
    b = np.sqrt(a ** 2 - (args.offset / 2) ** 2)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot((nx * h / 2 + a * np.cos(th)) / 1000, (z_sr + b * np.sin(th)) / 1000, color="k", lw=0.8, ls=":")
ax.set_xlabel("x (km)"); ax.set_ylabel("z (km)")
ax.set_title(f"sensitivity kernel, {args.window}, f0 = {args.f0:g} Hz", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.8, label="dE/dv")
fig.savefig(out / "kernel.png", dpi=150)

fig, axes = plt.subplots(2, 3, figsize=(12, 4.6), constrained_layout=True)
for c, n in enumerate(snap_n):
    for rr, (snaps, name) in enumerate(((fwd_snaps, "forward u"), (adj_snaps, "adjoint u_adj"))):
        a = snaps[n]
        cl = np.abs(a).max() or 1.0
        axes[rr, c].imshow(a.T, extent=ext, cmap="RdBu_r", vmin=-cl, vmax=cl, aspect="equal")
        axes[rr, c].set_title(f"{name}, t = {n * dt:.2f} s", fontsize=9)
        axes[rr, c].plot([x_s / 1000, x_r / 1000], [z_sr / 1000] * 2, "k.", ms=4)
fig.savefig(out / "kernel_snapshots.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 2.8), constrained_layout=True)
ax.plot(t, trace / np.abs(trace).max(), "k", lw=1, label="trace u(x_r, t)")
ax.plot(t, win, color="tab:blue", lw=1, label="window W(t)")
ax.set_xlabel("t (s)"); ax.legend(fontsize=8)
fig.savefig(out / "kernel_trace.png", dpi=150)

summary = {"window": args.window, "f0_Hz": args.f0, "offset_m": args.offset, "arrival_s": t_arr,
           "wavelength_m": lam, "fresnel_half_width_m": fresnel, "device": str(dev)}
(out / "kernel.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"{args.window}: arrival {t_arr:.2f} s (+ wavelet delay {t0:.2f} s), wavelength {lam:.0f} m, "
      f"first Fresnel half width at mid-offset {fresnel:.0f} m")
print(f"saved {out}/kernel*.png, kernel.json")
