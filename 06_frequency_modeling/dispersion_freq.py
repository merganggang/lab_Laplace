"""Numerical dispersion of the frequency-domain operators: phase and group velocity versus grid points per wavelength.

A plane wave u = exp(-i (k_x x + k_z z)) in the difference equation S u = 0 gives omega exactly
(there is no time step):

    (omega / v)^2 = -L(k) / M(k)
    L = [a (2 cos X + 2 cos Z - 4) + (1 - a) (4 cos X cos Z - 4) / 2] / h^2     Laplacian symbol
    M = c + d (2 cos X + 2 cos Z) + e 4 cos X cos Z                               mass-term weights
    X = k_x h, Z = k_z h, G = lambda / h = 2 pi / (k h)

5-point: a = 1, c = 1, d = e = 0, the same as the time-domain 2nd-order scheme in the limit C -> 0.
9-point: a, c, d of Jo, Shin & Suh (1996), e = (1 - c - 4 d) / 4. For comparison the time-domain
leapfrog with the 2nd-order stencil, sin^2(omega dt / 2) = C^2 [sin^2(X/2) + sin^2(Z/2)], is drawn too.

Run:  python3 dispersion_freq.py             defaults below ; python3 dispersion_freq.py -h lists the options
      python3 dispersion_freq.py --vmin 1500 --fmax 15
      python3 dispersion_freq.py --fit
Out:  results/dispersion_freq.png, results/dispersion_freq.json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- variables to change ---------------------------------------------------
STENCILS = [5, 9]        # frequency-domain operators to compare
COURANT = 0.3            # C of the time-domain reference curve (2nd-order stencil) ; 0: leave it out
ANGLES = [0.0, 45.0]     # propagation angles from the x axis (degrees)
TOL = 0.01               # allowed relative phase or group velocity error (1 %)
VMIN, FMAX = 1500.0, 15.0   # m/s, Hz: print the largest h for each operator ; 0: skip
FIT = False              # refit a, c, d of the 9-point operator and compare with Jo et al. (1996)
# -----------------------------------------------------------------------------

JO96 = (0.5461, 0.6248, 0.09381)
COEF = {5: (1.0, 1.0, 0.0), 9: JO96}

ap = argparse.ArgumentParser(description="Frequency-domain dispersion curves; options override the variables at the top.")
ap.add_argument("--stencil", type=int, nargs="+", choices=[5, 9], default=STENCILS)
ap.add_argument("--courant", type=float, default=COURANT)
ap.add_argument("--angle", type=float, nargs="+", default=ANGLES)
ap.add_argument("--tol", type=float, default=TOL)
ap.add_argument("--vmin", type=float, default=VMIN)
ap.add_argument("--fmax", type=float, default=FMAX)
ap.add_argument("--fit", action=argparse.BooleanOptionalAction, default=FIT)
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def omega_h_over_v(G, coef, angle_deg):
    """omega h / v of the frequency-domain operator for a plane wave with G grid points per wavelength."""
    a, c, d = coef
    e = (1.0 - c - 4.0 * d) / 4.0
    th = math.radians(angle_deg)
    kh = 2 * np.pi / np.asarray(G, float)
    X, Z = kh * math.cos(th), kh * math.sin(th)
    L = a * (2 * np.cos(X) + 2 * np.cos(Z) - 4) + (1 - a) * (4 * np.cos(X) * np.cos(Z) - 4) / 2
    M = c + d * (2 * np.cos(X) + 2 * np.cos(Z)) + e * 4 * np.cos(X) * np.cos(Z)
    return np.sqrt(np.maximum(-L / M, 0.0))


def velocities(G, coef, angle_deg):
    """Normalized phase and group velocity v_ph / v = omega / (v k), v_gr / v = d omega / d k / v."""
    kh = 2 * np.pi / G
    wh = omega_h_over_v(G, coef, angle_deg)
    return wh / kh, np.gradient(wh, kh)


def velocities_time(G, C, angle_deg):
    """Leapfrog in time, 2nd-order stencil: sin^2(omega dt / 2) = C^2 [sin^2(X/2) + sin^2(Z/2)]."""
    th = math.radians(angle_deg)
    kh = 2 * np.pi / G
    s = C * np.sqrt(np.sin(kh * math.cos(th) / 2) ** 2 + np.sin(kh * math.sin(th) / 2) ** 2)
    wdt = 2 * np.arcsin(np.clip(s, 0, 1))
    return wdt / (C * kh), np.gradient(wdt, kh) / C


def g_needed(vel, tol):
    """Smallest G with |v_ph/v - 1| and |v_gr/v - 1| below tol for every G above it."""
    G = np.linspace(2.0, 60.0, 5801)
    vph, vgr = vel(G)
    bad = (np.abs(vph - 1) > tol) | (np.abs(vgr - 1) > tol)
    return float(G[np.flatnonzero(bad).max() + 1]) if bad.any() else 2.0


summary = {"tol": args.tol, "stencils": {}, "angles_deg": args.angle}
curves = {f"{s}-point": (lambda G, a, s=s: velocities(G, COEF[s], a)) for s in args.stencil}
if args.courant > 0:
    curves[f"time domain, C = {args.courant:g}"] = lambda G, a: velocities_time(G, args.courant, a)
colors = {"5-point": "tab:red", "9-point": "tab:blue"}
styles = ["-", "--", ":", "-."]

print(f"tolerance {args.tol:.1%}; G = lambda / h needed at each angle")
print(f"{'operator':>24}" + "".join(f" {str(a) + ' deg':>9}" for a in args.angle))
G = np.linspace(2.0, 20.0, 1801)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
for name, vel in curves.items():
    need = {a: g_needed(lambda g: vel(g, a), args.tol) for a in args.angle}
    summary["stencils"][name] = {"G_needed": need}
    print(f"{name:>24}" + "".join(f" {need[a]:9.1f}" for a in args.angle))
    for a, ls in zip(args.angle, styles):
        vph, vgr = vel(G, a)
        kw = dict(color=colors.get(name, "0.5"), ls=ls, label=f"{name}, {a:g} deg")
        axes[0].plot(G, vph, **kw)
        axes[1].plot(G, vgr, **kw)
for ax, name in zip(axes, ["phase velocity  v_ph / v", "group velocity  v_gr / v"]):
    ax.axhspan(1 - args.tol, 1 + args.tol, color="0.9")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xlabel("grid points per wavelength  G = lambda / h")
    ax.set_title(name)
    ax.set_xlim(2, 20)
axes[0].set_ylim(0.7, 1.05)
axes[1].set_ylim(0.3, 1.1)
axes[0].legend(fontsize=9)
fig.suptitle("frequency-domain operators (no time step) and the time-domain reference")
fig.savefig(out / "dispersion_freq.png", dpi=150)

if args.vmin > 0 and args.fmax > 0:
    lam = args.vmin / args.fmax
    print(f"\nshortest wavelength  lambda_min = v_min / f_max = {lam:.1f} m")
    rec = {}
    for name in curves:
        Gn = max(summary["stencils"][name]["G_needed"].values())
        rec[name] = {"G_needed": Gn, "h_max_m": lam / Gn}
        print(f"{name:>24}: h <= lambda_min / {Gn:.1f} = {lam / Gn:.2f} m")
    summary["recommended"] = {"vmin": args.vmin, "fmax": args.fmax, "lambda_min_m": lam, **rec}

if args.fit:
    from scipy.optimize import minimize
    Gs = 1.0 / np.linspace(0.01, 0.25, 60)     # G from 4 to 100, as in Jo et al. (1996)
    ths = np.linspace(0.0, 45.0, 10)

    def misfit(p):
        return sum(np.sum((velocities(Gs, p, a)[0] - 1) ** 2) for a in ths)

    p = minimize(misfit, (0.5, 0.6, 0.1), method="Nelder-Mead",
                 options={"xatol": 1e-8, "fatol": 1e-14, "maxiter": 20000}).x
    err = lambda q: max(np.abs(velocities(Gs, q, a)[0] - 1).max() for a in ths)
    print(f"\nfit over G = 4 ~ 100, 0 ~ 45 deg (phase velocity, least squares):")
    print(f"  fit       a = {p[0]:.4f}, c = {p[1]:.4f}, d = {p[2]:.5f}, e = {(1 - p[1] - 4 * p[2]) / 4:.5f},"
          f"  max error {err(p):.2%}")
    print(f"  Jo (1996) a = {JO96[0]:.4f}, c = {JO96[1]:.4f}, d = {JO96[2]:.5f}, "
          f"e = {(1 - JO96[1] - 4 * JO96[2]) / 4:.5f},  max error {err(JO96):.2%}")
    summary["fit"] = {"a": p[0], "c": p[1], "d": p[2], "max_error": err(p), "jo96_max_error": err(JO96)}

(out / "dispersion_freq.json").write_text(json.dumps(summary, indent=2))
print(f"\nsaved {out / 'dispersion_freq.png'}")
