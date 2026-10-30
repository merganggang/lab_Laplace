"""Numerical decay of the 5-point Laplace-domain operator: decay rate error versus grid points per decay length.

The Laplace-domain field does not oscillate. A plane wave u = exp(-kappa (x cos phi + z sin phi)) in
the difference equation (s/v)^2 u - lap u = 0 gives the numerical decay rate kappa:

    [2 cosh(kappa h cos phi) + 2 cosh(kappa h sin phi) - 4] / h^2 = (s / v)^2

The exact rate is s / v. Along a grid axis kappa h = 2 asinh(s h / (2 v)), so the relative error is
about -(s h / v)^2 / 24 = -1 / (24 G_d^2), G_d = L_d / h grid points per decay length L_d = v / s.
kappa is smaller than s / v: the numerical field decays too slowly, and the error of ln u grows
with the distance, about |error| * s * r / v = |error| * s * t at the traveltime t.

Run:  python3 decay_laplace.py             defaults below ; python3 decay_laplace.py -h lists the options
      python3 decay_laplace.py --vmin 1500 --s 2 4 6 8 10 12
Out:  results/decay_laplace.png, results/decay_laplace.json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

# ---- variables to change ---------------------------------------------------
ANGLES = [0.0, 45.0]           # propagation angles from the x axis (degrees)
TOL = 0.01                     # allowed relative decay rate error (1 %)
VMIN = 1500.0                  # m/s: print the largest h for each damping constant ; 0: skip
SVALS = [2, 4, 6, 8, 10, 12]   # damping constants (1/s)
T_ARRIVAL = 3.0                # traveltime (s) for the accumulated error of ln u
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Laplace-domain decay rate error; options override the variables at the top.")
ap.add_argument("--angle", type=float, nargs="+", default=ANGLES)
ap.add_argument("--tol", type=float, default=TOL)
ap.add_argument("--vmin", type=float, default=VMIN)
ap.add_argument("--s", type=float, nargs="+", default=SVALS, help="damping constants (1/s)")
ap.add_argument("--t", type=float, default=T_ARRIVAL, help="traveltime (s) for the error of ln u")
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def rate_error(Gd, angle_deg):
    """Relative error kappa v / s - 1 of the 5-point operator, G_d = v / (s h) grid points per decay length."""
    c, sn = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    out = []
    for g in np.atleast_1d(Gd):
        a = 1.0 / g                                   # exact kappa h = s h / v
        # 2 cosh(x) - 2 = 4 sinh^2(x / 2), divided by a^2: no cancellation for small a
        f = lambda K: (4 * math.sinh(K * c / 2) ** 2 + 4 * math.sinh(K * sn / 2) ** 2) / (a * a) - 1.0
        out.append(brentq(f, 1e-6 * a, a, xtol=1e-14 * a) / a - 1.0)   # numerical kappa h is below the exact
    return np.array(out)


def gd_needed(angle_deg, tol):
    """Smallest G_d with |error| <= tol."""
    return brentq(lambda g: abs(rate_error(g, angle_deg)[0]) - tol, 0.05, 1e4)


summary = {"tol": args.tol, "angles_deg": args.angle, "Gd_needed": {}}
print(f"tolerance {args.tol:.1%} on the decay rate; G_d = (v / s) / h needed at each angle")
for a in args.angle:
    summary["Gd_needed"][f"{a:g}"] = gd_needed(a, args.tol)
    print(f"  {a:5.1f} deg: G_d >= {summary['Gd_needed'][f'{a:g}']:.2f}")

Gd = np.geomspace(0.5, 20, 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
for a, ls in zip(args.angle, ["-", "--", ":", "-."]):
    e = rate_error(Gd, a)
    axes[0].semilogx(Gd, 100 * e, color="tab:red", ls=ls, label=f"5-point, {a:g} deg")
axes[0].axhspan(-100 * args.tol, 100 * args.tol, color="0.9")
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set(xlabel="grid points per decay length  G_d = (v / s) / h", ylabel="decay rate error (%)",
            ylim=(-12, 1), title="kappa / (s / v) - 1")
axes[0].legend(fontsize=9)

if args.vmin > 0:
    worst = max(summary["Gd_needed"].values())
    rows = []
    print(f"\nv_min = {args.vmin:g} m/s; error of ln u at t = {args.t:g} s is about |error| * s * t")
    print(f"{'s (1/s)':>8} {'L_d (m)':>9} {'h max (m)':>10}   error and error of ln u at h = 12.5, 25, 50, 100 m")
    for s in args.s:
        Ld = args.vmin / s
        errs = {hh: abs(rate_error(Ld / hh, 0.0)[0]) for hh in (12.5, 25.0, 50.0, 100.0)}
        rows.append({"s": s, "L_d_m": Ld, "h_max_m": Ld / worst,
                     "error_at_h": {f"{hh:g}": e for hh, e in errs.items()},
                     "ln_error_at_t": {f"{hh:g}": e * s * args.t for hh, e in errs.items()}})
        print(f"{s:8g} {Ld:9.1f} {Ld / worst:10.1f}   "
              + "  ".join(f"{e:6.2%} ({e * s * args.t:.3f})" for e in errs.values()))
    summary["recommended"] = {"vmin": args.vmin, "t_s": args.t, "rows": rows}
    hs = np.array([r["h_max_m"] for r in rows])
    axes[1].plot(args.s, hs, "o-", color="tab:blue", label=f"Laplace, 5-point, {args.tol:.0%}")
    axes[1].set(xlabel="damping constant s (1/s)", ylabel="largest h (m)", yscale="log",
                title=f"v_min = {args.vmin:g} m/s")
    axes[1].legend(fontsize=9)
fig.savefig(out / "decay_laplace.png", dpi=150)

(out / "decay_laplace.json").write_text(json.dumps(summary, indent=2))
print(f"\nsaved {out / 'decay_laplace.png'}")
