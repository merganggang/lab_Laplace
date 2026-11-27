"""Checks of the frequency-domain derivatives: Jacobian column, dot-product test and gradient check.

Run:  python3 check_adjoint.py                      defaults below ; python3 check_adjoint.py -h lists the options
      python3 check_adjoint.py --stencil 5 --out results/check5
Out:  <out>/check_adjoint.png (gradient check error against the step alpha), check_jacobian.png (one Jacobian
      column, Born modeling against central difference), check_adjoint.json

Small model (part of Marmousi on the 48 m grid, 4 shots, 3 frequencies), double precision.
1. dot-product test  <J p, y> = <p, J^T y> with J p by Born modeling (virtual source solved with the LU of S)
   and J^T y by the adjoint solve (the same LU, residual conj(y) at the receivers). Round-off level: J^T is right.
2. Jacobian column   d(R u~)/dv_k for one shot and one node k: central difference of two forward solves against
   the Born solution for the virtual source -(dS/dv_k) u~ (lecture: 가상 송신원). The adjoint is not involved.
3. gradient check    [E(m + alpha dm) - E(m)] / alpha against g^T dm for a random dm over the free nodes.
4. gradient components  dE/dm_k by central difference at a few nodes against the same components of g. The last
   node is on the model edge: its gradient is exact too (the PML copy is differentiated), but the inversion keeps
   the edge fixed like the time-domain inversion.
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
import geoinv
import freqdomain as fd
import fwiproblem as F

# ---- variables to change ---------------------------------------------------
STENCIL = 9               # 9 (Jo et al. 1996) or 5
NSHOT = 4
FREQS = [2.0, 3.5, 5.0]   # Hz
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Dot-product test and gradient check; options override the variables at the top.")
ap.add_argument("--stencil", type=int, choices=[5, 9], default=STENCIL)
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--freqs", type=float, nargs="+", default=FREQS)
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

h = 48.0
v_true = F.marmousi(h)[48:144, :40]                 # 96 x 40 nodes, 4.6 km x 1.9 km, water on top
n_fixed = F.water_rows(v_true)
v0 = F.initial(v_true, h, "smooth", known_rows=n_fixed)
nx, nz = v_true.shape
src, rec = F.acquisition(nx, args.nshot, h)
freqs = np.array(args.freqs)
tmax = 4.0
damp = math.log(100.0) / tmax
dt, nt = F.time_axis(h, tmax)
wspec = fd.source_spectrum(F.wavelet("ricker", np.arange(nt) * dt), dt, 2 * np.pi * freqs - 1j * damp)
prob = F.FreqFWI(nx, nz, h, freqs, src, rec, wspec, stencil=args.stencil, damp=damp, fixed_rows=n_fixed)
print(f"{nx} x {nz} nodes, {args.nshot} shots, frequencies {', '.join(f'{f:g}' for f in freqs)} Hz, "
      f"{args.stencil}-point operator")

# ---- 1. dot-product test of J (Born) and J^T (adjoint)
m0 = v0.ravel() / 1000.0
J = prob.jacobian(m0)
dot = []
for seed in range(3):
    rel = geoinv.dot_product_test(J, seed=seed)
    dot.append(rel)
    print(f"dot-product test {seed}: |<J p, y> - <p, J^T y>| / |<J p, y>| = {rel:.1e}")

# ---- 2. one Jacobian column: Born modeling against central difference
ks = (nx // 2, 15)                                   # node k: middle of the model, 0.72 km deep
k = ks[0] * nz + ks[1]
shot = len(src) // 2
e = np.zeros(nx * nz)
e[k] = 1.0
J_ana = prob.to_complex(J.matvec(e))[:, shot]       # per km/s
dv = 1e-3                                            # km/s (1 m/s)
J_num = (prob.solve(m0 + dv * e) - prob.solve(m0 - dv * e))[:, shot] / (2 * dv)
prob.solve(m0)
jac_err = float(np.linalg.norm(J_num - J_ana) / np.linalg.norm(J_ana))
print(f"Jacobian column (shot {shot}, node {ks}, central difference dv = 1 m/s): relative difference {jac_err:.1e}")

# ---- 3. gradient check against the objective
d = prob.solve(v_true.ravel() / 1000.0)
obj = geoinv.Objective(prob, prob.to_real(d))
alphas = [10.0 ** p for p in range(-1, -9, -1)]      # km/s
rows = [{"alpha" if key == "h" else key: x for key, x in r.items()} for r in geoinv.gradient_check(obj, m0, alphas, seed=1)]
for r in rows:
    print(f"alpha = {r['alpha']:.0e} km/s: finite difference {r['fd']:+.6e}, g^T dm {r['gtv']:+.6e}, relative error {r['rel_err']:.1e}")

# ---- 4. single gradient components by central difference (the last node is on the model edge)
E0, g = obj.value_and_gradient(m0)
nodes = [(nx // 4, 6), (nx // 2, 15), (3 * nx // 4, 25), (nx // 2, 35), (nx // 2, nz - 1)]
comps = []
a = 1e-4
for ij in nodes:
    kk = ij[0] * nz + ij[1]
    ee = np.zeros(nx * nz)
    ee[kk] = 1.0
    fdv = (obj.value(m0 + a * ee) - obj.value(m0 - a * ee)) / (2 * a)
    comps.append({"i": ij[0], "j": ij[1], "z_km": ij[1] * h / 1000, "fd": float(fdv), "adjoint": float(g[kk]),
                  "rel_err": float(abs(fdv - g[kk]) / abs(fdv)), "edge": ij[1] == nz - 1})
    print(f"dE/dm at node {ij} (z = {ij[1] * h / 1000:.2f} km): finite difference {fdv:+.6e}, adjoint {g[kk]:+.6e}, "
          f"relative error {comps[-1]['rel_err']:.1e}{'  (model edge, fixed in the inversion)' if comps[-1]['edge'] else ''}")
n_solve = prob.n_solve
prob.close()

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.loglog([r["alpha"] for r in rows], [r["rel_err"] for r in rows], "o-", label=f"{args.stencil}-point")
ax.set_xlabel(r"$\alpha$ (km/s)")
ax.set_ylabel("|FD - g^T dm| / |g^T dm|")
ax.set_title(f"gradient check: dot-product {max(dot):.0e}", fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.savefig(out / "check_adjoint.png", dpi=150)

xr = rec[:, 0] * h / 1000
fig, axes = plt.subplots(len(freqs), 1, figsize=(7, 1.0 + 1.8 * len(freqs)), sharex=True, constrained_layout=True)
for ax, f, ja, jn in zip(np.atleast_1d(axes), freqs, J_ana, J_num):
    ax.plot(xr, ja.real, "-", lw=1.5, label="Born: virtual source solved with the LU")
    ax.plot(xr, jn.real, "--", lw=1.5, label="central difference, dv = 1 m/s")
    ax.plot(xr, 100 * (jn - ja).real, "-", lw=0.8, color="gray", label="difference x 100")
    ax.set_ylabel(f"Re, {f:g} Hz")
    ax.grid(alpha=0.3)
np.atleast_1d(axes)[0].set_title(f"Jacobian column, node x = {ks[0] * h / 1000:.2f} km, z = {ks[1] * h / 1000:.2f} km: "
                                 f"relative difference {jac_err:.0e}", fontsize=10)
np.atleast_1d(axes)[0].legend(fontsize=7)
np.atleast_1d(axes)[-1].set_xlabel("receiver x (km)")
fig.savefig(out / "check_jacobian.png", dpi=150)

summary = {"stencil": args.stencil, "nx": nx, "nz": nz, "nshot": args.nshot, "freqs_Hz": list(freqs),
           "dot_product": dot, "gradient_check": rows,
           "jacobian_column": {"shot": int(shot), "node": list(ks), "rel_diff": jac_err},
           "gradient_components": comps, "solves": n_solve}
(out / "check_adjoint.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"saved {out}/check_adjoint.png, check_jacobian.png, check_adjoint.json")
