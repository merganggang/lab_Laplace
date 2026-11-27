"""Checks of the Laplace-domain derivatives: dot-product test, gradient check and gradient components.

Run:  python3 check_adjoint.py                      defaults below ; python3 check_adjoint.py -h lists the options
      python3 check_adjoint.py --objective l2 --out results/check_l2
Out:  <out>/check_adjoint.png (gradient check error against the step alpha), check_adjoint.json

Small model (part of the salt model on the 40 m grid, 4 shots, 3 damping constants), double precision.
1. dot-product test  <J p, y> = <p, J^T y> with J p by Born modeling (virtual source solved with the LU of S)
   and J^T y by the adjoint solve (the same LU). With the log objective J is d ln|R u~| / dm, so both sides
   divide by the predicted value R u~: the test checks the 1 / R u~ of the adjoint source too.
2. gradient check    [E(m + alpha dm) - E(m)] / alpha against g^T dm for a random dm over the free nodes.
3. gradient components  dE/dm_k by central difference at a few nodes against the same components of g. The last
   node is on the model edge (fixed in the inversion, its gradient is exact too).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
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
import fwiproblem as F

# ---- variables to change ---------------------------------------------------
OBJECTIVE = "log"         # "log" (ln|R u~| against ln|d~|) or "l2"
NSHOT = 4
SVALS = [2.0, 6.0, 12.0]  # damping constants (1/s)
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Dot-product test and gradient check; options override the variables at the top.")
ap.add_argument("--objective", choices=["log", "l2"], default=OBJECTIVE)
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--s", type=float, nargs="+", default=SVALS, help="damping constants (1/s)")
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

h = 40.0
v_true = F.salt(h)[170:266, :40]                    # 96 x 40 nodes, 3.8 km x 1.6 km, water and salt top
n_fixed = F.water_rows(v_true)
v0 = F.initial(v_true, h, "smooth", known_rows=n_fixed)
nx, nz = v_true.shape
src, rec = F.acquisition(nx, args.nshot, h)
svals = np.array(args.s)
wlap = -np.ones(len(svals))                          # a negative source value, like the Ricker transform
prob = F.LaplaceFWI(nx, nz, h, svals, src, rec, wlap, fixed_rows=n_fixed, objective=args.objective)
print(f"{nx} x {nz} nodes, {args.nshot} shots, damping constants {', '.join(f'{s:g}' for s in svals)} 1/s, "
      f"{args.objective} objective")

# ---- 1. dot-product test of J (Born) and J^T (adjoint)
m0 = v0.ravel() / 1000.0
J = prob.jacobian(m0)
dot = []
for seed in range(3):
    rel = geoinv.dot_product_test(J, seed=seed)
    dot.append(rel)
    print(f"dot-product test {seed}: |<J p, y> - <p, J^T y>| / |<J p, y>| = {rel:.1e}")

# ---- 2. gradient check against the objective
d = prob.solve(v_true.ravel() / 1000.0)
obj = geoinv.Objective(prob, prob.data_vector(d))
alphas = [10.0 ** p for p in range(-1, -9, -1)]      # km/s
rows = [{"alpha" if key == "h" else key: x for key, x in r.items()} for r in geoinv.gradient_check(obj, m0, alphas, seed=1)]
for r in rows:
    print(f"alpha = {r['alpha']:.0e} km/s: finite difference {r['fd']:+.6e}, g^T dm {r['gtv']:+.6e}, relative error {r['rel_err']:.1e}")

# ---- 3. single gradient components by central difference (the last node is on the model edge)
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
ax.loglog([r["alpha"] for r in rows], [r["rel_err"] for r in rows], "o-", label=f"{args.objective} objective")
ax.set_xlabel(r"$\alpha$ (km/s)")
ax.set_ylabel("|FD - g^T dm| / |g^T dm|")
ax.set_title(f"gradient check: dot-product {max(dot):.0e}", fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.savefig(out / "check_adjoint.png", dpi=150)

summary = {"objective": args.objective, "nx": nx, "nz": nz, "nshot": args.nshot, "s": list(svals),
           "dot_product": dot, "gradient_check": rows, "gradient_components": comps, "solves": n_solve}
(out / "check_adjoint.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"saved {out}/check_adjoint.png, check_adjoint.json")
