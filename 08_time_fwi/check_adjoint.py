"""Checks of the derivatives: Jacobian column, dot-product test of the propagator and gradient check against E.

Run:  python3 check_adjoint.py                      defaults below ; python3 check_adjoint.py -h lists the options
      python3 check_adjoint.py --dtype float32 --out results/check32
Out:  <out>/check_adjoint.png (gradient check error against the step alpha), check_jacobian.png (one Jacobian column,
      numerical against analytic), check_adjoint.json

Small model (part of Marmousi on the 48 m grid, 4 shots) so that float64 runs quickly on the CPU or the GPU.
1. dot-product test  <F q, y> = <q, F^T y>: F maps a source time function to the receiver traces, F^T is the
   same time stepping run backward with y injected at the receivers. Round-off level: F^T is right.
2. Jacobian column   du/dv_k for one shot and one node k: central difference of two forward runs against the
   partial derivative wavefield, the same time stepping driven by the virtual source (2 / v_k) (u^{n+1} - 2 u^n
   + u^{n-1}) at node k (lecture: "속도로 미분"). This checks the linearisation; the adjoint is not involved.
3. gradient check    [E(m + alpha dm) - E(m)] / alpha against g^T dm for a random dm over the free nodes,
   and once more with dm only on the model edge (fixed in the inversion: the PML copies the edge velocity).
4. gradient components  dE/dm_k by central difference at a few nodes against the same components of g.
   One node costs two misfit evaluations; all M = nx * nz nodes would cost 2 M (the adjoint costs two runs).
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
sys.path.append(str(HERE.parent))     # geoinv when PYTHONPATH is not set (the container sets it)
import geoinv
import fwiproblem as F
import propagator as P

# ---- variables to change ---------------------------------------------------
DTYPE = "float64"         # "float64", "float32" (the inversion runs in float32)
NSHOT = 4
TMAX = 3.0                # record length (s)
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="Dot-product test and gradient check; options override the variables at the top.")
ap.add_argument("--dtype", choices=["float64", "float32"], default=DTYPE)
ap.add_argument("--nshot", type=int, default=NSHOT)
ap.add_argument("--tmax", type=float, default=TMAX)
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
ap.add_argument("--out", default="results", help="output folder")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
dev = P.device(args.device)
dtype = getattr(torch, args.dtype)

v_true = F.marmousi(48.0)[48:144, :40]             # 96 x 40 nodes, 4.6 km x 1.9 km
h = 48.0
v0 = F.initial(v_true, h, "smooth")
nx, nz = v_true.shape
dt, nt = F.time_axis(h, args.tmax)
t = np.arange(nt) * dt
src, rec = F.acquisition(nx, args.nshot)
wavelet = P.ricker(t, 3.0, 0.5)
print(f"{nx} x {nz} nodes, {nt} steps, {args.nshot} shots, {args.dtype}, device {dev}")

# ---- 1. dot-product test of the propagator
rng = np.random.default_rng(0)
dot = []
for trial in range(3):
    q = rng.standard_normal(nt)
    y = rng.standard_normal((len(rec), nt))
    s = src[trial % len(src)]
    Fq = P.propagate(v0, h, dt, nt, s[None, None], q[None, None], rec, vmax=F.VMAX, dtype=dtype, dev=dev)
    Fq = Fq["traces"][0].double().cpu().numpy()
    lay = P.Layout(nx, nz, 10)
    k = int(lay.flat(s))
    got = np.zeros(nt)

    def grab(n, u):
        got[n] = float(u.reshape(-1)[k])
    P.propagate(v0, h, dt, nt, np.broadcast_to(rec, (1,) + rec.shape), h ** 2 * y[None, :, ::-1], rec,
                vmax=F.VMAX, on_step=grab, dtype=dtype, dev=dev)
    Fty = got[::-1] / h ** 2
    Fty[-1] = 0.0                                   # the source at the last sample reaches no record
    lhs, rhs = float(np.sum(Fq * y)), float(np.sum(q * Fty))
    dot.append({"lhs": lhs, "rhs": rhs, "rel": abs(lhs - rhs) / abs(lhs)})
    print(f"dot-product test {trial}: <F q, y> = {lhs:+.6e}, <q, F^T y> = {rhs:+.6e}, relative {dot[-1]['rel']:.1e}")

# ---- 2. one Jacobian column: numerical against analytic (virtual source)
ks = (nx // 2, 15)                                   # node k: middle of the model, 0.72 km deep
shot = len(src) // 2
dv = 1.0 if args.dtype == "float64" else 20.0        # m/s
s1, w1 = src[shot][None, None], wavelet[None, None]


def traces(v, store=False):
    res = P.propagate(v, h, dt, nt, s1, w1, rec, vmax=F.VMAX, store=store, dtype=dtype, dev=dev)
    return res["traces"][0].double().cpu().numpy(), res["field"]


vp, vm = v0.copy(), v0.copy()
vp[ks] += dv
vm[ks] -= dv
J_num = (traces(vp)[0] - traces(vm)[0]) / (2 * dv)
_, field = traces(v0, store=True)
uk = field[:, 0, ks[0], ks[1]].double().cpu().numpy()
d2 = np.diff(np.concatenate(([0.0], uk, [0.0])), 2)         # u^{n+1} - 2 u^n + u^{n-1}, u^{-1} = 0
vk = float(v0[ks])
# propagate injects v_k^2 dt^2 / h^2 * wav at step n into u^{n+1}; the virtual source term is (2 / v_k) d2[n]
wav_k = (h ** 2 / (vk * dt) ** 2) * (2.0 / vk) * d2
J_ana = P.propagate(v0, h, dt, nt, np.array(ks)[None, None], wav_k[None, None], rec, vmax=F.VMAX,
                    dtype=dtype, dev=dev)["traces"][0].double().cpu().numpy()
jac_err = float(np.linalg.norm(J_num - J_ana) / np.linalg.norm(J_ana))
print(f"Jacobian column (shot {shot}, node {ks}, central difference dv = {dv:g} m/s): relative difference {jac_err:.1e}")

# ---- 3. gradient check
obs = F.TimeFWI(nx, nz, h, dt, nt, src, rec, wavelet, dtype=dtype, dev=dev)
d = obs.forward(v_true.ravel() / 1000.0)
prob = F.TimeFWI(nx, nz, h, dt, nt, src, rec, wavelet, dtype=dtype, dev=dev)
obj = geoinv.Objective(prob, d)
m0 = v0.ravel() / 1000.0
alphas = [10.0 ** e for e in range(-1, -9, -1)]      # km/s
# geoinv calls the step h; here h is the grid spacing, so the step is alpha as in the lecture
rows = [{"alpha" if k == "h" else k: x for k, x in r.items()} for r in geoinv.gradient_check(obj, m0, alphas, seed=1)]
for r in rows:
    print(f"alpha = {r['alpha']:.0e} km/s: finite difference {r['fd']:+.6e}, g^T dm {r['gtv']:+.6e}, relative error {r['rel_err']:.1e}")

# the model edge only: perturbing it moves the whole PML, where the backward run is not the exact transpose
E0, g = obj.value_and_gradient(m0)
dm = np.zeros(nx * nz)
edge = np.zeros((nx, nz), bool)
edge[0, 3:] = edge[-1, 3:] = edge[1:-1, -1] = True
dm[edge.ravel()] = rng.standard_normal(edge.sum())
alpha_e = 1e-4 if args.dtype == "float64" else 1e-2
fd_edge = (obj.value(m0 + alpha_e * dm) - obj.value(m0 - alpha_e * dm)) / (2 * alpha_e)
edge_err = abs(fd_edge - g @ dm) / abs(fd_edge)
print(f"model edge only (central difference, alpha = {alpha_e:.0e}): relative error {edge_err:.1e}  -> keep the edge fixed")

# ---- 4. single gradient components by central difference (the last node is on the model edge)
nodes = [(nx // 4, 6), (nx // 2, 15), (3 * nx // 4, 25), (nx // 2, 35), (nx // 2, nz - 1)]
comps = []
for ij in nodes:
    k = ij[0] * nz + ij[1]
    e = np.zeros(nx * nz)
    e[k] = 1.0
    fd = (obj.value(m0 + alpha_e * e) - obj.value(m0 - alpha_e * e)) / (2 * alpha_e)
    comps.append({"i": ij[0], "j": ij[1], "z_km": ij[1] * h / 1000, "fd": float(fd), "adjoint": float(g[k]),
                  "rel_err": float(abs(fd - g[k]) / abs(fd)), "edge": ij[1] == nz - 1})
    print(f"dE/dm at node {ij} (z = {ij[1] * h / 1000:.2f} km): finite difference {fd:+.6e}, adjoint {g[k]:+.6e}, "
          f"relative error {comps[-1]['rel_err']:.1e}{'  (model edge)' if comps[-1]['edge'] else ''}")

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.loglog([r["alpha"] for r in rows], [r["rel_err"] for r in rows], "o-", label=args.dtype)
ax.set_xlabel(r"$\alpha$ (km/s)")
ax.set_ylabel("|FD - g^T dm| / |g^T dm|")
ax.set_title(f"gradient check, {args.dtype}: dot-product {max(x['rel'] for x in dot):.0e}", fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.savefig(out / "check_adjoint.png", dpi=150)

tr = np.arange(nt) * dt
fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, constrained_layout=True)
for ax, r in zip(axes, (len(rec) // 4, 3 * len(rec) // 4)):
    ax.plot(tr, J_ana[r], "-", lw=1.5, label="analytic: virtual source")
    ax.plot(tr, J_num[r], "--", lw=1.5, label=f"numerical: central difference, dv = {dv:g} m/s")
    ax.plot(tr, 100 * (J_num[r] - J_ana[r]), "-", lw=0.8, color="gray", label="difference x 100")
    ax.set_ylabel(f"du/dv_k, x = {rec[r][0] * h / 1000:.2f} km")
    ax.grid(alpha=0.3)
axes[0].set_title(f"Jacobian column, node x = {ks[0] * h / 1000:.2f} km, z = {ks[1] * h / 1000:.2f} km: "
                  f"relative difference {jac_err:.0e}", fontsize=10)
axes[0].legend(fontsize=7)
axes[-1].set_xlabel("t (s)")
fig.savefig(out / "check_jacobian.png", dpi=150)

summary = {"dtype": args.dtype, "device": str(dev), "nx": nx, "nz": nz, "nt": nt, "nshot": args.nshot,
           "dot_product": dot, "gradient_check": rows, "edge_only": {"alpha": alpha_e, "rel_err": float(edge_err)},
           "jacobian_column": {"shot": int(shot), "node": list(ks), "dv": dv, "rel_diff": jac_err},
           "gradient_components": comps}
(out / "check_adjoint.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"saved {out}/check_adjoint.png, check_jacobian.png, check_adjoint.json")
