"""Gauss-Newton fit of an NMO hyperbola t(x) = sqrt(t0^2 + x^2 / v^2) to noisy traveltimes.

Run:  python3 nmo_gn.py
Out:  results/nmo_gn.png, results/nmo_gn.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# ---- variables to change ---------------------------------------------------
T0_TRUE, V_TRUE = 1.0, 2000.0    # true model (s, m/s)
NOISE = 0.005                    # noise standard deviation (s) ; try 0.0
SEED = 3                         # random seed for the noise
T0_INIT, V_INIT = 0.8, 2500.0    # starting model ; try V_INIT = 8000.0
LAMBDA = 0.0                     # 0: Gauss-Newton ; > 0: Levenberg-Marquardt, try 1e-3
N_ITER = 20
# -----------------------------------------------------------------------------

offsets = np.arange(0.0, 3001.0, 250.0)
rng = np.random.default_rng(SEED)
t_obs = np.sqrt(T0_TRUE ** 2 + (offsets / V_TRUE) ** 2) + rng.normal(0.0, NOISE, offsets.size)


def model(m):
    t0, v = m
    return np.sqrt(t0 ** 2 + (offsets / v) ** 2)


def residual(m):
    return model(m) - t_obs


def jacobian(m):
    t0, v = m
    t = model(m)
    return np.stack([t0 / t, -offsets ** 2 / (v ** 3 * t)], axis=1)


m = np.array([T0_INIT, V_INIT])
lam = LAMBDA
history = [(m.copy(), float(residual(m) @ residual(m)))]
print(f"{'iter':>4} {'t0 (s)':>9} {'v (m/s)':>10} {'|r|^2':>10} {'lambda':>8}")
print(f"{0:4d} {m[0]:9.4f} {m[1]:10.1f} {history[-1][1]:10.3e} {lam:8.1e}")
for k in range(1, N_ITER + 1):
    r, J = residual(m), jacobian(m)
    if np.linalg.norm(J.T @ r) < 1e-8:
        print(f"{k:4d} converged: |J^T r| < 1e-8")
        break
    A = J.T @ J
    try:
        step = -np.linalg.solve(A + lam * np.eye(2), J.T @ r)
    except np.linalg.LinAlgError:
        print(f"{k:4d} singular J^T J: raise LAMBDA or change the starting model")
        break
    m_new = m + step
    cost_new = float(residual(m_new) @ residual(m_new))
    if lam > 0 and cost_new > history[-1][1]:
        lam *= 10          # step made things worse: damp more and retry next iteration
        print(f"{k:4d} rejected, lambda -> {lam:.1e}")
        continue
    if lam > 0:
        lam = max(lam / 10, 1e-12)
    m = m_new
    history.append((m.copy(), cost_new))
    print(f"{k:4d} {m[0]:9.4f} {m[1]:10.1f} {cost_new:10.3e} {lam:8.1e}")

ref = least_squares(residual, [T0_INIT, V_INIT], jac=jacobian)
print(f"\nscipy least_squares: t0 = {ref.x[0]:.4f}, v = {ref.x[1]:.1f}, "
      f"|r|^2 = {2 * ref.cost:.3e}, nfev = {ref.nfev}, success = {ref.success}")
print(f"true model: t0 = {T0_TRUE}, v = {V_TRUE}")

out = Path("results")
out.mkdir(parents=True, exist_ok=True)
(out / "nmo_gn.json").write_text(json.dumps({
    "start": [T0_INIT, V_INIT], "lambda0": LAMBDA, "noise": NOISE, "seed": SEED,
    "iterations": [{"t0": float(h[0][0]), "v": float(h[0][1]), "cost": h[1]} for h in history],
    "scipy": {"t0": float(ref.x[0]), "v": float(ref.x[1]), "cost": float(2 * ref.cost)},
    "true": [T0_TRUE, V_TRUE]}, indent=2) + "\n", encoding="utf-8")
xx = np.linspace(0, 3000, 200)
fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
ax.plot(offsets, t_obs, "ko", mfc="w", label="observed")
for k, (mk, _) in enumerate(history):
    style = dict(color="tab:red", lw=2) if k == 0 else (
        dict(color="tab:blue", lw=2) if k == len(history) - 1 else dict(color="tab:blue", lw=0.8, ls="--"))
    ax.plot(xx, np.sqrt(mk[0] ** 2 + (xx / mk[1]) ** 2), label=f"iter {k}" if k in (0, len(history) - 1) else None, **style)
ax.set_xlabel("offset x (m)"); ax.set_ylabel("t (s)"); ax.invert_yaxis()
ax.set_title(f"start ({T0_INIT}, {V_INIT}), lambda0 = {LAMBDA}"); ax.legend()
fig.savefig(out / "nmo_gn.png", dpi=150)
print(f"Saved: {out / 'nmo_gn.json'}, {out / 'nmo_gn.png'}")
