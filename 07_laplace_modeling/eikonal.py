"""Local first-arrival reference for the Laplace-domain lab (import only).

Fast sweeping method (FSM; Zhao 2005), with the same cell-based updates and
bilinear sampling as the traveltime-tomography lab solver. Only the FSM and its helpers
are included here, so this lab has no runtime dependency on another lab folder.

Slowness s[i, j] belongs to cell (i, j), shape (nx, nz), in s/m.
Traveltimes T live on (nx + 1, nz + 1) nodes, in seconds, with spacing h in m.
sweet.py and compare_laplace.py treat the velocity nodes as these cell centres
and shift source and receiver coordinates by half a grid spacing.

    fsm(s, h, src)       first-arrival field for source (x, z), in m
    sample(T, h, pts)    bilinear sampling at receiver positions, in m
"""

import math

import numpy as np

INF = np.inf


def pad_cells(s):
    """Cell slowness with a border of inf: cell (i, j) is p[i + 1, j + 1], outside cells are inf."""
    p = np.full((s.shape[0] + 2, s.shape[1] + 2), INF)
    p[1:-1, 1:-1] = s
    return p


def source_cells(n, f):
    """Cell indices along one axis that contain coordinate f (in cells): two if f sits on a grid line."""
    k = int(math.floor(f))
    cells = {k - 1, k} if abs(f - round(f)) < 1e-9 else {k}
    return sorted(min(max(c, 0), n - 1) for c in cells)


def init_source(s, h, src):
    """Traveltime at the corners of the cell(s) holding the source: straight line times s_c * distance."""
    nx, nz = s.shape
    T = np.full((nx + 1, nz + 1), INF)
    x, z = src
    for ci in source_cells(nx, x / h):
        for cj in source_cells(nz, z / h):
            for i, j in [(ci, cj), (ci + 1, cj), (ci, cj + 1), (ci + 1, cj + 1)]:
                T[i, j] = min(T[i, j], s[ci, cj] * math.hypot(i * h - x, j * h - z))
    return T


def sample(T, h, pts):
    """Bilinear interpolation of a node field at points (N, 2)."""
    pts = np.atleast_2d(pts)
    fx = np.clip(pts[:, 0] / h, 0, T.shape[0] - 1 - 1e-9)
    fz = np.clip(pts[:, 1] / h, 0, T.shape[1] - 1 - 1e-9)
    i, j = fx.astype(int), fz.astype(int)
    wx, wz = fx - i, fz - j
    return ((1 - wx) * (1 - wz) * T[i, j] + wx * (1 - wz) * T[i + 1, j]
            + (1 - wx) * wz * T[i, j + 1] + wx * wz * T[i + 1, j + 1])


def _two(a, b, hs):
    """Two-point upwind update across one cell: (t - a)^2 + (t - b)^2 = (h s)^2, valid if |a - b| < h s."""
    with np.errstate(invalid="ignore"):
        d = a - b
        ok = np.abs(d) < hs
        t = 0.5 * (a + b + np.sqrt(np.maximum(2.0 * hs * hs - d * d, 0.0)))
    return np.where(ok, t, INF)


def _update(Tp, p, h, I, J):
    """Upwind estimate at nodes (I, J) from their four neighbours; Tp is T padded by one inf node."""
    tl, tr = Tp[I, J + 1], Tp[I + 2, J + 1]         # (i-1, j), (i+1, j)
    tu, td = Tp[I + 1, J], Tp[I + 1, J + 2]         # (i, j-1), (i, j+1)
    c00, c10 = p[I, J], p[I + 1, J]                 # cells (i-1, j-1), (i, j-1)
    c01, c11 = p[I, J + 1], p[I + 1, J + 1]         # cells (i-1, j), (i, j)
    t = np.minimum.reduce([tl + h * np.minimum(c00, c01), tr + h * np.minimum(c10, c11),
                           tu + h * np.minimum(c00, c10), td + h * np.minimum(c01, c11)])
    for a, b_, c in [(tl, tu, c00), (tr, tu, c10), (tl, td, c01), (tr, td, c11)]:
        t = np.minimum(t, _two(a, b_, h * c))
    return t


def fsm(s, h, src, eps=1e-9, maxit=100, return_iterations=False):
    """Fast sweeping (Zhao 2005): Gauss-Seidel sweeps in four orders until the largest change < eps (s).

    In one sweep order, nodes with the same i * si + j * sj depend only on earlier ones,
    so each such diagonal is updated at once with NumPy."""
    nx, nz = s.shape
    p = pad_cells(s)
    Tp = np.full((nx + 3, nz + 3), INF)
    Tp[1:-1, 1:-1] = init_source(s, h, src)
    free = ~np.isfinite(Tp[1:-1, 1:-1])
    I, J = np.nonzero(free)
    sweeps = []
    for si, sj in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        key = si * I + sj * J
        order = np.argsort(key, kind="stable")
        cuts = np.flatnonzero(np.diff(key[order])) + 1
        sweeps.append([(I[g], J[g]) for g in np.split(order, cuts)])
    for it in range(1, maxit + 1):
        old = Tp.copy()
        for groups in sweeps:
            for Ig, Jg in groups:
                Tp[Ig + 1, Jg + 1] = np.minimum(Tp[Ig + 1, Jg + 1], _update(Tp, p, h, Ig, Jg))
        fin = np.isfinite(Tp)
        change = np.abs(Tp[fin] - old[fin]).max()
        if change < eps:
            break
    T = Tp[1:-1, 1:-1].copy()
    return (T, it) if return_iterations else T
