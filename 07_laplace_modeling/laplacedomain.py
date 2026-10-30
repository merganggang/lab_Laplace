"""2-D acoustic finite-difference Laplace-domain modeling from the lecture (import only).

Laplace transform with the damping constant s > 0 (1/s):

    u~(s) = integral_0^inf u(t) exp(-s t) dt

It is the Fourier transform of the frequency-domain lab at the imaginary frequency omega = -i s. With u = du/dt = 0
at t = 0 (the medium is at rest until the source starts), d/dt -> s, so the time-domain equation
(1/v^2) d2u/dt2 = lap u + q (timedomain.py) becomes, for each damping constant,

    (s^2 / v^2) u~ - lap u~ = q~        and on the grid   S u = q

S is real, symmetric and positive definite: the mass term s^2 / v^2 has the opposite sign of the
frequency-domain term -omega^2 / v^2. Every diagonal entry is positive, every off-diagonal entry negative or 0,
and each row has |diagonal| >= sum |off-diagonal| (an M-matrix), so every entry of S^-1 is positive:
the Green's function (q~ = 1 / h^2 at one node) is positive everywhere. The wavefield has the sign of
the source term w~(s); a Ricker wavelet has w~(s) < 0 for every s > 0, so the lab uses -Ricker.
Everything is real double precision (float64): the field falls by tens of orders of magnitude.

The damping constant may also be complex, p = s + i omega: the equation (p^2 / v^2) u~ - lap u~ = q~ is
then the frequency-domain equation at the complex frequency omega - i s (the Fourier transform of
u(t) exp(-s t)), and S, u~ are complex. logtime.py uses this for the log method of traveltime.

Grid: u[i, j] = u(x_i, z_j), shape (nx, nz), node m = i NZ + j (the order of u.ravel()), 5-point
Laplacian. Boundaries (left, right, bottom; the top is a free surface u = 0 unless top="absorb"):
    dirichlet   u = 0 outside the grid
    ce          Clayton & Engquist (1977) A2 on the edge nodes: the frequency-domain condition at omega = -i s,
                du/dn + (s / v) u - (v / 2 s) d2u/dl2 = 0 (n outward normal, l along the edge), real
    pml         nb nodes of coordinate stretching d/dx -> (1 / gamma_x) d/dx, gamma_x = 1 + d_x / s,
                the frequency-domain gamma_x = 1 + d_x / (i omega) at omega = -i s, real and >= 1

The PML equation is multiplied by gamma_x gamma_z and written as
    -d/dx (gamma_z / gamma_x du/dx) - d/dz (gamma_x / gamma_z du/dz) + gamma_x gamma_z s^2 / v^2 u = q
with every coefficient evaluated between the two nodes it couples, so S^T = S.

    build_matrix(v, h, s, ...)          S and the layout
    dS_ds(v, h, s, ...)                 derivative of S with respect to s (mass term and PML stretch)
    factor(S, ordering)                 sparse LU (SuperLU) with a fill-reducing order of the unknowns
    solve(v, h, svals, src, rec, wlap)  receiver data for many damping constants and sources
    analytic_2d_laplace(r, s, v)        homogeneous whole-space Green's function K0(s r / v) / (2 pi)
    laplace_transform(u, dt, s)         u~(s) of sampled time signals (source wavelet or traces)
    sweet_time(G, dG)                   traveltime -dG/ds / G (Shin et al. 2002)
    log_time(G, omega)                  traveltime -Im[ln G] / omega at the complex frequency omega - i s
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.special import k0

from timedomain import Layout, damping


def build_matrix(v, h, s, bc="pml", top="free", nb=20):
    """Laplace-domain matrix S(s) for the model v (nx, nz) and the layout around it (float64, CSC).

    s : damping constant (1/s), > 0 ; or complex s + i omega (complex frequency omega - i s, complex S)
    Returns (S, layout, keep): keep holds the flat indices of the unknown nodes in the computational
    grid (the free-surface row j = 0 is u = 0 and not an unknown)."""
    if bc not in ("dirichlet", "ce", "pml"):
        raise ValueError(f"unknown bc {bc!r}")
    if np.real(s) <= 0:
        raise ValueError("the damping constant s must be positive")
    v = np.asarray(v, float)
    lay = Layout(*v.shape, bc, top, nb)
    NX, NZ = lay.NX, lay.NZ
    vmax = float(v.max())
    pml = bc == "pml"
    gx = lambda k: 1.0 + damping(lay, h, vmax, k, 0) / s if pml else np.ones_like(np.asarray(k, float))
    gz = lambda k: 1.0 + damping(lay, h, vmax, k, 1) / s if pml else np.ones_like(np.asarray(k, float))

    vv = lay.extend(v)
    I, J = np.meshgrid(np.arange(NX), np.arange(NZ), indexing="ij")
    mass = (gx(I) * gz(J) * s * s / vv ** 2).ravel()

    # ce: the rows of the edge nodes are halved (the half cell inside the grid), a quarter at corners
    ce = bc == "ce"
    absorb_top = top == "absorb"
    on_x = lambda P: (P == 0) | (P == NX - 1)
    on_z = lambda Q: (Q == NZ - 1) | ((Q == 0) & absorb_top)
    cell = (np.where(on_x(I), 0.5, 1.0) * np.where(on_z(J), 0.5, 1.0)).ravel() if ce else 1.0
    diag = mass * cell
    rows, cols, vals = [], [], []

    # 5-point Laplacian as a sum over grid edges: an edge between nodes p and p2 with weight w adds
    # w to both diagonals and -w to both off-diagonal entries; an edge reaching outside the grid
    # (u = 0 there) adds only to the diagonal of its inside node
    for di, dj in ((1, 0), (0, 1)):
        i0 = np.arange(-1, NX) if di else np.arange(NX)
        j0 = np.arange(-1, NZ) if dj else np.arange(NZ)
        P, Q = (a.ravel() for a in np.meshgrid(i0, j0, indexing="ij"))
        P2, Q2 = P + di, Q + dj
        in1 = (P >= 0) & (P < NX) & (Q >= 0) & (Q < NZ)
        in2 = (P2 < NX) & (Q2 < NZ)
        X, Z = P + 0.5 * di, Q + 0.5 * dj                       # midpoint of the edge
        w = (gz(Z) / gx(X) if di else gx(X) / gz(Z)) / h ** 2
        if ce:   # no edge leaves the grid (the ghost node is eliminated); an edge along a boundary is halved
            w = w * (in1 & in2) * np.where((P == P2) & on_x(P), 0.5, 1.0) * np.where((Q == Q2) & on_z(Q), 0.5, 1.0)
        m1 = np.where(in1, P * NZ + Q, 0)
        m2 = np.where(in2, P2 * NZ + Q2, 0)
        np.add.at(diag, m1[in1], w[in1])
        np.add.at(diag, m2[in2], w[in2])
        both = in1 & in2
        rows.extend([m1[both], m2[both]])
        cols.extend([m2[both], m1[both]])
        vals.extend([-w[both], -w[both]])

    if ce:   # boundary terms (s / v h) u and -(v / 2 s h) d2u/dl2 of each absorbing edge
        vflat = vv.ravel()
        corner = (on_x(I) & on_z(J)).ravel()
        sides = [(I == 0, 1), (I == NX - 1, 1), (J == NZ - 1, NZ)] + ([(J == 0, NZ)] if absorb_top else [])
        for side, step in sides:                   # step: flat index to the next node along the edge
            idx = np.flatnonzero(side.ravel())
            np.add.at(diag, idx, s / vflat[idx] / h * np.where(corner[idx], 0.5, 1.0))
            nxt = idx + step
            ok = np.isin(nxt, idx)
            p_, q_ = idx[ok], nxt[ok]
            wt = 0.5 * (vflat[p_] + vflat[q_]) / (2 * s * h ** 3)   # v / (2 s h) times the 1 / h^2 of d2u/dl2
            np.add.at(diag, p_, wt)
            np.add.at(diag, q_, wt)
            rows.extend([p_, q_])
            cols.extend([q_, p_])
            vals.extend([-wt, -wt])

    N = NX * NZ
    S = sp.coo_matrix((np.concatenate(vals + [diag]),
                       (np.concatenate(rows + [np.arange(N)]), np.concatenate(cols + [np.arange(N)]))),
                      shape=(N, N)).tocsc()
    keep = np.arange(N) if absorb_top else np.flatnonzero(J.ravel() > 0)
    if len(keep) < N:
        S = S[keep][:, keep]
    S = S.tocsc()
    S.sort_indices()
    return S, lay, keep


def dS_ds(v, h, s, bc="pml", top="free", nb=20, rel=1e-4):
    """dS/ds by a central difference of two assembled matrices (exact for the s^2 mass term; the PML
    stretch 1 + d / s and the ce boundary terms depend on s too). Same unknowns as build_matrix."""
    ds = rel * s
    Sp, _, _ = build_matrix(v, h, s + ds, bc, top, nb)
    Sm, _, _ = build_matrix(v, h, s - ds, bc, top, nb)
    return ((Sp - Sm) / (2 * ds)).tocsc()


# order in which LU eliminates the unknowns (SuperLU permc_spec); the order sets the fill-in
ORDERINGS = {"mindegree": "MMD_AT_PLUS_A",   # minimum degree on the pattern of S^T + S (S is symmetric)
             "colamd": "COLAMD",
             "natural": "NATURAL"}


def factor(S, ordering="mindegree"):
    """Sparse LU of the real matrix S with the pivots kept on the diagonal (S is positive definite)."""
    return spl.splu(S, permc_spec=ORDERINGS[ordering], diag_pivot_thresh=0.0,
                    options={"SymmetricMode": True})


def lu_bytes(lu):
    """Memory of the L and U factors: values (8 bytes real, 16 complex) plus row indices (4 bytes)."""
    return (lu.L.nnz + lu.U.nnz) * (lu.L.dtype.itemsize + 4)


def solve(v, h, svals, src, rec, wlap, bc="pml", top="free", nb=20, ordering="mindegree", fields=False,
          derivative=False):
    """Receiver data for all damping constants and sources: one LU per s, one solve per source.

    svals : damping constants (1/s) ; src (ns, 2), rec (nr, 2) model node indices (i, j)
    wlap  : Laplace transform of the source wavelet w~(s) at each s (ns,) or (ns_s, nsrc)
    derivative : also du~/ds from S du/ds = -(dS/ds) u with the same LU (virtual source)
    Returns dict: data (n_s, nsrc, nr), field (n_s, nx, nz) of the first source if fields, dfield if both,
    time_build_s, time_factor_s, time_solve_s (totals), lu_bytes (largest), residual (largest |S u - q| / |q|),
    nnz_S, n_unknowns, layout."""
    svals = np.atleast_1d(np.asarray(svals))
    dtype = complex if np.iscomplexobj(svals) else float     # complex s + i omega: complex frequency
    src, rec = np.atleast_2d(src), np.atleast_2d(rec)
    wlap = np.asarray(wlap, dtype).reshape(len(svals), -1)
    lay = Layout(*np.shape(v), bc, top, nb)
    flat = lambda ij: (ij[:, 0] + lay.pad) * lay.NZ + ij[:, 1] + lay.ptop
    out = {"data": np.zeros((len(svals), len(src), len(rec)), dtype), "time_build_s": 0.0, "time_factor_s": 0.0,
           "time_solve_s": 0.0, "lu_bytes": 0, "layout": lay}
    if fields:
        out["field"] = np.zeros((len(svals),) + np.shape(v), dtype)
        if derivative:
            out["dfield"] = np.zeros_like(out["field"])
    if derivative:
        out["ddata"] = np.zeros_like(out["data"])
    for k, s in enumerate(svals):
        tb = time.perf_counter()
        S, lay, keep = build_matrix(v, h, s, bc, top, nb)
        dS = dS_ds(v, h, s, bc, top, nb) if derivative else None
        pos = np.full(lay.NX * lay.NZ, -1)
        pos[keep] = np.arange(len(keep))
        t0 = time.perf_counter()
        lu = factor(S, ordering)
        t1 = time.perf_counter()
        q = np.zeros((len(keep), len(src)), dtype)
        q[pos[flat(src)], np.arange(len(src))] = wlap[k] / h ** 2   # point source: w~ / h^2 at one node
        u = lu.solve(q)
        du = lu.solve(-(dS @ u)) if derivative else None             # same LU, virtual source -(dS/ds) u
        t2 = time.perf_counter()
        out["residual"] = max(out.get("residual", 0.0),
                              float(np.linalg.norm(S @ u[:, 0] - q[:, 0]) / np.linalg.norm(q[:, 0])))
        out["time_build_s"] += t0 - tb
        out["time_factor_s"] += t1 - t0
        out["time_solve_s"] += t2 - t1
        out["lu_bytes"] = max(out["lu_bytes"], lu_bytes(lu))
        out["data"][k] = u[pos[flat(rec)]].T
        if derivative:
            out["ddata"][k] = du[pos[flat(rec)]].T
        if fields:
            for key, a in (("field", u), ("dfield", du)):
                if a is None:
                    continue
                full = np.zeros(lay.NX * lay.NZ, dtype)
                full[keep] = a[:, 0]
                out[key][k] = lay.inner(full.reshape(lay.NX, lay.NZ))
        out["nnz_S"], out["n_unknowns"] = S.nnz, S.shape[0]
    return out


def analytic_2d_laplace(r, s, v):
    """Homogeneous whole-space solution of (s/v)^2 u - lap u = delta(x) delta(z) at distance r.

    u = K0(s r / v) / (2 pi), K0 the modified Bessel function of the second kind: it does not oscillate
    and falls like sqrt(pi v / (2 s r)) exp(-s r / v) far away. It is the frequency-domain -(i/4) H0^(2)(omega r / v)
    at omega = -i s. Multiply by w~(s) for the source wavelet."""
    return k0(np.asarray(s, float) * np.asarray(r, float) / v) / (2.0 * np.pi)


def laplace_transform(u, dt, s):
    """u~(s) = sum_n u(t_n) exp(-s t_n) dt for signals sampled at t_n = n dt along the last axis.

    s may be an array: the result has the damping constants on the first axis."""
    u = np.asarray(u, float)
    t = np.arange(u.shape[-1]) * dt
    E = np.exp(-np.outer(np.atleast_1d(s), t)) * dt          # (n_s, nt)
    return np.tensordot(E, u, axes=([1], [u.ndim - 1]))


def sweet_time(G, dG):
    """Traveltime from the damped wavefield (SWEET, Shin et al. 2002): t = -(dG/ds) / G.

    For one arrival A exp(-s t_f) this is t_f; for a sum of arrivals it is their mean time weighted by
    their damped amplitudes. With an impulse source G is the Green's function; with a wavelet the
    weighted mean time of the wavelet is added."""
    return -np.asarray(dG) / np.asarray(G)


def log_time(G, omega):
    """Traveltime by the log method (Shin et al. 2002): G = A exp(-s t_f) exp(-i omega t_f) at the complex
    frequency omega - i s gives ln G = ln A - s t_f - i omega t_f, so t_f = -Im[ln G] / omega.

    np.angle is the principal value in (-pi, pi]: omega t_f must stay below pi, so omega is kept small
    (the textbook uses 0.01 Hz or less); above that the phase wraps and t_f jumps by 2 pi / omega."""
    return -np.angle(np.asarray(G)) / omega

