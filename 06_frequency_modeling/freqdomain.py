"""2-D acoustic finite-difference frequency-domain modeling from the lecture (import only).

Fourier transform with d/dt -> i omega (textbook appendix convention):

    u~(omega) = integral u(t) exp(-i omega t) dt,   u(t) = (1 / 2 pi) integral u~(omega) exp(i omega t) d omega

The time-domain equation (1/v^2) d2u/dt2 = lap u + q (timedomain.py) becomes, for each frequency,

    -lap u~ - (omega^2 / v^2) u~ = q~        and on the grid   S u = q

S is the impedance matrix (N x N, N = NX NZ nodes, node m = i NZ + j, the order of u.ravel()), u the
vector of u~ at all nodes and q the source vector: a point source is w~(omega) / h^2 at one node.

Stencils (lap and the mass term omega^2 / v^2 u):
    5   5-point Laplacian, mass term at the node only
    9   rotated 9-point operator of Jo, Shin & Suh (1996): a lap_0 + (1 - a) lap_45 and the mass term
        averaged over the node (c), its 4 axial (d) and 4 diagonal (e) neighbours, c + 4 d + 4 e = 1

Boundaries (left, right, bottom; the top is a free surface u = 0 unless top="absorb"):
    dirichlet   u = 0 outside the grid
    ce          Clayton & Engquist (1977) A2 on the edge nodes, no extra nodes:
                du/dn + i k u + (i / 2k) d2u/ds2 = 0, k = omega / v (n outward normal, s along the edge)
    pml         nb nodes of complex coordinate stretching d/dx -> (1 / gamma_x) d/dx,
                gamma_x = 1 + d_x / (i omega), same for z (Berenger 1994; Chew & Weedon 1994)

ce eliminates the ghost node outside the edge with the boundary condition (central difference for
du/dn) and halves the edge rows (quarter at the corners), so S stays symmetric; d2u/ds2 couples the
edge nodes along the edge, with no second derivative across a corner.

The PML equation is multiplied by gamma_x gamma_z and written as
    -d/dx (gamma_z / gamma_x du/dx) - d/dz (gamma_x / gamma_z du/dz) - gamma_x gamma_z omega^2 / v^2 u = q
and every coefficient is evaluated between the two nodes it couples, so S is complex symmetric
(S^T = S, not Hermitian). The week 10 gradient uses this.

Complex frequency: omega - i damp (damp = alpha_d, 1/s) is the transform of u(t) exp(-alpha_d t).
to_time() undoes the damping with exp(alpha_d t).

    build_matrix(v, h, omega, ...)       S and the layout
    factor(S, ordering)                  sparse LU (SuperLU) with a fill-reducing order of the unknowns
    solve(v, h, freqs, src, rec, wspec)  receiver data for many frequencies and sources, one LU per frequency
    analytic_2d_freq(r, omega, v)        homogeneous whole-space solution (Hankel function)
    source_spectrum(wav, dt, omega)      w~(omega) of a sampled source wavelet
    frequency_axis(tmax, dt, fmax)       frequencies of a record of length tmax ; to_time() inverts
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.special import hankel2

from timedomain import Layout, damping

JO96 = (0.5461, 0.6248, 0.09381)        # a, c, d of Jo et al. (1996) ; e = (1 - c - 4 d) / 4
STENCILS = {5: (1.0, 1.0, 0.0), 9: JO96}


def coefficients(stencil):
    """(a, c, d, e) of the 5- or 9-point operator; a tuple (a, c, d) is accepted too."""
    a, c, d = STENCILS[stencil] if stencil in STENCILS else stencil
    return a, c, d, (1.0 - c - 4.0 * d) / 4.0


def stretch(layout, h, vmax, k, axis, omega, bc):
    """gamma = 1 + d / (i omega) at node positions k along axis (1 outside the layer and for non-PML bc)."""
    if bc != "pml":
        return np.ones_like(np.asarray(k, float), dtype=complex)
    return 1.0 + damping(layout, h, vmax, k, axis) / (1j * omega)


def build_matrix(v, h, omega, stencil=5, bc="pml", top="free", nb=20, damp=0.0):
    """Impedance matrix S(omega) for the model v (nx, nz) and the layout around it.

    omega : angular frequency (rad/s) ; damp : alpha_d (1/s), the solve uses omega - i alpha_d
    Returns (S, layout, keep): S is CSC of the unknown nodes, keep the flat indices of those nodes
    in the computational grid (the free-surface row j = 0 is u = 0 and not an unknown)."""
    if bc not in ("dirichlet", "ce", "pml"):
        raise ValueError(f"unknown bc {bc!r}")
    w = complex(omega) - 1j * damp
    a, c, d, e = coefficients(stencil)
    v = np.asarray(v, float)
    lay = Layout(*v.shape, bc, top, nb)
    NX, NZ = lay.NX, lay.NZ
    vmax = float(v.max())
    gx = lambda k: stretch(lay, h, vmax, k, 0, w, bc)
    gz = lambda k: stretch(lay, h, vmax, k, 1, w, bc)

    # mass term per node: gamma_x gamma_z omega^2 / v^2
    vv = lay.extend(v)
    I, J = np.meshgrid(np.arange(NX), np.arange(NZ), indexing="ij")
    mass = (gx(I) * gz(J) * w * w / vv ** 2).ravel()

    # ce: the rows of the edge nodes are halved (the half cell inside the grid), a quarter at corners
    ce = bc == "ce"
    absorb_top = top == "absorb"
    on_x = lambda P: (P == 0) | (P == NX - 1)                     # on the left or right edge
    on_z = lambda Q: (Q == NZ - 1) | ((Q == 0) & absorb_top)      # on the bottom (or absorbing top) edge
    cell = (np.where(on_x(I), 0.5, 1.0) * np.where(on_z(J), 0.5, 1.0)).ravel() if ce else 1.0

    rows, cols, vals = [], [], []
    diag = -c * mass * cell                        # S = K - M ; K from the edges below
    diag = diag.astype(complex)

    def edges(di, dj):
        """All node pairs (p, p + (di, dj)) with at least one end in the grid, and their midpoints."""
        i0 = np.arange(-1, NX) if di else np.arange(NX)          # di is 0 or 1
        j0 = np.arange(-1, NZ) if dj > 0 else (np.arange(0, NZ + 1) if dj < 0 else np.arange(NZ))
        P, Q = np.meshgrid(i0, j0, indexing="ij")
        P, Q = P.ravel(), Q.ravel()
        P2, Q2 = P + di, Q + dj
        in1 = (P >= 0) & (P < NX) & (Q >= 0) & (Q < NZ)
        in2 = (P2 >= 0) & (P2 < NX) & (Q2 >= 0) & (Q2 < NZ)
        keep = in1 | in2
        P, Q, P2, Q2, in1, in2 = P[keep], Q[keep], P2[keep], Q2[keep], in1[keep], in2[keep]
        return P, Q, P2, Q2, in1, in2, 0.5 * (P + P2), 0.5 * (Q + Q2)

    def add(P, Q, P2, Q2, in1, in2, wk, mk):
        """Stiffness weight wk (edge Laplacian) and mass weight mk (between the two nodes) of each pair."""
        if ce:   # no pair reaches outside (the ghost node is eliminated); a pair along an edge is halved
            f = (np.where((P == P2) & on_x(P), 0.5, 1.0) * np.where((Q == Q2) & on_z(Q), 0.5, 1.0)
                 * (in1 & in2))
            wk, mk = wk * f, mk * f
        m1 = np.where(in1, P * NZ + Q, 0)
        m2 = np.where(in2, P2 * NZ + Q2, 0)
        np.add.at(diag, m1[in1], wk[in1])
        np.add.at(diag, m2[in2], wk[in2])
        both = in1 & in2
        off = -wk[both] - mk[both]
        rows.extend([m1[both], m2[both]])
        cols.extend([m2[both], m1[both]])
        vals.extend([off, off])

    def pair_mass(P, Q, P2, Q2, in1, in2, weight):
        if weight == 0.0:
            return np.zeros(len(P), complex)
        m1 = np.where(in1, P * NZ + Q, 0)
        m2 = np.where(in2, P2 * NZ + Q2, 0)
        return weight * 0.5 * (mass[m1] + mass[m2])

    # axial edges: A = gamma_z / gamma_x (x edges), B = gamma_x / gamma_z (z edges) at the midpoint
    for di, dj in ((1, 0), (0, 1)):
        E = edges(di, dj)
        X, Z = E[6], E[7]
        A, B = gz(Z) / gx(X), gx(X) / gz(Z)
        wk = (a * A + (1 - a) * (A - B) / 2) if di else (a * B + (1 - a) * (B - A) / 2)
        add(*E[:6], wk / h ** 2, pair_mass(*E[:6], d))
    # diagonal edges of lap_45 = (sum of the 4 diagonal neighbours - 4 u) / (2 h^2), weight (A + B) / 2
    if a != 1.0 or e != 0.0:
        for di, dj in ((1, 1), (1, -1)):
            E = edges(di, dj)
            X, Z = E[6], E[7]
            A, B = gz(Z) / gx(X), gx(X) / gz(Z)
            wk = (1 - a) * (A + B) / 2 / (2 * h ** 2)
            add(*E[:6], wk, pair_mass(*E[:6], e))

    if ce:   # boundary terms (i k / h) u and (i / 2k h) d2u/ds2 of each absorbing edge
        vflat = vv.ravel()
        corner = (on_x(I) & on_z(J)).ravel()
        edges_ = [(I == 0, 1), (I == NX - 1, 1), (J == NZ - 1, NZ)] + ([(J == 0, NZ)] if absorb_top else [])
        for side, step in edges_:            # step: flat index to the next node along the edge
            idx = np.flatnonzero(side.ravel())
            k = w / vflat[idx]
            np.add.at(diag, idx, 1j * k / h * np.where(corner[idx], 0.5, 1.0))
            nxt = idx + step
            ok = np.isin(nxt, idx)
            p_, q_ = idx[ok], nxt[ok]
            wt = 1j / (2 * (w / (0.5 * (vflat[p_] + vflat[q_]))) * h ** 3)
            np.add.at(diag, p_, -wt)
            np.add.at(diag, q_, -wt)
            rows.extend([p_, q_])
            cols.extend([q_, p_])
            vals.extend([wt, wt])

    N = NX * NZ
    S = sp.coo_matrix((np.concatenate(vals + [diag]),
                       (np.concatenate(rows + [np.arange(N)]), np.concatenate(cols + [np.arange(N)]))),
                      shape=(N, N)).tocsc()
    keep = np.arange(N) if top == "absorb" else np.flatnonzero((J.ravel() > 0))
    if len(keep) < N:
        S = S[keep][:, keep]
    return S.tocsc(), lay, keep


# order in which LU eliminates the unknowns (SuperLU permc_spec); the order sets the fill-in
ORDERINGS = {"mindegree": "MMD_AT_PLUS_A",   # minimum degree on the pattern of S^T + S (S is symmetric)
             "colamd": "COLAMD",             # approximate minimum degree for general matrices, SciPy default
             "natural": "NATURAL"}           # node order m = i NZ + j: S is a band of width NZ that fills in


def factor(S, ordering="mindegree"):
    """Sparse LU of S (SuperLU) with the unknowns reordered by ORDERINGS[ordering].

    The pivots stay on the diagonal (no row exchanges), so the fill-in is set by the ordering alone.
    With SuperLU's default partial pivoting the row exchanges grow at higher frequencies, where S
    is indefinite, and the LU can become ten times larger. solve() reports the residual |S u - q| / |q|."""
    return spl.splu(S, permc_spec=ORDERINGS[ordering], diag_pivot_thresh=0.0,
                    options={"SymmetricMode": True})


def lu_bytes(lu):
    """Memory of the L and U factors: values (16 bytes complex) plus row indices (4 bytes)."""
    return (lu.L.nnz + lu.U.nnz) * 20


def solve(v, h, freqs, src, rec, wspec, stencil=5, bc="pml", top="free", nb=20, damp=0.0,
          ordering="mindegree", fields=False):
    """Receiver data for all frequencies and sources: one LU per frequency, one solve per source.

    freqs : frequencies (Hz), > 0 unless damp > 0 ; src (ns, 2), rec (nr, 2) model node indices (i, j)
    wspec : source spectrum w~ at each frequency (nf,) or (nf, ns)
    Returns dict: data (nf, ns, nr) complex, field (nf, nx, nz) of the first source if fields,
    time_build_s, time_factor_s, time_solve_s (totals), lu_bytes (largest), residual (largest |S u - q| / |q|),
    nnz_S, n_unknowns, layout."""
    freqs = np.atleast_1d(np.asarray(freqs, float))
    src, rec = np.atleast_2d(src), np.atleast_2d(rec)
    wspec = np.asarray(wspec, complex).reshape(len(freqs), -1)
    lay = Layout(*np.shape(v), bc, top, nb)
    flat = lambda ij: (ij[:, 0] + lay.pad) * lay.NZ + ij[:, 1] + lay.ptop
    out = {"data": np.zeros((len(freqs), len(src), len(rec)), complex), "time_build_s": 0.0, "time_factor_s": 0.0,
           "time_solve_s": 0.0, "lu_bytes": 0, "layout": lay}
    if fields:
        out["field"] = np.zeros((len(freqs),) + np.shape(v), complex)
    for k, f in enumerate(freqs):
        tb = time.perf_counter()
        S, lay, keep = build_matrix(v, h, 2 * np.pi * f, stencil, bc, top, nb, damp)
        pos = np.full(lay.NX * lay.NZ, -1)
        pos[keep] = np.arange(len(keep))
        t0 = time.perf_counter()
        lu = factor(S, ordering)
        t1 = time.perf_counter()
        q = np.zeros((len(keep), len(src)), complex)
        q[pos[flat(src)], np.arange(len(src))] = wspec[k] / h ** 2   # point source: w~ / h^2 at one node
        u = lu.solve(q)
        t2 = time.perf_counter()
        out["residual"] = max(out.get("residual", 0.0),
                              float(np.linalg.norm(S @ u[:, 0] - q[:, 0]) / np.linalg.norm(q[:, 0])))
        out["time_build_s"] += t0 - tb
        out["time_factor_s"] += t1 - t0
        out["time_solve_s"] += t2 - t1
        out["lu_bytes"] = max(out["lu_bytes"], lu_bytes(lu))
        out["data"][k] = u[pos[flat(rec)]].T
        if fields:
            full = np.zeros(lay.NX * lay.NZ, complex)
            full[keep] = u[:, 0]
            out["field"][k] = lay.inner(full.reshape(lay.NX, lay.NZ))
        out["nnz_S"], out["n_unknowns"] = S.nnz, S.shape[0]
    return out


def analytic_2d_freq(r, omega, v):
    """Homogeneous whole-space solution of -lap u - (omega/v)^2 u = delta(x) delta(z) at distance r.

    u = -(i/4) H0^(2)(omega r / v): an outgoing cylindrical wave for exp(i omega t); omega may be complex
    (omega - i alpha_d). Multiply by w~(omega) for the source wavelet."""
    return -0.25j * hankel2(0, np.asarray(omega) * np.asarray(r, float) / v)


def source_spectrum(wav, dt, omega):
    """w~(omega) = sum_n w(t_n) exp(-i omega t_n) dt for a wavelet sampled at t_n = n dt (omega may be complex)."""
    t = np.arange(len(wav)) * dt
    return np.exp(-1j * np.outer(np.atleast_1d(omega), t)) @ (np.asarray(wav, float) * dt)


def frequency_axis(tmax, dt, fmax):
    """Record of nt samples t_n = n dt over one period T = nt dt >= tmax: df = 1 / T.

    Returns (nt, df, freqs) with freqs = k df for k = 0 .. up to fmax (the rest is set to 0)."""
    nt = int(np.ceil(tmax / dt))
    df = 1.0 / (nt * dt)
    nf = int(np.floor(fmax / df)) + 1
    return nt, df, np.arange(nf) * df


def to_time(U, nt, dt, damp=0.0):
    """Time traces from spectra U (..., nf) at f_k = k / (nt dt), k = 0 .. nf - 1, computed at omega - i damp.

    numpy.fft.irfft has the 1/N on the inverse: u_n exp(-damp t_n) = irfft(U)_n / dt. The negative
    frequencies are the complex conjugates of the positive ones (real signal), which irfft supplies."""
    U = np.asarray(U, complex)
    full = np.zeros(U.shape[:-1] + (nt // 2 + 1,), complex)
    nf = min(U.shape[-1], nt // 2 + 1)
    full[..., :nf] = U[..., :nf]
    t = np.arange(nt) * dt
    return np.fft.irfft(full, n=nt, axis=-1) / dt * np.exp(damp * t)
