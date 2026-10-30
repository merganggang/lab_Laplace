"""2-D acoustic finite-difference time-domain modeling, the reference for the Laplace-domain lab (import only).

Wave equation with constant density, wavefield u (pressure), velocity v, source term q:

    (1 / v^2) d2u/dt2 = d2u/dx2 + d2u/dz2 + q,   q = w(t) delta(x - x_s) delta(z - z_s)

Second-order central difference in time, 2nd- or 4th-order central difference in space:

    u^{n+1} = 2 u^n - u^{n-1} + dt^2 v^2 (L u^n + q^n),   q^n = w(n dt) / h^2 at the source node

Grid: u[i, j] = u(x_i, z_j), x_i = i h, z_j = j h, shape (nx, nz); i is x (distance), j is z (depth).
The velocity v[i, j] lives on the same nodes. Boundaries (left, right, bottom; the top is a free
surface u = 0 unless top="absorb"):

    dirichlet   u = 0 on the edge                       reflection coefficient -1
    pml         convolutional PML of nb nodes for the second-order equation (Roden & Gedney 2000;
                Komatitsch & Martin 2007; Pasalic & McGarry 2010): the same stretch d/dx -> (1/gamma_x) d/dx
                as in laplacedomain.py (gamma_x = 1 + d_x / s there), which in time needs the memory variables psi and zeta

pml adds nb nodes outside the model on the absorbing sides; the outer edge is u = 0.
laplacedomain.py solves the same equation one damping constant at a time with the same grid and layout;
compare_laplace.py transforms the traces of propagate() with exp(-s t) and compares the two.

    ricker(t, f0, t0)                   source wavelet w(t)
    second_diffs(u, h, order, free_top) d2u/dx2 and d2u/dz2, 2nd- or 4th-order
    laplacian(u, h, order, free_top)    their sum: 5-point (order 2) or 9-point cross (order 4)
    propagate(v, h, dt, nt, src, wav, rec, ...)   time stepping, returns traces and snapshots
    analytic_2d(r, t, v, f0, t0)        homogeneous whole-space solution for the same source
    stable_courant(order, ndim)         largest stable C = v_max dt / h
"""

import math

import numpy as np

# 4th-order second derivative: (-1/12, 4/3, -5/2, 4/3, -1/12) / h^2
C4 = (-5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0)
# 4th-order staggered first derivative to half nodes: (9/8 (u[i+1] - u[i]) - 1/24 (u[i+2] - u[i-1])) / h
S4 = (9.0 / 8.0, -1.0 / 24.0)


def ricker(t, f0, t0=None):
    """Ricker wavelet w(t) = (1 - 2 a) exp(-a), a = (pi f0 (t - t0))^2 ; peak frequency f0, delay t0."""
    t0 = 1.2 / f0 if t0 is None else t0
    a = (np.pi * f0 * (np.asarray(t) - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def stable_courant(order=2, ndim=2):
    """Largest C = v_max dt / h for leapfrog in time: C^2 * ndim * max(Q) <= 1,
    max(Q) = 1 for order 2 and 4/3 for order 4."""
    qmax = 1.0 if order == 2 else 4.0 / 3.0
    return 1.0 / math.sqrt(ndim * qmax)


def second_diffs(u, h, order=2, free_top=True):
    """d2u/dx2 and d2u/dz2 at the nodes (0 where the stencil leaves the grid).

    order 4 uses the 4th-order stencil from the second ring inward and the 2nd-order stencil on
    the second ring. With free_top the row j = 0 is u = 0 and the odd mirror u[:, -1] = -u[:, 1]
    keeps the 4th-order stencil at j = 1."""
    uxx, uzz = np.zeros_like(u), np.zeros_like(u)
    uxx[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / h ** 2
    uzz[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / h ** 2
    if order == 4:
        a0, a1, a2 = C4
        uxx[2:-2, :] = (a0 * u[2:-2, :] + a1 * (u[3:-1, :] + u[1:-3, :]) + a2 * (u[4:, :] + u[:-4, :])) / h ** 2
        uzz[:, 2:-2] = (a0 * u[:, 2:-2] + a1 * (u[:, 3:-1] + u[:, 1:-3]) + a2 * (u[:, 4:] + u[:, :-4])) / h ** 2
        if free_top:
            uzz[:, 1] = (a0 * u[:, 1] + a1 * (u[:, 2] + u[:, 0]) + a2 * (u[:, 3] - u[:, 1])) / h ** 2
    return uxx, uzz


def laplacian(u, h, order=2, free_top=True):
    """5-point (order 2) or 9-point cross (order 4) Laplacian; the outer ring is reset by the boundary."""
    uxx, uzz = second_diffs(u, h, order, free_top)
    return uxx + uzz


def _diff_half(u, h, axis, order):
    """First derivative at half nodes i + 1/2 along axis (length n - 1); 4th order away from the ends."""
    u = np.moveaxis(u, axis, 0)
    d = (u[1:] - u[:-1]) / h
    if order == 4 and u.shape[0] > 3:
        b1, b2 = S4
        d[1:-1] = (b1 * (u[2:-1] - u[1:-2]) + b2 * (u[3:] - u[:-3])) / h
    return np.moveaxis(d, 0, axis)


def _diff_node(p, h, axis, order):
    """First derivative at the nodes from half-node values p (length n - 1 along axis); 0 at the ends."""
    p = np.moveaxis(p, axis, 0)
    d = np.zeros((p.shape[0] + 1,) + p.shape[1:], p.dtype)
    d[1:-1] = (p[1:] - p[:-1]) / h
    if order == 4 and p.shape[0] > 3:
        b1, b2 = S4
        d[2:-2] = (b1 * (p[2:-1] - p[1:-2]) + b2 * (p[3:] - p[:-3])) / h
    return np.moveaxis(d, 0, axis)


class Layout:
    """Model grid inside a computational grid with nb extra nodes on the absorbing sides.

    Shared with laplacedomain.py: bc "pml" adds the layer, "dirichlet" and "ce" do not."""

    def __init__(self, nx, nz, bc, top, nb):
        self.pad = nb if bc == "pml" else 0
        self.ptop = self.pad if top == "absorb" else 0
        self.nx, self.nz = nx, nz
        self.NX, self.NZ = nx + 2 * self.pad, nz + self.pad + self.ptop

    def extend(self, a):
        """Model array (nx, nz) -> computational array, edge values copied into the layers."""
        return np.pad(a, ((self.pad, self.pad), (self.ptop, self.pad)), mode="edge")

    def inner(self, a):
        """Computational array -> model part (nx, nz)."""
        return a[self.pad:self.pad + self.nx, self.ptop:self.ptop + self.nz]

    def index(self, ij):
        """Model node indices (N, 2) -> computational indices."""
        ij = np.atleast_2d(ij)
        return ij[:, 0] + self.pad, ij[:, 1] + self.ptop

    def depth(self, k, axis):
        """Distance into the layer in nodes at (fractional) node positions k along axis; 0 inside the model."""
        n, lo, hi = (self.NX, self.pad, self.pad) if axis == 0 else (self.NZ, self.ptop, self.pad)
        k = np.asarray(k, float)
        d = np.zeros_like(k)
        if lo:
            d = np.maximum(d, lo - k)
        if hi:
            d = np.maximum(d, k - (n - 1 - hi))
        return d


def damping(layout, h, vmax, k, axis, R=1e-3, power=2):
    """d(x) = d0 (dist / L)^power at node positions k along axis, d0 = (power + 1) vmax ln(1/R) / (2 L), L = nb h.

    The same profile is the PML d_x, d_z here (CPML) and in laplacedomain.py (1/s)."""
    L = max(layout.pad, 1) * h
    d0 = (power + 1) * vmax * math.log(1.0 / R) / (2.0 * L)
    return d0 * (layout.depth(k, axis) * h / L) ** power


class CPML:
    """Convolutional PML for the second-order acoustic equation.

    In the layer d/dx becomes (1/s_x) d/dx with s_x = kappa + d_x / (alpha_x + i omega), kappa = 1
    (alpha_x = 0 and i omega = s give the gamma_x of laplacedomain.py). In time, multiplying by 1/s_x is a convolution
    with -d_x exp(-(d_x + alpha_x) t), updated recursively with b = exp(-(d + alpha) dt),
    a = d / (d + alpha) (b - 1):

        psi_x  = b psi_x  + a du/dx                       at half nodes i + 1/2
        zeta_x = b zeta_x + a (d2u/dx2 + dpsi_x/dx)       at the nodes
        d2u/dx~2 = d2u/dx2 + dpsi_x/dx + zeta_x

    and the same along z. alpha falls from alpha_max = pi f0 at the inner edge to 0 at the outer
    edge, which damps the low-frequency and grazing waves the plain PML lets through."""

    def __init__(self, layout, h, dt, vmax, f0, order, dtype=np.float32):
        self.h, self.order = h, order
        L = max(layout.pad, 1)
        pads = {0: (layout.pad, layout.pad), 1: (layout.ptop, layout.pad)}
        size = {0: layout.NX, 1: layout.NZ}
        self.slabs = []   # (axis, node slice, a/b at nodes, a/b at half nodes, psi, zeta)
        for axis in (0, 1):
            n, (lo, hi) = size[axis], pads[axis]
            ab = {}
            for half in (False, True):
                k = np.arange(n - 1) + 0.5 if half else np.arange(n, dtype=float)
                d = damping(layout, h, vmax, k, axis)
                frac = layout.depth(k, axis) / L
                alpha = np.where(d > 0, np.pi * f0 * (1.0 - frac), 0.0)
                b = np.exp(-(d + alpha) * dt)
                a = np.where(d > 0, d / np.maximum(d + alpha, 1e-30) * (b - 1.0), 0.0)
                ab[half] = (a, b)
            # the memory variables are 0 outside the layer; compute them on a slab of the layer
            # plus 4 nodes, enough for the 4th-order half-node stencils to reach 0
            for k0, k1 in (((0, lo + 4),) if lo else ()) + (((n - hi - 4, n),) if hi else ()):
                other = layout.NZ if axis == 0 else layout.NX
                shape = (-1, 1) if axis == 0 else (1, -1)
                an, bn = (x[k0:k1].reshape(shape).astype(dtype) for x in ab[False])
                ah, bh = (x[k0:k1 - 1].reshape(shape).astype(dtype) for x in ab[True])
                m = k1 - k0
                psi = np.zeros((m - 1, other) if axis == 0 else (other, m - 1), dtype)
                zeta = np.zeros((m, other) if axis == 0 else (other, m), dtype)
                sl = (slice(k0, k1), slice(None)) if axis == 0 else (slice(None), slice(k0, k1))
                self.slabs.append([axis, sl, an, bn, ah, bh, psi, zeta])

    def extra(self, u, uxx_uzz):
        """Correction added to the Laplacian: sum over x, z of dpsi/dx + zeta, nonzero only in the layer."""
        out = np.zeros_like(u)
        for s in self.slabs:
            axis, sl, an, bn, ah, bh, psi, zeta = s
            psi[:] = bh * psi + ah * _diff_half(u[sl], self.h, axis, self.order)
            dpsi = _diff_node(psi, self.h, axis, self.order)
            zeta[:] = bn * zeta + an * (uxx_uzz[axis][sl] + dpsi)
            out[sl] += dpsi + zeta
        return out


def propagate(v, h, dt, nt, src, wav, rec, order=2, bc="pml", top="free", nb=20, f0=None, snap_every=0,
              dtype=np.float32, laplace_s=None):
    """Time stepping of the 2-D acoustic wave equation.

    v      : velocity (nx, nz) on the model nodes (m/s)
    h, dt  : grid spacing (m), time step (s) ; nt time samples t_n = n dt, n = 0 .. nt - 1
    src    : source nodes (ns, 2) as (i, j) ; wav : source time functions (ns, nt) sampled at t_n
    rec    : receiver nodes (nr, 2) as (i, j)
    order  : 2 or 4 (spatial accuracy) ; bc : "dirichlet" or "pml" (left, right, bottom)
    top    : "free" (u = 0 at j = 0) or "absorb" (same as bc) ; nb : layer thickness in nodes
    f0     : peak frequency for the CPML alpha (Hz)
    snap_every : keep the model part of u every that many steps (0: none)
    laplace_s  : damping constants s (1/s): also accumulate the Laplace transform of the whole field,
                 sum_n u(t_n) exp(-s t_n) dt over the model part, while stepping (no snapshots kept)

    Returns dict: traces (nr, nt), snaps (list of (n, u)), laplace (n_s, nx, nz) if laplace_s, layout.
    u^0 = u^{-1} = 0: the medium is at rest before the source starts. All sources fire together."""
    if bc not in ("dirichlet", "pml"):
        raise ValueError(f"unknown bc {bc!r}")
    v = np.asarray(v, float)
    lay = Layout(*v.shape, bc, top, nb)
    vv = lay.extend(v).astype(dtype)
    v2dt2 = (vv * dt) ** 2
    si, sj = lay.index(src)
    ri, rj = lay.index(rec)
    wav = np.atleast_2d(np.asarray(wav, dtype))
    free_top = top == "free"
    cpml = CPML(lay, h, dt, float(v.max()), f0 or 10.0, order, dtype=dtype) if bc == "pml" else None

    u_prev = np.zeros((lay.NX, lay.NZ), dtype)
    u = np.zeros_like(u_prev)
    traces = np.zeros((len(ri), nt), dtype)
    snaps = []
    src_scale = v2dt2[si, sj] / h ** 2   # point source: delta(x) delta(z) -> 1 / h^2 at one node

    svals = None if laplace_s is None else np.atleast_1d(np.asarray(laplace_s, float))
    lap_field = None if svals is None else np.zeros((len(svals),) + v.shape)

    for n in range(nt):
        traces[:, n] = u[ri, rj]
        if svals is not None:
            inner = lay.inner(u)
            for k, s in enumerate(svals):
                lap_field[k] += math.exp(-s * n * dt) * dt * inner
        if snap_every and n % snap_every == 0:
            snaps.append((n, lay.inner(u).copy()))
        if n == nt - 1:
            break
        if cpml is None:
            lap = laplacian(u, h, order, free_top)
        else:
            uxx, uzz = second_diffs(u, h, order, free_top)
            lap = uxx + uzz + cpml.extra(u, (uxx, uzz))
        rhs = v2dt2 * lap
        np.add.at(rhs, (si, sj), src_scale * wav[:, n])
        u_next = 2.0 * u - u_prev + rhs
        u_next[0, :] = u_next[-1, :] = u_next[:, -1] = 0.0   # u = 0 on the outer edge
        u_next[:, 0] = 0.0                                    # free surface or outer edge of the top layer
        u_prev, u = u, u_next

    out = {"traces": traces, "snaps": snaps, "layout": lay}
    if svals is not None:
        out["laplace"] = lap_field
    return out


def analytic_2d(r, t, v, f0, t0=None, n_theta=4000):
    """Homogeneous whole-space solution of (1/v^2) u_tt = lap u + w(t) delta(x) delta(z) at distance r.

    Green's function of the 2-D wave equation convolved with w; with tau = (r/v) cosh(theta):
        u(r, t) = (1 / 2 pi) * integral_0^{acosh(v t / r)} w(t - (r/v) cosh theta) d theta
    The 2-D Green's function has a tail: u does not return to 0 right after the pulse passes.
    laplacedomain.analytic_2d_laplace is its Laplace transform for an impulse source."""
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    ok = v * t > r
    th_max = np.arccosh(v * t[ok] / r)
    s = np.linspace(0.0, 1.0, n_theta)
    th = th_max[:, None] * s[None, :]
    arg = t[ok, None] - (r / v) * np.cosh(th)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    out[ok] = trapz(ricker(arg, f0, t0), th, axis=1) / (2.0 * np.pi)
    return out
