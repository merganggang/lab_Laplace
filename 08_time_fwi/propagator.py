"""2-D acoustic time stepping with many shots at once on the GPU (PyTorch), forward and adjoint (import only).

Wave equation with constant density, wavefield u, velocity v, source term q:

    (1 / v^2) d2u/dt2 = d2u/dx2 + d2u/dz2 + q,   q = w(t) delta(x - x_s) delta(z - z_s)

Second-order central difference in time, 4th- or 8th-order central difference in space:

    u^{n+1} = 2 u^n - u^{n-1} + dt^2 v^2 (L u^n + q^n),   q^n = w(n dt) / h^2 at the source node

Grid: u[b, i, j] = u(x_i, z_j) for shot b, i is x (distance), j is z (depth), shape (B, nx, nz).
The spatial stencil is exactly symmetric up to the edges (zeros outside, odd mirror at the free surface
j = 0), so the same update run backward in time is the transpose (the adjoint) inside the model.
Boundaries: free surface on top, CPML of nb nodes on the left, right and bottom (Roden & Gedney 2000;
Komatitsch & Martin 2007), the damping fixed by vmax so that it does not move when v changes.
The PML layers copy the edge velocity of the model.

    device(name)                         "auto": cuda if a GPU is visible, else cpu
    ricker(t, f0, t0)                    source wavelet w(t)
    lowpass(x, fc, dt)                   causal Butterworth low-pass along the last axis (data and wavelet alike)
    propagate(v, h, dt, nt, src, wav, rec, ...)   time stepping of a batch of shots
    gradient(v, h, dt, nt, field, rec, resid, ...) adjoint run and the zero-lag cross-correlation dE/dv
"""

import math
import os

import numpy as np
import torch
from scipy.signal import butter, sosfilt

# second derivative: (c0 u[i] + sum_k c_k (u[i+k] + u[i-k])) / h^2
COEF = {4: (-5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0),
        8: (-205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)}
# 4th-order staggered first derivative to half nodes (CPML memory variables)
S4 = (9.0 / 8.0, -1.0 / 24.0)


def device(name="auto"):
    """torch.device: "auto" picks the GPU if one is visible. CPU threads follow LAB_CPUS (container CPU quota)."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cpu":
        torch.set_num_threads(int(os.environ.get("LAB_CPUS", os.environ.get("OMP_NUM_THREADS", "8"))))
    return torch.device(name)


def ricker(t, f0, t0=None):
    """Ricker wavelet w(t) = (1 - 2 a) exp(-a), a = (pi f0 (t - t0))^2 ; peak frequency f0, delay t0."""
    t0 = 1.2 / f0 if t0 is None else t0
    a = (np.pi * f0 * (np.asarray(t) - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def lowpass(x, fc, dt, order=6):
    """Causal Butterworth low-pass with corner fc (Hz) along the last axis.

    Causal, so nothing leaks before t = 0. The wave equation is linear and time invariant: filtering the
    source wavelet gives the same data as filtering the modeled traces, so the observed data and the
    wavelet get the same filter and the adjoint needs no change."""
    sos = butter(order, fc, fs=1.0 / dt, output="sos")
    return sosfilt(sos, np.asarray(x, float), axis=-1)


def stable_courant(order=4):
    """Largest C = v_max dt / h in 2-D for leapfrog in time: C^2 * 2 * max(-symbol) <= 4."""
    c = COEF[order]
    qmax = -(c[0] + 2 * sum(ck * (-1) ** k for k, ck in enumerate(c[1:], 1)))
    return 2.0 / math.sqrt(2.0 * qmax)


class Layout:
    """Model grid (nx, nz) inside the computational grid: nb PML nodes left, right and below, free surface on top."""

    def __init__(self, nx, nz, nb):
        self.nx, self.nz, self.nb = nx, nz, nb
        self.NX, self.NZ = nx + 2 * nb, nz + nb

    def extend(self, a):
        """Model array (nx, nz) -> computational array; the layers copy the edge values."""
        return torch.nn.functional.pad(a[None, None], (0, self.nb, self.nb, self.nb), mode="replicate")[0, 0]

    def inner(self, a):
        """Computational array (..., NX, NZ) -> model part (..., nx, nz)."""
        return a[..., self.nb:self.nb + self.nx, :self.nz]

    def flat(self, ij):
        """Model node indices (..., 2) -> flat computational indices i * NZ + j."""
        ij = torch.as_tensor(np.array(ij, copy=True), dtype=torch.long)
        return (ij[..., 0] + self.nb) * self.NZ + ij[..., 1]


def _tensor(x, dtype, dev):
    """NumPy array or tensor -> tensor on dev."""
    if isinstance(x, torch.Tensor):
        return x.to(device=dev, dtype=dtype)
    return torch.as_tensor(np.array(x, copy=True), dtype=dtype, device=dev)


def second_diffs(u, h, order):
    """d2u/dx2 and d2u/dz2 of u (B, NX, NZ): zeros outside the grid, odd mirror u[-k] = -u[k] above j = 0."""
    c = COEF[order]
    r = len(c) - 1
    B, NX, NZ = u.shape
    p = torch.nn.functional.pad(u, (r, r, r, r))
    p[:, :, :r] = -torch.flip(p[:, :, r + 1:2 * r + 1], dims=[2])
    mid = p[:, r:r + NX, r:r + NZ]
    uxx, uzz = c[0] * mid, c[0] * mid
    for k, ck in enumerate(c[1:], 1):
        uxx = uxx + ck * (p[:, r + k:r + k + NX, r:r + NZ] + p[:, r - k:r - k + NX, r:r + NZ])
        uzz = uzz + ck * (p[:, r:r + NX, r + k:r + k + NZ] + p[:, r:r + NX, r - k:r - k + NZ])
    return uxx / h ** 2, uzz / h ** 2


def _diff_half(u, h, dim):
    """First derivative at half nodes along dim (length n - 1); 4th order away from the ends."""
    u = torch.movedim(u, dim, 0)
    d = (u[1:] - u[:-1]) / h
    if u.shape[0] > 3:
        d[1:-1] = (S4[0] * (u[2:-1] - u[1:-2]) + S4[1] * (u[3:] - u[:-3])) / h
    return torch.movedim(d, 0, dim)


def _diff_node(p, h, dim):
    """First derivative at the nodes from half-node values p (length n - 1 along dim); 0 at the ends."""
    p = torch.movedim(p, dim, 0)
    d = torch.zeros((p.shape[0] + 1,) + p.shape[1:], dtype=p.dtype, device=p.device)
    d[1:-1] = (p[1:] - p[:-1]) / h
    if p.shape[0] > 3:
        d[2:-2] = (S4[0] * (p[2:-1] - p[1:-2]) + S4[1] * (p[3:] - p[:-3])) / h
    return torch.movedim(d, 0, dim)


class CPML:
    """Convolutional PML for the second-order acoustic equation (as in the time-domain modeling lab).

    In the layer d/dx becomes (1/s_x) d/dx with s_x = 1 + d_x / (alpha_x + i omega). In time, 1/s_x is a
    convolution updated recursively with b = exp(-(d + alpha) dt), a = d / (d + alpha) (b - 1):

        psi_x  = b psi_x  + a du/dx                       at half nodes i + 1/2
        zeta_x = b zeta_x + a (d2u/dx2 + dpsi_x/dx)       at the nodes
        d2u/dx~2 = d2u/dx2 + dpsi_x/dx + zeta_x

    d(x) = d0 (dist / L)^2, d0 = 3 vmax ln(1/R) / (2 L), L = nb h; alpha falls from pi f0 to 0 across the layer."""

    def __init__(self, lay, h, dt, vmax, f0, B, dtype, dev, R=1e-3):
        self.h = h
        L = lay.nb * h
        d0 = 3.0 * vmax * math.log(1.0 / R) / (2.0 * L)
        self.slabs = []
        for dim, n, lo, hi in ((1, lay.NX, lay.nb, lay.nb), (2, lay.NZ, 0, lay.nb)):
            ab = {}
            for half in (False, True):
                k = np.arange(n - 1 if half else n) + (0.5 if half else 0.0)
                dist = np.zeros_like(k)
                if lo:
                    dist = np.maximum(dist, lo - k)
                if hi:
                    dist = np.maximum(dist, k - (n - 1 - hi))
                frac = dist / lay.nb
                d = d0 * frac ** 2
                alpha = np.where(d > 0, np.pi * f0 * (1.0 - frac), 0.0)
                b = np.exp(-(d + alpha) * dt)
                a = np.where(d > 0, d / np.maximum(d + alpha, 1e-30) * (b - 1.0), 0.0)
                ab[half] = (a, b)
            # memory variables are 0 outside the layer: keep them on the layer plus 4 nodes
            for k0, k1 in (((0, lo + 4),) if lo else ()) + (((n - hi - 4, n),) if hi else ()):
                shape = (-1, 1) if dim == 1 else (1, -1)
                t = lambda x: torch.as_tensor(x.reshape(shape), dtype=dtype, device=dev)
                an, bn = (t(x[k0:k1]) for x in ab[False])
                ah, bh = (t(x[k0:k1 - 1]) for x in ab[True])
                m = k1 - k0
                other = lay.NZ if dim == 1 else lay.NX
                psi = torch.zeros((B, m - 1, other) if dim == 1 else (B, other, m - 1), dtype=dtype, device=dev)
                zeta = torch.zeros((B, m, other) if dim == 1 else (B, other, m), dtype=dtype, device=dev)
                sl = (slice(None), slice(k0, k1), slice(None)) if dim == 1 else (slice(None), slice(None), slice(k0, k1))
                self.slabs.append([dim, sl, an, bn, ah, bh, psi, zeta])

    def extra(self, u, uxx, uzz):
        """Correction added to the Laplacian, nonzero only in the layers."""
        out = torch.zeros_like(u)
        for s in self.slabs:
            dim, sl, an, bn, ah, bh, psi, zeta = s
            psi.copy_(bh * psi + ah * _diff_half(u[sl], self.h, dim))      # in place: CUDA Graph replays
            dpsi = _diff_node(psi, self.h, dim)
            zeta.copy_(bn * zeta + an * ((uxx if dim == 1 else uzz)[sl] + dpsi))
            out[sl] += dpsi + zeta
        return out


def propagate(v, h, dt, nt, src, wav, rec, order=4, nb=10, f0=3.0, vmax=None, store=False, on_step=None,
              dtype=torch.float32, dev=None, graph=True):
    """Time stepping of B shots at once.

    v      : velocity (nx, nz) on the model nodes (m/s), NumPy or torch
    h, dt  : grid spacing (m), time step (s) ; nt time samples t_n = n dt, n = 0 .. nt - 1
    src    : source nodes (B, K, 2) as (i, j) ; wav : source time functions (B, K, nt) sampled at t_n
             (K = 1 for shots; K = number of receivers for the adjoint run)
    rec    : receiver nodes (R, 2), shared by all shots
    order  : 4 or 8 (spatial accuracy) ; nb : CPML thickness (nodes) ; f0 : frequency for the CPML alpha
    vmax   : velocity for the CPML damping (fixed by the caller; default v.max())
    store  : keep the model part of u^n for every n: field (nt, B, nx, nz) on the device
    on_step: optional function (n, u) called with u^n on the computational grid before each update
    graph  : on the GPU, record one time step as a CUDA Graph and replay it. A step is about a hundred
             small array operations; launched one by one they take longer than the arithmetic itself.

    Returns dict: traces (B, R, nt), field, layout. u^0 = u^{-1} = 0: the medium is at rest."""
    dev = dev or device()
    v = _tensor(v, dtype, dev)
    nx, nz = v.shape
    lay = Layout(nx, nz, nb)
    vmax = float(v.max()) if vmax is None else float(vmax)
    v2dt2 = (lay.extend(v) * dt) ** 2
    wav = _tensor(wav, dtype, dev)
    B, K = wav.shape[:2]
    sidx = lay.flat(src).to(dev).reshape(B, K)
    sscale = v2dt2.reshape(-1)[sidx] / h ** 2          # point source: delta(x) delta(z) -> 1 / h^2 at one node
    ridx = lay.flat(rec).to(dev)
    cpml = CPML(lay, h, dt, vmax, f0, B, dtype, dev)

    u_prev = torch.zeros((B, lay.NX, lay.NZ), dtype=dtype, device=dev)
    u = torch.zeros_like(u_prev)
    wav_n = torch.zeros((B, K), dtype=dtype, device=dev)

    def step():
        """u^{n+1} from u^n, u^{n-1} and wav_n, written back in place: u_prev <- u, u <- u^{n+1}."""
        uxx, uzz = second_diffs(u, h, order)
        rhs = v2dt2 * (uxx + uzz + cpml.extra(u, uxx, uzz))
        rhs.view(B, -1).scatter_add_(1, sidx, sscale * wav_n)
        u_next = 2.0 * u - u_prev + rhs
        u_next[:, 0, :] = 0.0                            # outer edges of the layers and the free surface
        u_next[:, -1, :] = 0.0
        u_next[:, :, -1] = 0.0
        u_next[:, :, 0] = 0.0
        u_prev.copy_(u)
        u.copy_(u_next)

    run = step
    if graph and dev.type == "cuda":
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                    # warm-up on zero fields leaves them zero
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        run = g.replay

    traces = torch.zeros((nt, B, len(ridx)), dtype=dtype, device=dev)
    field = torch.zeros((nt, B, nx, nz), dtype=dtype, device=dev) if store else None
    for n in range(nt):
        traces[n] = u.reshape(B, -1)[:, ridx]
        if store:
            field[n] = lay.inner(u)
        if on_step is not None:
            on_step(n, u)
        if n == nt - 1:
            break
        wav_n.copy_(wav[:, :, n])
        run()
    return {"traces": traces.permute(1, 2, 0), "field": field, "layout": lay}


def gradient(v, h, dt, nt, field, rec, resid, order=4, nb=10, f0=3.0, vmax=None, dtype=torch.float32, dev=None):
    """dE/dv (nx, nz) for E = 1/2 sum r^2, from the stored forward field and the residual r (B, R, nt).

    Adjoint run: the same time stepping, the residual reversed in time injected at the receivers
    (time-reversed adjoint u_adj^k at step k belongs to t_{nt-1-k}). Zero-lag cross-correlation with the
    second time difference of the forward field:
        dE/dv = 2 / (v^3 dt^2) * sum_shots sum_n u_adj(t_{n+1}) (u^{n+1} - 2 u^n + u^{n-1})
    which is the discrete form of  integral (2 / v^3) u_adj d2u/dt2 dt."""
    dev = dev or device()
    v = _tensor(v, dtype, dev)
    resid = _tensor(resid, dtype, dev)
    B, R, _ = resid.shape
    rec_t = torch.as_tensor(rec, dtype=torch.long)
    src = rec_t[None].expand(B, R, 2)
    lay = Layout(*v.shape, nb)
    acc = torch.zeros((B,) + tuple(v.shape), dtype=dtype, device=dev)

    def xcorr(k, ua):
        n = nt - 1 - k
        if n <= nt - 2:
            d2 = field[n + 1] - 2.0 * field[n] + (field[n - 1] if n > 0 else 0.0)
            acc.add_(lay.inner(ua) * d2)

    propagate(v, h, dt, nt, src, h ** 2 * torch.flip(resid, dims=[2]), rec, order, nb, f0, vmax,
              on_step=xcorr, dtype=dtype, dev=dev)
    return 2.0 / (v ** 3 * dt ** 2) * acc.sum(0)
