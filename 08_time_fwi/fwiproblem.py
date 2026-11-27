"""Time-domain full waveform inversion as a geoinv.Problem (import only).

Model m: velocity of every model node in km/s, vector order k = i * nz + j (v.ravel()).
Data u(m): traces of every shot, shape (nshot, nrec, nt) flattened, from propagator.propagate.
Gradient J^T r: adjoint-state method (propagator.gradient), the forward fields kept from forward(m).

    TimeFWI(...)                 forward, gradient, illumination (pseudo-Hessian diagonal), fixed_mask
    marmousi(h), anomaly()       true models (m/s)
    initial(v, kind)             linear (in depth) or smooth starting model
    acquisition(nx, nshot)       sources and receivers one node below the free surface
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))     # geoinv when PYTHONPATH is not set (the container sets it)
import geoinv
import propagator as P

VMAX = 5500.0                         # CPML damping and the time step use this velocity, not v.max()
WATER = 144.0                         # water added on top of Marmousi (m): sources and receivers sit in water
V_WATER = 1500.0


class TimeFWI(geoinv.Problem):
    """Acoustic FWI on a fixed grid and acquisition; m in km/s.

    nx, nz, h      : model grid (nodes, m) ; dt, nt : time step (s) and samples
    src (S, 2)     : source nodes ; rec (R, 2) : receiver nodes ; wavelet (nt,) : source time function
    order, nb      : spatial order (4 or 8) and CPML thickness (nodes)
    batch          : shots stepped together; None picks the most whose stored fields fill a third of the
                     free memory (GPU) or 3 GiB (CPU)
    cache_gib      : forward fields kept for the gradient up to this size; beyond it gradient recomputes them
    fixed_mask     : not updated: the free-surface row and the rows down to fixed_depth (the gradient is
                     singular at the source and receiver nodes), and the model edge (left, right, bottom):
                     the PML copies the edge velocity, and the adjoint run is the exact transpose only inside."""

    def __init__(self, nx, nz, h, dt, nt, src, rec, wavelet, order=4, nb=10, f0=3.0, batch=None,
                 cache_gib=None, fixed_depth=2, dtype=torch.float32, dev=None):
        self.nx, self.nz, self.h, self.dt, self.nt = nx, nz, h, dt, nt
        self.src, self.rec = np.asarray(src), np.asarray(rec)
        self.wavelet = np.asarray(wavelet, float)
        self.order, self.nb, self.f0 = order, nb, f0
        self.dtype = dtype
        self.dev = dev or P.device()
        self.n_model = nx * nz
        mask = np.zeros((nx, nz), bool)
        mask[:, :fixed_depth + 1] = mask[0, :] = mask[-1, :] = mask[:, -1] = True
        self.fixed_mask = mask.ravel()
        if cache_gib is None:
            if self.dev.type == "cuda":
                cache_gib = 0.7 * torch.cuda.mem_get_info()[0] / 2 ** 30
            else:
                cache_gib = 6.0
        self.cache_gib = cache_gib
        if batch is None:
            free_gib = torch.cuda.mem_get_info()[0] / 2 ** 30 if self.dev.type == "cuda" else 9.0
            batch = max(1, int(free_gib / 3 / (self.field_gib() / len(self.src))))
        self.batch = min(batch, len(self.src))
        self._cache = None             # (m, list of fields per batch)
        self.n_prop = 0                # propagations (shots x runs), forward and adjoint

    def _v(self, m):
        return np.asarray(m, float).reshape(self.nx, self.nz) * 1000.0

    def _batches(self):
        S = len(self.src)
        return [np.arange(b, min(b + self.batch, S)) for b in range(0, S, self.batch)]

    def field_gib(self):
        nbytes = torch.finfo(self.dtype).bits // 8
        return len(self.src) * self.nt * self.nx * self.nz * nbytes / 2 ** 30

    def _run(self, v, idx, store):
        wav = np.broadcast_to(self.wavelet, (len(idx), 1, self.nt))
        res = P.propagate(v, self.h, self.dt, self.nt, self.src[idx][:, None], wav, self.rec, self.order, self.nb,
                          self.f0, VMAX, store=store, dtype=self.dtype, dev=self.dev)
        self.n_prop += len(idx)
        return res

    def forward(self, m):
        """Traces (S, R, nt) flattened; keeps the forward fields when they fit in cache_gib."""
        v = self._v(m)
        keep = self.field_gib() <= self.cache_gib
        self._cache = None
        fields, out = [], []
        for idx in self._batches():
            res = self._run(v, idx, keep)
            out.append(res["traces"].double().cpu().numpy())
            if keep:
                fields.append(res["field"])
        if keep:
            self._cache = (np.array(m, float, copy=True), fields)
        return np.concatenate(out).ravel()

    def _fields(self, m):
        """Forward fields per batch: the kept ones if m is the last forward model, else recomputed one by one."""
        if self._cache is not None and np.array_equal(self._cache[0], m):
            yield from zip(self._batches(), self._cache[1])
        else:
            v = self._v(m)
            for idx in self._batches():
                yield idx, self._run(v, idx, True)["field"]

    def gradient(self, m, r):
        """J^T r with respect to m (km/s): adjoint run per batch, cross-correlation, sum over shots."""
        v = self._v(m)
        r = np.asarray(r, float).reshape(len(self.src), len(self.rec), self.nt)
        g = torch.zeros((self.nx, self.nz), dtype=self.dtype, device=self.dev)
        for idx, field in self._fields(m):
            g += P.gradient(v, self.h, self.dt, self.nt, field, self.rec, r[idx], self.order, self.nb, self.f0,
                            VMAX, dtype=self.dtype, dev=self.dev)
            self.n_prop += len(idx)
        return 1000.0 * g.double().cpu().numpy().ravel()          # dE/dv (per m/s) -> per km/s

    def illumination(self, m):
        """Pseudo-Hessian diagonal H_P (Shin et al. 2001): sum over shots and time of the virtual source squared,
        (2 / v^3 d2u/dt2)^2. Large near the sources, small at depth (geometrical spreading)."""
        v = torch.as_tensor(self._v(m), dtype=self.dtype, device=self.dev)
        H = torch.zeros_like(v)
        for idx, f in self._fields(m):
            for n in range(1, self.nt - 1):
                H += ((f[n + 1] - 2.0 * f[n] + f[n - 1]) ** 2).sum(0)
        H *= (2.0 / (v ** 3 * self.dt ** 2)) ** 2
        return H.double().cpu().numpy().ravel()


def time_axis(h, tmax, courant=0.3):
    """dt = courant * h / VMAX and the number of samples for a record of length tmax."""
    dt = courant * h / VMAX
    return dt, int(round(tmax / dt)) + 1


def marmousi(h=48.0, water=WATER):
    """Marmousi P velocity (Versteeg 1994) on a 24 m or 48 m grid (m/s), shape (nx, nz), with water (m) added on
    top of its own thin water layer (about 40 m): sources and receivers one node below the surface are then in
    water, and the water rows are known and fixed, as in a marine survey."""
    path = HERE / "data" / "marmousi_24m.npy"
    if not path.is_file():
        sys.exit(f"{path} not found: copy the complete 08_time_fwi folder, including data/")
    v = np.load(path).astype(float)
    k = int(round(h / 24.0))
    v = v[::k, ::k]
    return np.pad(v, ((0, 0), (int(round(water / h)), 0)), constant_values=V_WATER)


def water_rows(v):
    """Number of top rows that are water everywhere."""
    n = 0
    while n < v.shape[1] and np.all(v[:, n] == V_WATER):
        n += 1
    return n


def anomaly(nx=128, nz=64, h=48.0, v0=1800.0, grad=0.5, dv=250.0, radius=300.0):
    """Background v0 + grad z (m/s; grad = 0 is homogeneous) with a fast and a slow disc at 1.2 km depth.
    With grad > 0 diving waves turn below the discs and carry their traveltime to the surface receivers."""
    x, z = np.meshgrid(np.arange(nx) * h, np.arange(nz) * h, indexing="ij")
    v = v0 + grad * z
    for xc, sign in ((nx * h / 3, 1.0), (2 * nx * h / 3, -1.0)):
        v[(x - xc) ** 2 + (z - 1200.0) ** 2 <= radius ** 2] += sign * dv
    return v


def initial(v_true, h, kind, sigma_m=300.0, known_rows=3, grad=0.5):
    """Starting model: "linear" is v(z) = v_top + grad (z - z_top) below the known rows (v_top: mean of the first
    free row), slower than the true model at depth,
    "smooth" is v_true smoothed with a Gaussian of sigma_m (m), "background" is the most common value of
    each row (the anomaly model without its discs). The top known_rows rows (water and the fixed rows down to
    the sources and receivers) are taken as known, as a water layer from bathymetry would be."""
    nx, nz = v_true.shape
    if kind == "linear":
        top = float(v_true[:, known_rows].mean())
        z = (np.arange(nz) - known_rows).clip(0) * h
        v = np.broadcast_to(top + grad * z, (nx, nz)).copy()
    elif kind == "smooth":
        v = gaussian_filter(v_true, sigma_m / h, mode="nearest")
    elif kind == "background":
        v = np.empty_like(v_true)
        for j in range(nz):
            vals, counts = np.unique(v_true[:, j], return_counts=True)
            v[:, j] = vals[counts.argmax()]
    else:
        raise ValueError(kind)
    v[:, :known_rows] = v_true[:, :known_rows]   # fixed rows keep their true values
    return v


def acquisition(nx, nshot, depth_node=1):
    """nshot sources evenly spread along the surface and a receiver at every node, depth_node below the surface."""
    xs = np.round(np.linspace(2, nx - 3, nshot)).astype(int)
    src = np.column_stack([xs, np.full(nshot, depth_node)])
    rec = np.column_stack([np.arange(nx), np.full(nx, depth_node)])
    return src, rec
