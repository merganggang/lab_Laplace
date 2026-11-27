"""Frequency-domain full waveform inversion as a geoinv.Problem (import only).

Model m: velocity of every model node in km/s, vector order k = i * nz + j (v.ravel()).
Data: complex receiver values d~ (nf, nshot, nrec), one per frequency, shot and receiver. geoinv works with
real vectors, so forward(m) returns [Re(R u~); Im(R u~)] flattened: (1/2) sum |r~|^2 is the same sum of
squares, and the gradient of the stacked real problem is g = Re[J^T conj(r~)] (lecture: 복소 잔차).

Per frequency (lecture symbols, S u~ = q~ for each shot):
    forward   u~ = S^-1 q~                       one LU of S, one solve per shot
    adjoint   u~+ = S^-1 R^T conj(r~)            the same LU: S^T = S
    gradient  g_k = -Re[(mass_k / v_k) (u~_k (W u~+)_k + (W u~)_k u~+_k)]   summed over shots and frequencies
    J p       R du~, du~ = -S^-1 [W (c u~) + c (W u~)],  c = mass p / v      (Born modeling, one solve per shot)
with S = K - (W D + D W) / 2, D = diag(mass), mass = gamma_x gamma_z omega^2 / v^2 (freqdomain.py).

Frequencies are independent: each worker process owns some frequencies, keeps their LU and forward fields,
and only the data, the gradient and J p cross between processes (lecture: 주파수 병렬).

    FreqFWI(...)                 forward, gradient, jacobian (LinearOperator), illumination, fixed_mask,
                                 wavelet estimation (srcest)
    observed(...)                time-domain traces of every shot (4th order, CPML) and their spectra
    wavelet(kind, t, f0)         source time functions: the assumed Ricker and a different "other" one
    marmousi(h), anomaly()       true models (m/s) ; initial(v, kind) ; acquisition(nx, nshot)
"""

import atexit
import math
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert
from scipy.sparse.linalg import LinearOperator

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))     # geoinv when PYTHONPATH is not set (the container sets it)
import geoinv
import freqdomain as fd
import timedomain as td

VMAX = 5500.0                         # PML damping velocity, fixed so that the PML does not move with v (m/s)
WATER = 144.0                         # water added on top of Marmousi (m): sources and receivers sit in water
V_WATER = 1500.0
SRC_DEPTH = 48.0                      # source and receiver depth (m)


def n_cpus():
    """CPU quota of the container (LAB_CPUS), else the machine, at most 8."""
    return max(1, min(int(os.environ.get("LAB_CPUS", os.cpu_count() or 1)), os.cpu_count() or 1, 8))


# ---- one process: the state of some frequencies ---------------------------------------------------------

class FreqSolver:
    """LU, mass parts and forward fields of the frequencies freqs[idx] for the current velocity."""

    def __init__(self, cfg, idx):
        self.cfg, self.idx = cfg, list(idx)
        self.v = None
        self.st = []                  # per frequency: dict S parts, lu, positions, u
        self.n_solve = 0

    def setup(self, v):
        """Matrices and LU for v (m/s, model grid); nothing to do if v is the current model."""
        v = np.asarray(v, float)
        if self.v is not None and np.array_equal(self.v, v):
            return
        c = self.cfg
        self.st = []
        for k in self.idx:
            S, lay, keep, W, mass, vv = fd.build_matrix(v, c["h"], 2 * np.pi * c["freqs"][k], c["stencil"], "pml",
                                                        "free", c["nb"], c["damp"], vmax=VMAX, parts=True)
            pos = np.full(lay.NX * lay.NZ, -1)
            pos[keep] = np.arange(len(keep))
            flat = lambda ij: (ij[:, 0] + lay.pad) * lay.NZ + ij[:, 1] + lay.ptop
            self.st.append({"lu": fd.factor(S, "mindegree"), "W": W, "mv": mass / vv, "keep": keep, "lay": lay,
                            "ps": pos[flat(c["src"])], "pr": pos[flat(c["rec"])], "u": None})
        self.v = v.copy()

    def forward(self, v, wspec):
        """Receiver data (nf_local, ns, nr) for source spectra wspec (nf_local, ns); keeps the fields u~."""
        self.setup(v)
        h, ns = self.cfg["h"], len(self.cfg["src"])
        out = []
        for s, w in zip(self.st, wspec):
            q = np.zeros((len(s["keep"]), ns), complex)
            q[s["ps"], np.arange(ns)] = w / h ** 2           # point source w~ / h^2 at one node
            s["u"] = s["lu"].solve(q)
            self.n_solve += ns
            out.append(s["u"][s["pr"]].T)
        return np.array(out)

    def scale(self, w):
        """u~ -> w u~ per frequency (w (nf_local,) or (nf_local, ns)): the fields of an estimated wavelet."""
        for s, wk in zip(self.st, w):
            s["u"] = s["u"] * wk

    def _grid(self, s, a):
        full = np.zeros(s["lay"].NX * s["lay"].NZ, a.dtype)
        full[s["keep"]] = a
        return full.reshape(s["lay"].NX, s["lay"].NZ)

    def gradient(self, r):
        """dE/dv (per m/s, model grid) for complex residuals r (nf_local, ns, nr): adjoint solve with the same LU,
        then the mass-term product summed over shots; the PML part is summed back onto the model edge."""
        g = None
        for s, rk in zip(self.st, r):
            b = np.zeros_like(s["u"])
            b[s["pr"]] = np.conj(rk).T                      # R^T conj(r~): the residual at the receivers
            a = s["lu"].solve(b)                             # u~+ = S^-1 R^T conj(r~)
            self.n_solve += b.shape[1]
            u = s["u"]
            gk = -np.real(s["mv"][:, None] * (u * (s["W"] @ a) + (s["W"] @ u) * a)).sum(1)
            gm = fd.to_model(s["lay"], self._grid(s, gk))
            g = gm if g is None else g + gm
        return g

    def jvec(self, dv):
        """R du~ (nf_local, ns, nr) for a velocity change dv (m/s, model grid): Born modeling."""
        out = []
        for s in self.st:
            dvv = s["lay"].extend(dv).ravel()[s["keep"]]
            c = (s["mv"] * dvv)[:, None]
            u = s["u"]
            du = -s["lu"].solve(s["W"] @ (c * u) + c * (s["W"] @ u))
            self.n_solve += u.shape[1]
            out.append(du[s["pr"]].T)
        return np.array(out)

    def illum(self):
        """Pseudo-Hessian diagonal (model grid): sum over shots of |2 mass_k / v_k u~_k|^2, the virtual source of
        the node itself (the neighbour weights of the 9-point mass term are left out)."""
        H = None
        for s in self.st:
            hk = (np.abs(2 * s["mv"][:, None] * s["u"]) ** 2).sum(1)
            hm = s["lay"].inner(self._grid(s, hk))
            H = hm if H is None else H + hm
        return H

    def field(self, j, shot=0, adjoint_r=None):
        """Model part of u~ (or of the adjoint field for residuals adjoint_r (ns, nr)) at local frequency j."""
        s = self.st[j]
        if adjoint_r is None:
            a = s["u"][:, shot]
        else:
            b = np.zeros_like(s["u"])
            b[s["pr"]] = np.conj(adjoint_r).T
            a = s["lu"].solve(b)[:, shot]
        return s["lay"].inner(self._grid(s, a))

    def count(self):
        return self.n_solve


def _serve(conn, cfg, idx):
    sol = FreqSolver(cfg, idx)
    while True:
        cmd, args = conn.recv()
        if cmd == "stop":
            break
        try:
            conn.send(("ok", getattr(sol, cmd)(*args)))
        except Exception:
            conn.send(("err", traceback.format_exc()))


class _Pool:
    """Frequencies split over worker processes (fork) that keep their state; workers = 1 runs in this process."""

    def __init__(self, cfg, nf, workers):
        workers = max(1, min(workers, nf))
        self.parts = [list(range(nf))[w::workers] for w in range(workers)]
        self.local = None
        self.conns, self.procs = [], []
        if workers == 1:
            self.local = FreqSolver(cfg, self.parts[0])
            return
        ctx = mp.get_context("fork")
        for idx in self.parts:
            a, b = ctx.Pipe()
            p = ctx.Process(target=_serve, args=(b, cfg, idx), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        atexit.register(self.close)

    def call(self, cmd, per_worker_args):
        """Run cmd on every worker with its own arguments; returns the list of results in worker order."""
        if self.local is not None:
            return [getattr(self.local, cmd)(*per_worker_args[0])]
        for c, a in zip(self.conns, per_worker_args):
            c.send((cmd, a))
        res = []
        for c in self.conns:
            status, val = c.recv()
            if status == "err":
                raise RuntimeError(f"worker failed in {cmd}:\n{val}")
            res.append(val)
        return res

    def close(self):
        for c in self.conns:
            try:
                c.send(("stop", ()))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=2)
        self.conns, self.procs = [], []


# ---- the inversion problem ------------------------------------------------------------------------------

class FreqFWI(geoinv.Problem):
    """Acoustic frequency-domain FWI on a fixed grid, acquisition and frequency set; m in km/s.

    freqs (nf,)     : frequencies (Hz), solved at omega - i damp (damp = alpha_d, 1/s)
    wspec (nf,)     : source spectrum w~ (the assumed wavelet), the same for every shot
    stencil, nb     : 5 or 9 (Jo et al. 1996) and PML thickness (nodes)
    fixed_rows      : top rows not updated (water, known)
    srcest, d_obs   : estimate w~ at every forward from d_obs (nf, ns, nr) with the current model (lecture:
                      송신파형 추정); E is then min over w~ and its gradient keeps the same formula
    mask (ns, nr)   : shot-receiver pairs used by the estimate (the near offsets left out of the inversion)
    fixed_mask      : the fixed rows and the model edge (left, right, bottom), whose velocity the PML copies."""

    def __init__(self, nx, nz, h, freqs, src, rec, wspec, stencil=9, nb=20, damp=0.0, fixed_rows=1,
                 srcest=False, d_obs=None, mask=None, workers=None):
        self.nx, self.nz, self.h = nx, nz, h
        self.freqs = np.atleast_1d(np.asarray(freqs, float))
        self.src, self.rec = np.asarray(src), np.asarray(rec)
        self.wspec = np.asarray(wspec, complex).reshape(len(self.freqs))
        self.srcest, self.d_obs = srcest, (None if d_obs is None else np.asarray(d_obs, complex))
        if srcest and d_obs is None:
            raise ValueError("srcest needs d_obs")
        self.mask = None if mask is None else np.asarray(mask, bool)
        self.n_model = nx * nz
        fixed = np.zeros((nx, nz), bool)
        fixed[:, :fixed_rows] = fixed[0, :] = fixed[-1, :] = fixed[:, -1] = True
        self.fixed_mask = fixed.ravel()
        self.shape = (len(self.freqs), len(self.src), len(self.rec))
        cfg = {"h": h, "freqs": self.freqs, "src": self.src, "rec": self.rec, "stencil": stencil, "nb": nb,
               "damp": damp}
        self.pool = _Pool(cfg, len(self.freqs), n_cpus() if workers is None else workers)
        self._m = None
        self.w_hat = self.wspec.copy()

    # stacked real data <-> complex (nf, ns, nr)
    def to_real(self, d):
        d = np.asarray(d, complex).ravel()
        return np.concatenate([d.real, d.imag])

    def to_complex(self, r):
        r = np.asarray(r, float)
        n = r.size // 2
        return (r[:n] + 1j * r[n:]).reshape(self.shape)

    def _v(self, m):
        return np.asarray(m, float).reshape(self.nx, self.nz) * 1000.0

    def _gather(self, parts):
        out = np.zeros(self.shape, complex)
        for idx, p in zip(self.pool.parts, parts):
            out[idx] = p
        return out

    def _split(self, a):
        return [a[idx] for idx in self.pool.parts]

    def solve(self, m, wspec=None):
        """Complex receiver data (nf, ns, nr) for model m and spectra wspec (nf,) (default: the assumed one)."""
        w = self.wspec if wspec is None else np.asarray(wspec, complex)
        wsp = np.broadcast_to(w[:, None], (len(self.freqs), len(self.src)))
        d = self._gather(self.pool.call("forward", [(self._v(m), wi) for wi in self._split(wsp)]))
        self._m = np.array(m, float, copy=True)
        return d

    def estimate_wavelet(self, G, d):
        """w~ per frequency minimizing sum |w~ G - d|^2 over shots and receivers: sum conj(G) d / sum |G|^2."""
        if self.mask is not None:
            G, d = G * self.mask, d * self.mask
        return (np.conj(G) * d).sum((1, 2)) / (np.abs(G) ** 2).sum((1, 2))

    def forward(self, m):
        if not self.srcest:
            return self.to_real(self.solve(m))
        G = self.solve(m, np.ones(len(self.freqs)))        # unit source: the Green's function responses
        self.w_hat = self.estimate_wavelet(G, self.d_obs)
        self.pool.call("scale", [(wi,) for wi in self._split(self.w_hat)])
        return self.to_real(self.w_hat[:, None, None] * G)

    def _ensure(self, m):
        if self._m is None or not np.array_equal(self._m, m):
            self.forward(m)

    def gradient(self, m, r):
        """J^T r (per km/s) for stacked real residuals r: one adjoint solve per shot and frequency."""
        self._ensure(m)
        rc = self.to_complex(r)
        g = sum(self.pool.call("gradient", [(ri,) for ri in self._split(rc)]))
        return 1000.0 * g.ravel()

    def jvec(self, m, p):
        self._ensure(m)
        dv = 1000.0 * np.asarray(p, float).reshape(self.nx, self.nz)
        return self.to_real(self._gather(self.pool.call("jvec", [(dv,)] * len(self.pool.parts))))

    def jacobian(self, m):
        """J as a LinearOperator (stacked real data x model): J p by Born modeling, J^T y by the adjoint.
        Never formed; with srcest the wavelet is held at its current estimate."""
        m = np.array(m, float, copy=True)
        self._ensure(m)
        n_data = 2 * int(np.prod(self.shape))
        return LinearOperator((n_data, self.n_model), matvec=lambda p: self.jvec(m, p),
                              rmatvec=lambda y: self.gradient(m, y), dtype=float)

    def illumination(self, m):
        """Pseudo-Hessian diagonal H_P (Shin et al. 2001), model grid flattened."""
        self._ensure(m)
        return sum(self.pool.call("illum", [()] * len(self.pool.parts))).ravel()

    def field(self, m, k, shot=0, residual=None):
        """u~ (or the adjoint field for complex residuals residual (ns, nr)) of frequency k, model grid."""
        self._ensure(m)
        w = next(i for i, idx in enumerate(self.pool.parts) if k in idx)
        j = self.pool.parts[w].index(k)
        if self.pool.local is not None:
            return self.pool.local.field(j, shot, residual)
        self.pool.conns[w].send(("field", (j, shot, residual)))
        status, val = self.pool.conns[w].recv()
        if status == "err":
            raise RuntimeError(val)
        return val

    @property
    def n_solve(self):
        return sum(self.pool.call("count", [()] * len(self.pool.parts)))

    def close(self):
        self.pool.close()


# ---- observed data: time-domain modeling --------------------------------------------------------------

def _shot(args):
    v, h, dt, nt, s, wav, rec, nb, f0 = args
    return td.propagate(v, h, dt, nt, s[None], wav[None], rec, order=4, bc="pml", top="free", nb=nb, f0=f0,
                        dtype=np.float64)["traces"]


def observed(v, h, src, rec, wav, dt, nt, nb=20, f0=6.0, workers=None):
    """Traces (ns, nr, nt) of every shot from the time-domain code: 4th-order space, 2nd-order time, CPML of nb
    nodes (a different discretization from the inversion: 9-point frequency-domain operator, PML)."""
    jobs = [(v, h, dt, nt, s, wav, rec, nb, f0) for s in np.asarray(src)]
    ctx = mp.get_context("fork")
    with ctx.Pool(n_cpus() if workers is None else workers) as pool:
        return np.array(pool.map(_shot, jobs))


FGRID = np.arange(1, 61) * 0.25        # frequencies of the kept observed spectra: 0.25 .. 15 Hz


def observed_spectra(name, v, h, src, rec, kind, f0, tmax, workers=None, folder=None):
    """Spectra (60, ns, nr) of the time-domain observed data at FGRID, at omega - i alpha_d with
    alpha_d = ln(100) / tmax. Kept in <folder>/obs/<name>_...npz (folder: results in the current directory) and
    reused by later runs; the first run computes every shot in the time domain."""
    folder = Path(folder) if folder else Path("results")
    path = folder / "obs" / f"{name}_h{h:g}_{kind}_f{f0:g}_T{tmax:g}_s{len(src)}.npz"
    if path.is_file():
        return np.load(path)["spectra"].astype(np.complex128), path, 0.0
    import time
    t0 = time.perf_counter()
    dt, nt = time_axis(h, tmax)
    tr = observed(v, h, src, rec, wavelet(kind, np.arange(nt) * dt, f0), dt, nt, f0=f0, workers=workers)
    D = fd.to_freq(tr, dt, FGRID, math.log(100.0) / tmax).astype(np.complex64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, spectra=D, freqs=FGRID, damp=math.log(100.0) / tmax)
    return D.astype(np.complex128), path, time.perf_counter() - t0


def time_axis(h, tmax, courant=0.3):
    """dt = courant * h / VMAX and the number of samples for a record of length tmax."""
    dt = courant * h / VMAX
    return dt, int(round(tmax / dt)) + 1


def wavelet(kind, t, f0=6.0, t0=None):
    """Source time function. "ricker": the wavelet the inversion assumes (peak f0, delay 1.5 / f0).
    "other": the true wavelet of the wavelet experiment, a Ricker of peak 0.8 f0 delayed by a further 0.04 s and
    rotated by 60 degrees in phase (u cos 60 + H[u] sin 60, H the Hilbert transform)."""
    t0 = 1.5 / f0 if t0 is None else t0
    if kind == "ricker":
        return td.ricker(t, f0, t0)
    if kind == "other":
        w = td.ricker(t, 0.8 * f0, t0 + 0.04)
        th = math.radians(60.0)
        return math.cos(th) * w - math.sin(th) * np.imag(hilbert(w))
    raise ValueError(kind)


# ---- models and acquisition (the same definitions as the time-domain inversion) ---------------------

def marmousi(h=24.0, water=WATER):
    """Marmousi P velocity (Versteeg 1994) on a 24 m or 48 m grid (m/s), shape (nx, nz), with water (m) added on
    top of its own thin water layer: sources and receivers at 48 m are then in water, and the water rows are
    known and fixed, as in a marine survey."""
    path = HERE / "data" / "marmousi_24m.npy"
    if not path.is_file():
        sys.exit(f"{path} not found: copy the complete 09_frequency_fwi folder, including data/")
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
    """Background v0 + grad z (m/s) with a fast and a slow disc at 1.2 km depth."""
    x, z = np.meshgrid(np.arange(nx) * h, np.arange(nz) * h, indexing="ij")
    v = v0 + grad * z
    for xc, sign in ((nx * h / 3, 1.0), (2 * nx * h / 3, -1.0)):
        v[(x - xc) ** 2 + (z - 1200.0) ** 2 <= radius ** 2] += sign * dv
    return v


def initial(v_true, h, kind, sigma_m=300.0, known_rows=3, grad=0.5):
    """Starting model: "linear" v(z) = v_top + grad (z - z_top) below the known rows, "smooth" v_true smoothed
    with a Gaussian of sigma_m (m), "background" the most common value of each row. The known rows keep their
    true values."""
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
    v[:, :known_rows] = v_true[:, :known_rows]
    return v


def acquisition(nx, nshot, h):
    """nshot sources evenly spread at SRC_DEPTH and a receiver at every node at the same depth."""
    jd = int(round(SRC_DEPTH / h))
    xs = np.round(np.linspace(2, nx - 3, nshot)).astype(int)
    src = np.column_stack([xs, np.full(nshot, jd)])
    rec = np.column_stack([np.arange(nx), np.full(nx, jd)])
    return src, rec
