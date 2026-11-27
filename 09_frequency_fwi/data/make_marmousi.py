"""Download the Marmousi P velocity model and save a 24 m version for the lab.

Source: Versteeg (1994), The Leading Edge; file marmvel.hh from the Madagascar data server.
The original file is not kept in the repository; only marmousi_24m.npy is.

Run:  python3 make_marmousi.py        (needs internet, run once by the instructor)
Out:  marmousi_24m.npy                 float32, shape (384, 126) = (nx, nz), dx = dz = 24 m
"""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np

URL = "https://www.ahay.org/data/marm/marmvel.hh"
SHA256 = "644597dd0789f16b832c61e3f7c35fc64633e54a54ac26419970b6daa92c99b0"
N1, N2, D = 751, 2301, 4.0     # n1 = z samples, n2 = x samples, 4 m spacing
STRIDE = 6                      # 4 m * 6 = 24 m

data = urlopen(URL, timeout=60).read()
digest = hashlib.sha256(data).hexdigest()
if digest != SHA256:
    raise SystemExit(f"SHA-256 mismatch: {digest}")
i = data.index(b"\x0c\x0c\x04")            # end of the text header
v = np.frombuffer(data[i + 3:], ">f4")     # xdr_float = big-endian float32
v = v[:N1 * N2].reshape(N2, N1)            # z is the fast axis, so rows are x: (nx, nz)
v24 = np.ascontiguousarray(v[::STRIDE, ::STRIDE], dtype=np.float32)
print(f"original {v.shape}, {D:.0f} m, v = {v.min():.0f} ~ {v.max():.0f} m/s")
print(f"decimated {v24.shape}, {D * STRIDE:.0f} m")
out = Path(__file__).resolve().parent / "marmousi_24m.npy"
np.save(out, v24)
print(f"Saved: {out}")
