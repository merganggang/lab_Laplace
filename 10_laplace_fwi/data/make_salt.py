"""Make salt_20m.npy, the SEG/EAGE salt model 2-D section A-A' on a 20 m grid, for the lab.

Source: Aminzadeh, Brac & Kunz (1997), 3-D Salt and Overthrust Models, SEG/EAGE. The 2-D section A-A' is
VEL_GRIDS/saltaa.h and saltaa@ in Salt_Model_3D.tar.gz of the SEG open data
(https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/Salt_Model_3D.tar.gz, the file the
Madagascar book/gallery/segsalt recipe fetches): big-endian (xdr) float32, 20 m spacing, x the fast axis (780 values
per depth row), velocity in m/s. Neither the archive nor saltaa@ is kept in the repository; only salt_20m.npy is.

The committed salt_20m.npy (779 x 209) was made from an instructor copy of the same section in km/s on the same
20 m grid, one sample short in x and z; SOURCE_COPY gives its SHA-256. Run this script on a downloaded archive to
rebuild the array from the SEG file and compare it with the committed one; it is replaced only with --save.

Run:  python3 make_salt.py Salt_Model_3D.tar.gz     (the archive downloaded once by the instructor)
      python3 make_salt.py --copy salt20m.bin        (the instructor copy: little-endian float32 km/s, (779, 209))
Out:  salt_20m.npy                                   float32, shape (nx, nz), dx = dz = 20 m, m/s
"""

import argparse
import hashlib
import tarfile
from pathlib import Path

import numpy as np

NX = 780                                     # x samples per depth row of saltaa@
SOURCE_COPY = "6f2dde964275a6028d14331aed9bafbe0fdc0d76cbc6c2803d15e045b797fc8a"   # salt20m.bin, (779, 209), km/s
HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description="Make salt_20m.npy from the SEG archive or the instructor copy.")
ap.add_argument("source", help="Salt_Model_3D.tar.gz, or with --copy the salt20m.bin copy")
ap.add_argument("--copy", action="store_true", help="the source is the instructor copy salt20m.bin")
ap.add_argument("--save", action="store_true", help="with the archive: replace the committed salt_20m.npy")
args = ap.parse_args()

if args.copy:
    raw = Path(args.source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_COPY:
        raise SystemExit(f"SHA-256 mismatch: {digest}")
    v = np.frombuffer(raw, "<f4").reshape(779, 209) * 1000.0          # km/s -> m/s, x is the slow axis
else:
    with tarfile.open(args.source) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("VEL_GRIDS/saltaa@"))
        raw = tar.extractfile(member).read()
    print(f"saltaa@ SHA-256 {hashlib.sha256(raw).hexdigest()}")
    v = np.frombuffer(raw, ">f4")
    v = v.reshape(-1, NX).T                                          # rows are depth, x fast: to (nx, nz)
v = np.ascontiguousarray(v, dtype=np.float32)
print(f"{v.shape}, 20 m, v = {v.min():.0f} ~ {v.max():.0f} m/s")
out = HERE / "salt_20m.npy"
if out.is_file() and not (args.copy or args.save):           # the archive only checks the committed array
    old = np.load(out)
    n = (min(old.shape[0], v.shape[0]), min(old.shape[1], v.shape[1]))
    print(f"committed {old.shape}: max |difference| on the common part "
          f"{np.abs(old[:n[0], :n[1]] - v[:n[0], :n[1]]).max():.3f} m/s (not overwritten; --save replaces it)")
else:
    np.save(out, v)
    print(f"Saved: {out}")
