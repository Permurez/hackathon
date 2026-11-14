"""Checks of haemo.py: exact round trip on a synthetic signal + agreement with MNE on a real file."""

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo  # noqa: E402

ok = True

# 1. forward model -> intensity -> back: must recover the planted HbO / HbR exactly
fs, T = 141.79, 60
t = np.arange(int(fs * T)) / fs
hbo = 0.5 * np.sin(2 * np.pi * 1.1 * t) + 0.2 * np.sin(2 * np.pi * 0.25 * t)   # uM
hbr = -0.15 * np.sin(2 * np.pi * 1.1 * t + 0.3)
L_cm, dpf = 3.0, haemo.dpf_scholkmann(np.array([760, 850]), 30)
od = np.array([(haemo.extinction(w)[0] * hbo + haemo.extinction(w)[1] * hbr) * 1e-6 * L_cm * d
               for w, d in zip((760, 850), dpf)])
I = 2.0 * 10 ** (-(od - od.mean(axis=1, keepdims=True)))  # intensity whose mean-referenced dOD = od - mean
od_back = haemo.optical_density(I.T).T
E = np.array([np.array(haemo.extinction(w)) * L_cm * d for w, d in zip((760, 850), dpf)])
c = np.linalg.solve(E, od_back) * 1e6
err = max(np.abs(c[0] - (hbo - hbo.mean())).max(), np.abs(c[1] - (hbr - hbr.mean())).max())
print(f"synthetic round trip: max error {err:.2e} uM")
ok &= err < 0.02  # the log of the mean intensity is not exactly the mean of the log

# 2. missing samples are interpolated
x = np.c_[np.arange(10.0), np.arange(10.0) * 2]
m = np.zeros(10, bool); m[[3, 4]] = True
x2 = x.copy(); x2[m] = 0
ok &= np.allclose(haemo.fill_missing(x2, m), x)

# 3. DPF formula (Scholkmann & Wolf 2013): ~6 at 760 nm, ~5 at 850 nm for a 25-year-old
d = haemo.dpf_scholkmann(np.array([760, 850]), 25)
print(f"DPF(25 y) 760 nm {d[0]:.2f}, 850 nm {d[1]:.2f}")
ok &= 5.5 < d[0] < 6.5 and 4.5 < d[1] < 5.5

# 4. real file: same HbO as MNE (same DPF/PPF), channels without dead samples
files = sorted(glob.glob(str(Path(__file__).resolve().parents[1] /
                             "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline/1_JZ*.snirf")))
if files:
    import mne
    h = haemo.convert(files[0], dpf=6.0)
    raw = mne.io.read_raw_snirf(files[0], preload=True, verbose="error")
    hb = mne.preprocessing.nirs.beer_lambert_law(mne.preprocessing.nirs.optical_density(raw, verbose="error"),
                                                 ppf=6.0)
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    for i, name in enumerate(h["pairs"]):
        if name in dead:
            print(f"  {name}: skipped (samples with zero intensity)")
            continue
        mm = hb.get_data(picks=[f"{name} hbo"])[0] * 1e6
        r = np.corrcoef(mm, h["hbo"][i])[0, 1]
        ratio = h["hbo"][i].std() / mm.std()
        print(f"  {name}: r = {r:.5f}, sd ours / MNE = {ratio:.4f}")
        ok &= r > 0.999 and abs(ratio - 1) < 0.01
else:
    print("real-file check skipped (no data)")

print("ALL OK" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
