"""Checks of the screening pipeline pieces on synthetic signals."""

import numpy as np

import screening as sc
import stiffness as st
from synthetic import make_recording

ok = True
FS = 141.79

# 1. late reflected wave (P2 > P1): P1 must be the early peak, not the global argmax
ph = np.linspace(0, 1, 200)
w = 0.8 * np.exp(-((ph - 0.18) / 0.06) ** 2) + 1.0 * np.exp(-((ph - 0.43) / 0.09) ** 2)
i1, kind = st.find_p1(w)
print(f"late P2: argmax phase {ph[np.argmax(w)]:.2f}, P1 phase {ph[i1]:.2f} ({kind})")
ok &= abs(ph[i1] - 0.18) < 0.03 and kind == "peak"

# 2. fixed_relax_area falls monotonically with planted stiffness
vals = []
for s in [0.0, 0.5, 1.0]:
    rec = make_recording(fs=FS, duration_s=300, stiffness=s, noise=0.15, seed=3)
    x = sc.cardiac(rec["hbo"] * 1e6, FS)
    tr, _ = sc.find_troughs(x, FS)
    f, _ = sc.layer_features(x, FS, sc.clean_beats(x, tr), np.random.default_rng(0), n_boot=50)
    vals.append(f["fixed_relax_area"])
    ok &= f["fixed_relax_area_lo"] <= f["fixed_relax_area"] <= f["fixed_relax_area_hi"]
print("fixed_relax_area vs stiffness", np.round(vals, 3))
ok &= bool(np.all(np.diff(vals) < 0))

# 3. +-20 % IBI rule: a missed beat (2x IBI) and a false beat (split IBI) must not inflate RMSSD
rng = np.random.default_rng(1)
ibi = 0.9 + 0.02 * rng.standard_normal(600)
clean, _ = sc.hrv(ibi, None, 0)
bad = ibi.copy()
bad[100], bad[300:302] = 1.8, [0.3, 0.6]
dirty, kept = sc.hrv(bad, None, 0)
print(f"RMSSD clean {clean['rmssd_ms']:.1f} ms, with artefacts raw {dirty['rmssd_raw_ms']:.1f} -> cleaned "
      f"{dirty['rmssd_ms']:.1f} ms, rejected {int((~kept).sum())}")
ok &= abs(dirty["rmssd_ms"] - clean["rmssd_ms"]) < 2 and (~kept).sum() == 3

print("ALL OK" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
