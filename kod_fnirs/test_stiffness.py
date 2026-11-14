"""Synthetic check: the stiffness index must grow with the planted artery stiffness."""

import numpy as np
import pandas as pd

import stiffness as st
from synthetic import make_recording

FS = 141.79
planted = [0.0, 0.25, 0.5, 0.75, 1.0]
rows = []
for s in planted:
    rec = make_recording(fs=FS, duration_s=300, stiffness=s, noise=0.15, seed=3)
    w, period, n = st.mean_pulse(rec["hbo"], FS)
    f = st.pulse_features(w)
    rows.append(f)
    print(f"planted {s:.2f}: rise {f['rise_frac']:.3f}  relax {f['relax_area']:+.4f}  b/a {f['b_a']:+.3f}  beats {n}")

df = st.add_index(pd.DataFrame(rows))
idx = df["stiffness_index"].to_numpy()
rho = np.corrcoef(planted, idx)[0, 1]
mono = all(np.all(np.diff(df[f]) * st.STIFF_SIGN[f] > 0) for f in st.INDEX_FEATURES)
ok = rho > 0.95 and mono
print(f"index {np.round(idx, 2)}  corr with planted {rho:.2f}")
print("ALL OK" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
