"""hypotheses.py - pre-specified literature hypotheses, one run, partial Spearman (age/sex/BMI), permutation, FDR in family."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, false_discovery_control
f = pd.read_csv("features.csv")
b = pd.read_csv("bands_supine.csv")
t = pd.read_csv("timing_index.csv")[["subject", "TI"]]
c = pd.read_csv("bh_co2.csv").groupby("subject").long_rise.mean().rename("bh_hbo_rise").reset_index()
df = f.merge(b, on="subject", how="left").merge(t, on="subject", how="left").merge(c, on="subject", how="left")
H = [("H1", "hr_bpm", "activity", -1), ("H2", "rmssd_ms", "activity", +1),
     ("H3", "l30_hbo_endothelial_rel", "bmi", -1), ("H4", "l30_hbo_endothelial_rel", "age", -1),
     ("H5", "l30_hbo_myogenic_rel", "age", -1), ("H6", "l30_hbo_neurogenic_rel", "age", +1),
     ("H7", "TI", "bmi", +1), ("H8", "bh_hbo_rise", "activity", +1), ("H9", "bh_hbo_rise", "age", -1),
     ("H10", "bh_max_s", "activity", +1)]
rng = np.random.default_rng(0); rows = []
for h, x, y, sign in H:
    cov = [v for v in ("age", "sex_male", "bmi") if v != y]
    d = df[[x, y] + cov].dropna(); r, p = spearmanr(d[x], d[y])
    R = d.rank(); X = np.c_[np.ones(len(R)), R[cov].values]; res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    rp, pp = pearsonr(res(R[x].values), res(R[y].values))
    perm = np.mean([abs(spearmanr(rng.permutation(d[x].values), d[y])[0]) >= abs(r) for _ in range(2000)])
    y60 = d[df.loc[d.index, "age"] < 60]; r60 = spearmanr(y60[x], y60[y])[0]
    rows.append(dict(H=h, param=x, vs=y, expected="+" if sign > 0 else "-", rho=r, p=p, perm_p=perm, partial_rho=rp, partial_p=pp,
                     rho_lt60=r60, direction_ok=np.sign(r) == sign, n=len(d)))
R = pd.DataFrame(rows); R["q_fdr"] = false_discovery_control(R.p)
pd.set_option("display.width", 220); print(R.round(3).to_string(index=False))
R.to_csv("hypotheses.csv", index=False)
