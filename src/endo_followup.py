"""endo_followup.py - (2) is the endothelial band a stable personal trait? 17 people x 4 rests a day (DOT, sitting);
(3) left-right forehead synchrony in the endothelial band vs activity (supine, pre-specified: higher in the active)."""
import glob, re, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import signal
from scipy.stats import spearmanr, pearsonr, friedmanchisquare
sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo
from features import load_meta
ROOT = Path(__file__).resolve().parents[1]
DOT = ROOT / "dane/infopack_and_data/dot_montage_sitting/data/RS"
SUP = ROOT / "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline"
LO, HI = 0.0095, 0.021

def rel_endo(x, fs):
    down = max(1, int(fs // 5)); x = signal.detrend(signal.decimate(x - x.mean(), down, ftype="fir")); fsd = fs / down
    f, p = signal.welch(x, fsd, nperseg=int(300 * fsd))
    P = lambda a, b: np.trapezoid(p[(f >= a) & (f <= b)], f[(f >= a) & (f <= b)])
    return P(LO, HI) / P(0.0095, 2.0 if fsd / 2 > 2 else fsd / 2 * 0.95)

def dot_file(path):
    sid, ses = re.search(r"(\d+)_[A-Z]+_RS(\d)", Path(path).name).groups()
    h = haemo.convert(path, dpf=6.0); fs = h["fs"]; n = int(600 * fs)
    dead = {c.split()[0] for c, k in zip(h["od_channels"], h["nonpositive"]) if k}
    ok = np.array([p not in dead for p in h["pairs"]]) & (h["distance_mm"] > 24)
    return dict(subject=int(sid), session=int(ses), endo=rel_endo(h["hbo"][ok, :n].mean(0), fs))

def sup_file(path):
    sid = int(re.match(r"(\d+)_", Path(path).name).group(1))
    h = haemo.convert(path, dpf=6.0); fs = h["fs"]
    dead = {c.split()[0] for c, k in zip(h["od_channels"], h["nonpositive"]) if k}
    idx = {n: i for i, n in enumerate(h["pairs"])}
    side = {}
    for nm, ch in (("R", ["S5_D3", "S5_D4"]), ("L", ["S6_D4", "S6_D5"])):
        g = [idx[c] for c in ch if c not in dead]
        if not g: return dict(subject=sid)
        x = h["hbo"][g].mean(0); down = int(fs // 5)
        side[nm] = signal.decimate(x - x.mean(), down, ftype="fir"); fsd = fs / down
    sos = signal.butter(2, [LO / (fsd / 2), HI / (fsd / 2)], btype="band", output="sos")
    a, b = signal.sosfiltfilt(sos, side["L"]), signal.sosfiltfilt(sos, side["R"])
    trim = int(60 * fsd)  # drop filter edges
    return dict(subject=sid, lr_sync=float(np.corrcoef(a[trim:-trim], b[trim:-trim])[0, 1]))

if __name__ == "__main__":
    with Pool(6) as p:
        D = pd.DataFrame(p.map(dot_file, sorted(glob.glob(str(DOT / "*.snirf")))))
        S = pd.DataFrame(p.map(sup_file, sorted(glob.glob(str(SUP / "*.snirf")))))
    # ---- (2) stability
    w = D.pivot(index="subject", columns="session", values="endo").dropna()
    z = w.values; gm = z.mean()
    ss_t = ((z - gm) ** 2).sum(); ss_p = (z.shape[1] * (z.mean(1) - gm) ** 2).sum(); ss_s = (z.shape[0] * (z.mean(0) - gm) ** 2).sum()
    k = z.shape[1]; msb = ss_p / (z.shape[0] - 1); msw = (ss_t - ss_p) / (z.size - z.shape[0])
    icc = (msb - msw) / (msb + (k - 1) * msw)
    print(f"(2) DOT, {len(w)} people x {k} rests: person {100*ss_p/ss_t:.0f}% | time of day {100*ss_s/ss_t:.0f}% | noise {100*(ss_t-ss_p-ss_s)/ss_t:.0f}% | ICC(1) {icc:.2f}")
    print(f"    session medians {np.round(np.median(z, 0), 3)}  Friedman p {friedmanchisquare(*z.T).pvalue:.3f}")
    print(f"    session 1+2 vs 3+4 between-person rho {spearmanr(z[:, :2].mean(1), z[:, 2:].mean(1))[0]:+.2f}")
    m = load_meta.__globals__["pd"].read_csv(ROOT / "dane/infopack_and_data/dot_montage_sitting/subjects_description.csv", decimal=",")
    m = m.rename(columns={"ID": "subject", "Ocena aktywności fizycznej": "activity"})[["subject", "activity"]]
    pm = w.mean(1).rename("endo").reset_index().merge(m, on="subject")
    r, pv = spearmanr(pm.endo, pm.activity); top = pm.activity == pm.activity.max()
    print(f"    DOT endo vs activity (n {len(pm)}, {top.sum()} at max): rho {r:+.2f} p {pv:.3f}; median max-group {pm[top].endo.median():.3f} vs rest {pm[~top].endo.median():.3f}")
    # ---- (3) left-right synchrony
    d = S.merge(load_meta(), on="subject").dropna(subset=["lr_sync"])
    for lab, x in (("all", d), ("18-42", d[d.age < 60])):
        r, pv = spearmanr(x.lr_sync, x.activity); rn, pn = spearmanr(x.lr_sync, x.NEG_CONTROL_group)
        R = x[["lr_sync", "activity", "age", "sex_male", "bmi"]].rank(); X = np.c_[np.ones(len(R)), R[["age", "sex_male", "bmi"]].values]
        res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]; rp, pp = pearsonr(res(R.lr_sync.values), res(R.activity.values))
        print(f"(3) L-R sync {lab} (n {len(x)}): median {x.lr_sync.median():.2f}; vs activity rho {r:+.2f} p {pv:.3f}; partial|age,sex,BMI {rp:+.2f} p {pp:.3f}; NEG control rho {rn:+.2f}")
    d.to_csv(ROOT / "endo_followup_supine.csv", index=False); D.to_csv(ROOT / "endo_followup_dot.csv", index=False)
