"""bands.py - relative power of the Stefanovska vasomotion bands (endothelial, neurogenic, myogenic, respiratory, cardiac)
in skin (10 mm) and forehead 30 mm channels, supine baseline; prints correlations with activity, age, BMI."""
import sys, glob, re, numpy as np, pandas as pd
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent)); import haemo
from features import load_meta, FOREHEAD, SHORT
from scipy import signal
from multiprocessing import Pool
D = str(Path(__file__).resolve().parents[1] / "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline") + "/"
BANDS = {"endothelial": (0.0095, 0.021), "neurogenic": (0.021, 0.052), "myogenic": (0.052, 0.145),
         "respiratory": (0.145, 0.6), "cardiac": (0.6, 2.0)}

def f(path):
    sid = int(re.match(r"(\d+)_", path.split("/")[-1]).group(1))
    h = haemo.convert(path, dpf=6.0); fs = h["fs"]
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    idx = {n: i for i, n in enumerate(h["pairs"])}
    out = {"subject": sid}
    for lay, chans in (("skin", SHORT), ("l30", FOREHEAD)):
        g = [idx[n] for n in chans if n not in dead]
        if not g: continue
        for chrom in ("hbo", "hbt"):
            x = h["hbo"][g].mean(0) + (h["hbr"][g].mean(0) if chrom == "hbt" else 0)
            down = int(fs // 5); x = signal.decimate(x - x.mean(), down, ftype="fir"); fsd = fs / down
            fq, p = signal.welch(signal.detrend(x), fsd, nperseg=int(300 * fsd))       # 300-s windows: resolves 0.0095 Hz
            tot = np.trapezoid(p[(fq >= 0.0095) & (fq <= 2.0)], fq[(fq >= 0.0095) & (fq <= 2.0)])
            for b, (lo, hi) in BANDS.items():
                m = (fq >= lo) & (fq <= hi)
                out[f"{lay}_{chrom}_{b}_rel"] = np.trapezoid(p[m], fq[m]) / tot
    return out

if __name__ == "__main__":
    from scipy.stats import spearmanr, false_discovery_control
    with Pool(6) as pl: F = pd.DataFrame(pl.map(f, sorted(glob.glob(D + "*.snirf"))))
    df = F.merge(load_meta(), on="subject")
    feats = [c for c in F if c != "subject"]
    rows = []
    for c in feats:
        for t in ("activity", "age", "bmi", "NEG_CONTROL_group"):
            d = df[list(dict.fromkeys([c, t, "age"]))].dropna()
            r, p = spearmanr(d[c], d[t]); rows.append(dict(feature=c, target=t, rho=r, p=p, n=len(d)))
            if t == "age":
                y = d[d.age < 60]; r, p = spearmanr(y[c], y.age); rows.append(dict(feature=c, target="age<60", rho=r, p=p, n=len(y)))
    R = pd.DataFrame(rows); R["q"] = false_discovery_control(R.p)
    print(f"{len(R)} tests; p<0.05: {(R.p<.05).sum()} (chance {0.05*len(R):.1f}); q<0.10: {(R.q<.1).sum()}")
    piv = R.pivot(index="feature", columns="target", values="rho").round(2)
    pv = R.pivot(index="feature", columns="target", values="p")
    star = pv.map(lambda p: "*" if p < .05 else "")
    print((piv.astype(str) + star).to_string())
    print("\nmedian relative power, skin vs 30 mm (HbO):")
    for b in BANDS: print(f"  {b:<12} skin {F[f'skin_hbo_{b}_rel'].median():.3f}   30mm {F[f'l30_hbo_{b}_rel'].median():.3f}")
