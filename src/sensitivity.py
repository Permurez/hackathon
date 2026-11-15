"""sensitivity.py - endothelial band vs activity: permutation test, random measurement error, misreported activity,
and how much systematic bias would be needed to create the observed correlation from nothing."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import bands, glob, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr
from features import load_meta
if __name__ == "__main__":
    with Pool(6) as pl: F = pd.DataFrame(pl.map(bands.f, sorted(glob.glob(bands.D + "*.snirf"))))
    F.to_csv(str(Path(__file__).resolve().parents[1] / "results/bands_supine.csv"), index=False)
    df = F.merge(load_meta(), on="subject")[["l30_hbo_endothelial_rel", "activity"]].dropna()
    x, a = df.iloc[:, 0].values, df.activity.values
    rho0 = spearmanr(x, a)[0]; rng = np.random.default_rng(0); N = 2000
    print(f"observed rho {rho0:+.2f}, n {len(x)}")
    perm = np.array([spearmanr(rng.permutation(x), a)[0] for _ in range(N)])
    print(f"permutation p (|rho| >= observed by chance): {np.mean(np.abs(perm) >= abs(rho0)):.4f};  chance rho 95% range "
          f"[{np.percentile(perm, 2.5):+.2f}, {np.percentile(perm, 97.5):+.2f}]")
    print("\n1. random measurement error, each person x (1 + e), e uniform +-k:")
    for k in (0.02, 0.05, 0.10, 0.20, 0.40):
        r = np.array([spearmanr(x * (1 + rng.uniform(-k, k, len(x))), a)[0] for _ in range(N)])
        print(f"   +-{100*k:>4.0f}%: rho median {np.median(r):+.2f}  [5-95%: {np.percentile(r,5):+.2f} .. {np.percentile(r,95):+.2f}]  still p<0.05 in {100*np.mean([spearmanr(x*(1+rng.uniform(-k,k,len(x))),a)[1]<.05 for _ in range(300)]):.0f}%")
    levels = np.sort(np.unique(a))
    print("\n2. misreported activity: a fraction of people moved one level up or down at random:")
    for frac in (0.1, 0.2, 0.3, 0.5):
        r = []
        for _ in range(N):
            aa = a.copy(); m = rng.random(len(a)) < frac
            i = np.searchsorted(levels, aa[m]) + rng.choice([-1, 1], m.sum()); aa[m] = levels[np.clip(i, 0, len(levels) - 1)]
            r.append(spearmanr(x, aa)[0])
        print(f"   {100*frac:>3.0f}% misreported: rho median {np.median(r):+.2f} [5-95%: {np.percentile(r,5):+.2f} .. {np.percentile(r,95):+.2f}]")
    print("\n3. bias needed to CREATE rho 0.40 from nothing (shuffled data + k% inflation for the >3 h/week group):")
    top = a == a.max()
    for k in (0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60):
        r = []
        for _ in range(N // 4):
            xs = rng.permutation(x).copy(); xs[top] *= (1 + k); r.append(spearmanr(xs, a)[0])
        r = np.array(r)
        print(f"   +{100*k:>3.0f}% bias: rho median {np.median(r):+.2f}, reaches >= {rho0:.2f} in {100*np.mean(r >= rho0):3.0f}% of runs")
    print(f"\n   observed difference >3 h/week vs rest: {100*(np.median(x[top])/np.median(x[~top])-1):+.0f}% (medians)")
