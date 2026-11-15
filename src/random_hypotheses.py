"""random_hypotheses.py - 5000 random band/signal/channel/target hypotheses (ages 18-42), each discovered on one random
half of the people and checked on the other half; plus how special the endothelial-activity result is among them."""
import glob, re, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy import signal, stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo
from features import load_meta
ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline"
GROUPS = {"forehead": ["S6_D4", "S5_D3", "S6_D5", "S5_D4"], "hair": ["S1_D1", "S2_D1", "S3_D2", "S4_D2"], "skin": ["S7_D4", "S8_D5"]}

def spectra(path):
    sid = int(re.match(r"(\d+)_", Path(path).name).group(1))
    h = haemo.convert(path, dpf=6.0); fs = h["fs"]; down = int(fs // 5); fsd = fs / down
    dead = {c.split()[0] for c, k in zip(h["od_channels"], h["nonpositive"]) if k}
    idx = {n: i for i, n in enumerate(h["pairs"])}; out = {"subject": sid}
    for gname, chans in GROUPS.items():
        g = [idx[c] for c in chans if c not in dead]
        if not g: continue
        for chrom in ("hbo", "hbr", "hbt"):
            x = (h["hbo"][g] + h["hbr"][g]).mean(0) if chrom == "hbt" else h[chrom][g].mean(0)
            x = signal.detrend(signal.decimate(x - x.mean(), down, ftype="fir"))
            f, p = signal.welch(x, fsd, nperseg=int(300 * fsd)); out[(gname, chrom)] = p; out["f"] = f
    return out

if __name__ == "__main__":
    with Pool(6) as pl: S = pl.map(spectra, sorted(glob.glob(str(SUP / "*.snirf"))))
    f = S[0]["f"]; meta = load_meta(); meta["subject"] = meta.subject.astype(int)
    meta = meta[meta.age < 60].set_index("subject")
    targets = ["activity", "age", "bmi", "sex_male", "height", "weight", "head_circ", "hair_darkness", "fitzpatrick", "treadmill_kmh", "NEG_CONTROL_group"]
    tot = (f >= 0.005) & (f <= 2.0)
    def value(s, key, lo, hi):
        p = s.get(key); 
        if p is None: return np.nan
        m = (f >= lo) & (f <= hi); return np.trapezoid(p[m], f[m]) / np.trapezoid(p[tot], f[tot])
    subs = [s["subject"] for s in S if s["subject"] in meta.index]
    Sd = {s["subject"]: s for s in S}
    rng = np.random.default_rng(42); rows = []
    for k in range(5000):
        lo = np.exp(rng.uniform(np.log(0.005), np.log(1.0))); hi = min(lo * rng.uniform(1.5, 4.0), 2.0)
        key = (rng.choice(list(GROUPS)), rng.choice(["hbo", "hbr", "hbt"])); t = rng.choice(targets)
        v = pd.Series({s: value(Sd[s], key, lo, hi) for s in subs}); d = pd.DataFrame({"v": v, "t": meta.loc[subs, t]}).dropna()
        if len(d) < 40 or d.t.nunique() < 2: continue
        perm = rng.permutation(len(d)); A, B = d.iloc[perm[: len(d) // 2]], d.iloc[perm[len(d) // 2:]]
        rA, pA = stats.spearmanr(A.v, A.t); rB, pB = stats.spearmanr(B.v, B.t); rF, pF = stats.spearmanr(d.v, d.t)
        rows.append(dict(lo=lo, hi=hi, group=key[0], chrom=key[1], target=t, rho=rF, p=pF, rho_disc=rA, p_disc=pA, rho_conf=rB, p_conf=pB, n=len(d)))
    R = pd.DataFrame(rows); R.to_csv(ROOT / "results/random_hypotheses.csv", index=False)
    print(f"{len(R)} random hypotheses (ages 18-42)")
    print(f"full-sample p<0.05: {100*np.mean(R.p<.05):.1f}%  (chance 5%)")
    disc = R[R.p_disc < 0.05]; rep = disc[(disc.p_conf < 0.05) & (np.sign(disc.rho_conf) == np.sign(disc.rho_disc))]
    print(f"discovered on half A: {len(disc)}; REPLICATED on unseen half B (same sign, p<0.05): {len(rep)} ({100*len(rep)/max(len(disc),1):.0f}%; chance ~2.5%)")
    print("\nreplicated hypotheses by target:"); print(rep.target.value_counts().to_string())
    print("\nhow special is endothelial-activity rho +0.46? share of random hypotheses with |rho| >= 0.46:",
          f"all targets {100*np.mean(np.abs(R.rho) >= 0.46):.2f}%, activity only {100*np.mean(np.abs(R[R.target=='activity'].rho) >= 0.46):.2f}% (n={int((R.target=='activity').sum())})")
    a = rep[rep.target.isin(["activity", "age", "treadmill_kmh", "bmi", "fitzpatrick", "hair_darkness"])].copy()
    a["band"] = a.apply(lambda r: f"{r.lo:.4f}-{r.hi:.4f} Hz", axis=1)
    print("\nreplicated, NON-body-size targets (the interesting ones):")
    print(a.sort_values("p")[["target", "group", "chrom", "band", "rho", "rho_disc", "rho_conf", "p"]].round(3).head(25).to_string(index=False))
    act = R[R.target == "activity"].copy(); act["mid"] = np.sqrt(act.lo * act.hi)
    bins = [0.005, 0.0095, 0.021, 0.052, 0.145, 0.6, 2.0]; act["zone"] = pd.cut(act.mid, bins, labels=["<0.0095", "endothelial", "neurogenic", "myogenic", "respiratory", "cardiac"])
    print("\nactivity hypotheses: median rho by band zone and channel group:")
    print(act.pivot_table(index="zone", columns="group", values="rho", aggfunc="median", observed=False).round(2).to_string())
