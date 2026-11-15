"""
features.py - every parameter we can compute from one resting fNIRS recording (supine 141.79 Hz baseline)
+ the breath-hold file, screened against every column of subjects_description.csv.

  python src/features.py          # -> features.csv, correlations.csv, correlations.png, summary in the terminal

Safeguards against finding noise:
  * the feature list is fixed here, before looking at any result
  * DPF fixed at 6 (an age-dependent DPF would write age into every uM value)
  * Spearman for every pair, Benjamini-Hochberg FDR over ALL tests together
  * NEGATIVE control: `Grupa` (counterbalancing group, random) must correlate with nothing
  * POSITIVE controls: light reaching the detector must depend on hair / skin, HR on sex
"""

from __future__ import annotations

import glob
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "dane/infopack_and_data/fnirs_high_freq_supine"
FOREHEAD = ["S6_D4", "S5_D3", "S6_D5", "S5_D4"]  # 30 mm, bare forehead
HAIR = ["S1_D1", "S2_D1", "S3_D2", "S4_D2"]      # 23-30 mm, around F1-F4: under the hair
SHORT = ["S7_D4", "S8_D5"]                       # 10 mm, skin
DPF = 6.0
BANDS = {"vlf": (0.01, 0.08), "mayer": (0.08, 0.15), "resp": (0.15, 0.5), "cardiac": (0.7, 2.5)}


# ---------------------------------------------------------------------------
def bp(x, fs, lo, hi, order=3):
    sos = signal.butter(order, [lo / (fs / 2), min(hi, 0.45 * fs) / (fs / 2)], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x)


def psd(x, fs, win=60):
    return signal.welch(x - x.mean(), fs, nperseg=int(min(len(x), win * fs)))


def band_power(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    return np.trapezoid(p[m], f[m])


def cardiac_quality(x, fs):
    f, p = psd(x, fs, 30)
    m = (f >= 0.7) & (f <= 2.5)
    return p[m].max() / p[m].mean()


def peak_freq(x, fs, lo, hi):
    f, p = psd(x, fs, 60)
    m = (f >= lo) & (f <= hi)
    return f[m][np.argmax(p[m])]


def beats(xc, fs):
    f0 = peak_freq(xc, fs, 0.7, 2.5)
    tr, _ = signal.find_peaks(-xc, distance=int(0.6 * fs / f0), prominence=0.3 * np.std(xc))
    return tr


def hrv(tr, xc, fs):
    d = np.gradient(xc)
    t = np.array([a + np.argmax(d[a:a + max(3, (b - a) // 2)]) for a, b in zip(tr[:-1], tr[1:])]) / fs
    ibi = np.diff(t)
    med = np.median(ibi)
    ok = np.abs(ibi - med) <= 0.2 * med
    pair = ok[1:] & ok[:-1]
    dd = np.diff(ibi)[pair]
    x, y = ibi[:-1][pair], ibi[1:][pair]
    sd1, sd2 = np.std(dd, ddof=1) / np.sqrt(2), np.std(x + y, ddof=1) / np.sqrt(2)
    return {"hr_bpm": 60 / ibi[ok].mean(), "rmssd_ms": 1000 * np.sqrt(np.mean(dd ** 2)),
            "sdnn_ms": 1000 * ibi[ok].std(ddof=1), "sd1_sd2": sd1 / sd2}


def pulse_shape(xc, tr):
    """Mean normalised beat -> fixed relaxation area (phase 0.25-1) and phase of the first systolic peak."""
    med = np.median(np.diff(tr))
    B = []
    for a, b in zip(tr[:-1], tr[1:]):
        if abs((b - a) - med) <= 0.2 * med:
            s = np.interp(np.linspace(a, b, 200), np.arange(a, b + 1), xc[a:b + 1])
            B.append((s - s.min()) / (np.ptp(s) + 1e-30))
    w = np.mean(B, 0)
    w = (w - w.min()) / np.ptp(w)
    ph = np.linspace(0, 1, 200)
    seg = w[50:]
    area = np.trapezoid(seg - np.linspace(seg[0], seg[-1], len(seg)), ph[50:])
    pk, _ = signal.find_peaks(w)
    pk = pk[(pk >= 20) & (pk <= 64)]
    return {"pulse_relax_area": area, "pulse_p1_phase": ph[pk[0]] if len(pk) else np.nan}


# ---------------------------------------------------------------------------
# HbO / HbR / HbT in three layers. CW fNIRS gives only CHANGES (mean = 0 by construction), so every
# feature describes fluctuations: their size per band, HbR-vs-HbO ratio, phase lag, coherence.
SLOW_BANDS = {"vlf": (0.01, 0.08), "mayer": (0.08, 0.15), "resp": (0.15, 0.5)}
RIGHT, LEFT = ["S5_D3", "S5_D4"], ["S6_D5", "S6_D4"]


def _band_csd(x, y, fs, lo, hi):
    f, pxy = signal.csd(x, y, fs, nperseg=int(min(len(x), 120 * fs)))
    _, cxy = signal.coherence(x, y, fs, nperseg=int(min(len(x), 120 * fs)))
    m = (f >= lo) & (f <= hi)
    return np.angle(pxy[m].sum()), float(cxy[m].mean())


def hb_features(h, idx, dead, fs):
    down = max(1, int(fs // 10))  # slow bands only -> 14 Hz is plenty, much faster spectra
    fs_d = fs / down

    def avg(chrom, names):
        g = [idx[n] for n in names if n not in dead]
        return h[chrom][g].mean(0) if g else None

    def slow(x):
        return signal.decimate(bp(x, fs, 0.01, 2.5), down, zero_phase=True) if x is not None else None

    long_o, long_r = slow(avg("hbo", FOREHEAD)), slow(avg("hbr", FOREHEAD))
    sk_o, sk_r = slow(avg("hbo", SHORT)), slow(avg("hbr", SHORT))
    layers = {"l30": (long_o, long_r)}
    if sk_o is not None:
        layers["skin"] = (sk_o, sk_r)
        bo = np.dot(long_o, sk_o) / np.dot(sk_o, sk_o)
        br = np.dot(long_r, sk_r) / np.dot(sk_r, sk_r)
        layers["brain"] = (long_o - bo * sk_o, long_r - br * sk_r)
    out = {}
    for lay, (o, r) in layers.items():
        sig = {"hbo": o, "hbr": r, "hbt": o + r}
        for bname, (lo, hi) in SLOW_BANDS.items():
            filt = {k: bp(v, fs_d, lo, hi) for k, v in sig.items()}
            for k, v in filt.items():
                out[f"{lay}_{k}_{bname}_uM"] = float(v.std())
            out[f"{lay}_hbr_over_hbo_{bname}"] = float(filt["hbr"].std() / filt["hbo"].std())
            if bname != "vlf":
                ph, coh = _band_csd(o, r, fs_d, lo, hi)
                out[f"{lay}_hbr_phase_{bname}_deg"] = float(np.degrees(ph))
                out[f"{lay}_hbo_hbr_coh_{bname}"] = coh
    # left vs right forehead, brain layer HbO (resting-state coupling of the two sides)
    if sk_o is not None:
        side = {}
        for nm, chans in (("R", RIGHT), ("L", LEFT)):
            xo = slow(avg("hbo", chans))
            if xo is None:
                return out
            side[nm] = xo - (np.dot(xo, sk_o) / np.dot(sk_o, sk_o)) * sk_o
        for bname in ("vlf", "mayer"):
            out[f"brain_LR_coh_{bname}"] = _band_csd(side["L"], side["R"], fs_d, *SLOW_BANDS[bname])[1]
    return out


# ---------------------------------------------------------------------------
def baseline_features(path):
    path = Path(path)
    sid = int(re.match(r"(\d+)_", path.name).group(1))
    r = haemo.read_snirf(path)
    h = haemo.convert(path, dpf=DPF)
    fs = h["fs"]
    idx = {n: i for i, n in enumerate(h["pairs"])}
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    I = haemo.fill_missing(r["intensity"], r["missing"])
    od_name = h["od_channels"]
    out = {"subject": sid, "missing_pct": 100 * r["missing"].mean(), "dead_channels": len(dead)}

    def chans(group):
        return [n for n in group if n not in dead]

    # --- optics: how much light comes back, and how clean the heartbeat is
    for gname, group in (("forehead", FOREHEAD), ("hair", HAIR), ("short", SHORT)):
        g = chans(group)
        for wl in (760, 850):
            cols = [od_name.index(f"{n} {wl}") for n in g]
            out[f"{gname}_log_intensity_{wl}"] = float(np.mean(np.log10(I[:, cols].mean(0)))) if cols else np.nan
        out[f"{gname}_cardiac_quality"] = float(np.mean([cardiac_quality(h["hbo"][idx[n]], fs) for n in g])) if g else np.nan
        cols = [od_name.index(f"{n} 850") for n in g]
        out[f"{gname}_pulse_od_850"] = float(np.mean([bp(h["od"][c], fs, 0.7, 2.5).std() for c in cols])) if cols else np.nan
    # pulse-oximetry-like ratio of ratios on the forehead: (AC/DC 760) / (AC/DC 850)
    g = chans(FOREHEAD)
    acdc = {wl: np.mean([bp(I[:, od_name.index(f"{n} {wl}")], fs, 0.7, 2.5).std() / I[:, od_name.index(f"{n} {wl}")].mean()
                         for n in g]) for wl in (760, 850)}
    out["ratio_of_ratios_R"] = acdc[760] / acdc[850]

    # --- haemodynamics (uM, fixed DPF)
    long_ = h["hbo"][[idx[n] for n in g]].mean(0)
    long_r = h["hbr"][[idx[n] for n in g]].mean(0)
    s = chans(SHORT)
    xc = bp(long_, fs, 0.5, 10)
    tr = beats(xc, fs)
    out.update(hrv(tr, xc, fs))
    out.update(pulse_shape(xc, tr))
    out["pulse_hbo_uM"] = float(bp(long_, fs, 0.7, 2.5).std() * 2 * np.sqrt(2))
    out["pulse_hbr_over_hbo"] = float(bp(long_r, fs, 0.7, 2.5).std() / bp(long_, fs, 0.7, 2.5).std())
    if s:
        sk = h["hbo"][[idx[n] for n in s]].mean(0)
        out["skin_pulse_hbo_uM"] = float(bp(sk, fs, 0.7, 2.5).std() * 2 * np.sqrt(2))
        a, b = bp(long_, fs, 0.7, 2.5), bp(sk, fs, 0.7, 2.5)
        out["skin_share_cardiac"] = float(np.corrcoef(a, b)[0, 1] ** 2)
        a, b = bp(long_, fs, 0.01, 0.5), bp(sk, fs, 0.01, 0.5)
        out["skin_share_slow"] = float(np.corrcoef(a, b)[0, 1] ** 2)
    f, p = psd(long_, fs, 120)
    tot = band_power(f, p, 0.01, 2.5)
    for k, (lo, hi) in BANDS.items():
        out[f"pow_{k}_rel"] = band_power(f, p, lo, hi) / tot
    out["resp_rate_bpm"] = 60 * peak_freq(long_, fs, 0.13, 0.6)
    out["mayer_freq_hz"] = peak_freq(long_, fs, 0.06, 0.15)
    out["hbo_hbr_slow_corr"] = float(np.corrcoef(bp(long_, fs, 0.01, 0.15), bp(long_r, fs, 0.01, 0.15))[0, 1])

    out.update(hb_features(h, idx, dead, fs))

    # --- head motion from the cap accelerometer
    acc = np.vstack([r["aux"][k] for k in ("acc-x", "acc-y", "acc-z") if k in r["aux"]])
    if acc.size:
        mag = np.linalg.norm(acc, axis=0)
        out["motion_acc_sd"] = float(np.std(np.diff(mag)))
    return out


def bh_features(path):
    sid = int(re.search(r"_V2_(\d+)_", Path(path).name).group(1))
    mk = haemo.read_snirf(path)["markers"]
    d = [next((t for t, c in mk[k + 1:] if c == "1"), np.nan) - t2 for k, (t2, c) in enumerate(mk) if c == "2"]
    d = [x for x in d if np.isfinite(x)]
    return {"subject": sid, "bh_max_s": max(d) if d else np.nan, "bh_mean_s": float(np.mean(d)) if d else np.nan}


def _safe(fn_path):
    fn, path = fn_path
    try:
        return fn(path)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {Path(path).name}: {e!r}")
        return None


# ---------------------------------------------------------------------------
def load_meta():
    m = pd.read_csv(D / "subjects_description.csv", decimal=",")
    hair = m["Kolor włosów"].str.strip().str.lower()
    dark = {"no hair": 0, "brak": 0, "siwe": 1, "blond": 2, "ciemny blond": 2.5, "brązowe": 3, "brazowe": 3,
            "brażowe": 3, "czarne": 4, "black": 4}
    typ = m["Typ włosów"].str.strip().str.lower().replace({"straigth": "straight", "brak": "no hair"})
    tex = m["Tekstura włosów"].str.strip().str.lower().replace({"brak": "no hair"})
    meds = m["Przyjmowane leki"].fillna("-").str.strip()
    t = pd.DataFrame({
        "subject": m["Nr osoby badanej"],
        "sex_male": (m["Płeć"] == "Mężczyzna").astype(int),
        "age": m["Wiek"], "height": m["Wzrost"], "weight": m["Waga"],
        "bmi": m["Waga"] / (m["Wzrost"] / 100) ** 2,
        "head_circ": m["Rozmiar głowy"],
        "hair_darkness": hair.map(dark),
        "hair_curl": typ.map({"no hair": 0, "straight": 1, "wavy": 2, "curly": 3, "kinky": 4}),
        "hair_thickness": tex.map({"no hair": 0, "fine": 1, "medium": 2, "coarse": 3}),
        "fitzpatrick": m["Fitzpatrick"].map({"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}),
        "medication_any": (meds != "-").astype(int),
        "antihypertensive": meds.str.contains("nadci").astype(int),
        "activity": m["Ocena aktywności fizycznej"],
        "treadmill_kmh": m["Prędkość bieżni"].str.replace(",", ".").str.extract(r"([\d.]+)")[0].astype(float),
        "NEG_CONTROL_group": m["Grupa"],
    })
    return t


def main():
    from scipy.stats import false_discovery_control, spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = [(baseline_features, f) for f in sorted(glob.glob(str(D / "data/baseline/*.snirf")))]
    jobs += [(bh_features, f) for f in sorted(glob.glob(str(D / "data/BH/*.snirf")))]
    with Pool(6) as p:
        res = [r for r in p.map(_safe, jobs) if r]
    base = pd.DataFrame([r for r in res if "hr_bpm" in r])
    bh = pd.DataFrame([r for r in res if "bh_max_s" in r])
    feat = base.merge(bh, on="subject", how="outer")
    meta = load_meta()
    df = feat.merge(meta, on="subject", how="left")
    df.to_csv(ROOT / "results/features.csv", index=False)
    F = [c for c in feat.columns if c != "subject"]
    T = [c for c in meta.columns if c != "subject"]

    rows = []
    for t in T + ["age_lt60"]:
        for f in F:
            d = df[list(dict.fromkeys([f, "age" if t == "age_lt60" else t, "age"]))].dropna()
            if t == "age_lt60":
                d = d[d["age"] < 60]
                x = d["age"]
            else:
                x = d[t]
            if x.nunique() < 2 or len(d) < 10:
                continue
            rho, p = spearmanr(d[f], x)
            rows.append(dict(feature=f, target=t, rho=rho, p=p, n=len(d)))
    res = pd.DataFrame(rows).dropna(subset=["p"])
    res["q_fdr"] = false_discovery_control(res["p"].values)
    res = res.sort_values("p")
    res.to_csv(ROOT / "results/correlations.csv", index=False)

    n = len(res)
    print(f"{len(df)} people, {len(F)} fNIRS parameters x {len(T) + 1} subject variables = {n} tests")
    print(f"p < 0.05: {(res.p < .05).sum()}  (expected by pure chance: {0.05 * n:.0f})   "
          f"FDR q < 0.05: {(res.q_fdr < .05).sum()}   q < 0.10: {(res.q_fdr < .10).sum()}")
    neg = res[res.target == "NEG_CONTROL_group"]
    print(f"NEGATIVE control (Grupa): smallest p {neg.p.min():.3f}, smallest q {neg.q_fdr.min():.2f} "
          f"-> {'OK, nothing' if neg.q_fdr.min() >= 0.1 else 'WARNING: false positives possible'}")
    print("\nsurvivors of FDR (q < 0.10):")
    print(res[res.q_fdr < 0.10].round(4).to_string(index=False))

    # heatmap: signed -log10 q
    piv = res[res.target != "age_lt60"].pivot(index="feature", columns="target", values="q_fdr")
    sg = res[res.target != "age_lt60"].pivot(index="feature", columns="target", values="rho")
    Z = -np.log10(piv.clip(lower=1e-6)) * np.sign(sg)
    fig, ax = plt.subplots(figsize=(11, 12))
    im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(Z.columns)), Z.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(Z.index)), Z.index)
    for (i, j), q in np.ndenumerate(piv.values):
        if np.isfinite(q) and q < 0.1:
            ax.text(j, i, f"{sg.values[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="±log10(q FDR)  (znak = kierunek korelacji)")
    ax.set_title("Parametry fNIRS vs dane osób (Spearman, FDR); liczby = rho dla q < 0,10")
    fig.tight_layout()
    fig.savefig(ROOT / "results/correlations.png", dpi=110)
    print(f"\n-> features.csv, correlations.csv, correlations.png")


if __name__ == "__main__":
    main()
