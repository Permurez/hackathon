"""
screening.py - circulation + autonomic screening from ONE resting fNIRS recording
(supine baseline, Cortivision Spectrum C23, 141.79 Hz, prefrontal montage).

Usage:
  python screening.py                         # whole baseline folder -> CSVs + Spearman summary
  python screening.py --file ONE.snirf        # one person, numbers printed
  python screening.py --refresh               # ignore the cache and recompute everything
  streamlit run app.py                        # dashboard (reads the cache written here)

Pipeline (identical for every person):
  1. FIXED channels, no per-person "top-k": 4 long forehead channels (30 mm, skin + cortex)
     S6_D4 S5_D3 S6_D5 S5_D4 and 2 short ones (10 mm, skin only) S7_D4 S8_D5. A channel is dropped
     for a person only if its cardiac_quality < 4. HbO in uM, zero-phase Butterworth 0.5-10 Hz.
  2. Three signal layers
       Raw_30mm        mean of the long channels
       Scalp_10mm      mean of the short channels
       Brain_Isolated  each long channel minus beta * nearest short channel (OLS), then averaged
     skin_share_pct = 100 * R^2 of the long channel explained by the short one (mean over channels).
  3. Beats: troughs of Raw_30mm; a beat is clean if its length is within +-20 % of the median and
     its shape correlates r > 0.9 with the median beat. The same beats cut all three layers.
  4. Shape per layer (stiffness.beat_shape), on the averaged beat:
       crest_time_ms, max_upslope_uMs, upslope_norm  in native time (NOT stretched), foot -> P1,
                     P1 = first local max in phase 0.10-0.32, else minimum of the 2nd derivative
       fixed_relax_area  PReFx-like area on the FIXED phase window 0.25-1.00 (no global argmax)
       hump_convexity / hump_present   dicrotic hump in phase 0.35-0.55 vs the chord P1 -> end
     95 % CI: bootstrap over the clean beats (N_BOOT resamples of the beat average).
     HR correction: cohort linear slope vs HR removed -> *_75bpm (fixed_relax_area, crest_time_ms).
  5. HRV / PRV from the steepest upstroke of every Raw_30mm beat; every IBI more than +-20 % from
     the person's median IBI is rejected BEFORE HR, RMSSD, SDNN, SD1/SD2; RMSSD/SD1/SD2 use only
     pairs of successive clean intervals. rmssd_raw_ms keeps the uncleaned value for comparison.
  6. Within-person change: first 3 min vs last 3 min of the same baseline (HR, RMSSD,
     fixed_relax_area_75bpm) + Wilcoxon signed-rank test over the cohort.
  7. Spearman of every layer's metrics vs age (N=all and < 60 y), activity, BMI, antihypertensives.

Every number is RELATIVE to this cohort (mostly healthy 18-42 y). Screening demo, not a diagnosis.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fnirs_toolkit as fk  # noqa: E402
import stiffness as st  # noqa: E402

LONG = ["S6_D4", "S5_D3", "S6_D5", "S5_D4"]  # 30 mm forehead (no hair: F1/F2 channels left out)
SHORT = ["S7_D4", "S8_D5"]                   # 10 mm, skin only
LAYERS = ["Raw_30mm", "Scalp_10mm", "Brain_Isolated"]
LAYER_PL = {"Raw_30mm": "30 mm (skóra + kora)", "Scalp_10mm": "skóra 10 mm", "Brain_Isolated": "mózg (30 mm − skóra)"}
MIN_Q = 4.0
N_POINTS = st.N_POINTS
IBI_TOL = 0.20      # +-20 % of the median IBI
SHAPE_R = 0.9
N_BOOT = 300
SEG_S = 180.0       # first / last 3 min
HR_REF = 75.0
SHAPE_COLS = ["crest_time_ms", "max_upslope_uMs", "upslope_norm", "fixed_relax_area", "hump_convexity", "amp_uM"]
HR_CORRECTED = ["fixed_relax_area", "crest_time_ms"]
ELAST_LAYER = "Brain_Isolated"  # main elasticity index (see validation printout)
LOW_PCT = 25
SKIN_SHARE_HIGH = 92.0  # cohort 75th percentile of skin_share_pct (median 85 %)
CACHE = HERE / "screening_cache.pkl"
OUT_DIR = HERE / "cardio_out"


# ---------------------------------------------------------------------------
# Signal layers
# ---------------------------------------------------------------------------
def cardiac(x, fs):
    return fk.bandpass(x, fs, 0.5, min(10.0, 0.45 * fs), order=3)


def load_layers(path):
    d = fk.load_snirf_haemo(str(path))
    fs = d["fs"]
    names = [c.split()[0] for c in d["ch_hbo"]]
    missing = [c for c in LONG + SHORT if c not in names]
    if missing:
        raise ValueError(f"missing channels {missing}")
    info = d["raw_haemo"].info
    pos = {n: info["chs"][info["ch_names"].index(c)]["loc"][:3] for n, c in zip(names, d["ch_hbo"])}
    sig = {n: d["hbo"][names.index(n)] * 1e6 for n in LONG + SHORT}  # mol/L -> uM
    q = {n: fk.cardiac_quality(sig[n], fs) for n in sig}
    xc = {n: cardiac(sig[n], fs) for n in sig}
    long_ok = [n for n in LONG if np.nan_to_num(q[n], nan=-1) >= MIN_Q]
    short_ok = [n for n in SHORT if np.nan_to_num(q[n], nan=-1) >= MIN_Q]
    if not long_ok:
        raise ValueError("no 30 mm channel with cardiac_quality >= 4")
    layers = {"Raw_30mm": np.mean([xc[n] for n in long_ok], axis=0),
              "Scalp_10mm": np.mean([xc[n] for n in short_ok], axis=0) if short_ok else None,
              "Brain_Isolated": None}
    r2 = []
    if short_ok:
        res = []
        for n in long_ok:
            s = min(short_ok, key=lambda m: np.linalg.norm(pos[m] - pos[n]))
            xl, xs = xc[n], xc[s]
            beta = float(np.dot(xl, xs) / np.dot(xs, xs))
            res.append(xl - beta * xs)
            r2.append(1.0 - np.var(res[-1]) / np.var(xl))
        layers["Brain_Isolated"] = np.mean(res, axis=0)
    meta = {"fs": fs, "n_long": len(long_ok), "n_short": len(short_ok), "long_used": " ".join(long_ok),
            "short_used": " ".join(short_ok), "skin_share_pct": 100 * float(np.mean(r2)) if r2 else np.nan,
            **{f"q_{n}": q[n] for n in LONG + SHORT}}
    return fs, layers, meta


# ---------------------------------------------------------------------------
# Beats
# ---------------------------------------------------------------------------
def resample(x, a, b):
    return np.interp(np.linspace(a, b, N_POINTS), np.arange(a, b + 1), x[a:b + 1])


def find_troughs(x, fs):
    period = 60.0 / fk.heart_rate(x, fs)
    tr, _ = signal.find_peaks(-x, distance=max(1, int(0.6 * period * fs)), prominence=0.3 * np.std(x))
    return tr, period


def clean_beats(x, tr):
    """(a, b) trough pairs with length within +-20 % of the median and r > 0.9 vs the median shape."""
    dur = np.diff(tr)
    med = np.median(dur)
    pairs = [(a, b) for a, b in zip(tr[:-1], tr[1:]) if abs((b - a) - med) <= IBI_TOL * med]
    if len(pairs) < 30:
        return pairs
    S = np.array([resample(x, a, b) for a, b in pairs])
    S = (S - S.min(1, keepdims=True)) / (np.ptp(S, 1, keepdims=True) + 1e-30)
    m = np.median(S, 0)
    r = np.array([np.corrcoef(s, m)[0, 1] for s in S])
    return [p for p, ok in zip(pairs, r > SHAPE_R) if ok]


def layer_arrays(x, beats, L):
    beats = [(a, b) for a, b in beats if a + L <= len(x)]
    nat = np.array([x[a:a + L] for a, _ in beats])
    strt = np.array([resample(x, a, b) for a, b in beats])
    amp = np.array([np.ptp(x[a:b + 1]) for a, b in beats])
    dur = np.array([b - a for a, b in beats])
    return nat, strt, amp, dur


def shape_of_mean(nat_m, str_m, fs, D):
    """Shape features of an averaged beat. The foot is re-found on the layer itself (the skin or
    residual foot can sit a few samples off the Raw_30mm trough)."""
    foot = int(np.argmin(nat_m[: max(2, int(0.15 * D))]))
    f = st.beat_shape(nat_m[foot:], fs, D, str_m)
    f["amp_uM"] = float(np.ptp(nat_m[foot:foot + D + 1]))
    return f


def layer_features(x, fs, beats, rng, n_boot=N_BOOT):
    """Point estimate + bootstrap 95 % CI of every shape feature for one layer."""
    if x is None or len(beats) < 30:
        return {}, None
    D = int(np.median([b - a for a, b in beats]))
    L = int(1.35 * D) + 2
    nat, strt, amp, _ = layer_arrays(x, beats, L)
    if len(nat) < 30:
        return {}, None
    m_nat, m_str = nat.mean(0), strt.mean(0)
    out = shape_of_mean(m_nat, m_str, fs, D)
    out["amp_uM"] = float(np.mean(amp))  # mean single-beat peak-to-peak (not blurred by jitter)
    out["n_beats"] = len(nat)
    if n_boot:
        n = len(nat)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot) / n
        Bn, Bs, Ba = W @ nat, W @ strt, W @ amp
        boot = pd.DataFrame([shape_of_mean(Bn[i], Bs[i], fs, D) for i in range(n_boot)])
        boot["amp_uM"] = Ba
        for c in SHAPE_COLS:
            lo, hi = np.nanpercentile(boot[c].astype(float), [2.5, 97.5])
            out[f"{c}_lo"], out[f"{c}_hi"] = float(lo), float(hi)
        out["hump_present_frac"] = float(boot["hump_present"].mean())
    w = (m_str - m_str.min()) / (np.ptp(m_str) + 1e-30)
    return out, w


# ---------------------------------------------------------------------------
# HRV / PRV
# ---------------------------------------------------------------------------
def beat_fiducials(x, fs, tr):
    """Time [s] of the steepest upstroke after every trough (parabolic sub-sample refinement)."""
    d = np.gradient(x)
    t = []
    for a, b in zip(tr[:-1], tr[1:]):
        seg = d[a:a + max(3, (b - a) // 2)]
        i = int(np.argmax(seg))
        off = 0.0
        if 0 < i < len(seg) - 1:
            y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        t.append((a + i + off) / fs)
    return np.unique(np.array(t))


def hrv(ibi, rng=None, n_boot=N_BOOT):
    """IBI [s] -> HR, RMSSD, SDNN, SD1, SD2 after the +-20 %-of-median rule (+ bootstrap CI)."""
    nan = dict(hr_bpm=np.nan, rmssd_ms=np.nan, sdnn_ms=np.nan, sd1_ms=np.nan, sd2_ms=np.nan, sd1_sd2=np.nan,
               rmssd_raw_ms=np.nan, n_ibi=len(ibi), ibi_clean_frac=np.nan)
    if len(ibi) < 20:
        return nan, np.zeros(len(ibi), bool)
    med = np.median(ibi)
    ok = np.abs(ibi - med) <= IBI_TOL * med
    pair = ok[1:] & ok[:-1]
    x, y = ibi[:-1][pair], ibi[1:][pair]
    if len(x) < 10:
        return nan, ok
    dd = y - x
    sd1 = np.std(dd, ddof=1) / np.sqrt(2)
    sd2 = np.std(x + y, ddof=1) / np.sqrt(2)
    out = dict(hr_bpm=60.0 / ibi[ok].mean(), rmssd_ms=1000 * np.sqrt(np.mean(dd ** 2)),
               sdnn_ms=1000 * ibi[ok].std(ddof=1), sd1_ms=1000 * sd1, sd2_ms=1000 * sd2,
               sd1_sd2=float(sd1 / sd2), rmssd_raw_ms=1000 * np.sqrt(np.mean(np.diff(ibi) ** 2)),
               n_ibi=int(ok.sum()), ibi_clean_frac=float(ok.mean()))
    if rng is not None and n_boot:
        g = ibi[ok]
        bh = [60.0 / rng.choice(g, len(g)).mean() for _ in range(n_boot)]
        br = [1000 * np.sqrt(np.mean(rng.choice(dd, len(dd)) ** 2)) for _ in range(n_boot)]
        out["hr_bpm_lo"], out["hr_bpm_hi"] = np.percentile(bh, [2.5, 97.5])
        out["rmssd_ms_lo"], out["rmssd_ms_hi"] = np.percentile(br, [2.5, 97.5])
    return out, ok


# ---------------------------------------------------------------------------
# One recording
# ---------------------------------------------------------------------------
def analyse(path, n_boot=N_BOOT, seed=0):
    path = Path(path)
    rng = np.random.default_rng(seed)
    fs, layers, meta = load_layers(path)
    raw = layers["Raw_30mm"]
    tr, _ = find_troughs(raw, fs)
    beats = clean_beats(raw, tr)
    T = len(raw) / fs
    out = {"file": path.name, "subject": st.subject_id(path), "duration_s": T, **meta,
           "n_troughs": len(tr), "n_clean_beats": len(beats)}
    for lay in LAYERS:
        f, w = layer_features(layers[lay], fs, beats, rng, n_boot)
        out.update({f"{lay}_{k}": v for k, v in f.items()})
        out[f"_wave_{lay}"] = w
    tb = beat_fiducials(raw, fs, tr)
    ibi = np.diff(tb)
    h, ok = hrv(ibi, rng, n_boot)
    out.update(h)
    out["_ibi"], out["_ibi_ok"], out["_tb"] = ibi, ok, tb[1:]
    if out.get("Raw_30mm_amp_uM") and out.get("Scalp_10mm_amp_uM"):
        out["skin_amp_ratio_pct"] = 100 * out["Scalp_10mm_amp_uM"] / out["Raw_30mm_amp_uM"]

    # within-person: first vs last 3 minutes
    for tag, (t0, t1) in {"first3": (0.0, SEG_S), "last3": (T - SEG_S, T)}.items():
        sb = [(a, b) for a, b in beats if a / fs >= t0 and b / fs <= t1]
        for lay in ("Raw_30mm", "Brain_Isolated"):
            f, _ = layer_features(layers[lay], fs, sb, rng, n_boot=0)
            out[f"{tag}_{lay}_fixed_relax_area"] = f.get("fixed_relax_area", np.nan)
        m = (tb[1:] >= t0) & (tb[1:] <= t1)
        hs, _ = hrv(ibi[m], None, 0)
        out[f"{tag}_hr_bpm"], out[f"{tag}_rmssd_ms"] = hs["hr_bpm"], hs["rmssd_ms"]
    return out


def _safe(path):
    try:
        return analyse(path)
    except Exception as e:  # noqa: BLE001
        return {"file": Path(path).name, "subject": st.subject_id(Path(path)), "error": str(e)}


def run_folder(files, jobs):
    with Pool(jobs) as pool:
        rows = pool.map(_safe, [str(f) for f in files])
    for r in rows:
        if "error" in r:
            print(f"FAIL {r['file']}: {r['error']}")
    return pd.DataFrame([r for r in rows if "error" not in r])


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
def load_meta(path=st.DEFAULT_META):
    m = pd.read_csv(path, decimal=",")
    m = m.rename(columns={"Nr osoby badanej": "subject", "Wiek": "age", "Płeć": "sex", "Wzrost": "height_cm",
                          "Waga": "weight_kg", "Ocena aktywności fizycznej": "activity", "Przyjmowane leki": "meds"})
    m["bmi"] = m["weight_kg"] / (m["height_cm"] / 100) ** 2
    meds = m["meds"].fillna("-").str.lower()
    m["antihypertensive"] = meds.str.contains("nadci").astype(int)
    m["beta_blocker"] = meds.str.contains("nebilet").astype(int)
    return m[["subject", "age", "sex", "height_cm", "weight_kg", "bmi", "activity", "meds",
              "antihypertensive", "beta_blocker"]]


def fit_hr(df) -> dict:
    """Cohort slope of each HR-dependent shape metric vs HR (per layer)."""
    fits = {}
    for lay in LAYERS:
        for c in HR_CORRECTED:
            col = f"{lay}_{c}"
            ok = df[[col, "hr_bpm"]].notna().all(axis=1)
            if ok.sum() < 10:
                continue
            slope = float(np.polyfit(df.loc[ok, "hr_bpm"], df.loc[ok, col], 1)[0])
            rho, p = spearmanr(df.loc[ok, "hr_bpm"], df.loc[ok, col])
            fits[col] = dict(slope=slope, rho=float(rho), p=float(p))
    return fits


def apply_hr(df, fits) -> pd.DataFrame:
    df = df.copy()
    for col, f in fits.items():
        shift = f["slope"] * (df["hr_bpm"] - HR_REF)
        df[f"{col}_75bpm"] = df[col] - shift
        for s in ("lo", "hi"):
            if f"{col}_{s}" in df:
                df[f"{col}_75bpm_{s}"] = df[f"{col}_{s}"] - shift
        if col.endswith("fixed_relax_area"):
            lay = col[: -len("_fixed_relax_area")]
            for tag in ("first3", "last3"):
                c2 = f"{tag}_{lay}_fixed_relax_area"
                if c2 in df:
                    df[f"{c2}_75bpm"] = df[c2] - f["slope"] * (df[f"{tag}_hr_bpm"] - HR_REF)
    return df


def pct(values, x):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not np.isfinite(x) or len(values) == 0:
        return np.nan
    return 100.0 * (np.sum(values < x) + 0.5 * np.sum(values == x)) / len(values)


PILLARS = {
    # name: (column, higher_is_better)
    "elast": (f"{ELAST_LAYER}_fixed_relax_area_75bpm", True),
    "rmssd": ("rmssd_ms", True),
    "hr": ("hr_bpm", False),
    "perf": ("Scalp_10mm_amp_uM", True),
}


def score(df: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Percentile 0-100 (100 = best) of each pillar and its 95 % CI, relative to the cohort."""
    df = df.copy()
    for k, (col, up) in PILLARS.items():
        ref = cohort[col].values
        f = (lambda v: pct(ref, v)) if up else (lambda v: 100 - pct(ref, v))
        df[f"{k}_pct"] = [f(v) for v in df[col]]
        lo, hi = (f"{col}_lo", f"{col}_hi") if up else (f"{col}_hi", f"{col}_lo")
        if lo in df:
            df[f"{k}_pct_lo"] = [f(v) for v in df[lo]]
            df[f"{k}_pct_hi"] = [f(v) for v in df[hi]]
    df["auton_pct"] = df[["rmssd_pct", "hr_pct"]].mean(axis=1)
    df["auton_pct_lo"] = df[["rmssd_pct_lo", "hr_pct_lo"]].mean(axis=1)
    df["auton_pct_hi"] = df[["rmssd_pct_hi", "hr_pct_hi"]].mean(axis=1)
    df["irregular"] = df["sd1_sd2"] > 0.75
    return df


def screening_text(r) -> list[str]:
    """Short rule-based interpretation that combines the three pillars."""
    def low(k):
        v = r.get(f"{k}_pct", np.nan)
        return bool(np.isfinite(v) and v < LOW_PCT)
    el, au, pe = low("elast"), low("auton"), low("perf")
    share = r.get("skin_share_pct", np.nan)
    txt = []
    if el and not pe:
        txt.append("Kształt pulsu po odjęciu skóry jest sztywniejszy niż u 3/4 grupy, przy prawidłowym ukrwieniu "
                   "skóry – obraz bardziej pasujący do obniżonej elastyczności głębszych naczyń niż do chwilowej "
                   "reakcji skórnej. Warto zmierzyć ciśnienie tętnicze.")
    elif el and pe:
        txt.append("Sztywniejszy kształt pulsu współwystępuje ze słabym pulsem skóry czoła – możliwe chwilowe "
                   "obkurczenie naczyń (chłód, stres) zniekształcające pomiar. Zalecane powtórzenie w spokoju i cieple.")
    elif pe:
        txt.append("Słaby puls skóry czoła przy prawidłowym kształcie pulsu głębszego – najpewniej chwilowe "
                   "obkurczenie naczyń skóry (chłód, stres), a nie cecha naczyń głębszych.")
    else:
        txt.append("Elastyczność naczyń i ukrwienie skóry w typowym zakresie grupy.")
    if au:
        txt.append("Niska zmienność rytmu i/lub wyższe tętno spoczynkowe – przewaga napięcia współczulnego "
                   "(stres, zmęczenie, kofeina, słabsza kondycja).")
    elif r.get("auton_pct", 0) > 75:
        txt.append("Wysoka zmienność rytmu przy spokojnym tętnie – dobra przewaga przywspółczulna (regeneracja).")
    if r.get("irregular"):
        txt.append("Chmura Poincaré jest okrągła (SD1/SD2 > 0,75): odstępy zmieniają się chaotycznie. U młodych "
                   "to zwykle silna modulacja oddechowa; u starszych może to być nieregularny rytm – rozstrzyga EKG.")
    if np.isfinite(share) and share > SKIN_SHARE_HIGH:
        txt.append(f"Aż {share:.0f}% pulsu w kanałach 30 mm pochodzi ze skóry – wynik „mózgowy” ma większą niepewność.")
    dh = r.get("last3_hr_bpm", np.nan) - r.get("first3_hr_bpm", np.nan)
    dr = r.get("last3_rmssd_ms", np.nan) - r.get("first3_rmssd_ms", np.nan)
    if np.isfinite(dh) and np.isfinite(dr):
        if dh < -1 and dr > 0:
            txt.append(f"W trakcie 10 min spoczynku organizm się wyciszał: tętno {dh:+.1f} bpm, RMSSD {dr:+.0f} ms.")
        elif dh > 1 and dr < 0:
            txt.append(f"W trakcie spoczynku brak wyciszenia: tętno {dh:+.1f} bpm, RMSSD {dr:+.0f} ms.")
    return txt


DISCLAIMER = ("Wskaźnik PRZESIEWOWY z jednego 10-minutowego nagrania fNIRS, odniesiony do tej grupy "
              "(głównie zdrowi 18–42 l.). To nie jest diagnoza medyczna – przy niepokojącym wyniku: "
              "pomiar ciśnienia, EKG i konsultacja lekarska.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
TARGETS = [("age", "wiek N", None), ("age", "wiek <60", 60), ("activity", "aktywność", None),
           ("bmi", "BMI", None), ("antihypertensive", "leki nadciśn.", None)]


def validation(df) -> pd.DataFrame:
    metrics = [f"{lay}_{c}" for lay in LAYERS for c in
               ("fixed_relax_area_75bpm", "crest_time_ms_75bpm", "upslope_norm", "hump_convexity", "amp_uM")]
    metrics += ["skin_share_pct", "hr_bpm", "rmssd_ms", "rmssd_raw_ms", "sdnn_ms", "sd1_sd2"]
    rows = []
    for m in metrics:
        if m not in df:
            continue
        row = {"metric": m}
        for col, label, age_max in TARGETS:
            d = df.dropna(subset=[m, col])
            if age_max:
                d = d[d["age"] < age_max]
            r, p = spearmanr(d[m], d[col]) if len(d) > 5 else (np.nan, np.nan)
            row[f"rho {label}"], row[f"p {label}"], row[f"n {label}"] = r, p, len(d)
        rows.append(row)
    return pd.DataFrame(rows)


def print_validation(val):
    print("\nSpearman rho (p)  [* p<0.05]")
    head = f"{'metric':<38}" + "".join(f"{lab:>20}" for _, lab, _ in TARGETS)
    print(head)
    ns = "".join(f"{'n=' + str(int(val[f'n {lab}'].max())):>20}" for _, lab, _ in TARGETS)
    print(f"{'':<38}{ns}")
    for _, r in val.iterrows():
        cells = []
        for _, lab, _ in TARGETS:
            rho, p = r[f"rho {lab}"], r[f"p {lab}"]
            cells.append(f"{rho:+.2f} ({p:.3f}){'*' if p < 0.05 else ' '}" if np.isfinite(rho) else "-")
        print(f"{r['metric']:<38}" + "".join(f"{c:>20}" for c in cells))


def intra_subject(df) -> pd.DataFrame:
    rows = []
    for name, a, b in [("tętno [bpm]", "first3_hr_bpm", "last3_hr_bpm"),
                       ("RMSSD [ms]", "first3_rmssd_ms", "last3_rmssd_ms"),
                       ("fixed_relax_area_75 (30 mm)", "first3_Raw_30mm_fixed_relax_area_75bpm",
                        "last3_Raw_30mm_fixed_relax_area_75bpm"),
                       ("fixed_relax_area_75 (mózg)", "first3_Brain_Isolated_fixed_relax_area_75bpm",
                        "last3_Brain_Isolated_fixed_relax_area_75bpm")]:
        d = df.dropna(subset=[a, b])
        delta = d[b] - d[a]
        p = wilcoxon(d[a], d[b]).pvalue if len(d) > 5 else np.nan
        rows.append(dict(metric=name, n=len(d), first3_median=d[a].median(), last3_median=d[b].median(),
                         delta_median=delta.median(), frac_increase=float((delta > 0).mean()), p_wilcoxon=p))
    return pd.DataFrame(rows)


def build(data_dir=st.DEFAULT_DIR, cache=CACHE, jobs=6, refresh=False):
    """Cohort table with metadata, HR correction and percentiles (cached raw analysis)."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        raw = pd.read_pickle(cache)
    else:
        raw = run_folder(sorted(Path(data_dir).glob("*.snirf")), jobs)
        raw.to_pickle(cache)
    df = raw.merge(load_meta(), on="subject", how="left")
    fits = fit_hr(df)
    df = apply_hr(df, fits)
    df = score(df, df)
    return df.sort_values("subject").reset_index(drop=True), fits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(st.DEFAULT_DIR))
    ap.add_argument("--file", help="analyse one .snirf and print its numbers")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--refresh", action="store_true", help="recompute even if the cache exists")
    args = ap.parse_args()

    if args.file:
        r = analyse(args.file)
        for k, v in r.items():
            if not k.startswith("_"):
                print(f"{k:<45} {v}")
        return

    bh = Path(args.data_dir).parent / "BH"
    n_bh = len(list(bh.glob("*.snirf"))) if bh.exists() else 0
    print(f"baseline: {len(list(Path(args.data_dir).glob('*.snirf')))} files, breath-hold (BH): {n_bh} files")
    df, fits = build(args.data_dir, args.cache, args.jobs, args.refresh)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = [c for c in df.columns if not c.startswith("_")]
    df[cols].to_csv(out / "screening_cohort.csv", index=False)
    print(f"{len(df)} people -> {out / 'screening_cohort.csv'}")
    print("channels used: long", df["n_long"].value_counts().sort_index().to_dict(),
          " short", df["n_short"].value_counts().sort_index().to_dict())
    print(f"IBI kept after +-20 % rule: median {100 * df['ibi_clean_frac'].median():.1f} %;  RMSSD before -> after "
          f"cleaning: median {df['rmssd_raw_ms'].median():.0f} -> {df['rmssd_ms'].median():.0f} ms, "
          f"max {df['rmssd_raw_ms'].max():.0f} -> {df['rmssd_ms'].max():.0f} ms")
    for lay in LAYERS:
        c = f"{lay}_p1_is_peak"
        if c in df:
            print(f"{lay:<15} P1 = local peak in {100 * df[c].mean():.0f} % (rest: shoulder);  "
                  f"global argmax after phase 0.32 in {100 * (df[f'{lay}_argmax_phase'] > 0.32).mean():.0f} %")
    print("\nHR slopes (removed -> *_75bpm):")
    for k, f in fits.items():
        print(f"  {k:<36} {f['slope']:+.5f}/bpm  (rho vs HR {f['rho']:+.2f}, p {f['p']:.3f})")
    val = validation(df)
    val.to_csv(out / "screening_validation.csv", index=False)
    print_validation(val)
    intra = intra_subject(df)
    intra.to_csv(out / "screening_intra.csv", index=False)
    print("\nwithin-person change, first 3 min vs last 3 min of the baseline (Wilcoxon):")
    print(intra.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
