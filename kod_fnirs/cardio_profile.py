"""
cardio_profile.py - three-module circulation profile from ONE resting fNIRS recording
(supine 141.79 Hz baseline, Cortivision C23, prefrontal montage).

Usage:
  python cardio_profile.py                                  # whole cohort -> CSV + methods PDF + all cards
  python cardio_profile.py --subjects 25 60                 # cohort is still the reference, cards only for 25, 60
  python cardio_profile.py --files new_person.snirf         # score a NEW recording against the cohort
  python cardio_profile.py --no-report --no-cards           # numbers only (CSV)

Output (in --out-dir, default cardio_out/):
  cardio_profile.csv         one row per person, all numbers + percentiles + category
  raport_metody_wyniki.pdf   step-by-step explanation, cohort results, validation, conclusions
  karty/karta_<ID>.pdf       one-page card per selected person (scales vs the cohort)

Modules (all from the same fixed set of 4 forehead channels + 2 short channels):
  1. ELASTICITY  ("stan rur")      relax_area of the averaged beat (PReFx-like, Fabiani 2014),
                                   corrected to 75 bpm with the cohort HR slope, + second hump present?
                                   -> T-score vs young reference (18-30 y, no medication) and percentile
  2. AUTONOMIC   ("stan sterownika") HR and pulse-rate variability (RMSSD, beat-to-beat, max-upstroke
                                   fiducial, parabolic sub-sample timing, 20 % artefact rule)
  3. SKIN vs DEEP ("10 vs 30 mm")  short-channel (skin) pulse: amplitude + relax_area; share of the
                                   long-channel cardiac signal explained by the skin (R^2); deep pulse =
                                   long - beta*short, averaged beat-synchronously -> deep relax_area
Every score is RELATIVE to this cohort (mostly healthy 18-42 y). It is a screening demo, not a diagnosis.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fnirs_toolkit as fk  # noqa: E402
import stiffness as st  # noqa: E402

FOREHEAD = ["S6_D4", "S5_D3", "S6_D5", "S5_D4"]  # best pulse + age relation (see channel survey)
SHORT = ["S7_D4", "S8_D5"]  # 10 mm, skin only
N_POINTS = 200
REF_AGE = (18, 30)  # young reference group for T-scores (like bone densitometry)
LOW_PCT = 25  # "obniżone" = lowest quarter of the cohort


# ---------------------------------------------------------------------------
# Beat-level signal processing
# ---------------------------------------------------------------------------
def cardiac(x, fs):
    """Zero-phase band-pass 0.5-10 Hz: keeps the heartbeat and its harmonics (the shape)."""
    return fk.bandpass(x, fs, 0.5, min(10.0, 0.45 * fs), order=3)


def find_beats(xc, fs, period):
    """Troughs (beat starts) -> list of (a, b) sample pairs of plausible beats."""
    tr, _ = signal.find_peaks(-xc, distance=max(1, int(0.6 * period * fs)), prominence=0.3 * np.std(xc))
    return [(a, b) for a, b in zip(tr[:-1], tr[1:]) if 0.7 * period < (b - a) / fs < 1.3 * period]


def resample(xc, a, b):
    return np.interp(np.linspace(a, b, N_POINTS), np.arange(a, b + 1), xc[a:b + 1])


def mean_beat(xc, fs, beats, ref=None):
    """Average beat. Each beat: trough->trough, 200 points, 0..1; keep beats with r>0.9 vs the median.
    ref (optional): signal that defines which beats are good (used for the deep residual, whose single
    beats are too noisy to judge). Returns wave 0..1, median duration [s], n kept, median amplitude."""
    if len(beats) < 30:
        return None, np.nan, len(beats), np.nan
    raw = np.array([resample(xc, a, b) for a, b in beats])
    judge = raw if ref is None else np.array([resample(ref, a, b) for a, b in beats])
    rng = np.ptp(judge, axis=1, keepdims=True)
    norm = (judge - judge.min(axis=1, keepdims=True)) / np.where(rng > 0, rng, 1)
    med = np.median(norm, axis=0)
    keep = np.array([np.corrcoef(b, med)[0, 1] for b in norm]) > 0.9
    if keep.sum() < 30:
        return None, np.nan, int(keep.sum()), np.nan
    if ref is None:
        w = norm[keep].mean(axis=0)
    else:  # residual: average first, normalise after (single residual beats are noise-dominated)
        w = raw[keep].mean(axis=0)
        w = w - np.linspace(w[0], w[-1], N_POINTS)  # remove the linear drift across the beat
    w = (w - w.min()) / (np.ptp(w) + 1e-30)
    durs = np.array([(b - a) / fs for a, b in beats])[keep]
    return w, float(np.median(durs)), int(keep.sum()), float(np.median(np.ptp(raw[keep], axis=1)))


def second_hump(w):
    """Height of the second (reflected / dicrotic) peak above the dip before it, 0 if there is none."""
    k = int(np.argmax(w))
    fall = w[k:]
    pk, pr = signal.find_peaks(fall, prominence=0.003)
    return float(pr["prominences"].max()) if len(pk) else 0.0


def pulse_intervals(xc, fs, beats):
    """Inter-beat intervals [s] from the steepest point of each upstroke (sub-sample, parabolic)."""
    d = np.gradient(xc)
    t = []
    for a, b in beats:
        seg = d[a:a + max(3, (b - a) // 2)]
        i = int(np.argmax(seg))
        off = 0.0
        if 0 < i < len(seg) - 1:
            y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        t.append((a + i + off) / fs)
    t = np.unique(np.array(t))
    return np.diff(t), t


def hrv(ibi):
    """Clean intervals (20 % rule) -> HR, RMSSD, SDNN [ms]. Only successive pairs of clean beats count."""
    if len(ibi) < 60:
        return dict(hr_bpm=np.nan, rmssd_ms=np.nan, sdnn_ms=np.nan, n_ibi=len(ibi))
    med = np.median(ibi)
    ok = (ibi > 0.7 * med) & (ibi < 1.3 * med)
    ok[1:] &= np.abs(np.diff(ibi)) < 0.2 * ibi[:-1]
    good = ibi[ok]
    pair = ok[1:] & ok[:-1]
    sd = np.diff(ibi)[pair]
    rmssd, sdnn = np.sqrt(np.mean(sd ** 2)), good.std(ddof=1)
    # Poincare: SD1 = beat-to-beat spread, SD2 = slow spread. Regular sinus rhythm (even with strong
    # breathing-related variation) has SD1/SD2 well below 1; random irregular rhythm (e.g. atrial
    # fibrillation, many ectopic beats) pushes it towards 1 - then a high RMSSD is NOT good regulation.
    sd1 = rmssd / np.sqrt(2)
    sd2 = np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 1e-12))
    return dict(hr_bpm=60.0 / good.mean(), rmssd_ms=1000 * rmssd, sdnn_ms=1000 * sdnn,
                sd1_sd2=float(sd1 / sd2), n_ibi=int(ok.sum()), clean_frac=float(ok.mean()))


# ---------------------------------------------------------------------------
# One recording
# ---------------------------------------------------------------------------
def analyse(path, part=None):
    """part=None: whole recording; part=(i, n): only the i-th of n equal pieces (reliability check)."""
    path = Path(path)
    d = fk.load_snirf_haemo(str(path))
    fs = d["fs"]
    if part is not None:
        i, n = part
        L = d["hbo"].shape[1] // n
        d["hbo"] = d["hbo"][:, i * L:(i + 1) * L]
    names = [c.split()[0] for c in d["ch_hbo"]]
    info = d["raw_haemo"].info
    mid = {n: info["chs"][info["ch_names"].index(c)]["loc"][:3] for n, c in zip(names, d["ch_hbo"])}
    sig = {n: d["hbo"][i] * 1e6 for i, n in enumerate(names)}  # mol/L -> umol/L
    missing = [c for c in FOREHEAD + SHORT if c not in sig]
    if missing:
        raise ValueError(f"missing channels {missing}")
    xc = {n: cardiac(sig[n], fs) for n in FOREHEAD + SHORT}
    q = {n: fk.cardiac_quality(sig[n], fs) for n in FOREHEAD + SHORT}
    best = max(FOREHEAD, key=lambda n: np.nan_to_num(q[n], nan=-1))
    hr0 = fk.heart_rate(sig[best], fs)
    period = 60.0 / hr0
    out = {"file": path.name, "subject": st.subject_id(path), "fs": fs, "best_channel": best,
           "quality_best": q[best]}

    # Module 1: elasticity from the 4 forehead channels
    waves, durs, nb, beats_of = [], [], [], {}
    for n in FOREHEAD:
        beats_of[n] = find_beats(xc[n], fs, period)
        w, dur, k, _ = mean_beat(xc[n], fs, beats_of[n])
        if w is not None and q[n] >= 4:
            waves.append(w); durs.append(dur); nb.append(k)
    out["n_channels"] = len(waves)
    if waves:
        w = np.mean(waves, axis=0)
        w = (w - w.min()) / np.ptp(w)
        out.update(st.pulse_features(w))
        out["hump"] = second_hump(w)
        out["beat_ms"] = 1000 * float(np.median(durs))
        out["n_beats"] = int(np.median(nb))
        out["_wave"] = w

    # Module 2: autonomic balance from the best forehead channel
    ibi, tb = pulse_intervals(xc[best], fs, beats_of[best])
    out.update(hrv(ibi))
    out["_ibi"], out["_tb"] = ibi, tb

    # Module 3: skin (10 mm) vs deep (30 mm)
    skin, deep, r2, amp_s, amp_l = [], [], [], [], []
    for n in FOREHEAD:
        s = min(SHORT, key=lambda m: np.linalg.norm(mid[m] - mid[n]))
        xl, xs = xc[n], xc[s]
        beta = float(np.dot(xl, xs) / np.dot(xs, xs))
        resid = xl - beta * xs
        r2.append(1 - np.var(resid) / np.var(xl))
        wd, _, k, _ = mean_beat(resid, fs, beats_of[n], ref=xl)
        if wd is not None:
            deep.append(wd)
        _, _, _, a = mean_beat(xl, fs, beats_of[n])
        amp_l.append(a)
    for s in SHORT:
        bs = find_beats(xc[s], fs, period)
        ws, _, k, a = mean_beat(xc[s], fs, bs)
        if ws is not None and q[s] >= 4:
            skin.append(ws); amp_s.append(a)
    out["skin_share"] = float(np.median(r2))  # part of the 30 mm pulse that the skin channel explains
    out["skin_amp_uM"] = float(np.median(amp_s)) if amp_s else np.nan
    out["long_amp_uM"] = float(np.nanmedian(amp_l))
    if skin:
        ws = np.mean(skin, axis=0)
        out["skin_relax_area"] = st.pulse_features(ws)["relax_area"]
        out["skin_hump"] = second_hump(ws)
        out["_skin_wave"] = ws
    if deep:
        wd = np.mean(deep, axis=0)
        wd = (wd - wd.min()) / np.ptp(wd)
        out["deep_relax_area"] = st.pulse_features(wd)["relax_area"]
        out["deep_hump"] = second_hump(wd)
        out["_deep_wave"] = wd
    return out


def _safe(arg):
    path, part = arg if isinstance(arg, tuple) else (arg, None)
    try:
        return analyse(path, part)
    except Exception as e:  # noqa: BLE001
        return {"file": Path(path).name, "subject": st.subject_id(Path(path)), "error": str(e)}


def run_files(files, jobs, part=None):
    with Pool(jobs) as pool:
        rows = pool.map(_safe, [(str(f), part) for f in files])
    for r in rows:
        if "error" in r:
            print(f"FAIL {r['file']}: {r['error']}")
    return pd.DataFrame([r for r in rows if "error" not in r])


# ---------------------------------------------------------------------------
# Cohort scoring
# ---------------------------------------------------------------------------
def fit_reference(ref: pd.DataFrame) -> dict:
    """Everything a new person is scored against: HR slopes, young reference mean/SD, cohort samples.
    Main elasticity index = DEEP relax_area (30 mm minus skin), corrected to 75 bpm: on the real
    supine data it is the most age-related (also < 60 y and without medication) and repeatable index."""
    fits = {}
    for col in ("relax_area", "deep_relax_area"):
        ok = ref[[col, "hr_bpm"]].notna().all(axis=1)
        slope, _ = np.polyfit(ref.loc[ok, "hr_bpm"], ref.loc[ok, col], 1)
        rho, p = spearmanr(ref.loc[ok, "hr_bpm"], ref.loc[ok, col])
        fits[col] = dict(slope=float(slope), rho=float(rho), p=float(p))
    ra75 = ref["deep_relax_area"] - fits["deep_relax_area"]["slope"] * (ref["hr_bpm"] - 75)
    young = ref["age"].between(*REF_AGE) & ref["meds"].fillna("-").str.strip().eq("-")
    return {"hr_fit": fits, "ref_mean": float(ra75[young].mean()), "ref_sd": float(ra75[young].std()),
            "n_ref": int(young.sum()),
            "cohort": {"elast75": ra75.dropna().values, "rmssd_ms": ref["rmssd_ms"].dropna().values,
                       "skin_amp_uM": ref["skin_amp_uM"].dropna().values,
                       "skin_share": ref["skin_share"].dropna().values}}


def pct(values, x):
    """Percentile of x within the cohort values (0-100, mid-rank for ties)."""
    if not np.isfinite(x) or len(values) == 0:
        return np.nan
    return 100.0 * (np.sum(values < x) + 0.5 * np.sum(values == x)) / len(values)


CATEGORIES = {
    "optimal": ("Profil optymalny",
                "Naczynia elastyczne na tle grupy, spokojny rytm serca z dobrą zmiennością."),
    "skin": ("Chwilowa reakcja skórna / napięcie",
             "Głębszy puls elastyczny, ale słaby puls skóry czoła i niska zmienność rytmu: "
             "obraz pasujący do stresu lub chłodu w trakcie badania. Warto powtórzyć pomiar w spokoju."),
    "autonomic": ("Podwyższone napięcie autonomiczne",
                  "Naczynia elastyczne, ale niska zmienność rytmu serca (przewaga układu współczulnego: "
                  "stres, zmęczenie, kofeina, słaba kondycja)."),
    "eva": ("Sygnał ostrzegawczy: sztywniejsze naczynia",
            "Kształt pulsu sztywniejszy niż u 3/4 grupy. Warto zmierzyć ciśnienie "
            "i skonsultować wynik; regularny wysiłek aerobowy poprawia elastyczność tętnic."),
    "overload": ("Przeciążenie naczyniowo-autonomiczne",
                 "Jednocześnie sztywniejszy kształt pulsu i niska zmienność rytmu serca. "
                 "Warto zmierzyć ciśnienie i skonsultować wynik z lekarzem."),
    "irregular": ("Zmienność rytmu do weryfikacji",
                  "Odstępy między uderzeniami zmieniają się szybko i mało regularnie (wysokie SD1/SD2). "
                  "U młodych, aktywnych osób to zwykle silna modulacja oddechowa (dobry znak), u starszych "
                  "może to być nieregularny rytm (np. pobudzenia dodatkowe) - rozstrzyga dopiero EKG."),
}
IRREGULAR_SD1_SD2 = 0.75  # cohort: 3 of 4 people aged 68-78 (all on antihypertensives) are above it


def score(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    df = df.copy()
    c = ref["cohort"]
    f = ref["hr_fit"]
    df["relax75"] = df["relax_area"] - f["relax_area"]["slope"] * (df["hr_bpm"] - 75)
    df["elast75"] = df["deep_relax_area"] - f["deep_relax_area"]["slope"] * (df["hr_bpm"] - 75)
    df["elast_T"] = (df["elast75"] - ref["ref_mean"]) / ref["ref_sd"]
    df["elast_pct"] = [pct(c["elast75"], x) for x in df["elast75"]]
    df["rmssd_pct"] = [pct(c["rmssd_ms"], x) for x in df["rmssd_ms"]]
    df["skin_pct"] = [pct(c["skin_amp_uM"], x) for x in df["skin_amp_uM"]]
    df["skin_share_pct"] = [pct(c["skin_share"], x) for x in df["skin_share"]]
    df["hump_present"] = df["hump"] > 0.01
    df["irregular"] = df["sd1_sd2"] > IRREGULAR_SD1_SD2
    cats = []
    for _, r in df.iterrows():
        if not np.isfinite(r["elast_pct"]) or not np.isfinite(r["rmssd_pct"]):
            cats.append("brak danych"); continue
        stiff, tense, skin_low = r["elast_pct"] < LOW_PCT, r["rmssd_pct"] < LOW_PCT, r["skin_pct"] < LOW_PCT
        if r["irregular"]:
            key = "eva" if stiff else "irregular"
        else:
            key = ("overload" if stiff and tense else "eva" if stiff else
                   ("skin" if skin_low else "autonomic") if tense else "optimal")
        cats.append(key)
    df["category_key"] = cats
    df["category"] = [CATEGORIES[k][0] if k in CATEGORIES else k for k in cats]
    return df


def validation(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman of every index with age (all, <60) and activity; partial with activity | age."""
    rows = []
    for col, label in [("relax_area", "relax_area (surowe)"), ("relax75", "relax_area @75 bpm"),
                       ("elast75", "ELASTYCZNOŚĆ: głęboki @75 bpm"),
                       ("hump", "wysokość 2. garbu"), ("hr_bpm", "tętno"), ("rmssd_ms", "RMSSD"),
                       ("sdnn_ms", "SDNN"), ("sd1_sd2", "SD1/SD2 (nieregularność)"), ("skin_relax_area", "relax_area skóry (10 mm)"),
                       ("deep_relax_area", "relax_area głęboki (30 mm - skóra)"),
                       ("skin_share", "udział skóry w pulsie (R²)"), ("skin_amp_uM", "amplituda pulsu skóry")]:
        d = df.dropna(subset=[col, "age"])
        y = d[d["age"] < 60]
        r, p = spearmanr(d["age"], d[col])
        ry, py = spearmanr(y["age"], y[col])
        ra, pa = spearmanr(d["activity"], d[col], nan_policy="omit")
        rp, pp = fk.partial_corr(d[col], d["activity"], d["age"])
        rows.append(dict(index=label, col=col, n=len(d), rho_age=r, p_age=p, rho_age_lt60=ry, p_age_lt60=py,
                         rho_act=ra, p_act=pa, rho_act_given_age=rp, p_act_given_age=pp))
    return pd.DataFrame(rows)


RELIABILITY_COLS = [("relax_area", "relax_area (30 mm)"), ("hump", "wysokość 2. garbu"),
                    ("deep_relax_area", "relax_area głęboki"), ("skin_relax_area", "relax_area skóry"),
                    ("skin_share", "udział skóry (R²)"), ("skin_amp_uM", "amplituda pulsu skóry"),
                    ("hr_bpm", "tętno"), ("rmssd_ms", "RMSSD"), ("sd1_sd2", "SD1/SD2")]


def reliability(halves: pd.DataFrame) -> pd.DataFrame:
    """Same person, first 5 min vs last 5 min: is the index a stable trait or noise?"""
    w = halves.pivot_table(index="subject", columns="half", values=[c for c, _ in RELIABILITY_COLS])
    rows = []
    for c, label in RELIABILITY_COLS:
        d = w[c].dropna()
        r, p = spearmanr(d[1], d[2])
        rows.append(dict(index=label, col=c, n=len(d), rho_halves=r, p=p))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(st.DEFAULT_DIR), help="reference cohort (.snirf folder)")
    ap.add_argument("--meta", default=str(st.DEFAULT_META))
    ap.add_argument("--subjects", type=int, nargs="*", help="cohort IDs that get a card (default: all)")
    ap.add_argument("--files", nargs="*", default=[], help="extra recordings scored against the cohort")
    ap.add_argument("--out-dir", default="cardio_out")
    ap.add_argument("--cache", default="cardio_cache.pkl", help="cohort results, reused on the next run")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--no-cards", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = st.load_meta(Path(args.meta))
    cache = Path(args.cache)
    if cache.exists():
        cohort = pd.read_pickle(cache)
    else:
        cohort = run_files(sorted(Path(args.data_dir).glob("*.snirf")), args.jobs)
        cohort.to_pickle(cache)
    halves_cache = cache.with_name(cache.stem + "_halves.pkl")
    if halves_cache.exists():
        halves = pd.read_pickle(halves_cache)
    else:
        files = sorted(Path(args.data_dir).glob("*.snirf"))
        h = [run_files(files, args.jobs, part=(i, 2)).assign(half=i + 1) for i in range(2)]
        halves = pd.concat(h, ignore_index=True)
        halves.to_pickle(halves_cache)
    rel = reliability(halves)
    cohort = cohort.merge(meta, on="subject", how="left")
    ref = fit_reference(cohort)
    cohort = score(cohort, ref)
    cohort["source"] = "cohort"

    extra = pd.DataFrame()
    if args.files:
        extra = run_files([Path(f) for f in args.files], args.jobs)
        extra = score(extra.merge(meta, on="subject", how="left"), ref)
        extra["source"] = "new"

    allp = pd.concat([cohort, extra], ignore_index=True)
    cols = [c for c in allp.columns if not c.startswith("_")]
    allp[cols].sort_values(["source", "subject"]).to_csv(out / "cardio_profile.csv", index=False)
    val = validation(cohort)
    val.to_csv(out / "walidacja.csv", index=False)
    print(f"{len(cohort)} cohort + {len(extra)} new -> {out / 'cardio_profile.csv'}")
    for k, v in ref["hr_fit"].items():
        print(f"HR slope of {k}: {v['slope']:+.5f} per bpm (rho {v['rho']:+.2f}, p {v['p']:.3f})")
    print(f"young reference (age {REF_AGE[0]}-{REF_AGE[1]}, no medication) n={ref['n_ref']}")
    print(val.round(3).to_string(index=False))
    rel.to_csv(out / "powtarzalnosc.csv", index=False)
    print("\nsplit-half reliability (first vs second half of the recording):")
    print(rel.round(3).to_string(index=False))
    print(cohort["category"].value_counts().to_string())

    if args.no_report and args.no_cards:
        return
    import cardio_report as rep  # matplotlib-only PDF builder
    if not args.no_report:
        rep.methods_report(cohort, ref, val, rel, args.data_dir, out / "raport_metody_wyniki.pdf")
        print(f"report -> {out / 'raport_metody_wyniki.pdf'}")
    if not args.no_cards:
        sel = cohort if args.subjects is None else cohort[cohort["subject"].isin(args.subjects)]
        sel = pd.concat([sel, extra], ignore_index=True)
        (out / "karty").mkdir(exist_ok=True)
        for _, r in sel.iterrows():
            name = f"karta_{int(r['subject']):02d}.pdf" if pd.notna(r["subject"]) else f"karta_{Path(r['file']).stem}.pdf"
            rep.person_card(r, cohort, ref, out / "karty" / name)
        print(f"{len(sel)} cards -> {out / 'karty'}")


if __name__ == "__main__":
    main()
