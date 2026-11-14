"""
stiffness.py - per-person arterial stiffness index from the fNIRS pulse wave.

Best input: fnirs_high_freq_supine baseline (141.79 Hz, 10 min rest) - the pulse SHAPE is resolved.

Usage:
  python stiffness.py                                   # whole supine baseline folder -> stiffness.csv
  python stiffness.py --data-dir DIR --out out.csv --fig stiffness.png
  python stiffness.py --file ONE.snirf                  # one person, features printed

Per recording:
  1. HbO of the long channels with the clearest heartbeat (top-k by cardiac quality)
  2. band-pass 0.5-10 Hz (keeps the harmonics that make the pulse shape), cut trough->trough,
     reject odd beats (length or shape outliers), average -> one clean pulse per channel
  3. pulse features (median over channels):
       rise_frac  share of the beat spent rising to the peak       stiff -> lower  (sharper upstroke)
       relax_area area between the decay and a straight line        stiff -> lower  (PReFx-like,
                  peak -> end of beat (pulse relaxation function)                    Fabiani 2014)
       b_a        second-derivative (SDPPG) b/a ratio               stiff -> higher (Takazawa 1998,
                                                                                     rises with age)
  4. stiffness_index = -z(relax_area) across the cohort. Real supine baseline (59 people):
     relax_area vs age Spearman -0.37 (p=0.004), still -0.27 (p=0.045) without the 4 people
     aged 68-78; rise_frac and b_a showed no age relation (|rho| < 0.12), so they are only reported.
     Higher = stiffer RELATIVE to the other people in the folder. It is not a calibrated PWV.

Validation printed at the end: Spearman of every feature with age and physical activity, and
activity | age partial correlation (fnirs_toolkit.partial_corr).
"""

from __future__ import annotations

import argparse
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fnirs_toolkit as fk  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE.parent / "dane/infopack_and_data/fnirs_high_freq_supine/data/baseline"
DEFAULT_META = HERE.parent / "dane/infopack_and_data/fnirs_high_freq_supine/subjects_description.csv"
FEATURES = ["rise_frac", "relax_area", "b_a"]
INDEX_FEATURES = ["relax_area"]  # only feature that tracked age on the real supine data (see docstring)
STIFF_SIGN = {"rise_frac": -1, "relax_area": -1, "b_a": +1}  # direction in which stiffness grows
N_POINTS = 200


def mean_pulse(x: np.ndarray, fs: float):
    """Ensemble-averaged pulse (normalised 0..1, N_POINTS samples), median beat period [s], n beats."""
    xc = fk.bandpass(x, fs, 0.5, min(10.0, 0.45 * fs), order=3)
    hr = fk.heart_rate(x, fs)
    if not np.isfinite(hr):
        return None, np.nan, 0
    period = 60.0 / hr
    troughs, _ = signal.find_peaks(-xc, distance=max(1, int(0.6 * period * fs)), prominence=0.3 * np.std(xc))
    grid = np.linspace(0, 1, N_POINTS)
    beats, durs = [], []
    for a, b in zip(troughs[:-1], troughs[1:]):
        dur = (b - a) / fs
        if not (0.7 * period < dur < 1.3 * period):
            continue
        seg = np.interp(np.linspace(a, b, N_POINTS), np.arange(a, b + 1), xc[a:b + 1])
        rng = seg.max() - seg.min()
        if rng > 0:
            beats.append((seg - seg.min()) / rng)
            durs.append(dur)
    if len(beats) < 30:
        return None, np.nan, len(beats)
    B = np.array(beats)
    med = np.median(B, axis=0)
    r = np.array([np.corrcoef(b, med)[0, 1] for b in B])
    keep = r > 0.9  # drop beats with a broken shape (motion, missed trough)
    if keep.sum() < 30:
        return None, np.nan, int(keep.sum())
    w = B[keep].mean(axis=0)
    w = (w - w.min()) / (w.max() - w.min())
    return w, float(np.median(np.array(durs)[keep])), int(keep.sum())


def pulse_features(w: np.ndarray) -> dict:
    ph = np.linspace(0, 1, len(w))
    k = int(np.argmax(w))
    fall = w[k:]
    line = np.linspace(w[k], w[-1], len(fall))
    relax_area = float(np.trapezoid(fall - line, ph[k:]))
    # SDPPG: a = first (positive) peak of the 2nd derivative on the upstroke, b = the dip after it
    d2 = np.gradient(np.gradient(signal.savgol_filter(w, 15, 3)))
    up = d2[: max(k, 3)]
    ia = int(np.argmax(up))
    a = up[ia]
    b = d2[ia:k + 1].min() if k > ia else np.nan
    return {"rise_frac": float(ph[k]), "relax_area": relax_area,
            "b_a": float(b / a) if a > 0 else np.nan}


def subject_id(path: Path):
    m = re.search(r"(?:^|_)(\d+)_[A-Z]{2}_", path.name)
    return int(m.group(1)) if m else None


def features_for_file(path: Path, top_k: int = 3, min_quality: float = 4.0) -> dict:
    d = fk.load_snirf_haemo(str(path))
    fs = d["fs"]
    long_idx = np.flatnonzero(~d["short_mask"])
    q = np.array([fk.cardiac_quality(d["hbo"][i], fs) for i in long_idx])
    order = [long_idx[i] for i in np.argsort(-np.nan_to_num(q, nan=-1))[:top_k] if q[i] >= min_quality]
    rows, waves = [], []
    for i in order:
        w, period, n = mean_pulse(d["hbo"][i], fs)
        if w is None:
            continue
        f = pulse_features(w)
        f.update(period_s=period, n_beats=n)
        rows.append(f)
        waves.append(w)
    out = {"file": path.name, "subject": subject_id(path), "fs": fs, "n_channels": len(rows)}
    if rows:
        for k in rows[0]:
            out[k] = float(np.nanmedian([r[k] for r in rows]))
        out["hr_bpm"] = 60.0 / out["period_s"]
        out["_wave"] = np.mean(waves, axis=0)
    return out


def _safe(args):
    path, top_k = args
    try:
        return features_for_file(path, top_k)
    except Exception as e:
        return {"file": path.name, "subject": subject_id(path), "error": str(e)}


def add_index(df):
    ok = df[INDEX_FEATURES].notna().all(axis=1)
    z = [(STIFF_SIGN[f] * (df[f] - df.loc[ok, f].mean()) / df.loc[ok, f].std()) for f in INDEX_FEATURES]
    df["stiffness_index"] = np.where(ok, np.mean(z, axis=0), np.nan)
    df["stiffness_rank"] = df["stiffness_index"].rank(ascending=False)  # 1 = stiffest
    return df


def load_meta(path: Path):
    import pandas as pd
    m = pd.read_csv(path, decimal=",")
    m = m.rename(columns={"Nr osoby badanej": "subject", "Wiek": "age",
                          "Ocena aktywności fizycznej": "activity", "Przyjmowane leki": "meds", "Płeć": "sex"})
    return m[["subject", "age", "sex", "activity", "meds"]]


def report(df) -> None:
    from scipy.stats import spearmanr
    d = df.dropna(subset=["stiffness_index"])
    print(f"\n{len(d)}/{len(df)} recordings usable")
    if "age" in d:
        print("Spearman rho (p)        age              activity         activity|age")
        for f in FEATURES + ["stiffness_index", "hr_bpm"]:
            cells = []
            for col in ("age", "activity"):
                r, p = spearmanr(d[f], d[col], nan_policy="omit")
                cells.append(f"{r:+.2f} ({p:.3f})")
            r, p = fk.partial_corr(d[f], d["activity"], d["age"])
            cells.append(f"{r:+.2f} ({p:.3f})")
            print(f"  {f:<20}  " + "   ".join(cells))
    cols = [c for c in ("subject", "age", "activity", "stiffness_index") if c in d]
    s = d.sort_values("stiffness_index", ascending=False)[cols]
    print("\nstiffest 5:\n" + s.head(5).to_string(index=False))
    print("most elastic 5:\n" + s.tail(5).to_string(index=False))


def figure(df, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.dropna(subset=["stiffness_index"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ph = np.linspace(0, 1, N_POINTS)
    for _, r in d.sort_values("stiffness_index").iterrows():
        c = plt.cm.coolwarm((r["stiffness_index"] - d["stiffness_index"].min())
                            / (np.ptp(d["stiffness_index"]) + 1e-12))
        ax[0].plot(ph, r["_wave"], color=c, lw=0.8, alpha=0.8)
    ax[0].set_title("mean pulse per person (red = stiffer index)")
    ax[0].set_xlabel("phase of heartbeat")
    if "age" in d:
        sc = ax[1].scatter(d["age"], d["stiffness_index"], c=d["activity"], cmap="viridis")
        fig.colorbar(sc, ax=ax[1], label="physical activity score")
        ax[1].set_xlabel("age [years]")
    ax[1].set_ylabel("stiffness index (z, relative)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"figure -> {path}")


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DIR), help="folder with .snirf files")
    ap.add_argument("--meta", default=str(DEFAULT_META), help="subjects_description.csv ('' to skip)")
    ap.add_argument("--file", help="analyse a single .snirf and print its features")
    ap.add_argument("--out", default="stiffness.csv")
    ap.add_argument("--fig", default=None, help="PNG with pulse shapes and index vs age")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    if args.file:
        r = features_for_file(Path(args.file), args.top_k)
        r.pop("_wave", None)
        for k, v in r.items():
            print(f"{k:<12} {v}")
        return

    files = sorted(Path(args.data_dir).glob("*.snirf"))
    if not files:
        raise SystemExit(f"no .snirf in {args.data_dir}")
    with Pool(args.jobs) as pool:
        rows = pool.map(_safe, [(f, args.top_k) for f in files])
    for r in rows:
        if "error" in r:
            print(f"FAIL {r['file']}: {r['error']}")
    df = pd.DataFrame(rows)
    if args.meta and Path(args.meta).exists():
        df = df.merge(load_meta(Path(args.meta)), on="subject", how="left")
    df = add_index(df)
    df.drop(columns=["_wave"], errors="ignore").to_csv(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}")
    report(df)
    if args.fig:
        figure(df, args.fig)


if __name__ == "__main__":
    main()
