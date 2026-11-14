"""
plot_pulse_age.py - how the fNIRS pulse wave changes with age (supine 141.79 Hz baseline).

Usage:
  python plot_pulse_age.py --save pulse_age.png        # computes all subjects (cached in --cache)

Panels:
  top row    : decay (relax_area), rise time [ms], max upstroke speed [1/s] vs age, with Spearman rho
  bottom row : averaged pulse of the youngest and the oldest people on a real time axis [ms],
               5 s of the raw pulse (cardiac band) of the youngest vs the oldest person,
               one pulse with the relax_area shaded (what the number measures)
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fnirs_toolkit as fk  # noqa: E402
import stiffness as st  # noqa: E402


def collect(files, cache: Path, jobs: int) -> pd.DataFrame:
    if cache.exists():
        return pd.read_pickle(cache)
    with Pool(jobs) as pool:
        rows = pool.map(st._safe, [(f, 3) for f in files])
    df = pd.DataFrame([r for r in rows if "_wave" in r])
    df["path"] = [str(next(f for f in files if f.name == n)) for n in df["file"]]
    df.to_pickle(cache)
    return df


def upstroke(w: np.ndarray, period_s: float):
    """Rise time to peak [ms] and max slope of the normalised pulse [amplitude per second]."""
    k = int(np.argmax(w))
    dt = period_s / (len(w) - 1)
    return k * dt * 1000, float(np.max(np.gradient(w[: k + 1], dt))) if k > 1 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(st.DEFAULT_DIR))
    ap.add_argument("--meta", default=str(st.DEFAULT_META))
    ap.add_argument("--cache", default="pulse_cache.pkl", help="per-subject pulses, reused on next run")
    ap.add_argument("--save", default="pulse_age.png")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(Path(args.data_dir).glob("*.snirf"))
    df = collect(files, Path(args.cache), args.jobs)
    df = df.merge(st.load_meta(Path(args.meta)), on="subject", how="left").dropna(subset=["age"])
    up = [upstroke(w, p) for w, p in zip(df["_wave"], df["period_s"])]
    df["rise_ms"] = [u[0] for u in up]
    df["upslope"] = [u[1] for u in up]

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    old = df["age"] >= 60
    for a, col, label in zip(ax[0], ["relax_area", "rise_ms", "upslope"],
                             ["decay: relax_area (lower = straighter decay)",
                              "rise time to peak [ms]", "max upstroke speed [1/s]"]):
        rho, p = spearmanr(df["age"], df[col])
        rho_y, p_y = spearmanr(df.loc[~old, "age"], df.loc[~old, col])
        sc = a.scatter(df["age"], df[col], c=df["activity"], cmap="viridis", s=35)
        a.set_xlabel("age [years]")
        a.set_ylabel(label)
        a.set_title(f"all n={len(df)}: rho={rho:+.2f} (p={p:.3f})\n"
                    f"age<60 n={(~old).sum()}: rho={rho_y:+.2f} (p={p_y:.3f})", fontsize=9)
    fig.colorbar(sc, ax=ax[0, 2], label="physical activity score")

    # averaged pulses on a real time axis: 4 youngest vs 4 oldest
    young = df.nsmallest(4, "age")
    oldest = df.nlargest(4, "age")
    for grp, cmap in ((young, plt.cm.Blues), (oldest, plt.cm.Reds)):
        for j, (_, r) in enumerate(grp.iterrows()):
            t = np.linspace(0, r["period_s"] * 1000, len(r["_wave"]))
            ax[1, 0].plot(t, r["_wave"], color=cmap(0.5 + 0.12 * j), lw=1.5,
                          label=f"#{int(r['subject'])}, {int(r['age'])} y")
    ax[1, 0].set_xlabel("time from beat start [ms]")
    ax[1, 0].set_ylabel("normalised HbO pulse")
    ax[1, 0].set_title("averaged pulse: 4 youngest (blue) vs 4 oldest (red)", fontsize=9)
    ax[1, 0].legend(fontsize=7)

    # 5 s of raw cardiac-band pulse, youngest vs oldest person
    for r, c in ((young.iloc[0], "C0"), (oldest.iloc[0], "C3")):
        d = fk.load_snirf_haemo(r["path"])
        fs = d["fs"]
        long_idx = np.flatnonzero(~d["short_mask"])
        best = long_idx[fk.rank_channels(d["hbo"][long_idx], fs)[0]]
        x = fk.bandpass(d["hbo"][best], fs, 0.5, 10)
        s0 = int(120 * fs)
        seg = x[s0:s0 + int(5 * fs)]
        ax[1, 1].plot(np.arange(len(seg)) / fs, seg / np.std(seg), c, lw=1,
                      label=f"#{int(r['subject'])}, {int(r['age'])} y")
    ax[1, 1].set_xlabel("time [s] (5 s from 120 s)")
    ax[1, 1].set_ylabel("pulse (z)")
    ax[1, 1].set_title("raw pulse, best channel, 0.5-10 Hz", fontsize=9)
    ax[1, 1].legend(fontsize=8)

    # what relax_area measures
    r = df.loc[df["relax_area"].idxmax()]
    w = r["_wave"]
    ph = np.linspace(0, 1, len(w))
    k = int(np.argmax(w))
    line = np.linspace(w[k], w[-1], len(w) - k)
    ax[1, 2].plot(ph, w, "k", lw=1.5)
    ax[1, 2].plot(ph[k:], line, "k--", lw=1)
    ax[1, 2].fill_between(ph[k:], line, w[k:], color="C1", alpha=0.4, label="relax_area")
    ax[1, 2].axvline(ph[k], color="C2", lw=1, label="peak (rise_frac)")
    ax[1, 2].set_xlabel("phase of heartbeat")
    ax[1, 2].set_title(f"relax_area = area between decay and straight line\n(example #{int(r['subject'])}, "
                       f"{int(r['age'])} y; elastic arteries -> bulging decay)", fontsize=9)
    ax[1, 2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.save, dpi=110)
    print(f"figure -> {args.save}")
    for col in ("relax_area", "rise_ms", "upslope"):
        print(f"{col:<10} vs age: rho {spearmanr(df['age'], df[col])[0]:+.2f}, "
              f"p {spearmanr(df['age'], df[col])[1]:.3f}")


if __name__ == "__main__":
    main()
