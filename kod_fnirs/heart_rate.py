"""
heart_rate.py - heart rate over time from one .snirf recording (fNIRS pulse only, no ECG).

Usage:
  python heart_rate.py FILE.snirf                          # figure window
  python heart_rate.py FILE.snirf --save hr.png --csv hr.csv
  python heart_rate.py FILE.snirf --zoom 100 120 --top-k 5

How:
  1. intensity -> HbO (fnirs_toolkit.load_snirf_haemo)
  2. take the --top-k long channels with the clearest heartbeat, z-score their cardiac band
     and average them (beats are in sync across the head, noise is not)
  3. detect each beat (fk.beat_times), beat-to-beat interval -> instantaneous HR in bpm

Figure: (1) pulse signal with detected beats in the --zoom window, (2) heart rate over the
whole recording with markers, (3) slow HbO (< 0.08 Hz) for context, e.g. breath-hold response.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fnirs_toolkit as fk  # noqa: E402


def pulse_signal(d: dict, top_k: int):
    """Average of the z-scored cardiac band of the top_k long channels with the clearest heartbeat."""
    fs = d["fs"]
    long_idx = np.flatnonzero(~d["short_mask"])
    best = long_idx[fk.rank_channels(d["hbo"][long_idx], fs)[:top_k]]
    parts = []
    for i in best:
        x = fk.bandpass(d["hbo"][i], fs, 0.5, min(3.0, 0.45 * fs))
        parts.append((x - x.mean()) / (x.std() + 1e-20))
    return np.mean(parts, axis=0), [d["ch_hbo"][i].split()[0] for i in best]


def beats_and_hr(x: np.ndarray, fs: float):
    """Beat times, beat-to-beat intervals and HR, dropping physiologically impossible intervals."""
    tb = fk.beat_times(x, fs)
    ibi = np.diff(tb)
    t_ibi = tb[1:]
    ok = (ibi > 0.33) & (ibi < 1.5)  # 40-180 bpm
    # also drop single intervals that jump >30% from the local median (missed / extra beat)
    med = np.array([np.median(ibi[max(0, k - 5):k + 6]) for k in range(len(ibi))])
    ok &= np.abs(ibi - med) < 0.3 * med
    return tb, t_ibi[ok], ibi[ok], 60.0 / ibi[ok]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help=".snirf file")
    ap.add_argument("--top-k", type=int, default=5, help="how many best channels to average")
    ap.add_argument("--zoom", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                    help="time window [s] for the beat-detection panel (default: first 20 s)")
    ap.add_argument("--smooth", type=float, default=10.0, help="moving-median window for the HR trend [s]")
    ap.add_argument("--save", help="write the figure to this PNG instead of opening a window")
    ap.add_argument("--csv", help="write beat times, intervals and HR to this CSV")
    args = ap.parse_args()

    d = fk.load_snirf_haemo(args.file)
    fs = d["fs"]
    x, chans = pulse_signal(d, args.top_k)
    tb, t_hr, ibi, hr = beats_and_hr(x, fs)
    if len(hr) < 10:
        raise SystemExit("too few clean beats - check channel quality (view_snirf.py --info)")

    # moving median of HR on a regular 4 Hz grid
    grid = np.arange(t_hr[0], t_hr[-1], 0.25)
    hr_grid = np.interp(grid, t_hr, hr)
    w = max(1, int(args.smooth * 4))
    trend = np.array([np.median(hr_grid[max(0, k - w // 2):k + w // 2 + 1]) for k in range(len(grid))])

    sdnn = np.std(ibi) * 1000
    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000
    print(f"file      : {Path(args.file).name}  (fs {fs:.2f} Hz)")
    print(f"channels  : {', '.join(chans)}")
    print(f"beats     : {len(tb)} detected, {len(hr)} clean intervals ({100 * len(hr) / max(1, len(tb) - 1):.0f}%)")
    print(f"HR        : mean {hr.mean():.1f} bpm, median {np.median(hr):.1f}, "
          f"range (5-95%) {np.percentile(hr, 5):.0f}-{np.percentile(hr, 95):.0f} bpm")
    print(f"HRV       : SDNN {sdnn:.0f} ms, RMSSD {rmssd:.0f} ms"
          + ("  (low fs: treat as rough)" if fs < 50 else ""))

    if args.csv:
        import pandas as pd
        pd.DataFrame({"t_s": t_hr, "ibi_s": ibi, "hr_bpm": hr}).to_csv(args.csv, index=False)
        print(f"beats     -> {args.csv}")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(x)) / fs
    z0, z1 = args.zoom if args.zoom else (t[0], min(t[-1], t[0] + 20))
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1.3, 2, 1.3]})
    fig.suptitle(f"{Path(args.file).name}   heart rate from fNIRS pulse   "
                 f"(mean {hr.mean():.0f} bpm, {len(chans)} channels averaged)", fontsize=11)

    m = (t >= z0) & (t <= z1)
    ax[0].plot(t[m], x[m], "C3", lw=0.9)
    bm = (tb >= z0) & (tb <= z1)
    ax[0].plot(tb[bm], np.interp(tb[bm], t, x), "kv", ms=6, label="detected beat")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("pulse (z)")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"beat detection, {z0:.0f}-{z1:.0f} s", fontsize=9)

    ax[1].plot(t_hr, hr, ".", color="C0", ms=3, alpha=0.5, label="beat-to-beat")
    ax[1].plot(grid, trend, "C0", lw=2, label=f"{args.smooth:.0f} s median")
    ax[1].set_ylabel("heart rate [bpm]")
    ax[1].legend(fontsize=8, loc="upper right")

    slow = np.mean([fk.bandpass(d["hbo"][d["ch_hbo"].index(c + " hbo")], fs, 0.005, 0.08, order=2)
                    for c in chans], axis=0)
    ax[2].plot(t, slow * 1e6, "C2", lw=1)
    ax[2].set_ylabel("slow HbO [µM]")
    ax[2].set_xlabel("time [s]")
    ax[2].sharex(ax[1])

    ann = d["annotations"]
    for a in (ax[1], ax[2]):
        for onset, desc in zip(ann.onset, ann.description):
            a.axvline(onset, color="k", lw=0.6, alpha=0.4)
    for onset, desc in zip(ann.onset, ann.description):
        ax[1].text(onset, 1.0, desc, transform=ax[1].get_xaxis_transform(), fontsize=7, va="bottom")

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"figure    -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
