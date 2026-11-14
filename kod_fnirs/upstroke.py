"""
upstroke.py - pulse wave over time and how fast each beat's upstroke rises (beat by beat).

fNIRS does NOT measure blood pressure in mmHg. The HbO pulse is an optical blood-VOLUME pulse
(like a finger PPG); its shape follows the pressure wave, so its upstroke speed and rise time are
usable, but amplitudes are relative, not mmHg.

Usage:
  python upstroke.py FILE.snirf --save up.png [--csv beats.csv] [--zoom 120 130] [--top-k 3]

Per beat (foot of the wave -> 90% of its amplitude):
  rise_ms     time from the foot of the wave to 90% of the beat amplitude
  upslope     max slope of the upstroke divided by the beat amplitude [1/s]
              (amplitude-free, so channels and people can be compared)
Figure: (1) pulse in the --zoom window with each upstroke highlighted, (2) upslope per beat over
the recording, (3) rise time per beat over the recording; markers as vertical lines.
Best on the 141.79 Hz supine data; at 18 Hz the upstroke has only ~4 samples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fnirs_toolkit as fk  # noqa: E402


def pulse(d: dict, top_k: int):
    """Average of z-scored 0.5-10 Hz HbO of the top_k long channels with the clearest heartbeat."""
    fs = d["fs"]
    long_idx = np.flatnonzero(~d["short_mask"])
    best = long_idx[fk.rank_channels(d["hbo"][long_idx], fs)[:top_k]]
    parts = [fk.bandpass(d["hbo"][i], fs, 0.5, min(10.0, 0.45 * fs), order=3) for i in best]
    return np.mean([(p - p.mean()) / p.std() for p in parts], axis=0), [d["ch_hbo"][i].split()[0] for i in best]


def beats(x: np.ndarray, fs: float):
    """Per-beat (trough index, peak index, rise_ms, upslope 1/s), implausible beats removed."""
    period = 60.0 / fk.heart_rate(x, fs)
    tr, _ = signal.find_peaks(-x, distance=max(1, int(0.6 * period * fs)), prominence=0.3 * np.std(x))
    dx = np.gradient(x) * fs
    out = []
    for a, b in zip(tr[:-1], tr[1:]):
        if not (0.6 * period < (b - a) / fs < 1.4 * period):
            continue
        amp = x[a:b].max() - x[a]
        if amp <= 0:
            continue
        # foot -> 90% of the beat amplitude: robust when the top is flat or has two humps
        k = a + int(np.argmax(x[a:b] >= x[a] + 0.9 * amp))
        if k - a < 2:
            continue
        out.append((a, k, (k - a) / fs * 1000, dx[a:k + 1].max() / amp))
    arr = np.array(out)
    # drop outliers (> 3 MAD from the median) in either feature
    keep = np.ones(len(arr), bool)
    for c in (2, 3):
        med = np.median(arr[:, c])
        mad = np.median(np.abs(arr[:, c] - med)) + 1e-12
        keep &= np.abs(arr[:, c] - med) < 3 * 1.4826 * mad
    return arr[keep]


def trend(t, v, win_s=20.0):
    return np.array([np.median(v[np.abs(t - ti) <= win_s / 2]) for ti in t])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--zoom", type=float, nargs=2, default=(120, 130), metavar=("T0", "T1"))
    ap.add_argument("--save", help="PNG instead of a window")
    ap.add_argument("--csv", help="per-beat table")
    args = ap.parse_args()

    d = fk.load_snirf_haemo(args.file)
    fs = d["fs"]
    x, chans = pulse(d, args.top_k)
    B = beats(x, fs)
    t_beat = B[:, 0] / fs
    print(f"{Path(args.file).name}: fs {fs:.1f} Hz, channels {', '.join(chans)}, {len(B)} clean beats")
    print(f"rise time  median {np.median(B[:, 2]):.0f} ms (IQR {np.percentile(B[:, 2], 25):.0f}-{np.percentile(B[:, 2], 75):.0f})")
    print(f"upslope    median {np.median(B[:, 3]):.1f} /s (IQR {np.percentile(B[:, 3], 25):.1f}-{np.percentile(B[:, 3], 75):.1f})")
    if args.csv:
        import pandas as pd
        pd.DataFrame({"t_s": t_beat, "rise_ms": B[:, 2], "upslope_per_s": B[:, 3]}).to_csv(args.csv, index=False)
        print(f"beats -> {args.csv}")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1.4, 1, 1]})
    fig.suptitle(f"{Path(args.file).name}   pulse upstroke beat by beat   (optical volume pulse, not mmHg)",
                 fontsize=11)
    z0, z1 = args.zoom
    m = (t >= z0) & (t <= z1)
    ax[0].plot(t[m], x[m], "k", lw=1)
    for a, k, _, _ in B:
        a, k = int(a), int(k)
        if z0 <= a / fs <= z1:
            ax[0].plot(t[a:k + 1], x[a:k + 1], "C3", lw=2.5)
    ax[0].plot([], [], "C3", lw=2.5, label="upstroke (foot -> 90% amplitude)")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("pulse (z)")
    ax[0].legend(fontsize=8, loc="upper right")

    for a_, col, lab, c in ((ax[1], 3, "upstroke speed [1/s]", "C0"), (ax[2], 2, "rise time to 90% [ms]", "C2")):
        a_.plot(t_beat, B[:, col], ".", color=c, ms=3, alpha=0.4, label="beat")
        a_.plot(t_beat, trend(t_beat, B[:, col]), color=c, lw=2, label="20 s median")
        a_.set_ylabel(lab)
        a_.legend(fontsize=8, loc="upper right")
        for onset in d["annotations"].onset:
            a_.axvline(onset, color="k", lw=0.6, alpha=0.4)
    ax[2].sharex(ax[1])
    ax[2].set_xlabel("time [s]")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"figure -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
