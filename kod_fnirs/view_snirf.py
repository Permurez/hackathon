"""
view_snirf.py - look inside a .snirf recording: summary + 4-panel figure.

Usage:
  python view_snirf.py FILE.snirf                       # summary + figure window
  python view_snirf.py FILE.snirf --save out.png        # write PNG instead (Colab / no display)
  python view_snirf.py FILE.snirf --info                # summary only
  python view_snirf.py FILE.snirf --ch S1_D1 S7_D4 --band cardiac --tmin 100 --tmax 130

Figure panels:
  1. HbO (solid) / HbR (dashed) of the chosen channels, filtered to --band, markers as lines
  2. accelerometer magnitude (movement, end of breath holds)
  3. power spectrum of HbO with the physiological bands shaded
  4. ensemble-averaged pulse wave (shape ~ arterial elasticity; best at 141.79 Hz)

--band: raw | vlf | mayer | resp | cardiac | LO-HI (e.g. 0.5-8)
Without --ch the 3 long channels with the clearest heartbeat are shown.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fnirs_toolkit as fk  # noqa: E402
from validate_real import read_aux  # noqa: E402

BAND_COLORS = {"vlf": "#cccccc", "mayer": "#f4c27a", "resp": "#9ecae1", "cardiac": "#f4a6a6"}


def parse_band(band: str):
    if band == "raw":
        return None
    if band in fk.BANDS:
        return fk.BANDS[band]
    lo, hi = (float(v) for v in band.split("-"))
    return lo, hi


def meta_tags(path: str) -> dict:
    out = {}
    with h5py.File(path, "r") as f:
        g = f["nirs/metaDataTags"]
        for k in ("SubjectID", "Task", "Block", "MeasurementDate", "MeasurementTime"):
            if k in g:
                v = g[k][()]
                v = v.decode() if isinstance(v, bytes) else np.atleast_1d(v)[0]
                out[k] = v.decode() if isinstance(v, bytes) else v
    return out


def pair(ch_name: str) -> str:
    return ch_name.split()[0]  # 'S1_D1 hbo' -> 'S1_D1'


def print_summary(path: str, d: dict) -> None:
    fs, hbo = d["fs"], d["hbo"]
    ann = d["annotations"]
    print(f"file      : {path}")
    for k, v in meta_tags(path).items():
        print(f"{k:<10}: {v}")
    print(f"fs        : {fs:.2f} Hz, duration {hbo.shape[1] / fs:.1f} s")
    n_short = int(d["short_mask"].sum())
    print(f"channels  : {len(d['ch_hbo'])} pairs ({len(d['ch_hbo']) - n_short} long, {n_short} short <15 mm), "
          f"distances {d['dist_m'].min() * 1000:.0f}-{d['dist_m'].max() * 1000:.0f} mm")
    if d["sci"] is not None:
        sci = np.asarray(d["sci"])
        print(f"SCI       : {int((sci >= 0.7).sum())}/{len(sci)} wavelength channels >= 0.7 (good scalp contact)")
    if len(ann):
        counts = Counter(ann.description)
        print("markers   : " + ", ".join(f"'{k}' x{v}" for k, v in sorted(counts.items())))
        for desc in sorted(counts):
            onsets = ann.onset[ann.description == desc]
            print(f"  '{desc}' at s: " + " ".join(f"{t:.1f}" for t in onsets[:12]) + (" ..." if len(onsets) > 12 else ""))
    else:
        print("markers   : none")
    print("aux       : " + ", ".join(read_aux(path).keys()))
    long_idx = np.flatnonzero(~d["short_mask"])
    order = long_idx[fk.rank_channels(hbo[long_idx], fs)]
    q = [fk.cardiac_quality(hbo[i], fs) for i in order[:5]]
    print("best heartbeat channels: " + ", ".join(f"{pair(d['ch_hbo'][i])} (q={v:.0f})" for i, v in zip(order, q)))
    print(f"heart rate ~{fk.heart_rate(hbo[order[0]], fs):.0f} bpm, "
          f"breathing ~{fk.respiration_rate(hbo[order[0]], fs):.1f} /min (best channel)")


def pick_channels(d: dict, wanted: list[str] | None, n: int = 3) -> list[int]:
    names = [pair(c) for c in d["ch_hbo"]]
    if wanted:
        missing = [w for w in wanted if w not in names]
        if missing:
            raise SystemExit(f"unknown channel(s) {missing}; available: {' '.join(names)}")
        return [names.index(w) for w in wanted]
    long_idx = np.flatnonzero(~d["short_mask"])
    return list(long_idx[fk.rank_channels(d["hbo"][long_idx], d["fs"])[:n]])


def plot(path: str, d: dict, chans: list[int], band, tmin: float, tmax: float | None, save: str | None) -> None:
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = d["fs"]
    n = d["hbo"].shape[1]
    t = np.arange(n) / fs
    tmax = t[-1] if tmax is None else min(tmax, t[-1])
    win = (t >= tmin) & (t <= tmax)

    fig, ax = plt.subplots(4, 1, figsize=(13, 13), gridspec_kw={"height_ratios": [3, 1, 2, 2]})
    fig.suptitle(f"{Path(path).name}   fs={fs:.2f} Hz   band={band or 'raw'}", fontsize=11)

    for k, i in enumerate(chans):
        c = f"C{k}"
        hbo, hbr = d["hbo"][i], d["hbr"][i]
        if band:
            hbo, hbr = fk.bandpass(hbo, fs, *band), fk.bandpass(hbr, fs, *band)
        ax[0].plot(t[win], hbo[win] * 1e6, c, lw=0.8, label=f"{pair(d['ch_hbo'][i])} HbO")
        ax[0].plot(t[win], hbr[win] * 1e6, c, lw=0.8, ls="--", alpha=0.7, label=f"{pair(d['ch_hbo'][i])} HbR")
    ann = d["annotations"]
    for onset, desc in zip(ann.onset, ann.description):
        if tmin <= onset <= tmax:
            ax[0].axvline(onset, color="k", lw=0.6, alpha=0.5)
            ax[0].text(onset, 1.0, desc, transform=ax[0].get_xaxis_transform(), fontsize=7, va="bottom")
    ax[0].set_ylabel("Δ concentration [µM]")
    ax[0].legend(fontsize=7, ncol=len(chans), loc="upper right")

    aux = read_aux(path)
    acc = [aux[k] for k in ("acc-x", "acc-y", "acc-z") if k in aux]
    if acc:
        ta = acc[0][0]
        mag = np.sqrt(sum(v[1] ** 2 for v in acc))
        m = (ta >= tmin) & (ta <= tmax)
        ax[1].plot(ta[m], mag[m], "k", lw=0.6)
    ax[1].set_ylabel("|acc|")
    ax[1].set_xlabel("time [s]")
    ax[1].sharex(ax[0])

    for name, (lo, hi) in fk.BANDS.items():
        ax[2].axvspan(lo, hi, color=BAND_COLORS[name], alpha=0.35, label=name)
    for k, i in enumerate(chans):
        f, p = fk.psd(d["hbo"][i][win], fs)
        ax[2].semilogy(f[1:], p[1:], f"C{k}", lw=0.8)
    ax[2].set_xlim(0, min(3.0, fs / 2))
    ax[2].set_xlabel("frequency [Hz]")
    ax[2].set_ylabel("HbO PSD")
    ax[2].legend(fontsize=7, loc="upper right")

    for k, i in enumerate(chans):
        ph, w, nb = fk.pulse_waveform(d["hbo"][i][win], fs)
        if nb:
            ax[3].plot(ph, w, f"C{k}", label=f"{pair(d['ch_hbo'][i])} ({nb} beats)")
    ax[3].set_xlabel("phase of heartbeat (trough -> trough)")
    ax[3].set_ylabel("normalised HbO pulse")
    ax[3].legend(fontsize=7)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)
        print(f"figure -> {save}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help=".snirf file")
    ap.add_argument("--info", action="store_true", help="print summary only, no figure")
    ap.add_argument("--ch", nargs="+", help="channel pairs to show, e.g. S1_D1 S7_D4")
    ap.add_argument("--band", default="raw", help="raw | vlf | mayer | resp | cardiac | LO-HI in Hz")
    ap.add_argument("--tmin", type=float, default=0.0, help="start of time window [s]")
    ap.add_argument("--tmax", type=float, default=None, help="end of time window [s]")
    ap.add_argument("--no-tddr", action="store_true", help="skip TDDR motion correction")
    ap.add_argument("--save", help="write the figure to this PNG instead of opening a window")
    args = ap.parse_args()

    d = fk.load_snirf_haemo(args.file, tddr=not args.no_tddr)
    print_summary(args.file, d)
    if not args.info:
        plot(args.file, d, pick_channels(d, args.ch), parse_band(args.band), args.tmin, args.tmax, args.save)


if __name__ == "__main__":
    main()
