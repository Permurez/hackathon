"""
validate_real.py - check our fNIRS-only estimates against REAL reference sensors
(respiration belt, PPG) stored as SNIRF 'aux' channels.

Built for the open BU dataset (von Luhmann et al. 2020, Front. Neurosci., NITRC
'luhmann20synhrf', Dataset I: RESP belt + PPG + BP, 50 Hz), but works for any SNIRF
whose aux names contain 'resp' / 'ppg' (case-insensitive).

Usage:
  python validate_real.py --data-dir BU_DATASET_I/ --out validation.csv

For each file and each 60 s window it compares:
  breathing rate  : belt spectrum  vs  fk.respiration_rate / respiration_rate_fused (HbO)
  heart rate      : PPG spectrum   vs  fk.heart_rate (HbO)
  inhale/exhale   : belt rising    vs  fk.inhale_mask(fk.breath_phase(HbO)), best global offset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import fnirs_toolkit as fk


def read_aux(path: str) -> dict:
    """Return {name_lower: (time_s, values)} for every /nirs*/aux* group in a SNIRF file."""
    out = {}
    with h5py.File(path, "r") as f:
        for nirs_key in [k for k in f.keys() if k.startswith("nirs")]:
            g = f[nirs_key]
            for aux_key in [k for k in g.keys() if k.startswith("aux")]:
                a = g[aux_key]
                name = a["name"][()]
                name = name.decode() if isinstance(name, bytes) else str(np.atleast_1d(name)[0])
                if isinstance(name, bytes):
                    name = name.decode()
                vals = np.asarray(a["dataTimeSeries"][()], float).squeeze()
                t = np.asarray(a["time"][()], float).squeeze()
                if t.size == 2 and vals.size > 2:  # SNIRF allows [start, step] time spec
                    t = t[0] + np.arange(vals.size) * t[1]
                out[name.strip().lower()] = (t, vals)
    return out


def find_aux(aux: dict, key: str):
    for name, v in aux.items():
        if key in name:
            return v
    return None


def resample_to(t_src, x_src, t_dst):
    return np.interp(t_dst, t_src, x_src)


def validate_file(path: Path, win_s: float = 60.0, step_s: float = 30.0) -> pd.DataFrame:
    d = fk.load_snirf_haemo(str(path))
    fs = d["fs"]
    long_idx = np.flatnonzero(~d["short_mask"])
    hbo = d["hbo"][long_idx]
    best = fk.rank_channels(hbo, fs)[0]
    x = hbo[best]
    t = np.arange(x.size) / fs
    aux = read_aux(str(path))
    resp = find_aux(aux, "resp")
    ppg = find_aux(aux, "ppg")
    belt = resample_to(resp[0], resp[1], t) if resp is not None else None
    pulse = resample_to(ppg[0], ppg[1], t) if ppg is not None else None

    # breathing phase over the whole recording, offset calibrated once per file
    ph, _, _ = fk.breath_phase(x, fs)
    phase_acc, phase_off = np.nan, np.nan
    if belt is not None:
        b = fk.bandpass(belt, fs, 0.08, 0.6, order=2)
        inhale_ref = np.gradient(b) > 0
        phase_acc, phase_off = max((np.mean(fk.inhale_mask(ph, o) == inhale_ref), o)
                                   for o in np.linspace(-np.pi, np.pi, 73))

    rows = []
    n_win, n_step = int(win_s * fs), int(step_s * fs)
    for s in range(0, x.size - n_win + 1, n_step):
        seg = slice(s, s + n_win)
        r = {"file": path.name, "t0_s": s / fs, "channel": d["ch_hbo"][long_idx[best]]}
        r["rr_fnirs"] = fk.respiration_rate(x[seg], fs)
        try:
            r["rr_fnirs_fused"] = fk.respiration_rate_fused(x[seg], fs)["rr_bpm"]
        except Exception:
            r["rr_fnirs_fused"] = np.nan
        r["hr_fnirs"] = fk.heart_rate(x[seg], fs)
        if belt is not None:
            r["rr_belt"] = 60 * fk.dominant_freq(belt[seg], fs, 0.08, 0.6, win_s=win_s)
        if pulse is not None:
            r["hr_ppg"] = 60 * fk.dominant_freq(pulse[seg], fs, 0.7, 2.5, win_s=20)
        r["phase_acc_file"] = phase_acc
        r["phase_offset_rad"] = phase_off
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    def mae(a, b):
        m = df[[a, b]].dropna()
        return (m[a] - m[b]).abs().mean() if len(m) else np.nan
    print("\n=== summary over all windows ===")
    if "rr_belt" in df:
        print(f"breathing MAE  HbO peak : {mae('rr_fnirs', 'rr_belt'):.2f} /min")
        print(f"breathing MAE  fused    : {mae('rr_fnirs_fused', 'rr_belt'):.2f} /min")
    if "hr_ppg" in df:
        print(f"heart rate MAE vs PPG   : {mae('hr_fnirs', 'hr_ppg'):.2f} bpm")
    if "phase_acc_file" in df:
        pa = df.groupby("file")["phase_acc_file"].first()
        print(f"inhale/exhale accuracy  : mean {pa.mean():.2f} (per file: {np.round(pa.values, 2)})")
        po = df.groupby("file")["phase_offset_rad"].first()
        print(f"calibrated phase offset : mean {np.angle(np.mean(np.exp(1j * po.values))):.2f} rad, "
              f"spread {np.std(po.values):.2f} rad (small spread = one offset fits everyone)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder with .snirf files (searched recursively)")
    ap.add_argument("--out", default="validation.csv", help="per-window results CSV")
    ap.add_argument("--win", type=float, default=60.0, help="window length in seconds")
    args = ap.parse_args()
    files = sorted(Path(args.data_dir).rglob("*.snirf"))
    if not files:
        raise SystemExit(f"no .snirf files in {args.data_dir}")
    parts = []
    for p in files:
        try:
            parts.append(validate_file(p, args.win))
            print(f"ok    {p.name}")
        except Exception as e:
            print(f"FAIL  {p.name}: {e}")
    if not parts:
        raise SystemExit("nothing validated")
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(args.out, index=False)
    summarize(df)
    print(f"\nper-window results -> {args.out}")


if __name__ == "__main__":
    main()
