"""fnirs_vascular.core - one resting-state fNIRS recording (SNIRF) -> vascular markers as a plain, JSON-ready dict.

Markers (same definitions as the analysis in src/, values reproduce bands_supine.csv and ti_followup.csv):
  vascular_regulation  share of the slow vasomotion band 0.0095-0.021 Hz (attributed to endothelial, NO-dependent
                       activity; Kvernmo et al. 1999, 2003) in the forehead HbO spectrum, 0.0095-2 Hz
  arterial_timing      reflected-wave timing index TI = 1 / (t_reflected - t_systolic) of the averaged pulse
                       (Afkhami et al. 2021), mean over the long channels with a clear heartbeat

Percentiles and the activity probability are given only when the recording comes from the reference montage;
for any other montage the raw values are returned with a warning.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy import signal

from . import haemo

__version__ = "0.1.0"
BAND_HZ = (0.0095, 0.021)
REFERENCE_CHANNELS = ("S6_D4", "S5_D3", "S6_D5", "S5_D4")  # 30 mm, bare forehead, in the reference montage
LONG_MM = (20.0, 35.0)       # channels used for the pulse timing (and as fallback for the band)
MIN_FS_TIMING = 50.0         # the reflected wave is ~100 ms after the systolic peak: needs fast sampling
MIN_MINUTES_BAND = 8.0       # the band has a ~1-min period: shorter rests give too few cycles
DISCLAIMER = ("Group-level markers compared with healthy adults 18-42 y (supine, 10-min rest). "
              "Not a medical diagnosis; one 10-min recording has limited test-retest reliability (ICC ~0.36 "
              "for the band).")


# --------------------------------------------------------------------------- signal helpers (as in src/)
def _bp(x, fs, lo, hi):
    return signal.sosfiltfilt(signal.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos"), x)


def _cardiac_quality(x, fs):
    f, p = signal.welch(x - x.mean(), fs, nperseg=int(min(len(x), 30 * fs)))
    m = (f >= 0.7) & (f <= 2.5)
    return float(p[m].max() / p[m].mean())


def _pulse_times(w, fs):
    """t_systolic, t_reflected [s] from one averaged beat (foot at index 0)."""
    s = (w - w.min()) / (np.ptp(w) + 1e-30)
    s2 = np.gradient(np.gradient(signal.savgol_filter(s, max(7, int(0.03 * fs) | 1), 3)))
    zc = np.flatnonzero(np.diff(np.sign(s2)) != 0)
    if len(zc) < 4:
        return np.nan, np.nan
    pk, _ = signal.find_peaks(s, prominence=0.002)
    t_sys = min(pk[0], zc[1]) if len(pk) else zc[1]
    a, b = zc[2], zc[3]
    pk2 = pk[(pk > a) & (pk <= b)]
    t_refl = pk2[0] if len(pk2) else a + int(np.argmin(s2[a:b + 1]))
    return t_sys / fs, t_refl / fs


def _band_share(x, fs):
    down = max(1, int(fs // 5))
    xd = signal.detrend(signal.decimate(x - x.mean(), down, ftype="fir") if down > 1 else x - x.mean())
    fsd = fs / down
    f, p = signal.welch(xd, fsd, nperseg=int(min(len(xd), 300 * fsd)))
    top = min(2.0, 0.95 * fsd / 2)
    band, tot = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1]), (f >= BAND_HZ[0]) & (f <= top)
    return float(np.trapezoid(p[band], f[band]) / np.trapezoid(p[tot], f[tot]))


def _beats(ref, fs):
    """Heart period [s] and beat onsets (troughs) of the band-passed reference signal."""
    fw, pw = signal.welch(ref, fs, nperseg=int(min(len(ref), 20 * fs)))
    m = (fw > 0.7) & (fw < 2.5)
    period = 1 / fw[m][np.argmax(pw[m])]
    tr, _ = signal.find_peaks(-ref, distance=int(0.6 * period * fs), prominence=0.3 * ref.std())
    L = int(np.median(np.diff(tr)))
    tr = [a for a, b in zip(tr[:-1], tr[1:]) if abs((b - a) - L) < 0.2 * L and a + L < len(ref)]
    return period, tr, L


def _timing_index(h, ref, fs, dead):
    _, tr, L = _beats(ref, fs)
    ti = []
    for i, name in enumerate(h["pairs"]):
        if name in dead or not (LONG_MM[0] <= h["distance_mm"][i] <= LONG_MM[1]):
            continue
        seg = h["hbo"][i]
        if _cardiac_quality(seg, fs) < 4:
            continue
        x = _bp(seg, fs, 0.5, 5)
        w = np.mean([x[a:a + L] for a in tr], axis=0)
        w = w - np.linspace(w[0], w[-1], L)
        ts, tf = _pulse_times(w, fs)
        if np.isfinite(ts) and 0.040 < tf - ts < 0.400:
            ti.append(1 / (tf - ts))
    ti = np.array(ti)
    if len(ti) < 3:
        return np.nan, len(ti)
    keep = np.abs(ti - ti.mean()) <= 1.5 * ti.std() if ti.std() > 0 else np.ones(len(ti), bool)
    return (float(ti[keep].mean()) if keep.sum() >= 3 else np.nan), int(keep.sum())


# --------------------------------------------------------------------------- reference group
@lru_cache(maxsize=1)
def reference() -> dict:
    return json.loads(files(__package__).joinpath("reference.json").read_text())


def _percentile(values, v):
    return None if v is None else round(100.0 * float(np.mean(np.asarray(values) < v)), 1)


def _same_montage(h) -> bool:
    ref = reference()["montage"]
    got = {n: float(d) for n, d in zip(h["pairs"], h["distance_mm"])}
    return set(got) == set(ref) and all(abs(got[n] - ref[n]) <= 3.0 for n in ref)


def _num(x):
    return None if x is None or not math.isfinite(x) else float(x)


# --------------------------------------------------------------------------- public API
def _as_path(recording) -> Path:
    """Path / str to a .snirf file, or an MNE Raw read from one (mne.io.read_raw_snirf)."""
    if hasattr(recording, "filenames"):
        recording = recording.filenames[0]
    p = Path(recording)
    if p.suffix.lower() != ".snirf":
        raise ValueError(f"expected a .snirf file, got {p.name}")
    return p


def analyze(recording, channels=None) -> dict:
    """One resting-state recording -> dict of markers, quality and warnings (JSON-serialisable).

    recording: path to a .snirf file, or an MNE Raw object read from one.
    channels:  optional list of pair names ("S5_D3", ...) for the band; default = forehead channels of the
               reference montage, else every long (20-35 mm) channel with a clear heartbeat.
    """
    path = _as_path(recording)
    h = haemo.convert(path, dpf=6.0)
    fs = float(h["fs"])
    minutes = float((h["time"][-1] - h["time"][0]) / 60)
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    idx = {n: i for i, n in enumerate(h["pairs"])}
    warnings = []
    comparable = _same_montage(h)

    if channels:
        names = [c for c in channels if c in idx and c not in dead]
        source = "user"
    elif comparable:
        names = [c for c in REFERENCE_CHANNELS if c not in dead]
        source = "reference forehead channels"
    else:
        names = [n for n, d in zip(h["pairs"], h["distance_mm"])
                 if n not in dead and LONG_MM[0] <= d <= LONG_MM[1] and _cardiac_quality(h["hbo"][idx[n]], fs) >= 4]
        source = "all long channels with a clear heartbeat"
        warnings.append("Montage differs from the reference montage: raw values only, no comparison with the "
                        "reference group.")
    if not names:
        raise ValueError(f"{path.name}: no usable channel (dead or missing)")
    if channels and comparable and set(names) != set(REFERENCE_CHANNELS):
        comparable = False
        warnings.append("Custom channel selection: raw values only, no comparison with the reference group.")
    if minutes < MIN_MINUTES_BAND:
        warnings.append(f"Recording is {minutes:.1f} min; the band needs >= {MIN_MINUTES_BAND:.0f} min of rest "
                        "(about one cycle per minute), so its value is unreliable.")

    x = h["hbo"][[idx[n] for n in names]].mean(0)
    band = _band_share(x, fs)
    ref_sig = _bp(x, fs, 0.5, min(5.0, 0.45 * fs))
    period, beats, _ = _beats(ref_sig, fs)
    hr = 60.0 / period

    if fs >= MIN_FS_TIMING:
        ti, n_ti = _timing_index(h, _bp(x, fs, 0.5, 5), fs, dead)
        if not math.isfinite(ti):
            warnings.append("Pulse timing index not computed: fewer than 3 long channels with a clear heartbeat.")
    else:
        ti, n_ti = np.nan, 0
        warnings.append(f"Sampling {fs:.1f} Hz is below {MIN_FS_TIMING:.0f} Hz: pulse timing index not computed.")

    R = reference()
    m = R["activity_model"]
    prob = None
    if comparable:
        zval = (band - m["endo_mean"]) / m["endo_sd"]
        prob = 1.0 / (1.0 + math.exp(-(m["intercept"] + m["coef"] * zval)))

    return {
        "file": path.name,
        "module_version": __version__,
        "recording": {"duration_min": round(minutes, 2), "sampling_hz": round(fs, 2), "pairs": len(h["pairs"]),
                      "dead_pairs": sorted(dead), "missing_samples_pct": round(100 * float(h["missing"].mean()), 2),
                      "reference_montage": comparable},
        "heart_rate_bpm": round(hr, 1),
        "beats_used": len(beats),
        "vascular_regulation": {
            "endothelial_band_share": round(band, 4),
            "band_hz": list(BAND_HZ),
            "channels": names, "channel_source": source,
            "percentile_vs_reference": _percentile(R["endothelial_band_share"], band) if comparable else None,
            "prob_aerobic_gt_3h_week": None if prob is None else round(prob, 3),
            "model_auc_leave_one_out": round(m["auc_leave_one_out"], 2),
        },
        "arterial_timing": {
            "timing_index_per_s": None if _num(ti) is None else round(ti, 3),
            "channels_used": n_ti,
            "percentile_vs_reference": _percentile(R["timing_index_per_s"], _num(ti)) if comparable else None,
        },
        "reference_group": {"description": R["description"], "n": R["n"]},
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }


def find_recordings(paths) -> list[Path]:
    """Expand files and folders (searched recursively, e.g. a BIDS dataset) into a sorted list of .snirf files."""
    out = []
    for p in map(Path, paths):
        out += sorted(p.rglob("*.snirf")) if p.is_dir() else [p]
    return out


def analyze_many(paths, channels=None) -> list[dict]:
    """analyze() over files and folders; a failing file gives {"file", "error"} instead of stopping the batch."""
    res = []
    for p in find_recordings(paths):
        try:
            res.append(analyze(p, channels))
        except Exception as e:  # noqa: BLE001 - one bad file must not stop a batch
            res.append({"file": p.name, "error": f"{type(e).__name__}: {e}"})
    return res
