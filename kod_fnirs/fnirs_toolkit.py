"""
fnirs_toolkit.py - feature extraction for "systemic physiology hidden in fNIRS".

Everything works on plain numpy arrays (1 channel = 1D array, sampling rate fs in Hz),
so it does not matter whether data comes from MNE, a CSV or a .mat file.

Main ideas:
  - cardiac band   ~0.7-2.5 Hz  -> heart rate, pulse shape (arterial stiffness proxy)
  - respiratory    ~0.08-0.6 Hz -> breathing rate, breathing phase, apnea detection
  - Mayer waves    ~0.08-0.15 Hz -> blood pressure oscillations (overlaps with 6 breaths/min!)
  - breath hold    slow rise of HbO/HbT 10-30 s after onset -> cerebrovascular reactivity

Python >= 3.9, numpy, scipy, scikit-learn. MNE only for load_snirf_haemo().
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

# ---------------------------------------------------------------------------
# Frequency bands (Hz). Tune them once you see the real spectra.
# ---------------------------------------------------------------------------
BANDS = {
    "vlf": (0.01, 0.08),      # very slow vasomotion / drifts
    "mayer": (0.08, 0.15),    # Mayer waves (blood pressure, ~0.1 Hz)
    "resp": (0.15, 0.50),     # spontaneous breathing (9-30 breaths/min)
    "cardiac": (0.70, 2.50),  # heart rate 42-150 bpm
}


# ---------------------------------------------------------------------------
# Basic signal processing
# ---------------------------------------------------------------------------
def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (no time shift, important for phase/latency)."""
    nyq = fs / 2.0
    hi = min(hi, 0.99 * nyq)
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def psd(x: np.ndarray, fs: float, win_s: float = 60.0):
    """Welch power spectral density. Long window = fine frequency resolution."""
    nperseg = int(min(len(x), win_s * fs))
    f, p = signal.welch(x - np.mean(x), fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return f, p


def dominant_freq(x: np.ndarray, fs: float, fmin: float, fmax: float, win_s: float = 60.0) -> float:
    """Frequency (Hz) of the biggest PSD peak inside [fmin, fmax], with parabolic refinement."""
    f, p = psd(x, fs, win_s)
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return np.nan
    idx = np.flatnonzero(m)
    k = idx[np.argmax(p[idx])]
    if 0 < k < len(p) - 1:  # parabolic interpolation around the peak bin
        a, b, c = p[k - 1], p[k], p[k + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return float(f[k] + shift * (f[1] - f[0]))
    return float(f[k])


def band_powers(x: np.ndarray, fs: float, bands: dict = BANDS, relative: bool = True) -> dict:
    """Power in each band (optionally divided by total 0.01-2.5 Hz power)."""
    f, p = psd(x, fs)
    out = {}
    total = np.trapezoid(p[(f >= 0.01) & (f <= 2.5)], f[(f >= 0.01) & (f <= 2.5)])
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f <= hi)
        val = np.trapezoid(p[m], f[m]) if np.any(m) else np.nan
        out[f"pow_{name}"] = val / total if relative and total > 0 else val
    return out


# ---------------------------------------------------------------------------
# Signal quality
# ---------------------------------------------------------------------------
def cardiac_quality(x: np.ndarray, fs: float) -> float:
    """
    Simple quality index: how 'peaky' the cardiac band is (peak power / mean power
    in 0.7-2.5 Hz). Good optode contact => clear heartbeat => high value (> ~5).
    """
    f, p = psd(x, fs, win_s=30)
    m = (f >= BANDS["cardiac"][0]) & (f <= BANDS["cardiac"][1])
    if not np.any(m):
        return np.nan
    return float(p[m].max() / (p[m].mean() + 1e-20))


def rank_channels(data: np.ndarray, fs: float) -> np.ndarray:
    """Return channel indices sorted from best to worst cardiac_quality. data: (n_ch, n_t)."""
    q = np.array([cardiac_quality(ch, fs) for ch in data])
    return np.argsort(-np.nan_to_num(q, nan=-1))


# ---------------------------------------------------------------------------
# Heart
# ---------------------------------------------------------------------------
def heart_rate(x: np.ndarray, fs: float) -> float:
    """Mean heart rate in beats per minute from the cardiac-band spectral peak."""
    return 60.0 * dominant_freq(x, fs, *BANDS["cardiac"], win_s=20)


def beat_times(x: np.ndarray, fs: float, hr_bpm: float | None = None) -> np.ndarray:
    """Times (s) of pulse peaks. Uses the mean HR to set the minimum distance between peaks."""
    if hr_bpm is None or np.isnan(hr_bpm):
        hr_bpm = heart_rate(x, fs)
    # narrow band around the fundamental: harmonics (pulse shape) create false double peaks,
    # +-40% keeps the beat-to-beat frequency modulation (RSA) intact
    f_hr = hr_bpm / 60.0
    xc = bandpass(x, fs, max(0.5, 0.65 * f_hr), min(1.5 * f_hr, 0.45 * fs), order=3)
    min_dist = max(1, int(0.6 * fs * 60.0 / hr_bpm))  # 60% of a mean beat period
    peaks, _ = signal.find_peaks(xc, distance=min_dist, prominence=0.3 * np.std(xc))
    # parabolic refinement => sub-sample peak times (matters at 7-10 Hz sampling)
    t = peaks.astype(float)
    for i, k in enumerate(peaks):
        if 0 < k < len(xc) - 1:
            a, b, c = xc[k - 1], xc[k], xc[k + 1]
            d = a - 2 * b + c
            if d != 0:
                t[i] = k + 0.5 * (a - c) / d
    return t / fs


def instantaneous_hr(x: np.ndarray, fs: float, out_fs: float = 4.0):
    """
    Beat-to-beat heart rate resampled to a regular grid.
    Returns (t, hr_bpm). HR goes up on inhale and down on exhale
    (respiratory sinus arrhythmia, RSA) -> an independent breathing signal.
    """
    tb = beat_times(x, fs)
    ibi = np.diff(tb)
    tm = tb[1:]
    ok = (ibi > 0.33) & (ibi < 1.5)  # 40-180 bpm, drop missed/extra beats
    tm, ibi = tm[ok], ibi[ok]
    t = np.arange(tm[0], tm[-1], 1.0 / out_fs)
    hr = 60.0 / np.interp(t, tm, ibi)
    return t, hr


def hrv_features(x: np.ndarray, fs: float) -> dict:
    """Very rough HRV. At 7-10 Hz sampling treat it as a hint, not a clinical value."""
    tb = beat_times(x, fs)
    ibi = np.diff(tb)
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]
    if len(ibi) < 10:
        return {"hrv_sdnn_ms": np.nan, "hrv_rmssd_ms": np.nan}
    return {
        "hrv_sdnn_ms": float(np.std(ibi) * 1000),
        "hrv_rmssd_ms": float(np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000),
    }


def pulse_waveform(x: np.ndarray, fs: float, n_points: int = 100):
    """
    Ensemble-averaged single heartbeat (trough -> trough), each beat stretched to the
    same length. Averaging many beats sampled at different phases gives a smooth
    shape even at low fs. Returns (phase 0..1, mean_wave normalised to 0..1, n_beats).
    """
    xc = bandpass(x, fs, 0.5, min(5.0, 0.45 * fs))
    hr = heart_rate(x, fs)
    min_dist = max(1, int(0.6 * fs * 60.0 / hr))
    troughs, _ = signal.find_peaks(-xc, distance=min_dist, prominence=0.3 * np.std(xc))
    t = np.arange(len(xc)) / fs
    beats = []
    grid = np.linspace(0, 1, n_points)
    for a, b in zip(troughs[:-1], troughs[1:]):
        dur = (b - a) / fs
        if not (0.4 < dur < 1.5):
            continue
        seg_t = np.linspace(t[a], t[b], n_points)
        seg = np.interp(seg_t, t, xc)  # linear interpolation onto common grid
        rng = seg.max() - seg.min()
        if rng <= 0:
            continue
        beats.append((seg - seg.min()) / rng)
    if not beats:
        return grid, np.full(n_points, np.nan), 0
    return grid, np.mean(beats, axis=0), len(beats)


def pulse_shape_features(x: np.ndarray, fs: float) -> dict:
    """
    Pulse-shape features linked to arterial elasticity (see Fabiani/Gratton 'PReFx').
      rise_frac   : share of the beat spent rising (stiff arteries -> faster rise, narrower peak)
      relax_area  : area between the falling part of the pulse and a straight line from
                    peak to end of beat, a simple PReFx-like proxy. More convex decay ~ more elastic.
      width_half  : fraction of the beat above half amplitude
    """
    ph, w, n = pulse_waveform(x, fs)
    if n == 0 or np.any(np.isnan(w)):
        return {"pulse_rise_frac": np.nan, "pulse_relax_area": np.nan, "pulse_width_half": np.nan, "pulse_n_beats": 0}
    k = int(np.argmax(w))
    rise_frac = ph[k]
    fall = w[k:]
    line = np.linspace(w[k], w[-1], len(fall))
    relax_area = float(np.trapezoid(fall - line, ph[k:]))
    width_half = float(np.mean(w > 0.5))
    return {"pulse_rise_frac": float(rise_frac), "pulse_relax_area": relax_area,
            "pulse_width_half": width_half, "pulse_n_beats": int(n)}


# ---------------------------------------------------------------------------
# Breathing
# ---------------------------------------------------------------------------
def respiration_rate(x: np.ndarray, fs: float, fmin: float = 0.13, fmax: float = 0.6) -> float:
    """
    Breaths per minute from the spectral peak.
    Default fmin=0.13 Hz skips Mayer waves (~0.1 Hz), which are often STRONGER than
    breathing in HbO and would be picked by mistake. For slow paced breathing
    (e.g. 6/min = 0.1 Hz) pass fmin=0.05 and use the known pace to sanity-check.
    """
    return 60.0 * dominant_freq(x, fs, fmin, fmax, win_s=90)


def respiration_rate_fused(x: np.ndarray, fs: float, fmin: float = 0.13, fmax: float = 0.6) -> dict:
    """
    Breathing rate from two independent routes, combined:
      1) breathing wave directly in HbO (blood volume follows chest pressure)
      2) RSA: beat-to-beat heart rate speeds up on inhale, slows on exhale
    Spectra are normalised and multiplied: only a frequency present in BOTH survives,
    which suppresses Mayer waves / drifts that show up in just one route.
    """
    f1, p1 = psd(x, fs, 90)
    t_hr, hr = instantaneous_hr(x, fs, out_fs=4.0)
    f2, p2 = psd(hr - hr.mean(), 4.0, 90)
    grid = np.arange(fmin, fmax, 0.002)
    a = np.interp(grid, f1, p1)
    b = np.interp(grid, f2, p2)
    a, b = a / a.max(), b / b.max()
    comb = a * b
    return {"rr_bpm": float(60 * grid[np.argmax(comb)]),
            "rr_bpm_hbo": float(60 * grid[np.argmax(a)]),
            "rr_bpm_rsa": float(60 * grid[np.argmax(b)])}


def respiration_signal(x: np.ndarray, fs: float, f_resp: float, rel_bw: float = 0.4) -> np.ndarray:
    """Narrow band-pass around the breathing frequency (+-40%) => clean breathing wave."""
    lo, hi = f_resp * (1 - rel_bw), f_resp * (1 + rel_bw)
    return bandpass(x, fs, max(lo, 0.02), hi, order=2)


def breath_phase(x: np.ndarray, fs: float, f_resp: float | None = None):
    """
    Instantaneous breathing phase via the Hilbert transform.
    Returns (phase_rad in [-pi, pi], resp_wave, envelope).
    Which phase = inhale has to be CALIBRATED: compare against a reference belt,
    the paced-breathing cue, or the RSA (HR rises on inhale, see instantaneous_hr).
    """
    if f_resp is None:
        f_resp = respiration_rate(x, fs) / 60.0
    r = respiration_signal(x, fs, f_resp)
    analytic = signal.hilbert(r)
    return np.angle(analytic), r, np.abs(analytic)


def inhale_mask(phase: np.ndarray, offset_rad: float = 0.0) -> np.ndarray:
    """
    Boolean mask 'inhale'. Default assumes inhale = FALLING half of the HbO breathing wave
    (Hilbert phase 0..pi): inhaling lowers intrathoracic pressure, venous blood drains from
    the head, head HbO/HbT drop; on exhale they rise (Elwell et al. 1996).
    Always CALIBRATE offset_rad on paced breathing / a belt: vascular delay shifts it.
    """
    p = np.angle(np.exp(1j * (phase - offset_rad)))
    return p >= 0


def detect_apnea(x: np.ndarray, fs: float, f_resp: float | None = None,
                 thr: float = 0.35, min_dur_s: float = 8.0, smooth_s: float = 6.0):
    """
    Unsupervised breath-hold detector: during apnea the respiratory oscillation vanishes,
    so the respiratory-band envelope drops. Returns list of (start_s, end_s).
    thr = fraction of the median envelope below which we call it 'no breathing'.
    """
    _, _, env = breath_phase(x, fs, f_resp)
    win = max(1, int(smooth_s * fs))
    env_s = np.convolve(env, np.ones(win) / win, mode="same")
    low = env_s < thr * np.median(env_s)
    events, start = [], None
    for i, v in enumerate(low):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (i - start) / fs >= min_dur_s:
                events.append((start / fs, i / fs))
            start = None
    if start is not None and (len(low) - start) / fs >= min_dur_s:
        events.append((start / fs, len(low) / fs))
    return events


def breath_hold_response(x: np.ndarray, fs: float, onsets_s, pre_s: float = 10.0, post_s: float = 45.0,
                         lowpass_hz: float = 0.08, hold_s: float | None = None) -> dict:
    """
    Event-locked average of the slow haemodynamic response to breath holds.
    x should be HbO, HbR or HbT (HbO+HbR). Returns the averaged curve and features:
      bh_amp        : peak change vs pre-onset baseline
      bh_ttp_s      : time to peak after onset
      bh_latency_s  : first time the response exceeds baseline + 2 SD (onset delay)
      bh_slope      : max rising slope (units/s)
      bh_auc        : area under the curve from 0 to post_s
      bh_index      : bh_amp / hold duration (if hold_s given) - 'breath-hold index' used in
                      fNIRS studies of trained divers vs controls
    """
    nyq = fs / 2
    sos = signal.butter(3, lowpass_hz / nyq, btype="low", output="sos")
    xs = signal.sosfiltfilt(sos, x)
    n_pre, n_post = int(pre_s * fs), int(post_s * fs)
    epochs = []
    for t0 in onsets_s:
        k = int(round(t0 * fs))
        if k - n_pre < 0 or k + n_post > len(xs):
            continue
        ep = xs[k - n_pre:k + n_post]
        epochs.append(ep - ep[:n_pre].mean())
    if not epochs:
        return {"bh_n": 0}
    E = np.array(epochs)
    mean = E.mean(axis=0)
    t = (np.arange(len(mean)) - n_pre) / fs
    post = t >= 0
    base_sd = E[:, :n_pre].std() + 1e-20
    k_peak = np.argmax(np.abs(mean[post]))
    above = np.flatnonzero(np.abs(mean[post]) > 2 * base_sd)
    d = np.gradient(mean, 1 / fs)
    return {
        "bh_n": len(epochs),
        "bh_amp": float(mean[post][k_peak]),
        "bh_ttp_s": float(t[post][k_peak]),
        "bh_latency_s": float(t[post][above[0]]) if len(above) else np.nan,
        "bh_slope": float(d[post].max()),
        "bh_auc": float(np.trapezoid(mean[post], t[post])),
        "bh_index": float(mean[post][k_peak] / hold_s) if hold_s else np.nan,
        "_t": t, "_mean": mean, "_epochs": E,
    }


# ---------------------------------------------------------------------------
# Coupling between rhythms
# ---------------------------------------------------------------------------
def coherence_band(x: np.ndarray, y: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Mean magnitude-squared coherence between two signals in a band (0 = unrelated, 1 = locked)."""
    nper = int(min(len(x), 60 * fs))
    f, c = signal.coherence(x, y, fs=fs, nperseg=nper)
    m = (f >= lo) & (f <= hi)
    return float(c[m].mean()) if np.any(m) else np.nan


def hbo_hbr_phase_lag(hbo: np.ndarray, hbr: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Mean phase difference (rad) HbO vs HbR in a band. Cerebral vs scalp origin changes it."""
    a = np.angle(signal.hilbert(bandpass(hbo, fs, lo, hi, 2)))
    b = np.angle(signal.hilbert(bandpass(hbr, fs, lo, hi, 2)))
    return float(np.angle(np.mean(np.exp(1j * (a - b)))))


def paced_phasor(hbo: np.ndarray, hbr: np.ndarray, fs: float, f0: float) -> dict:
    """
    Lock-in ('phasor') analysis at a KNOWN driving frequency f0, e.g. paced breathing 0.1 Hz.
    Idea from coherent hemodynamics spectroscopy (CHS, Fantini et al.): paced breathing drives
    coherent oscillations; the HbR-vs-HbO phase and amplitude ratio reflect blood flow vs
    volume balance and have been linked to cerebral autoregulation.
    Returns amplitudes, HbR-HbO phase difference (deg), amplitude ratio, HbT amplitude and
    how much of the signal power sits at f0 (coherence-like 'locking' 0..1).
    """
    t = np.arange(len(hbo)) / fs
    ref = np.exp(-2j * np.pi * f0 * t)

    def lockin(x):
        xb = bandpass(x - np.mean(x), fs, max(f0 * 0.6, 0.02), f0 * 1.4, order=2)
        return 2 * np.mean(xb * ref), xb

    o, ob = lockin(hbo)
    d, db = lockin(hbr)
    tt, tb = lockin(hbo + hbr)
    lock_o = (np.abs(o) ** 2 / 2) / (np.var(ob) + 1e-20)
    return {
        "pp_amp_hbo": float(np.abs(o)), "pp_amp_hbr": float(np.abs(d)), "pp_amp_hbt": float(np.abs(tt)),
        "pp_ratio_hbr_hbo": float(np.abs(d) / (np.abs(o) + 1e-20)),
        "pp_phase_hbr_minus_hbo_deg": float(np.degrees(np.angle(d / (o + 1e-20)))),
        "pp_locking_hbo": float(min(lock_o, 1.0)),
    }


def rsa_amplitude(x: np.ndarray, fs: float, f_resp: float | None = None) -> float:
    """
    Respiratory sinus arrhythmia amplitude (bpm, peak-to-trough HR swing per breath) from
    fNIRS pulse alone. In HRV-biofeedback research RSA is largest near ~6 breaths/min
    (resonance, 4.5-6.5/min individually) and reflects vagal/baroreflex function.
    """
    t_hr, hr = instantaneous_hr(x, fs, out_fs=4.0)
    if f_resp is None:
        f_resp = respiration_rate(x, fs) / 60.0
    hr_b = bandpass(hr - hr.mean(), 4.0, max(f_resp * 0.6, 0.03), f_resp * 1.4, order=2)
    return float(2 * np.sqrt(2) * np.std(hr_b))  # sine amplitude*2 = peak-to-trough


# ---------------------------------------------------------------------------
# One-call feature vector for a resting/free-breathing recording
# ---------------------------------------------------------------------------
def subject_features(hbo: np.ndarray, hbr: np.ndarray, fs: float, prefix: str = "") -> dict:
    """All 'static' features from one channel (or channel average) of a recording."""
    f = {}
    f["hr_bpm"] = heart_rate(hbo, fs)
    f["rr_bpm"] = respiration_rate(hbo, fs)
    f["quality_cardiac"] = cardiac_quality(hbo, fs)
    f.update(hrv_features(hbo, fs))
    f.update(pulse_shape_features(hbo, fs))
    f.update({k + "_hbo": v for k, v in band_powers(hbo, fs).items()})
    f.update({k + "_hbr": v for k, v in band_powers(hbr, fs).items()})
    f["coh_resp_hbo_hbr"] = coherence_band(hbo, hbr, fs, *BANDS["resp"])
    f["coh_mayer_hbo_hbr"] = coherence_band(hbo, hbr, fs, *BANDS["mayer"])
    f["pulse_amp_hbo"] = float(np.std(bandpass(hbo, fs, *BANDS["cardiac"])))
    try:
        f["rsa_amp_bpm"] = rsa_amplitude(hbo, fs, f["rr_bpm"] / 60.0)
    except Exception:
        f["rsa_amp_bpm"] = np.nan
    return {prefix + k: v for k, v in f.items()}


# ---------------------------------------------------------------------------
# Statistics / validation for small N
# ---------------------------------------------------------------------------
def partial_corr(x, y, covar):
    """
    Spearman correlation between x and y after removing the linear effect of covar
    (e.g. age). Use it to show a biomarker tracks fitness and not just age.
    Returns (rho, p).
    """
    x, y, c = map(lambda a: np.asarray(a, float), (x, y, covar))
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(c))
    x, y, c = stats.rankdata(x[ok]), stats.rankdata(y[ok]), stats.rankdata(c[ok])
    C = np.column_stack([np.ones_like(c), c])
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)


def evaluate_subject_level(X, y, task: str = "regression", model=None, groups=None, n_perm: int = 200, seed: int = 0):
    """
    Honest evaluation for small datasets.
      - If groups (subject ids) are given: Leave-One-Subject-Out (LOSO).
        Otherwise one row = one subject and we use Leave-One-Out.
      - Compares against a dummy baseline (predict mean / most frequent class).
      - Permutation test: shuffles y n_perm times to get a p-value.
    Returns dict with score, baseline score, p-value and out-of-fold predictions.
    """
    from sklearn.base import clone
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
    from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X, y = np.asarray(X, float), np.asarray(y)
    if model is None:
        est = RidgeCV(alphas=np.logspace(-2, 3, 20)) if task == "regression" else LogisticRegression(C=0.5, max_iter=2000)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    cv = LeaveOneGroupOut() if groups is not None else LeaveOneOut()
    if task == "regression":
        dummy, metric, better = DummyRegressor(), mean_absolute_error, "lower"
    else:
        dummy, metric, better = DummyClassifier(strategy="most_frequent"), balanced_accuracy_score, "higher"

    def run(m, yy):
        pred = cross_val_predict(clone(m), X, yy, cv=cv, groups=groups)
        return metric(yy, pred), pred

    score, pred = run(model, y)
    base, _ = run(dummy, y)
    rng = np.random.default_rng(seed)
    null = np.array([run(model, rng.permutation(y))[0] for _ in range(n_perm)])
    p = (np.sum(null <= score) + 1) / (n_perm + 1) if better == "lower" else (np.sum(null >= score) + 1) / (n_perm + 1)
    return {"metric": metric.__name__, "score": float(score), "baseline": float(base), "p_perm": float(p), "pred": pred}


# ---------------------------------------------------------------------------
# Loading real data (MNE)
# ---------------------------------------------------------------------------
def load_snirf_haemo(path: str, tddr: bool = True, ppf: float = 6.0):
    """
    SNIRF file -> HbO/HbR numpy arrays.
    Pipeline: raw intensity -> optical density -> (TDDR motion correction) -> Beer-Lambert.
    NO filtering here on purpose: cardiac and respiratory bands are the signal we want.
    Returns dict(hbo, hbr (n_ch, n_t), fs, ch_hbo, ch_hbr, sci, short_mask, raw_haemo).
    If the file already contains concentrations, Beer-Lambert is skipped.
    """
    import mne

    raw = mne.io.read_raw_snirf(path, preload=True, verbose="error")
    return haemo_from_raw(raw, tddr=tddr, ppf=ppf)


def haemo_from_raw(raw, tddr: bool = True, ppf: float = 6.0) -> dict:
    """Same as load_snirf_haemo but for an already loaded MNE Raw (NIRx folder, .fif, ...)."""
    import mne

    raw = raw.copy().load_data()
    types = set(raw.get_channel_types())
    sci = None
    if "fnirs_cw_amplitude" in types:
        od = mne.preprocessing.nirs.optical_density(raw)
        sci = mne.preprocessing.nirs.scalp_coupling_index(od)
        if tddr:
            od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(od)
        haemo = mne.preprocessing.nirs.beer_lambert_law(od, ppf=ppf)
    elif "fnirs_od" in types:
        od = raw
        sci = mne.preprocessing.nirs.scalp_coupling_index(od)
        if tddr:
            od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(od)
        haemo = mne.preprocessing.nirs.beer_lambert_law(od, ppf=ppf)
    else:
        haemo = raw
    picks_hbo = mne.pick_types(haemo.info, fnirs="hbo")
    picks_hbr = mne.pick_types(haemo.info, fnirs="hbr")
    dist = mne.preprocessing.nirs.source_detector_distances(haemo.info, picks=picks_hbo)
    data = haemo.get_data()
    return {
        "hbo": data[picks_hbo], "hbr": data[picks_hbr], "fs": haemo.info["sfreq"],
        "ch_hbo": [haemo.ch_names[i] for i in picks_hbo],
        "ch_hbr": [haemo.ch_names[i] for i in picks_hbr],
        "sci": sci, "short_mask": dist < 0.015, "dist_m": dist,
        "annotations": haemo.annotations, "raw_haemo": haemo,
    }
