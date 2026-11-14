"""
synthetic.py - fake fNIRS recordings with KNOWN ground truth.

Use it to test the pipeline before the real data arrives and to sanity-check
every feature: if a function can't recover a value we planted, it won't work on real data.
"""

from __future__ import annotations

import numpy as np


def _pulse_shape(phase: np.ndarray, stiffness: float = 0.5) -> np.ndarray:
    """
    One heartbeat as a function of phase (0..1 per beat).
    stiffness 0 = elastic artery (slow rise, clear dicrotic bump),
    stiffness 1 = stiff artery (sharp peak, bump merges into the decay).
    """
    p = np.mod(phase, 1.0)
    rise = 0.12 + 0.10 * (1 - stiffness)            # time to systolic peak
    main = np.exp(-0.5 * ((p - rise) / (0.06 + 0.04 * (1 - stiffness))) ** 2)
    notch_pos = rise + 0.25
    dicrotic = (0.35 * (1 - stiffness) + 0.05) * np.exp(-0.5 * ((p - notch_pos) / 0.07) ** 2)
    decay = 0.25 * np.exp(-(p - rise).clip(0) / 0.35) * (p > rise)
    return main + dicrotic + decay


def make_recording(
    fs: float = 10.0,
    duration_s: float = 600.0,
    hr_bpm: float = 66.0,
    rr_bpm: float = 15.0,
    rsa_bpm: float = 4.0,
    stiffness: float = 0.5,
    breath_holds: list[tuple[float, float]] | None = None,
    bh_amp: float = 1.0,
    bh_delay_s: float = 6.0,
    mayer_amp: float = 0.4,
    noise: float = 0.15,
    seed: int = 0,
) -> dict:
    """
    Returns dict(t, hbo, hbr, fs, resp_ref, inhale_ref, truth).
    resp_ref  : 'belt' signal (lung volume, rises during inhale)
    Sign convention from literature: inhale lowers intrathoracic pressure and drains venous
    blood from the head, so head HbO/HbT FALL on inhale and RISE on exhale (Elwell 1996).
    inhale_ref: True while inhaling
    Units are arbitrary (think micromolar-ish).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fs)
    n = len(t)

    # --- breathing: lung volume, flat during breath holds -------------------
    f_r = rr_bpm / 60.0
    resp_phase = 2 * np.pi * f_r * t
    lung = -np.cos(resp_phase)                  # rising half (-pi..0 of phase-pi) = inhale
    breathing = np.ones(n)
    for s, e in breath_holds or []:
        breathing[(t >= s) & (t < e)] = 0.0
    sm = int(2 * fs)                            # soften start/end of holds
    breathing = np.convolve(breathing, np.ones(sm) / sm, mode="same")
    lung = lung * breathing
    dlung = np.gradient(lung)
    inhale_ref = (dlung > 0) & (breathing > 0.5)

    # --- heart: RSA = HR faster on inhale, slower on exhale ------------------
    hr_inst = hr_bpm + rsa_bpm * np.sin(resp_phase) * breathing  # sin leads -cos by 90 deg = peaks mid-inhale
    beat_phase = np.cumsum(hr_inst / 60.0) / fs
    cardiac = _pulse_shape(beat_phase, stiffness)
    cardiac = (cardiac - cardiac.mean()) / cardiac.std()

    # --- slow systemic components --------------------------------------------
    mayer = mayer_amp * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2 * np.pi))
    vlf = 0.3 * np.cumsum(rng.normal(size=n)) / np.sqrt(fs * 60)

    # --- breath-hold response: CO2 builds up -> vasodilation -> HbO/HbT rise ---
    bh = np.zeros(n)
    for s, e in breath_holds or []:
        dur = e - s
        tt = t - (s + bh_delay_s)
        ramp = np.clip(tt / dur, 0, 1)                    # grows while CO2 accumulates
        recov = np.exp(-np.clip(tt - dur, 0, None) / 8.0)  # washes out after breathing restarts
        bh += bh_amp * ramp * recov * (tt > 0)

    hbo = (0.5 * cardiac - 0.35 * lung + mayer + vlf + bh
           + noise * rng.normal(size=n))
    hbr = (-0.15 * cardiac + 0.12 * lung - 0.4 * mayer + 0.3 * vlf - 0.35 * bh
           + noise * rng.normal(size=n))
    return {
        "t": t, "hbo": hbo, "hbr": hbr, "fs": fs,
        "resp_ref": lung, "inhale_ref": inhale_ref,
        "truth": {"hr_bpm": hr_bpm, "rr_bpm": rr_bpm, "stiffness": stiffness,
                  "breath_holds": breath_holds or [], "bh_delay_s": bh_delay_s},
    }


def make_cohort(n_subjects: int = 30, seed: int = 1):
    """
    Fake cohort: age and physical activity drive stiffness and resting HR.
    Fitter -> lower HR, more elastic arteries; older -> stiffer arteries.
    Returns list of recordings and arrays age, activity (0..1).
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(19, 65, n_subjects)
    activity = np.clip(rng.normal(0.5, 0.2, n_subjects) - 0.003 * (age - 40), 0, 1)
    recs = []
    for i in range(n_subjects):
        stiff = np.clip(0.2 + 0.01 * (age[i] - 20) - 0.4 * activity[i] + rng.normal(0, 0.05), 0, 1)
        hr = 75 - 15 * activity[i] + rng.normal(0, 3)
        recs.append(make_recording(hr_bpm=hr, rr_bpm=rng.uniform(11, 18), stiffness=stiff,
                                   duration_s=300, seed=100 + i))
    return recs, age, activity
