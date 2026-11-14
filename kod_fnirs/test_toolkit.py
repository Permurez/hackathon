"""
test_toolkit.py - checks that every feature recovers what synthetic.py planted.
Run:  python test_toolkit.py      (all lines should say OK)
"""

import numpy as np

import fnirs_toolkit as fk
from synthetic import make_cohort, make_recording


def check(name, cond, info=""):
    print(f"[{'OK ' if cond else 'FAIL'}] {name} {info}")
    return cond


def main():
    ok = True
    holds = [(200, 220), (320, 340), (440, 460)]
    rec = make_recording(hr_bpm=66, rr_bpm=15, breath_holds=holds, seed=3)
    x, y, fs = rec["hbo"], rec["hbr"], rec["fs"]

    hr = fk.heart_rate(x, fs)
    ok &= check("heart rate", abs(hr - 66) < 2, f"{hr:.1f} bpm (true 66)")

    rr = fk.respiration_rate(x, fs)
    ok &= check("respiration rate", abs(rr - 15) < 1, f"{rr:.2f} /min (true 15)")
    naive = fk.respiration_rate(x, fs, fmin=0.05)
    print(f"       (naive band from 0.05 Hz would say {naive:.1f}/min -> Mayer waves trap)")
    fused = fk.respiration_rate_fused(x, fs)
    ok &= check("respiration rate fused (HbO x RSA)", abs(fused["rr_bpm"] - 15) < 1, str({k: round(v, 1) for k, v in fused.items()}))

    ev = fk.detect_apnea(x, fs)
    hits = sum(any(abs(s - hs) < 6 for s, _ in ev) for hs, _ in holds)
    ok &= check("apnea detection", hits == 3 and len(ev) == 3, f"found {[(round(a), round(b)) for a, b in ev]}")

    resp = fk.breath_hold_response(x, fs, [h[0] for h in holds], hold_s=20)
    ok &= check("breath-hold response", resp["bh_amp"] > 0.5 and 15 < resp["bh_ttp_s"] < 40,
                f"amp={resp['bh_amp']:.2f} index={resp['bh_index']:.3f}/s ttp={resp['bh_ttp_s']:.1f}s latency={resp['bh_latency_s']:.1f}s")

    # breathing phase vs belt reference (outside breath holds)
    rec2 = make_recording(hr_bpm=70, rr_bpm=12, seed=4)
    ph, r, env = fk.breath_phase(rec2["hbo"], fs)
    best = max(((np.mean(fk.inhale_mask(ph, off) == rec2["inhale_ref"]), off)
                for off in np.linspace(-np.pi, np.pi, 37)))
    ok &= check("breath phase (inhale/exhale)", best[0] > 0.85,
                f"accuracy={best[0]:.2f} at calibrated offset={best[1]:.2f} rad")

    # RSA: instantaneous HR should oscillate at the breathing rate
    t_hr, hr_i = fk.instantaneous_hr(rec2["hbo"], fs)
    f_rsa = 60 * fk.dominant_freq(hr_i - hr_i.mean(), 4.0, 0.05, 0.5, win_s=90)
    ok &= check("RSA (HR follows breathing)", abs(f_rsa - 12) < 1.5, f"{f_rsa:.1f} /min (true 12)")

    # slow paced breathing: 6/min overlaps Mayer waves (0.1 Hz)
    rec3 = make_recording(rr_bpm=6, mayer_amp=0.2, seed=5)
    rr3 = fk.respiration_rate(rec3["hbo"], fs, fmin=0.05)
    ok &= check("slow paced breathing 6/min", abs(rr3 - 6) < 0.7, f"{rr3:.2f} /min")

    # paced breathing phasor + RSA amplitude
    pp = fk.paced_phasor(rec3["hbo"], rec3["hbr"], fs, 0.1)
    # planted: HbR moves opposite to HbO at the breathing frequency -> phase ~180 deg
    ok &= check("paced-breathing phasor at 0.1 Hz", abs(abs(pp["pp_phase_hbr_minus_hbo_deg"]) - 180) < 25 and pp["pp_locking_hbo"] > 0.5,
                f"{ {k: round(v, 2) for k, v in pp.items()} }")
    rsa_big = fk.rsa_amplitude(make_recording(rr_bpm=6, rsa_bpm=8, seed=9)["hbo"], fs, 0.1)
    rsa_small = fk.rsa_amplitude(make_recording(rr_bpm=6, rsa_bpm=2, seed=9)["hbo"], fs, 0.1)
    ok &= check("RSA amplitude tracks planted RSA", rsa_big > 2 * rsa_small,
                f"planted 16 vs 4 bpm p-t -> {rsa_big:.1f} vs {rsa_small:.1f}")

    # pulse shape tracks stiffness
    stiff = [0.0, 0.5, 1.0]
    rf = [fk.pulse_shape_features(make_recording(stiffness=s, seed=7)["hbo"], fs)["pulse_rise_frac"] for s in stiff]
    ok &= check("pulse rise fraction falls with stiffness", rf[0] > rf[1] > rf[2], f"{np.round(rf, 3)}")

    # full feature vector
    feats = fk.subject_features(x, y, fs)
    ok &= check("subject_features", len(feats) > 15 and not np.isnan(feats["hr_bpm"]), f"{len(feats)} features")

    # cohort: can we predict activity better than baseline? and is it more than age?
    recs, age, act = make_cohort(30)
    import pandas as pd
    X = pd.DataFrame([fk.subject_features(r["hbo"], r["hbr"], r["fs"]) for r in recs])
    cols = ["hr_bpm", "pulse_rise_frac", "pulse_relax_area", "pulse_width_half"]
    res = fk.evaluate_subject_level(X[cols].values, act, task="regression", n_perm=100)
    ok &= check("LOO regression beats baseline", res["score"] < res["baseline"] and res["p_perm"] < 0.05,
                f"MAE={res['score']:.3f} baseline={res['baseline']:.3f} p={res['p_perm']:.3f}")
    rho, p = fk.partial_corr(X["hr_bpm"], act, age)
    ok &= check("partial corr HR~activity | age", rho < -0.4 and p < 0.05, f"rho={rho:.2f} p={p:.3g}")

    # quality ranking: noisy channel should be last
    good = make_recording(seed=8)["hbo"]
    bad = np.random.default_rng(0).normal(size=len(good))
    order = fk.rank_channels(np.vstack([bad, good]), fs)
    ok &= check("channel quality ranking", order[0] == 1)

    print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
