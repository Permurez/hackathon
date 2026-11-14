"""
demo_figure.py - one overview figure on synthetic data: what each feature looks like.
Run:  python demo_figure.py  -> demo_overview.png
Use the same 4 panels on real data on day 1 to see what is actually in the signal.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fnirs_toolkit as fk
from synthetic import make_recording

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, INK2 = "#1f1f1e", "#5c5b55"

holds = [(200, 220), (320, 340), (440, 460)]
rec = make_recording(hr_bpm=66, rr_bpm=15, breath_holds=holds, seed=3)
x, fs, t = rec["hbo"], rec["fs"], rec["t"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)

# 1) raw HbO: heartbeat riding on breathing
a = ax[0, 0]
m = (t > 100) & (t < 130)
a.plot(t[m], x[m], color=BLUE, lw=1.2, label="HbO (synthetic)")
a.plot(t[m], fk.respiration_signal(x, fs, 0.25)[m], color=ORANGE, lw=2, label="breathing band 0.15-0.35 Hz")
a.set(title="1. 30 s of HbO: fast pulse on top of slow breathing", xlabel="time [s]", ylabel="a.u.")
a.set_ylim(x[m].min() - 0.2, x[m].max() + 0.9)
a.legend(frameon=False, loc="upper left", ncol=2)

# 2) spectrum with bands
a = ax[0, 1]
f, p = fk.psd(x, fs, 90)
a.semilogy(f, p, color=BLUE, lw=1.5)
for name, (lo, hi), c, yy in [("Mayer\n~0.1 Hz", fk.BANDS["mayer"], GREY, 0.80),
                              ("breathing", fk.BANDS["resp"], ORANGE, 0.93),
                              ("heart (+ harmonics)", fk.BANDS["cardiac"], AQUA, 0.93)]:
    a.axvspan(lo, hi, color=c, alpha=0.12, lw=0)
    a.text((lo + hi) / 2 if name != "Mayer\n~0.1 Hz" else hi + 0.02, yy, name, transform=a.get_xaxis_transform(),
           ha="center" if name != "Mayer\n~0.1 Hz" else "left", color=INK2, fontsize=9)
a.set(xlim=(0, 2.5), title="2. Spectrum: three physiological peaks", xlabel="frequency [Hz]", ylabel="power")

# 3) breath-hold response
a = ax[1, 0]
r = fk.breath_hold_response(x, fs, [h[0] for h in holds])
for ep in r["_epochs"]:
    a.plot(r["_t"], ep, color=GREY, lw=0.8, alpha=0.6)
a.plot(r["_t"], r["_mean"], color=BLUE, lw=2.2, label="mean of 3 holds")
a.axvspan(0, 20, color=ORANGE, alpha=0.12, lw=0)
a.text(10, -0.15, "breath held (20 s)", ha="center", color=INK2)
a.axvline(r["bh_ttp_s"], color=INK2, lw=1, ls=":")
a.text(r["bh_ttp_s"] + 1, r["bh_amp"] * 0.5, f"peak {r['bh_ttp_s']:.0f} s", color=INK2)
a.set(title="3. Breath hold: delayed HbO rise (CO2 -> vessels dilate)", xlabel="time from hold onset [s]",
      ylabel="change vs baseline [a.u.]")
a.legend(frameon=False, loc="upper right")

# 4) breathing phase vs reference belt
a = ax[1, 1]
rec2 = make_recording(hr_bpm=70, rr_bpm=12, seed=4)
ph, rw, env = fk.breath_phase(rec2["hbo"], fs)
m = (rec2["t"] > 60) & (rec2["t"] < 90)
inh = fk.inhale_mask(ph)
acc = np.mean(inh == rec2["inhale_ref"])
a.plot(rec2["t"][m], rec2["resp_ref"][m], color=GREY, lw=2, label="reference belt (lung volume)")
a.plot(rec2["t"][m], rw[m] / np.std(rw) * 0.7, color=BLUE, lw=1.5, label="breathing wave from HbO")
a.fill_between(rec2["t"][m], -1.6, 1.6, where=inh[m], color=AQUA, alpha=0.15, lw=0, label="detected inhale")
a.set(title=f"4. Breathing phase from HbO (inhale accuracy {acc:.0%})", xlabel="time [s]", ylabel="a.u.",
      ylim=(-1.6, 1.9))
a.legend(frameon=False, loc="upper right", ncol=1, fontsize=8)

fig.suptitle("fNIRS: what is hidden besides brain activity (synthetic data, known ground truth)",
             color=INK, fontsize=13)
fig.savefig("demo_overview.png", dpi=130)
print("saved demo_overview.png")
