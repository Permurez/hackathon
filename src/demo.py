"""
demo.py - one recording in, the endothelial-band marker out (for the demo video).

  python src/demo.py dane/infopack_and_data/fnirs_high_freq_supine/data/baseline/7_SR_BS_2025-07-03-14-13-26.snirf

Steps shown on screen:
  1. raw intensity -> HbO (own conversion, src/haemo.py), forehead channels
  2. spectrum of the 10-min resting signal, endothelial band 0.0095-0.021 Hz highlighted
  3. the person's band power vs the cohort (percentile) and the model's probability of
     ">3 h aerobic exercise per week" (logistic regression trained on the OTHER people only)
Group-level marker, not a diagnosis.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo  # noqa: E402
from features import FOREHEAD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LO, HI = 0.0095, 0.021


def main():
    path = Path(sys.argv[1])
    m = re.match(r"(\d+)_", path.name)
    sid = int(m.group(1)) if m else None
    print(f"[1/3] {path.name}: raw light intensity -> HbO ...")
    h = haemo.convert(path, dpf=6.0)
    fs = h["fs"]
    dead = {c.split()[0] for c, n in zip(h["od_channels"], h["nonpositive"]) if n}
    idx = {n: i for i, n in enumerate(h["pairs"])}
    x = h["hbo"][[idx[n] for n in FOREHEAD if n not in dead]].mean(0)
    down = int(fs // 5)
    xd = signal.detrend(signal.decimate(x - x.mean(), down, ftype="fir"))
    fsd = fs / down
    print("[2/3] spectrum of the resting signal ...")
    f, p = signal.welch(xd, fsd, nperseg=int(300 * fsd))
    band = (f >= LO) & (f <= HI)
    tot = (f >= 0.0095) & (f <= 2.0)
    endo = np.trapezoid(p[band], f[band]) / np.trapezoid(p[tot], f[tot])

    print("[3/3] compare with the cohort and the model ...")
    coh = pd.read_csv(ROOT / "bands_supine.csv")[["subject", "l30_hbo_endothelial_rel"]] \
        .merge(pd.read_csv(ROOT / "features.csv")[["subject", "age", "activity"]], on="subject")
    coh = coh[(coh.age < 60) & (coh.subject != sid)].dropna()
    pct = 100 * np.mean(coh.l30_hbo_endothelial_rel < endo)
    y = (coh.activity == coh.activity.max()).astype(int)
    mu, sd = coh.l30_hbo_endothelial_rel.mean(), coh.l30_hbo_endothelial_rel.std()
    model = LogisticRegression().fit(((coh.l30_hbo_endothelial_rel - mu) / sd).values[:, None], y)
    prob = model.predict_proba([[(endo - mu) / sd]])[0, 1]

    print("\n=== RESULT ===")
    print(f"endothelial band power: {100 * endo:.1f} % of the signal")
    print(f"percentile in the cohort (18-42 y, n={len(coh)}): {pct:.0f}")
    print(f"model: probability of >3 h aerobic exercise / week = {100 * prob:.0f} %  (cohort AUC ~0.70)")
    print("Group-level marker, not a diagnosis.")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    t = np.arange(len(xd)) / fsd / 60
    ax[0].plot(t, xd, lw=0.6, color="#2a78d6")
    ax[0].set_xlabel("czas [min]"); ax[0].set_ylabel("HbO [µM]"); ax[0].set_title("1. Sygnał z czoła, 10 min spoczynku")
    ax[1].semilogx(f[1:], p[1:], color="#14213d")
    ax[1].fill_between(f[band], p[band], color="#d97706", alpha=0.6, label="pasmo śródbłonkowe")
    ax[1].set_xlim(0.005, 2); ax[1].set_xlabel("częstotliwość [Hz]"); ax[1].set_title("2. Widmo: z czego składa się sygnał"); ax[1].legend()
    act_lab = {0.0: "brak", 0.32: "lekka", 1.06: "<1 h", 1.76: "1–3 h", 3.03: ">3 h"}
    lv = sorted(coh.activity.unique())
    for i, a in enumerate(lv):
        v = 100 * coh[coh.activity == a].l30_hbo_endothelial_rel
        ax[2].scatter(np.full(len(v), i), v, color="#9aa5b1", s=22)
    ax[2].axhline(100 * endo, color="#d97706", lw=2.5, label=f"ta osoba: {pct:.0f}. percentyl")
    ax[2].set_xticks(range(len(lv)), [act_lab.get(a, a) for a in lv]); ax[2].set_xlabel("aeroby tygodniowo (ankieta)")
    ax[2].set_ylabel("moc pasma [% sygnału]"); ax[2].set_title(f"3. Na tle grupy · model: {100 * prob:.0f}% szans >3 h/tydz"); ax[2].legend()
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Ślad treningu w spoczynkowym fNIRS (wskaźnik grupowy, nie diagnoza)", fontsize=13)
    fig.tight_layout()
    out = ROOT / f"demo_{path.stem}.png"
    fig.savefig(out, dpi=120)
    print(f"figure -> {out}")
    if "--show" in sys.argv:  # this Python has no GUI toolkit: open the saved PNG in the system image viewer
        import subprocess
        subprocess.Popen(["xdg-open", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
