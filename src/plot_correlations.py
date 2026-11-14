"""
plot_correlations.py - readable version of the correlation screen (reads correlations.csv from features.py).

Colour = Spearman rho (blue = negative, red = positive). Only cells that survive FDR (q < 0.10) are
solid and labelled; everything else is faded. Rows and columns are grouped and named in Polish.

  python src/plot_correlations.py      # -> correlations_pl.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

ROWS = [  # (section, [(feature, Polish label)])
    ("Serce", [("hr_bpm", "tętno spoczynkowe"), ("rmssd_ms", "zmienność rytmu RMSSD"),
               ("sdnn_ms", "zmienność rytmu SDNN"), ("sd1_sd2", "nieregularność rytmu SD1/SD2")]),
    ("Fala tętna", [("pulse_p1_phase", "faza 1. szczytu fali (P1)"), ("pulse_relax_area", "wybrzuszenie opadania fali"),
                    ("pulse_hbo_uM", "amplituda pulsu 30 mm"), ("skin_pulse_hbo_uM", "amplituda pulsu skóry 10 mm"),
                    ("pulse_hbr_over_hbo", "puls HbR / HbO")]),
    ("Oddech\ni wolne fale", [("resp_rate_bpm", "częstość oddechu"), ("pow_resp_rel", "moc pasma oddechowego"),
                             ("pow_mayer_rel", "moc fal Mayera (0,1 Hz)"), ("mayer_freq_hz", "częstotliwość fal Mayera"),
                             ("pow_vlf_rel", "moc wolnych fal (VLF)"), ("pow_cardiac_rel", "moc pasma sercowego"),
                             ("hbo_hbr_slow_corr", "zgodność HbO i HbR (wolne)")]),
    ("Skóra\nvs głębiej", [("skin_share_cardiac", "udział skóry w pulsie 30 mm"),
                          ("skin_share_slow", "udział skóry w wolnych zmianach")]),
    ("Bezdech\n(z markerów,\nnie z fNIRS)", [("bh_max_s", "najdłuższy bezdech"), ("bh_mean_s", "średni bezdech")]),
    ("Pomiar /\nurządzenie\n(nie fizjologia)", [
        ("forehead_log_intensity_760", "światło czoło 760 nm"), ("forehead_log_intensity_850", "światło czoło 850 nm"),
        ("hair_log_intensity_760", "światło pod włosami 760 nm"), ("hair_log_intensity_850", "światło pod włosami 850 nm"),
        ("short_log_intensity_850", "światło kanały krótkie 850 nm"),
        ("forehead_pulse_od_850", "puls w gęstości optycznej czoło"), ("hair_pulse_od_850", "puls w gęstości opt. włosy"),
        ("short_pulse_od_850", "puls w gęstości opt. krótkie"), ("forehead_cardiac_quality", "jakość pulsu czoło"),
        ("hair_cardiac_quality", "jakość pulsu pod włosami"), ("short_cardiac_quality", "jakość pulsu krótkie"),
        ("ratio_of_ratios_R", "R (jak w pulsoksymetrze)"), ("motion_acc_sd", "ruch głowy"),
        ("missing_pct", "brakujące próbki"), ("dead_channels", "martwe kanały")]),
]
COLS = [  # (section, [(target, Polish label)])
    ("Wielkość ciała", [("sex_male", "płeć (+ = wyżej u M)"), ("height", "wzrost"), ("weight", "waga"),
                        ("bmi", "BMI"), ("head_circ", "obwód głowy")]),
    ("Wiek i zdrowie", [("age", "wiek (wszyscy)"), ("age_lt60", "wiek (< 60 l.)"), ("medication_any", "jakiekolwiek leki"),
                        ("antihypertensive", "leki na nadciśnienie")]),
    ("Styl życia", [("activity", "aktywność fizyczna"), ("treadmill_kmh", "prędkość na bieżni")]),
    ("Włosy i skóra", [("hair_darkness", "ciemność włosów"), ("hair_curl", "skręt włosów"),
                       ("hair_thickness", "grubość włosa"), ("fitzpatrick", "karnacja (Fitzpatrick)")]),
    ("Kontrola", [("NEG_CONTROL_group", "grupa losowa\n(ma być pusto)")]),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d6d5d0"
# diverging: blue arm <- gray midpoint -> red arm (reference palette)
CMAP = LinearSegmentedColormap.from_list("div", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
                                                 "#f4b4b3", "#e34948", "#9c2424"])


def draw(ROWS, out, title, csv="correlations.csv"):
    c = pd.read_csv(ROOT / csv)
    rho = c.pivot(index="feature", columns="target", values="rho")
    q = c.pivot(index="feature", columns="target", values="q_fdr")
    rows = [(s, f, lab) for s, items in ROWS for f, lab in items]
    cols = [(s, t, lab) for s, items in COLS for t, lab in items]
    R = np.array([[rho.get(t, {}).get(f, np.nan) for _, t, _ in cols] for _, f, _ in rows])
    Q = np.array([[q.get(t, {}).get(f, np.nan) for _, t, _ in cols] for _, f, _ in rows])

    fig = plt.figure(figsize=(14, max(8.0, 3.2 + 0.33 * len(rows))), facecolor="#fcfcfb")
    ax = fig.add_axes([0.28, 0.08, 0.66, 0.78])
    ax.set_facecolor("#fcfcfb")
    norm = plt.Normalize(-0.7, 0.7)
    for i in range(len(rows)):
        for j in range(len(cols)):
            r, qq = R[i, j], Q[i, j]
            if not np.isfinite(r):
                continue
            sig = np.isfinite(qq) and qq < 0.10
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=CMAP(norm(r)),
                                       alpha=1.0 if sig else 0.22, lw=0))
            if sig:
                ax.text(j + 0.5, i + 0.52, f"{r:+.2f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                        color="white" if abs(r) > 0.45 else INK, weight="bold" if qq < 0.05 else "normal")
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(cols)) + 0.5, [lab for *_, lab in cols], rotation=55, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(rows)) + 0.5, [lab for *_, lab in rows], fontsize=9)
    ax.tick_params(length=0)
    ax.xaxis.tick_top(); plt.setp(ax.get_xticklabels(), rotation=55, ha="left")
    for s in ax.spines.values():
        s.set_visible(False)

    # section separators and labels
    k = 0
    for s, items in ROWS:
        if k:
            ax.axhline(k, color=MUTED, lw=1.2)
        fig.text(0.012, 0.86 - 0.78 * (k + len(items) / 2) / len(rows), s, fontsize=9.5, weight="bold",
                 color=MUTED if "nie" in s else INK, va="center")
        k += len(items)
    if "fizjologia" in ROWS[-1][0]:
        ax.axhspan(len(rows) - len(ROWS[-1][1]), len(rows), color="#e9e8e4", alpha=0.45, zorder=-1)
    k = 0
    for s, items in COLS:
        if k:
            ax.axvline(k, color=MUTED, lw=1.2)
        ax.text(k + len(items) / 2, len(rows) + 0.6, s, ha="center", va="top", fontsize=9.5, weight="bold",
                color=INK)
        k += len(items)

    fig.text(0.012, 0.975, title,
             fontsize=15, weight="bold", color=INK)
    fig.text(0.012, 0.955, "Kolor = korelacja Spearmana (czerwony: im więcej cechy, tym wyższy parametr; niebieski: "
                           "tym niższy). Pełne pola z liczbą = wynik przetrwał korektę na "
                           f"{len(c)} testów (FDR q < 0,10; pogrubione q < 0,05).", fontsize=9, color=MUTED, wrap=True)
    note = "Blade pola = brak dowodu na związek."
    if "fizjologia" in ROWS[-1][0]:
        note += " Szare tło = parametry pomiaru, nie fizjologia (reagują na grupę losową)."
    fig.text(0.012, 0.94, note, fontsize=9, color=MUTED)
    cax = fig.add_axes([0.70, 0.035, 0.24, 0.011])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_label("ρ Spearmana", fontsize=8.5, color=MUTED); cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    fig.text(0.012, 0.02, "Wielkość ciała (płeć, wzrost, waga, obwód głowy) to w dużej mierze jeden czynnik: "
                          "mężczyźni są tu wyżsi i ciężsi.\nLeki na nadciśnienie = te same 4 osoby w wieku 68–78 l., "
                          "nie do oddzielenia od wieku.", fontsize=8.5, color=MUTED)
    fig.savefig(ROOT / out, dpi=130, facecolor=fig.get_facecolor())
    print(f"-> {ROOT / out}")


LAYER = {"skin": "skóra 10 mm", "l30": "30 mm", "brain": "mózg"}
BAND = {"vlf": "wolne", "mayer": "Mayer", "resp": "oddech"}


def hb_rows():
    rows = []
    for lay in ("skin", "l30", "brain"):
        items = []
        for k, lab in (("hbo", "HbO"), ("hbr", "HbR"), ("hbt", "HbT")):
            items += [(f"{lay}_{k}_{b}_uM", f"wahania {lab}, {BAND[b]}") for b in BAND]
        items += [(f"{lay}_hbr_over_hbo_{b}", f"HbR/HbO, {BAND[b]}") for b in BAND]
        items += [(f"{lay}_hbr_phase_{b}_deg", f"opóźnienie HbR, {BAND[b]}") for b in ("mayer", "resp")]
        items += [(f"{lay}_hbo_hbr_coh_{b}", f"spójność HbO–HbR, {BAND[b]}") for b in ("mayer", "resp")]
        if lay == "brain":
            items += [("brain_LR_coh_vlf", "spójność lewa–prawa, wolne"), ("brain_LR_coh_mayer", "spójność lewa–prawa, Mayer")]
        rows.append((LAYER[lay], items))
    return rows


if __name__ == "__main__":
    draw(ROWS, "correlations_pl.png", "Co sygnał fNIRS z czoła mówi o osobie? Parametry ogólne × 16 cech, 59 osób")
    draw(hb_rows(), "correlations_hb.png", "HbO, HbR i HbT w trzech warstwach × 16 cech, 59 osób (tylko ZMIANY: CW nie mierzy poziomu)")
