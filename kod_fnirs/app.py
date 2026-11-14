"""
app.py - Streamlit demo of the fNIRS circulation / autonomic screening.

  python screening.py          # once: computes the cohort cache (screening_cache.pkl)
  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screening as sc  # noqa: E402

INK, MUTED, ACCENT, SKIN, BAD = "#1f2933", "#7b8794", "#2563eb", "#d97706", "#b91c1c"
AGE_GROUPS = [(18, 24, "18–24 l."), (25, 59, "25–59 l."), (60, 120, "60+ l.")]  # 35-59 has only 2 people
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

st.set_page_config(page_title="fNIRS: przesiew krążenia", layout="wide")


@st.cache_resource
def cohort():
    if not sc.CACHE.exists():
        st.error("Brak cache. Uruchom najpierw: python screening.py")
        st.stop()
    return sc.build()


def age_group(age):
    for lo, hi, name in AGE_GROUPS:
        if lo <= age <= hi:
            return lo, hi, name
    return 0, 200, "wszyscy"


def fnum(x, d=0, unit=""):
    return f"{x:.{d}f}{unit}".replace(".", ",") if np.isfinite(x) else "–"


df, fits = cohort()
labels = {int(r.subject): f"#{int(r.subject)}  ·  {int(r.age)} l.  ·  {r.sex}" for r in df.itertuples()}
st.sidebar.title("Przesiew krążenia z fNIRS")
sid = st.sidebar.selectbox("Osoba", list(labels), format_func=labels.get)
st.sidebar.caption(f"Grupa odniesienia: {len(df)} osób, spoczynek na leżąco 10 min, Cortivision C23 141,79 Hz.")
r = df[df.subject == sid].iloc[0]

st.title(f"Osoba #{sid} – {int(r.age)} l., {r.sex}")
st.caption(f"plik {r.file}  ·  kanały 30 mm: {r.long_used}  ·  kanały 10 mm: {r.short_used or 'brak'}  ·  "
           f"czystych uderzeń: {int(r.n_clean_beats)}  ·  aktywność {fnum(r.activity, 2)}  ·  BMI {fnum(r.bmi, 1)}"
           f"{'  ·  leki na nadciśnienie' if r.antihypertensive else ''}")

# ---------------------------------------------------------------- 1. three pillars
st.subheader("1. Trzy filary (percentyl w grupie, 100 = najkorzystniej; kreska = 95% CI)")
el = f"{sc.ELAST_LAYER}_fixed_relax_area_75bpm"
pillars = [
    ("Elastyczność naczyń", "elast", f"pole relaksacji (mózg, @75 bpm) {fnum(r[el], 3)} "
     f"[{fnum(r[el + '_lo'], 3)}–{fnum(r[el + '_hi'], 3)}]  ·  Crest Time {fnum(r[f'{sc.ELAST_LAYER}_crest_time_ms_75bpm'])} ms"
     f"  ·  garb dykrotyczny: {'tak' if r[f'{sc.ELAST_LAYER}_hump_present'] else 'nie'}"),
    ("Balans autonomiczny", "auton", f"tętno {fnum(r.hr_bpm)} bpm [{fnum(r.hr_bpm_lo)}–{fnum(r.hr_bpm_hi)}]  ·  "
     f"RMSSD {fnum(r.rmssd_ms)} ms [{fnum(r.rmssd_ms_lo)}–{fnum(r.rmssd_ms_hi)}] (przed czyszczeniem "
     f"{fnum(r.rmssd_raw_ms)})  ·  SD1/SD2 {fnum(r.sd1_sd2, 2)}"),
    ("Ukrwienie skóry i udział skóry", "perf", f"amplituda pulsu skóry {fnum(r.Scalp_10mm_amp_uM, 2)} µM  ·  "
     f"30 mm {fnum(r.Raw_30mm_amp_uM, 2)} µM  ·  skóra = {fnum(r.skin_share_pct)}% sygnału 30 mm"),
]
fig, axes = plt.subplots(3, 1, figsize=(11, 3.4))
for ax, (name, k, txt) in zip(axes, pillars):
    ax.imshow(np.linspace(0, 1, 256)[None], extent=(0, 100, 0, 1), aspect="auto", cmap="RdYlGn", alpha=0.6)
    ax.axvline(sc.LOW_PCT, color=INK, lw=0.7, ls=":")
    p, lo, hi = r[f"{k}_pct"], r.get(f"{k}_pct_lo", np.nan), r.get(f"{k}_pct_hi", np.nan)
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([min(lo, hi), max(lo, hi)], [0.5, 0.5], color=INK, lw=4, solid_capstyle="butt")
    if np.isfinite(p):
        ax.plot([p], [0.5], "o", ms=11, color="white", mec=INK, mew=2)
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title(f"{name}: {fnum(p)}. percentyl", loc="left", fontsize=10, weight="bold")
    ax.text(0, -0.85, txt, fontsize=7.5, color=MUTED, transform=ax.transData)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout(h_pad=2.2)
st.pyplot(fig)

# ---------------------------------------------------------------- 2. pulse curves vs age norm
lo_a, hi_a, gname = age_group(r.age)
peers = df[(df.age.between(lo_a, hi_a)) & (df.subject != sid)]
st.subheader(f"2. Kształt uderzenia: skóra 10 mm vs mózg (30 mm − skóra), tło = norma {gname} (n={len(peers)})")
ph = np.linspace(0, 1, sc.N_POINTS)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, lay, col in [(axes[0], "Scalp_10mm", SKIN), (axes[1], "Brain_Isolated", ACCENT)]:
    W = np.array([w for w in peers[f"_wave_{lay}"] if isinstance(w, np.ndarray)])
    if len(W):
        m, s = W.mean(0), W.std(0)
        ax.fill_between(ph, m - s, m + s, color=MUTED, alpha=0.2, label=f"norma {gname} (±1 SD)")
        ax.plot(ph, m, color=MUTED, lw=1)
    w = r[f"_wave_{lay}"]
    if isinstance(w, np.ndarray):
        ax.plot(ph, w, color=col, lw=2.2, label=f"osoba #{sid}")
        p1 = r[f"{lay}_p1_phase"]
        ax.axvline(p1, color=col, lw=0.8, ls="--")
        ax.text(p1, 1.02, "P1", color=col, fontsize=8, ha="center")
    for a, b, c in [(*sc.st.RELAX_WIN, "#e5e7eb"), (*sc.st.HUMP_WIN, "#fde68a")]:
        ax.axvspan(a, b, color=c, alpha=0.25, lw=0)
    ax.set_title(sc.LAYER_PL[lay]); ax.set_xlabel("faza uderzenia (0 = stopa, 1 = następna stopa)")
    ax.legend(fontsize=7, loc="upper right")
axes[0].set_ylabel("puls HbO (znormalizowany)")
fig.tight_layout()
st.pyplot(fig)
st.caption("Szare tło na fazie 0,25–1,00 = okno pola relaksacji; żółte 0,35–0,55 = okno garbu dykrotycznego. "
           "Bardziej wybrzuszone opadanie = elastyczniejsze naczynia.")

# ---------------------------------------------------------------- 3. Poincare + 4. text
c1, c2 = st.columns([1, 1.2])
with c1:
    st.subheader("3. Wykres Poincaré (po filtrze ±20%)")
    ibi, ok = 1000 * r["_ibi"], r["_ibi_ok"]
    pair = ok[1:] & ok[:-1]
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    bad = ~pair
    ax.plot(ibi[:-1][bad], ibi[1:][bad], "x", ms=3, color=BAD, alpha=0.5, label=f"odrzucone ({bad.sum()})")
    ax.plot(ibi[:-1][pair], ibi[1:][pair], ".", ms=3, color=INK, alpha=0.5, label=f"czyste ({pair.sum()})")
    g = ibi[ok]
    lim = np.percentile(g, [0.5, 99.5]) + np.array([-100, 100])
    ax.plot(lim, lim, color=MUTED, lw=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("odstęp n [ms]"); ax.set_ylabel("odstęp n+1 [ms]")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"SD1 {fnum(r.sd1_ms)} ms · SD2 {fnum(r.sd2_ms)} ms", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    st.subheader("4. Opis przesiewowy")
    for t in sc.screening_text(r.to_dict()):
        st.markdown(f"- {t}")
    st.warning(sc.DISCLAIMER)
    st.markdown("**Zmiana w trakcie spoczynku (pierwsze 3 min → ostatnie 3 min)**")
    rows = [("tętno [bpm]", "first3_hr_bpm", "last3_hr_bpm", 1),
            ("RMSSD [ms]", "first3_rmssd_ms", "last3_rmssd_ms", 0),
            ("pole relaksacji @75 (mózg)", "first3_Brain_Isolated_fixed_relax_area_75bpm",
             "last3_Brain_Isolated_fixed_relax_area_75bpm", 3)]
    st.table({"wskaźnik": [a for a, *_ in rows],
              "pierwsze 3 min": [fnum(r[b], d) for _, b, _, d in rows],
              "ostatnie 3 min": [fnum(r[c], d) for _, _, c, d in rows],
              "zmiana": [fnum(r[c] - r[b], d) for _, b, c, d in rows]})

with st.expander("Wszystkie liczby tej osoby"):
    st.dataframe(r[[c for c in df.columns if not c.startswith("_")]].astype(str), width="stretch")
