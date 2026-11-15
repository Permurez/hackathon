"""
app.py - interactive demo: one 10-minute resting fNIRS recording -> two axes of vascular health.

  ~/.venvs/fnirs/bin/streamlit run src/app.py

Pick a recording from the dataset or upload a .snirf file; everything is computed live by vascular.py.
Group-level markers, not a diagnosis.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vascular as V  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Puls pod czepkiem", layout="wide")


@st.cache_data(show_spinner=False)
def run(path: str) -> dict:
    return V.analyse(path)


@st.cache_data(show_spinner=False)
def load_cohort():
    return V.cohort()


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


coh = load_cohort()
files = sorted(V.BASE.glob("*.snirf"), key=lambda p: V.subject_of(p) or 0)


def label(p: Path) -> str:
    s = V.subject_of(p)
    r = coh[coh.subject == s]
    if len(r) and np.isfinite(r.age.iloc[0]):
        return f"Osoba {s} · {int(r.age.iloc[0])} l. · ankieta: {V.ACT.get(r.activity.iloc[0], '?')}"
    return p.name


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Nagranie")
mode = st.sidebar.radio("Źródło", ["Z zestawu danych", "Wgraj plik .snirf"])
if mode == "Z zestawu danych":
    if not files:
        st.error("Brak danych. Uruchom ./download_data.sh albo wgraj plik .snirf.")
        st.stop()
    start = next((i for i, p in enumerate(files) if V.subject_of(p) == 17), 0)
    path = str(st.sidebar.selectbox("Osoba", files, index=start, format_func=label))
else:
    up = st.sidebar.file_uploader("Plik .snirf: 10 min spoczynku, Cortivision C23", type=["snirf"])
    if up is None:
        st.info("Wgraj nagranie .snirf w panelu po lewej.")
        st.stop()
    tmp = Path(tempfile.gettempdir()) / up.name
    tmp.write_bytes(up.getvalue())
    path = str(tmp)
st.sidebar.caption("Odniesienie: zdrowi 18–42 l. bez leków (n = 52), zbiór Cortivision na leżąco. "
                   "Osoba analizowana jest wyłączona z grupy odniesienia.")

# ---------------------------------------------------------------- header + analysis
st.title("Puls pod czepkiem")
st.markdown("**Dwie osie zdrowia naczyń z 10 minut spoczynkowego fNIRS:** jak aktywnie naczynia się regulują "
            "(silniejsze u trenujących) i jak sztywne są tętnice (rośnie z wiekiem).")

with st.status("Analiza nagrania…", expanded=False) as status:
    status.write("1. Surowe światło → gęstość optyczna → hemoglobina HbO (własna konwersja, zgodna z MNE)")
    res = run(path)
    status.write("2. Widmo sygnału z czoła → moc wolnego rytmu naczyń 0,0095–0,021 Hz")
    status.write(f"3. {res['n_beats']} uderzeń serca uśrednionych → fala z serca i fala odbita → indeks TI")
    status.write("4. Porównanie ze zdrową grupą i model aktywności")
    status.update(label=f"Gotowe: {res['file']} ({res['duration_min']:.1f} min)", state="complete")
sc = V.score(res, coh)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regulacja naczyń", f"{sc['pct_endo']:.0f}. percentyl",
          help="Moc wolnego rytmu naczyń (pasmo przypisywane śródbłonkowi) na tle zdrowej grupy.")
c2.metric("Sztywność tętnic (TI)", f"{sc['pct_ti']:.0f}. percentyl" if np.isfinite(sc["pct_ti"]) else "–",
          help="Wyżej = fala odbita wraca szybciej = sztywniejsze tętnice niż u rówieśników.")
c3.metric("Model: aeroby > 3 h/tydz", f"{100 * sc['prob_active']:.0f}%",
          help="Regresja logistyczna na paśmie śródbłonkowym; AUC ≈ 0,70 na osobach spoza uczenia.")
c4.metric("Tętno spoczynkowe", f"{res['hr']:.0f} bpm")

row = coh[coh.subject == res["subject"]]
if len(row) and np.isfinite(row.activity.iloc[0]):
    st.caption(f"Dla porównania, ankieta tej osoby: {V.ACT.get(row.activity.iloc[0], '?')}, "
               f"wiek {int(row.age.iloc[0])} l. (model tego nie widzi).")

tab1, tab2, tab3 = st.tabs(["Sygnał i fala tętna", "Dwie osie na tle grupy", "Dowody"])
with tab1:
    a, b = st.columns(2)
    with a:
        show(V.fig_signal(res))
    with b:
        show(V.fig_spectrum(res))
    a, b = st.columns([1.4, 1])
    with a:
        show(V.fig_pulse(res))
    with b:
        st.subheader("Co widać w tym nagraniu")
        for line in V.interpret(res, sc):
            st.markdown(f"- {line}")
with tab2:
    a, b = st.columns([1.3, 1])
    with a:
        show(V.fig_two_axes(coh, res))
    with b:
        st.subheader("Dwie niezależne informacje")
        st.markdown(
            "- **Pionowo, regulacja:** wolny rytm naczyń, ok. raz na minutę. U osób trenujących ponad 3 h "
            "tygodniowo jest średnio **1,49× silniejszy**; u sportowców w laserze dopplerowskim skóry 1,5× "
            "(Kvernmo i in. 2003).\n"
            "- **Poziomo, struktura:** indeks fali odbitej. W sztywniejszych tętnicach fala odbita wraca szybciej. "
            "Rośnie z wiekiem już u osób w wieku 18–42 lat (r = +0,36), powtarzalność 0,95.\n"
            "- **Osie są niezależne** (korelacja ok. 0): trening i wiek zostawiają w sygnale dwa różne ślady.")
with tab3:
    st.markdown("""
| | Regulacja (pasmo śródbłonkowe) | Struktura (indeks fali odbitej TI) |
|---|---|---|
| Związek | z aktywnością: ρ = +0,46, p = 0,0004 (18–42 l., n = 55) | z wiekiem: r = +0,36, p = 0,009 (bez leków, n = 52) |
| Wielkość | 1,49× u trenujących > 3 h/tydz (literatura: 1,5×) | — |
| Odporność | 17 testów, 270 wariantów analizy, 5000 losowych hipotez | ten sam kierunek w 8/8 kanałach |
| Powtarzalność | 0,36 (jeden 10-min pomiar) | 0,95 |
| Literatura | Stefanovska 1999, Kvernmo 2003 | Afkhami i in. 2021 |
""")
    a, b = st.columns(2)
    for col, name in ((a, "fig_endothelial_activity.png"), (b, "fig_ti_age.png")):
        if (ROOT / name).exists():
            col.image(str(ROOT / name))

st.warning(V.DISCLAIMER + " Pokazuje trend w grupie, nie liczbę godzin treningu ani sztywność tętnic konkretnej osoby.")
