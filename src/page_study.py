"""page_study.py - many resting recordings at once: one table of markers, a group comparison with a plain-language
conclusion, and a CSV export. The use case: every fNIRS study records a rest; these markers come free with it and
show whether the compared groups also differ in their blood vessels (a confound for brain responses)."""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))
sys.path.insert(0, str(ROOT / "src"))
import fnirs_vascular as fv  # noqa: E402
import vascular as V  # noqa: E402
from features import load_meta  # noqa: E402

MARKERS = {"rytm naczyń [% sygnału]": "wolny rytm naczyń (pasmo śródbłonkowe)", "TI [1/s]": "indeks fali odbitej TI"}


def pl(v, f=".2f") -> str:
    """Number with a Polish decimal comma."""
    return format(v, f).replace(".", ",")


@st.cache_data(show_spinner=False, persist="disk")
def run_many(paths: tuple[str, ...]) -> list[dict]:
    return fv.analyze_many(list(paths))


def to_table(res: list[dict]) -> pd.DataFrame:
    rows = []
    for r in res:
        if "error" in r:
            rows.append({"plik": r["file"], "ostrzeżenia": "BŁĄD: " + r["error"]})
            continue
        v, t = r["vascular_regulation"], r["arterial_timing"]
        rows.append({"plik": r["file"], "tętno [bpm]": r["heart_rate_bpm"],
                     "rytm naczyń [% sygnału]": round(100 * v["endothelial_band_share"], 1),
                     "percentyl rytmu": v["percentile_vs_reference"],
                     "TI [1/s]": t["timing_index_per_s"], "percentyl TI": t["percentile_vs_reference"],
                     "długość [min]": r["recording"]["duration_min"],
                     "ostrzeżenia": " | ".join(r["warnings"])})
    return pd.DataFrame(rows)


def cliffs_delta(a, b):
    return float((np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b)))


def compare(df: pd.DataFrame, col: str, groups: list[str]):
    a = df.loc[df.grupa == groups[0], col].dropna().to_numpy(float)
    b = df.loc[df.grupa == groups[1], col].dropna().to_numpy(float)
    if len(a) < 3 or len(b) < 3:
        return None
    return dict(a=a, b=b, p=float(stats.mannwhitneyu(a, b).pvalue), med_a=float(np.median(a)),
                med_b=float(np.median(b)), delta=cliffs_delta(a, b))


def fig_groups(df, col, groups, c):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    rng = np.random.default_rng(0)
    for i, (g, x, color) in enumerate(((groups[0], c["a"], V.ORANGE), (groups[1], c["b"], V.GREY))):
        ax.boxplot(x, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color=V.INK, lw=2))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(x)), x, s=26, color=color, edgecolor="white", zorder=3)
    ax.set_xticks([0, 1], [f"{groups[0]}\n(n = {len(c['a'])})", f"{groups[1]}\n(n = {len(c['b'])})"])
    ax.set_ylabel(col)
    ax.set_title(f"{MARKERS[col]}\np = {pl(c['p'], '.4f')} (Mann–Whitney)", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- input
st.title("Badanie: wiele nagrań naraz")
st.markdown("Każde badanie fNIRS i tak nagrywa spoczynek. Ten widok liczy z niego **dwie miary naczyń dla każdego "
            "uczestnika** i sprawdza, czy porównywane grupy różnią się także naczyniowo. Taka różnica może "
            "udawać różnicę w aktywności mózgu, więc warto ją uwzględnić jako zmienną kontrolną.")

src = st.sidebar.radio("Nagrania", ["Zestaw danych hackathonu (60 osób, leżąc)", "Wgraj własne pliki .snirf"])
groups_df, order = None, None
if src.startswith("Zestaw"):
    paths = sorted(str(p) for p in V.BASE.glob("*.snirf"))
    if not paths:
        st.error("Brak danych. Uruchom ./download_data.sh albo wgraj pliki.")
        st.stop()
    split = st.sidebar.selectbox("Podział na grupy", ["Ankieta: aeroby > 3 h/tydz vs mniej",
                                                      "Wiek: od 30 lat vs poniżej 30", "Płeć"])
    meta = load_meta()
    meta["plik_id"] = meta.subject.astype(int)
    if split.startswith("Ankieta"):
        order = ["aeroby > 3 h/tydz", "mniej"]
        meta["grupa"] = np.where(meta.activity == meta.activity.max(), *order)
    elif split.startswith("Wiek"):
        order = ["od 30 lat", "poniżej 30"]
        meta["grupa"] = np.where(meta.age >= 30, *order)
    else:
        order = ["mężczyźni", "kobiety"]
        meta["grupa"] = np.where(meta.sex_male == 1, *order)
    groups_df = meta[["plik_id", "grupa"]]
else:
    ups = st.sidebar.file_uploader("Pliki .snirf (spoczynek, ok. 10 min)", type=["snirf"], accept_multiple_files=True)
    gcsv = st.sidebar.file_uploader("Opcjonalnie CSV z grupami: kolumny plik, grupa", type=["csv"])
    if not ups:
        st.info("Wgraj pliki .snirf w panelu po lewej. Opcjonalnie dodaj CSV z kolumnami `plik,grupa`.")
        st.stop()
    tmp = Path(tempfile.mkdtemp(prefix="fnirs_study_"))
    paths = []
    for u in ups:
        (tmp / u.name).write_bytes(u.getvalue())
        paths.append(str(tmp / u.name))
    if gcsv is not None:
        groups_df = pd.read_csv(gcsv)

with st.spinner(f"Liczę {len(paths)} nagrań…"):
    res = run_many(tuple(paths))
df = to_table(res)
ok = df["ostrzeżenia"].fillna("").eq("").sum() if "ostrzeżenia" in df else len(df)

c1, c2, c3 = st.columns(3)
c1.metric("Nagrania", len(df))
c2.metric("Bez ostrzeżeń", int(ok))
c3.metric("Czas obliczeń", "ok. 0,2 s / nagranie")

# ---------------------------------------------------------------- groups
if groups_df is not None:
    if "plik_id" in groups_df:
        df["grupa"] = df.plik.str.extract(r"^(\d+)_")[0].astype(float).map(
            groups_df.set_index("plik_id").grupa)
    else:
        df["grupa"] = df.plik.map(groups_df.set_index("plik").grupa)
    names = order or sorted(map(str, df.grupa.dropna().unique()))
    if len(names) == 2:
        st.subheader("Czy grupy różnią się naczyniowo?")
        cols = st.columns(2)
        verdicts = []
        for col_ui, col in zip(cols, MARKERS):
            c = compare(df, col, names)
            if c is None:
                col_ui.info(f"{MARKERS[col]}: za mało nagrań w grupach.")
                continue
            with col_ui:
                st.pyplot(fig_groups(df, col, names, c), clear_figure=True)
            name = MARKERS[col][0].upper() + MARKERS[col][1:]
            if c["p"] < 0.05:
                verdicts.append(f"**{name}: grupy się różnią** (mediana {pl(c['med_a'])} vs "
                                f"{pl(c['med_b'])}, p = {pl(c['p'], '.4f')}, delta Cliffa {pl(c['delta'], '+.2f')}). Porównując "
                                "między tymi grupami odpowiedź mózgu, dodaj tę miarę jako zmienną kontrolną.")
            else:
                verdicts.append(f"{name}: brak wyraźnej różnicy między grupami (p = {pl(c['p'])}).")
        st.markdown("#### Wniosek dla tego badania")
        for v in verdicts:
            st.markdown(f"- {v}")
        st.caption("Porównanie grup to właściwe użycie tych miar: są wiarygodne dla grup, nie dla pojedynczej osoby.")
    elif len(names):
        st.info(f"Porównanie działa dla dwóch grup, a w danych jest {len(names)}: {', '.join(map(str, names))}.")

# ---------------------------------------------------------------- table + export
st.subheader("Wszystkie nagrania")
st.dataframe(df, hide_index=True, width="stretch")
buf = io.StringIO()
df.to_csv(buf, index=False)
st.download_button("Pobierz tabelę (CSV)", buf.getvalue(), "miary_naczyniowe.csv", "text/csv")
st.caption("Percentyle tylko dla montażu zgodnego z grupą odniesienia (zdrowi 18–42 l., n = 52). Nagrania z tego "
           "zestawu należą do grupy odniesienia, więc ich percentyl liczony jest na tle grupy, która je zawiera.")
with st.expander("Ten sam wynik bez aplikacji (moduł fnirs-vascular)"):
    st.code("pip install ./plugin\nfnirs-vascular folder_badania/ --out miary.csv", language="bash")
