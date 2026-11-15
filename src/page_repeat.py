"""page_repeat.py - the same person recorded several times: how stable is the slow vasomotion band, how many
recordings must be averaged, and the smallest change that is more than noise (the numbers a training study needs).
Demo data: DOT dataset, 17 people x 4 ten-minute rests over one day (sitting, other montage -> raw values)."""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))
sys.path.insert(0, str(ROOT / "src"))
import fnirs_vascular as fv  # noqa: E402
import vascular as V  # noqa: E402

DOT = ROOT / "dane/infopack_and_data/dot_montage_sitting/data/RS"
ICC_ALL_LONG = 0.36  # src/endo_followup.py: every long channel, no heartbeat criterion
ICC_MODULE = 0.56    # this module on the same DOT rests (long channels with a clear heartbeat)


@st.cache_data(show_spinner=False, persist="disk")
def run_many(paths: tuple[str, ...]) -> list[dict]:
    return fv.analyze_many(list(paths))


def pl(v, f=".2f") -> str:
    """Number with a Polish decimal comma."""
    return format(v, f).replace(".", ",")


def icc1(w: np.ndarray):
    """ICC(1,1) and within-person SD for a people x sessions array."""
    n, k = w.shape
    gm = w.mean()
    msb = k * ((w.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((w - w.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw)), float(np.sqrt(msw))


def spearman_brown(icc, k):
    return k * icc / (1 + (k - 1) * icc)


def k_needed(icc, target):
    return int(np.ceil(target * (1 - icc) / (icc * (1 - target))))


st.title("Pomiar powtarzany: ile nagrań potrzeba?")
st.markdown("Żeby śledzić **jedną osobę**, np. przed i po treningu, miara musi być powtarzalna. Ten widok liczy, "
            "jak bardzo wolny rytm naczyń zmienia się między nagraniami tej samej osoby, ile nagrań trzeba "
            "uśrednić i jaka zmiana jest już czymś więcej niż szumem.")

src = st.sidebar.radio("Nagrania", ["Zestaw DOT (17 osób × 4 spoczynki w ciągu dnia)",
                                    "Wgraj kilka nagrań jednej osoby"])

if src.startswith("Zestaw"):
    paths = sorted(str(p) for p in DOT.glob("*_RS*.snirf"))
    if not paths:
        st.error("Brak danych DOT. Uruchom ./download_data.sh.")
        st.stop()
    with st.spinner(f"Liczę {len(paths)} nagrań…"):
        res = run_many(tuple(paths))
    rows = []
    for r in res:
        m = re.match(r"(\d+)_[A-Z]+_RS(\d)", r["file"])
        if m and "error" not in r:
            rows.append(dict(osoba=int(m[1]), sesja=int(m[2]), rytm=100 * r["vascular_regulation"]["endothelial_band_share"]))
    d = pd.DataFrame(rows)
    w = d.pivot(index="osoba", columns="sesja", values="rytm").dropna()
    icc, sw = icc1(w.to_numpy())
    k75, k80 = k_needed(icc, 0.75), k_needed(icc, 0.80)
    mdc1 = 1.96 * np.sqrt(2) * sw

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Powtarzalność 1 nagrania (ICC)", pl(icc),
              help=f"ICC(1,1), {len(w)} osób × {w.shape[1]} spoczynki. Analiza na wszystkich długich kanałach "
                   f"(src/endo_followup.py) daje {pl(ICC_ALL_LONG)}.")
    c2.metric("Średnia z 4 nagrań", pl(spearman_brown(icc, 4)))
    c3.metric("Nagrań do ICC 0,75", f"{k75}")
    c4.metric("Najmniejsza wykrywalna zmiana", f"{pl(mdc1, '.1f')} pp",
              help="MDC95 dla porównania dwóch pojedynczych nagrań, w punktach procentowych udziału pasma.")

    who = st.sidebar.selectbox("Osoba", list(w.index), format_func=lambda s: f"Osoba {s}")
    a, b = st.columns([1.4, 1])
    with a:
        order = w.mean(axis=1).sort_values().index
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        for i, s in enumerate(order):
            hi = s == who
            ax.plot([i] * w.shape[1], w.loc[s], "o", ms=7 if hi else 5, color=V.ORANGE if hi else V.GREY,
                    alpha=1 if hi else 0.8, zorder=3 if hi else 2)
            ax.plot([i - 0.3, i + 0.3], [w.loc[s].mean()] * 2, color=V.INK if hi else V.MUTED, lw=2 if hi else 1)
        ax.set_xticks(range(len(order)), [str(s) for s in order], fontsize=8)
        ax.set_xlabel("osoba (posortowane po średniej)")
        ax.set_ylabel("wolny rytm naczyń [% sygnału]")
        ax.set_title("4 spoczynki tej samej osoby w ciągu dnia (kropki) i ich średnia (kreska)", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with b:
        ks = np.arange(1, 9)
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        ax.plot(ks, spearman_brown(icc, ks), "o-", color=V.BLUE, label=f"ten moduł (ICC {pl(icc)})")
        ax.plot(ks, spearman_brown(ICC_ALL_LONG, ks), "o--", color=V.GREY, label=f"wszystkie kanały ({pl(ICC_ALL_LONG)})")
        ax.axhline(0.75, color=V.MUTED, lw=0.8, ls=":")
        ax.text(8, 0.73, "dobra powtarzalność (0,75)", ha="right", va="top", fontsize=8, color=V.MUTED)
        ax.set_xlabel("liczba uśrednionych nagrań")
        ax.set_ylabel("powtarzalność")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Ile nagrań uśredniać", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    x = w.loc[who]
    st.markdown(f"**Osoba {who}:** " + "; ".join(pl(v, ".1f") for v in x) +
                f" % → średnia {pl(x.mean(), '.1f')} %, rozrzut ± {pl(x.std(), '.1f')} pp w ciągu jednego dnia.")
    st.markdown("#### Wniosek")
    st.markdown(
        f"- Jedno 10-minutowe nagranie ma powtarzalność **{pl(icc)}**: to za mało, żeby oceniać jedną osobę "
        f"na podstawie jednego pomiaru.\n"
        f"- Średnia z **{k75} nagrań** daje powtarzalność 0,75, a z {k80} nagrań 0,80.\n"
        f"- Różnica między dwoma pojedynczymi nagraniami mniejsza niż **{pl(mdc1, '.1f')} pp** to szum. Przy średnich "
        f"z {k75} nagrań przed i po treningu próg spada do **{pl(mdc1 / np.sqrt(k75), '.1f')} pp**. Dla porównania: "
        "trenujący i nietrenujący różnią się w medianie o ok. 12 pp.\n"
        "- To są liczby potrzebne do zaplanowania badania treningowego: ile nagrań na osobę i jakiej zmiany szukać.")
    st.caption("Dane DOT: siedząco, inny montaż niż grupa odniesienia, więc tylko surowe wartości (bez percentyli). "
               "Pasmo liczone na długich kanałach z wyraźnym tętnem; próbkowanie 18 Hz nie pozwala liczyć TI.")
else:
    ups = st.sidebar.file_uploader("Nagrania .snirf jednej osoby", type=["snirf"], accept_multiple_files=True)
    if not ups or len(ups) < 2:
        st.info("Wgraj co najmniej 2 nagrania spoczynkowe tej samej osoby.")
        st.stop()
    tmp = Path(tempfile.mkdtemp(prefix="fnirs_repeat_"))
    paths = []
    for u in ups:
        (tmp / u.name).write_bytes(u.getvalue())
        paths.append(str(tmp / u.name))
    res = [r for r in run_many(tuple(paths)) if "error" not in r]
    x = np.array([100 * r["vascular_regulation"]["endothelial_band_share"] for r in res])
    st.dataframe(pd.DataFrame({"plik": [r["file"] for r in res], "rytm naczyń [% sygnału]": x.round(1),
                               "ostrzeżenia": [" | ".join(r["warnings"]) for r in res]}), hide_index=True)
    st.markdown(f"Średnia z {len(x)} nagrań: **{pl(x.mean(), '.1f')} %** (rozrzut ± {pl(x.std(ddof=1), '.1f')} pp). "
                f"Powtarzalność takiej średniej (przy ICC jednego nagrania {pl(ICC_ALL_LONG)}–{pl(ICC_MODULE)}): "
                f"{pl(spearman_brown(ICC_ALL_LONG, len(x)))}–{pl(spearman_brown(ICC_MODULE, len(x)))}.")
