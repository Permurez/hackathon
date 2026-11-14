"""
cardio_report.py - PDF builders for cardio_profile.py (matplotlib only, no extra packages).

  methods_report(cohort, ref, val, rel, data_dir, path)  step-by-step methods, results, conclusions
  person_card(row, cohort, ref, path)                    one-page card: scales vs the cohort
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import cardio_profile as cp  # noqa: E402
import fnirs_toolkit as fk  # noqa: E402
import stiffness as st  # noqa: E402

A4 = (8.27, 11.69)
INK, MUTED, ACCENT = "#1f2933", "#616e7c", "#2563eb"
GOOD, WARN, BAD = "#15803d", "#b45309", "#b91c1c"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": "#9aa5b1",
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


# ---------------------------------------------------------------------------
# Text layout
# ---------------------------------------------------------------------------
class TextFlow:
    """Writes headings / paragraphs / bullets top-down on A4 pages, opening a new page when full."""

    def __init__(self, pdf, title_prefix=""):
        self.pdf, self.prefix, self.fig, self.y = pdf, title_prefix, None, 0

    def _new(self):
        self.flush()
        self.fig = plt.figure(figsize=A4)
        self.y = 0.95
        if self.prefix:
            self.fig.text(0.08, 0.975, self.prefix, fontsize=7, color=MUTED)

    def flush(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None

    def _need(self, h):
        if self.fig is None or self.y - h < 0.05:
            self._new()

    def h1(self, t):
        self._need(0.08)
        self.fig.text(0.08, self.y, t, fontsize=16, weight="bold", color=INK, va="top")
        self.y -= 0.045

    def h2(self, t):
        self._need(0.06)
        self.y -= 0.008
        self.fig.text(0.08, self.y, t, fontsize=12, weight="bold", color=ACCENT, va="top")
        self.y -= 0.03

    def p(self, t, width=100, size=9.2, color=INK, indent=0.08, bullet=None):
        lines = textwrap.wrap(t, width)
        h = 0.0165 * size / 9.2
        for i, line in enumerate(lines):
            self._need(h)
            if bullet and i == 0:
                self.fig.text(indent - 0.02, self.y, bullet, fontsize=size, color=ACCENT, va="top")
            self.fig.text(indent, self.y, line, fontsize=size, color=color, va="top")
            self.y -= h
        self.y -= 0.006

    def b(self, t):
        self.p(t, width=95, indent=0.105, bullet="•")

    def n(self, i, t):
        self.p(t, width=94, indent=0.11, bullet=f"{i}.")

    def table(self, rows, col_x, header=True, size=8.2):
        h = 0.0155
        for j, r in enumerate(rows):
            self._need(h)
            for x, cell in zip(col_x, r):
                self.fig.text(x, self.y, str(cell), fontsize=size, va="top",
                              weight="bold" if header and j == 0 else "normal", color=INK)
            self.y -= h
        self.y -= 0.008


def fig_page(pdf, fig, caption=None):
    if caption:
        for i, line in enumerate(textwrap.wrap(caption, 120)):
            fig.text(0.07, 0.035 - i * 0.014, line, fontsize=8, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def f2(x, n=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def fp(p):
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def fr(r):
    return f"{r:+.2f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_pipeline(path, pdf):
    """Steps 1-8 on one real recording."""
    d = fk.load_snirf_haemo(str(path))
    fs = d["fs"]
    names = [c.split()[0] for c in d["ch_hbo"]]
    n = cp.FOREHEAD[0]
    x = d["hbo"][names.index(n)] * 1e6
    xc = cp.cardiac(x, fs)
    period = 60 / fk.heart_rate(x, fs)
    beats = cp.find_beats(xc, fs, period)
    w, dur, k, amp = cp.mean_beat(xc, fs, beats)
    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots(4, 1, figsize=A4, gridspec_kw={"hspace": 0.55})
    fig.suptitle(f"Krok po kroku na jednym nagraniu (osoba #{st.subject_id(Path(path))}, kanał {n})",
                 weight="bold", y=0.97)
    ax[0].plot(t, x, lw=0.4, color=INK)
    ax[0].set_title("① HbO po konwersji (OD → TDDR → Beer-Lambert), całe 10 min: dominują wolne wahania")
    ax[0].set_xlabel("czas [s]"); ax[0].set_ylabel("HbO [µM]")
    s0, s1 = int(120 * fs), int(126 * fs)
    ax[1].plot(t[s0:s1], x[s0:s1] - x[s0:s1].mean(), lw=0.8, color=MUTED, label="surowe HbO")
    ax[1].plot(t[s0:s1], xc[s0:s1], lw=1.2, color=ACCENT, label="po filtrze 0,5–10 Hz")
    tr = [a for a, _ in beats if s0 <= a < s1]
    ax[1].plot(t[tr], xc[tr], "o", color=BAD, ms=4, label="dołki = początki uderzeń")
    ax[1].set_title("② 6 s sygnału: filtr usuwa oddech i dryf, zostaje tętno; ③ dołki wyznaczają uderzenia")
    ax[1].set_xlabel("czas [s]"); ax[1].legend(fontsize=7, loc="upper right")
    raw = np.array([cp.resample(xc, a, b) for a, b in beats[:400]])
    norm = (raw - raw.min(1, keepdims=True)) / np.ptp(raw, 1, keepdims=True)
    ph = np.linspace(0, 1, cp.N_POINTS)
    ax[2].plot(ph, norm[::4].T, lw=0.3, color=ACCENT, alpha=0.25)
    ax[2].plot(ph, w, lw=2.2, color=INK, label=f"średnia z {k} uderzeń (r>0,9 z medianą)")
    ax[2].set_title("④ każde uderzenie → 200 punktów i skala 0–1; ⑤ odrzucenie zniekształconych; ⑥ średnia")
    ax[2].set_xlabel("faza uderzenia (0 = początek, 1 = następne uderzenie)"); ax[2].legend(fontsize=7)
    kpk = int(np.argmax(w))
    line = np.linspace(w[kpk], w[-1], len(w) - kpk)
    ax[3].plot(ph, w, lw=2, color=INK)
    ax[3].plot(ph[kpk:], line, "--", color=MUTED, lw=1)
    ax[3].fill_between(ph[kpk:], line, w[kpk:], color=ACCENT, alpha=0.3, label="relax_area (pole nad prostą)")
    ax[3].axvline(ph[kpk], color=GOOD, lw=1, label="szczyt skurczowy")
    ax[3].set_title("⑦ relax_area: jak mocno opadanie 'wybrzusza się' ponad prostą szczyt → koniec uderzenia")
    ax[3].set_xlabel("faza uderzenia"); ax[3].legend(fontsize=7)
    fig_page(pdf, fig, "Rysunek 1. Ta sama procedura działa na każdym z 4 kanałów czołowych; ich średnie "
                       "kształty są następnie uśredniane w jeden puls na osobę.")


def ref_wave(cohort, col="_wave"):
    young = cohort["age"].between(*cp.REF_AGE) & cohort["meds"].fillna("-").str.strip().eq("-")
    W = np.array([w for w in cohort.loc[young, col] if isinstance(w, np.ndarray)])
    return W.mean(0), np.percentile(W, 25, axis=0), np.percentile(W, 75, axis=0)


def fig_module1(cohort, ref, pdf):
    fig, ax = plt.subplots(2, 2, figsize=A4, gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    fig.suptitle("Moduł 1 – elastyczność naczyń", weight="bold", y=0.97)
    ph = np.linspace(0, 1, cp.N_POINTS)
    for col, a, title in (("_wave", ax[0, 0], "puls 30 mm (skóra + głębiej)"),
                          ("_deep_wave", ax[0, 1], "puls głęboki (30 mm − skóra)")):
        m, lo, hi = ref_wave(cohort, col)
        a.fill_between(ph, lo, hi, color=ACCENT, alpha=0.15)
        a.plot(ph, m, color=ACCENT, lw=2, label=f"referencja 18–30 l. (n={ref['n_ref']}), IQR")
        old = cohort[cohort["age"] >= 60]
        for _, r in old.iterrows():
            a.plot(ph, r[col], color=BAD, lw=1, alpha=0.8)
        a.plot([], [], color=BAD, label="osoby 68–78 l.")
        a.set_title(title); a.set_xlabel("faza uderzenia"); a.legend(fontsize=7)
    d = cohort.dropna(subset=["elast75", "age"])
    sc = ax[1, 0].scatter(d["age"], d["elast75"], c=d["activity"], cmap="viridis", s=22)
    r, p = spearmanr(d["age"], d["elast75"])
    y = d[d["age"] < 60]
    ry, py = spearmanr(y["age"], y["elast75"])
    ax[1, 0].set_title(f"elastyczność vs wiek\nwszyscy rho={fr(r)} (p={fp(p)}), <60 l. rho={fr(ry)} (p={fp(py)})",
                       fontsize=9)
    ax[1, 0].set_xlabel("wiek [lata]"); ax[1, 0].set_ylabel("głęboki relax_area @75 bpm")
    fig.colorbar(sc, ax=ax[1, 0], label="aktywność fizyczna")
    v = d["elast75"]
    ax[1, 1].hist(v, bins=18, color=ACCENT, alpha=0.6)
    for q, lab in ((25, "P25 – próg 'obniżona'"), (50, "mediana")):
        ax[1, 1].axvline(np.percentile(v, q), color=BAD if q == 25 else INK, ls="--", lw=1)
        ax[1, 1].text(np.percentile(v, q), ax[1, 1].get_ylim()[1] * 0.92, " " + lab, fontsize=7)
    ax[1, 1].set_title("rozkład w grupie = skala populacyjna"); ax[1, 1].set_xlabel("głęboki relax_area @75 bpm")
    fig_page(pdf, fig, "Rysunek 2. Góra: średni kształt pulsu młodej grupy referencyjnej (niebieski, z przedziałem "
                       "międzykwartylowym) i 4 osób najstarszych (czerwone). Dół: elastyczność maleje z wiekiem "
                       "także bez osób po 60. roku życia; histogram to skala, na której umieszczamy każdą osobę.")


def fig_module2(cohort, pdf):
    fig, ax = plt.subplots(2, 2, figsize=A4, gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    fig.suptitle("Moduł 2 – balans autonomiczny", weight="bold", y=0.97)
    young = cohort[cohort["age"] < 30].sort_values("rmssd_ms").iloc[len(cohort[cohort["age"] < 30]) // 2]
    irr = cohort.sort_values("sd1_sd2").iloc[-1]
    for r, c in ((young, ACCENT), (irr, BAD)):
        ibi = r["_ibi"]
        tb = r["_tb"][1:]
        ax[0, 0].plot(tb[:300], 1000 * ibi[:300], lw=0.8, color=c, label=f"#{int(r['subject'])}, {int(r['age'])} l.")
        ax[0, 1].plot(1000 * ibi[:-1], 1000 * ibi[1:], ".", ms=2, color=c, alpha=0.5,
                      label=f"#{int(r['subject'])}: SD1/SD2={r['sd1_sd2']:.2f}".replace(".", ","))
    ax[0, 0].set_title("odstępy między uderzeniami (pierwsze 300)"); ax[0, 0].set_xlabel("czas [s]")
    ax[0, 0].set_ylabel("odstęp [ms]"); ax[0, 0].legend(fontsize=7)
    ax[0, 1].set_title("wykres Poincaré: odstęp n vs n+1"); ax[0, 1].set_xlabel("odstęp n [ms]")
    ax[0, 1].set_ylabel("odstęp n+1 [ms]"); ax[0, 1].legend(fontsize=7)
    d = cohort.dropna(subset=["rmssd_ms"])
    irr_m = d["irregular"]
    ax[1, 0].scatter(d.loc[~irr_m, "age"], d.loc[~irr_m, "rmssd_ms"], s=22, color=ACCENT, label="rytm regularny")
    ax[1, 0].scatter(d.loc[irr_m, "age"], d.loc[irr_m, "rmssd_ms"], s=40, marker="x", color=BAD,
                     label=f"SD1/SD2 > {cp.IRREGULAR_SD1_SD2}".replace(".", ","))
    y = d[d["age"] < 60]
    r, p = spearmanr(y["age"], y["rmssd_ms"])
    ax[1, 0].set_title(f"RMSSD vs wiek (<60 l.: rho={fr(r)}, p={fp(p)})", fontsize=9)
    ax[1, 0].set_xlabel("wiek [lata]"); ax[1, 0].set_ylabel("RMSSD [ms]"); ax[1, 0].legend(fontsize=7)
    r, p = spearmanr(d["activity"], d["rmssd_ms"], nan_policy="omit")
    ax[1, 1].scatter(d["activity"], d["rmssd_ms"], s=22, color=GOOD)
    ax[1, 1].set_title(f"RMSSD vs aktywność fizyczna (rho={fr(r)}, p={fp(p)})", fontsize=9)
    ax[1, 1].set_xlabel("ocena aktywności fizycznej"); ax[1, 1].set_ylabel("RMSSD [ms]")
    fig_page(pdf, fig, "Rysunek 3. Regularny rytm daje na wykresie Poincaré wydłużoną 'cygaro' wzdłuż przekątnej "
                       "(SD1/SD2 małe); chaotyczne odstępy dają okrągłą chmurę (SD1/SD2 bliskie 1). "
                       "Wysokie RMSSD przy okrągłej chmurze nie jest oznaką dobrej regulacji.")


def fig_module3(cohort, val, pdf):
    fig, ax = plt.subplots(2, 2, figsize=A4, gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    fig.suptitle("Moduł 3 – skóra (10 mm) vs warstwa głębsza (30 mm)", weight="bold", y=0.97)
    ph = np.linspace(0, 1, cp.N_POINTS)
    r = cohort.loc[cohort["age"].idxmin()]
    for col, c, lab in (("_skin_wave", MUTED, "skóra 10 mm"), ("_wave", INK, "30 mm (surowy)"),
                        ("_deep_wave", ACCENT, "głęboki = 30 mm − β·skóra")):
        if isinstance(r[col], np.ndarray):
            ax[0, 0].plot(ph, r[col], color=c, lw=1.8, label=lab)
    ax[0, 0].set_title(f"przykład: osoba #{int(r['subject'])}, {int(r['age'])} l.")
    ax[0, 0].set_xlabel("faza uderzenia"); ax[0, 0].legend(fontsize=7)
    ax[0, 1].hist(100 * cohort["skin_share"].dropna(), bins=15, color=MUTED)
    ax[0, 1].set_title(f"ile pulsu 30 mm tłumaczy skóra (mediana {100 * cohort['skin_share'].median():.0f}%)")
    ax[0, 1].set_xlabel("R² [%]")
    sel = val.set_index("col").loc[["skin_relax_area", "relax75", "elast75"]]
    labs = ["skóra 10 mm", "30 mm surowy\n@75 bpm", "głęboki\n@75 bpm"]
    xs = np.arange(3)
    ax[1, 0].bar(xs - 0.18, sel["rho_age"], 0.36, color=INK, label="wszyscy (n=59)")
    ax[1, 0].bar(xs + 0.18, sel["rho_age_lt60"], 0.36, color=ACCENT, label="tylko <60 l. (n=55)")
    for i, (a, b) in enumerate(zip(sel["p_age"], sel["p_age_lt60"])):
        ax[1, 0].text(i - 0.18, min(a, 0) - 0.02, "*" if a < 0.05 else "", ha="center", va="top")
        ax[1, 0].text(i + 0.18, min(b, 0) - 0.02, "*" if b < 0.05 else "", ha="center", va="top")
    ax[1, 0].axhline(0, color=MUTED, lw=0.6)
    ax[1, 0].set_xticks(xs, labs); ax[1, 0].set_ylabel("rho Spearmana z wiekiem")
    ax[1, 0].set_title("odjęcie skóry wzmacnia związek z wiekiem (* p<0,05)"); ax[1, 0].legend(fontsize=7)
    d = cohort.dropna(subset=["skin_relax_area", "deep_relax_area"])
    sc = ax[1, 1].scatter(d["skin_relax_area"], d["deep_relax_area"], c=d["age"], cmap="magma_r", s=22)
    r_, p_ = spearmanr(d["skin_relax_area"], d["deep_relax_area"])
    ax[1, 1].set_title(f"skóra vs głęboki: rho={fr(r_)} (p={fp(p_)})", fontsize=9)
    ax[1, 1].set_xlabel("relax_area skóry"); ax[1, 1].set_ylabel("relax_area głęboki")
    fig.colorbar(sc, ax=ax[1, 1], label="wiek")
    fig_page(pdf, fig, "Rysunek 4. Kanał 10 mm widzi tylko skórę czoła; jego puls 'zanieczyszcza' kanały 30 mm. "
                       "Po regresyjnym odjęciu skóry zostaje część pulsu z głębszych warstw – i to ona najlepiej "
                       "różnicuje osoby wg wieku.")


def fig_reliability(rel, cohort, pdf):
    fig, ax = plt.subplots(2, 1, figsize=A4, gridspec_kw={"hspace": 0.45, "height_ratios": [1, 1.1]})
    fig.suptitle("Powtarzalność i podział na kategorie", weight="bold", y=0.97)
    rr = rel.sort_values("rho_halves")
    ax[0].barh(rr["index"], rr["rho_halves"], color=[GOOD if v > 0.8 else WARN for v in rr["rho_halves"]])
    for i, v in enumerate(rr["rho_halves"]):
        ax[0].text(v + 0.01, i, f"{v:.2f}".replace(".", ","), va="center", fontsize=8)
    ax[0].set_xlim(0, 1.1); ax[0].set_xlabel("rho: pierwsze 5 min vs ostatnie 5 min (te same osoby)")
    ax[0].set_title("czy wskaźnik jest cechą osoby, a nie szumem?")
    cnt = cohort["category"].value_counts()
    cols = {cp.CATEGORIES[k][0]: c for k, c in (("optimal", GOOD), ("skin", WARN), ("autonomic", WARN),
                                                ("eva", BAD), ("overload", BAD), ("irregular", MUTED))}
    ax[1].barh(cnt.index[::-1], cnt.values[::-1], color=[cols.get(k, MUTED) for k in cnt.index[::-1]])
    for i, v in enumerate(cnt.values[::-1]):
        ax[1].text(v + 0.3, i, str(v), va="center")
    ax[1].set_xlabel("liczba osób"); ax[1].set_title(f"kategorie w grupie (n={len(cohort)})")
    fig_page(pdf, fig, "Rysunek 5. Góra: korelacja wyniku z pierwszej i drugiej połowy tego samego nagrania. "
                       "Dół: ile osób trafiło do każdej kategorii. Progi (najniższa ćwiartka grupy) są umowne – "
                       "z definicji ok. 1/4 osób ma 'obniżony' wynik w każdym module.")


# ---------------------------------------------------------------------------
# Methods report
# ---------------------------------------------------------------------------
def methods_report(cohort, ref, val, rel, data_dir, path):
    V = val.set_index("col")
    R = rel.set_index("col")
    e, raw = V.loc["elast75"], V.loc["relax75"]
    rm = V.loc["rmssd_ms"]
    irr = cohort[cohort["irregular"]]
    old = cohort[cohort["age"] >= 60]
    n = len(cohort)
    with PdfPages(path) as pdf:
        T = TextFlow(pdf, "Profil krążenia z fNIRS – metody, wyniki, wnioski")
        T.h1("Profil krążenia z jednego nagrania fNIRS")
        T.p("Trzy moduły oceny układu krążenia z 10-minutowego spoczynkowego nagrania fNIRS czoła "
            "(Cortivision C23, 141,79 Hz, pozycja leżąca), policzone dla całej grupy "
            f"(n = {n}, wiek 18–78 l.) i dla pojedynczych osób.", color=MUTED)
        T.h2("Najważniejsze wyniki w skrócie")
        T.b(f"Elastyczność (moduł 1): kształt pulsu z warstwy głębszej, po odjęciu skóry i korekcie tętna, "
            f"słabnie z wiekiem: rho = {fr(e.rho_age)} (p = {fp(e.p_age)}); także bez 4 osób po 60. r.ż.: "
            f"rho = {fr(e.rho_age_lt60)} (p = {fp(e.p_age_lt60)}). Wynik jest powtarzalny "
            f"(pierwsza vs druga połowa nagrania: rho = {R.loc['deep_relax_area', 'rho_halves']:.2f}).")
        T.b(f"Ten sam wskaźnik bez odjęcia skóry jest wyraźnie słabszy (rho = {fr(raw.rho_age)}, "
            f"<60 l.: {fr(raw.rho_age_lt60)}, p = {fp(raw.p_age_lt60)}). Separacja skóry (moduł 3) "
            "nie jest więc ozdobnikiem – to ona wyciąga sygnał związany z wiekiem.")
        T.b(f"Balans autonomiczny (moduł 2): RMSSD rośnie z aktywnością fizyczną (rho = {fr(rm.rho_act)}, "
            f"p = {fp(rm.p_act)}) i maleje z wiekiem u osób <60 l. (rho = {fr(rm.rho_age_lt60)}, "
            f"p = {fp(rm.p_age_lt60)}) – zgodnie z literaturą HRV.")
        T.b(f"U {int(old['irregular'].sum())} z {len(old)} osób w wieku 68–78 l. odstępy między uderzeniami są "
            "chaotyczne (SD1/SD2 > 0,75). Ich wysokie RMSSD to raczej nieregularny rytm niż dobra regulacja – "
            "dlatego system tego nie nagradza.")
        T.b("Każda osoba dostaje percentyl i T-score względem grupy oraz jedną z kategorii przesiewowych. "
            "To narzędzie porównawcze w obrębie tej grupy, nie diagnoza medyczna.")

        T.h2("1. Dane")
        T.b("Zbiór: fnirs_high_freq_supine/baseline – 59 nagrań, 10 min spoczynku na leżąco, oczy otwarte.")
        T.b("Montaż: czoło (Fp1, Fpz, Fp2, AF7, AF8, F1–F4); 8 kanałów długich (23–30 mm) "
            "i 2 krótkie (10 mm: S7–D4, S8–D5). Dwie długości fali: 760 i 850 nm.")
        T.b("Metadane: wiek, płeć, wzrost, waga, leki, ocena aktywności fizycznej. Cztery osoby w wieku 68–78 l. "
            "biorą leki na nadciśnienie, jedna (20 l.) beta-bloker.")
        T.h2("2. Przygotowanie sygnału (wspólne dla wszystkich modułów)")
        T.n(1, "Surowe natężenie światła → gęstość optyczna: OD = −ln(I / średnie I).")
        T.n(2, "TDDR (Fishburn i in. 2019): naprawa skoków od ruchów głowy.")
        T.n(3, "Prawo Beera-Lamberta (ppf = 6) → stężenie HbO i HbR. Dalej używamy HbO w µM. "
               "Bez filtrowania na tym etapie – tętno i oddech to właśnie sygnał, którego szukamy.")
        T.n(4, "Stały zestaw kanałów dla wszystkich: 4 kanały czołowe 30 mm (S6–D4, S5–D3, S6–D5, S5–D4) "
               "+ 2 krótkie 10 mm. Wybrane na podstawie przeglądu wszystkich 10 kanałów: najwyraźniejsze tętno "
               "(czoło bez włosów) i jedyne, na których kształt pulsu wiązał się z wiekiem. Stały zestaw oznacza, "
               "że każdą osobę mierzymy w tym samym miejscu.")
        T.n(5, "Filtr pasmowy Butterwortha 0,5–10 Hz, rzędu 3, przód-tył (bez przesunięcia w czasie). "
               "Usuwa oddech (~0,25 Hz), fale Mayera (~0,1 Hz) i dryf; zostawia tętno z harmonicznymi, "
               "które tworzą kształt pulsu.")
        T.n(6, "Średnie tętno z widma (szczyt w 0,7–2,5 Hz). Dołki = początki uderzeń; odstęp co najmniej "
               "0,6 okresu, głębokość co najmniej 0,3 odchylenia standardowego.")
        T.n(7, "Uderzenie = odcinek od dołka do dołka. Odrzucamy uderzenia krótsze niż 0,7 lub dłuższe niż 1,3 "
               "średniego okresu. Każde przeskalowujemy do 200 punktów i do zakresu 0–1.")
        T.n(8, "Liczymy medianowy kształt; zostawiamy tylko uderzenia o korelacji > 0,9 z nim (zwykle ~650 z ~700). "
               "Średnia z nich = puls kanału. Średnia z 4 kanałów = puls osoby.")
        T.flush()
        fig_pipeline(Path(data_dir) / cohort.sort_values("quality_best").iloc[n // 2]["file"], pdf)

        T = TextFlow(pdf, "Profil krążenia z fNIRS – metody, wyniki, wnioski")
        T.h1("Moduł 1 – elastyczność naczyń („stan rur”)")
        T.h2("Zjawisko")
        T.p("Każde uderzenie serca wysyła falę ciśnienia, która odbija się od rozgałęzień tętnic i wraca. "
            "W elastycznych tętnicach fala biegnie wolno i odbicie wraca dopiero w rozkurczu – na opadającej "
            "części pulsu widać 'wybrzuszenie' lub drugi garb. W sztywnych tętnicach (wiek, nadciśnienie) "
            "fala biegnie szybciej, odbicie zlewa się ze szczytem, a opadanie robi się proste. "
            "Prędkość fali tętna jest uznanym, niezależnym czynnikiem ryzyka sercowo-naczyniowego "
            "(Laurent 2006; Vlachopoulos 2010). Ten sam efekt w pulsie fNIRS opisała grupa Fabiani i Gratton "
            "(wskaźnik PReFx, Fabiani 2014; Chiarelli 2017; Tan 2017).")
        T.h2("Co liczymy")
        T.n(1, "relax_area = pole między opadającą częścią uśrednionego pulsu a prostą od szczytu do końca "
               "uderzenia (Rys. 1, krok ⑦). Duże pole = wybrzuszone opadanie = elastyczne naczynia.")
        T.n(2, "Liczymy je na pulsie głębokim (moduł 3: kanał 30 mm minus skóra), bo ten okazał się najbardziej "
               "związany z wiekiem i powtarzalny.")
        T.n(3, f"Korekta tętna: kształt pulsu zależy też od tempa serca, więc sprowadzamy wynik do 75 bpm "
               f"nachyleniem z grupy (wynik_75 = wynik − b·(HR − 75); b = "
               f"{ref['hr_fit']['deep_relax_area']['slope']:+.4f} na bpm). Dla pulsu głębokiego zależność od tętna "
               f"jest słaba (rho = {fr(ref['hr_fit']['deep_relax_area']['rho'])}), dla surowego 30 mm wyraźna "
               f"(rho = {fr(ref['hr_fit']['relax_area']['rho'])}, p = {fp(ref['hr_fit']['relax_area']['p'])}).")
        T.n(4, "Dodatkowo: wysokość drugiego garbu (0, gdy go nie ma) – opisowo na karcie osoby.")
        T.h2("Skala populacyjna – porównanie człowieka z grupą")
        T.p(f"Każda osoba dostaje dwie liczby: (a) percentyl w całej grupie (0–100; np. 20 = 80% osób ma bardziej "
            f"elastyczny kształt pulsu) oraz (b) T-score: o ile odchyleń standardowych różni się od średniej "
            f"młodej grupy referencyjnej ({cp.REF_AGE[0]}–{cp.REF_AGE[1]} l., bez leków, n = {ref['n_ref']}). "
            "To ta sama logika co w densytometrii kości: T = 0 – jak młody zdrowy dorosły, T = −1 – o jedno "
            "odchylenie gorzej. Próg 'obniżona elastyczność' = najniższa ćwiartka grupy (percentyl < 25).")
        T.h2("Wyniki")
        T.table([["wskaźnik", "rho z wiekiem", "p", "rho <60 l.", "p", "powtarz."]] + [
            [V.loc[c, "index"].replace("ELASTYCZNOŚĆ: ", "")[:34], fr(V.loc[c, "rho_age"]), fp(V.loc[c, "p_age"]), fr(V.loc[c, "rho_age_lt60"]),
             fp(V.loc[c, "p_age_lt60"]),
             f"{R.loc[k, 'rho_halves']:.2f}" if k in R.index else "–"]
            for c, k in (("elast75", "deep_relax_area"), ("relax75", "relax_area"), ("relax_area", "relax_area"),
                         ("hump", "hump"), ("skin_relax_area", "skin_relax_area"))],
            [0.08, 0.44, 0.56, 0.65, 0.76, 0.84])
        T.p(f"Związek z aktywnością fizyczną po uwzględnieniu wieku nie wyszedł (rho = {fr(e.rho_act_given_age)}, "
            f"p = {fp(e.p_act_given_age)}). W tej grupie 18–42-latków różnice w kondycji są prawdopodobnie zbyt "
            "małe albo ankieta aktywności zbyt zgrubna, żeby je zobaczyć w kształcie pulsu.", color=MUTED)
        T.flush()
        fig_module1(cohort, ref, pdf)

        T = TextFlow(pdf, "Profil krążenia z fNIRS – metody, wyniki, wnioski")
        T.h1("Moduł 2 – balans autonomiczny („stan sterownika”)")
        T.h2("Zjawisko")
        T.p("Serce nie bije jak metronom. Nerw błędny (układ przywspółczulny – spokój, regeneracja) z uderzenia "
            "na uderzenie wydłuża i skraca odstępy, zwłaszcza w rytmie oddechu. Przewaga układu współczulnego "
            "(stres, zmęczenie, kofeina, słaba kondycja) podnosi tętno i 'usztywnia' rytm. Szybka zmienność "
            "między kolejnymi uderzeniami (RMSSD) jest standardową miarą aktywności nerwu błędnego "
            "(Task Force 1996; Shaffer i Ginsberg 2017). W spoczynku zmienność liczona z fali tętna (PRV) "
            "dobrze zgadza się z HRV z EKG (Schäfer i Vagedes 2013).")
        T.h2("Co liczymy")
        T.n(1, "Na najlepszym kanale czołowym: dla każdego uderzenia moment najszybszego narastania pulsu "
               "(maksimum pochodnej) z dokładnością poniżej próbki (interpolacja paraboliczna; próbka = 7 ms).")
        T.n(2, "Odstępy między kolejnymi uderzeniami (~700 na osobę). Odrzucamy odstępy spoza 0,7–1,3 mediany "
               "oraz te, które różnią się od poprzedniego o więcej niż 20% (pominięte lub fałszywe uderzenie).")
        T.n(3, "HR = 60 / średni odstęp; RMSSD = pierwiastek ze średniego kwadratu różnic kolejnych odstępów [ms]; "
               "SDNN = odchylenie standardowe odstępów.")
        T.n(4, "SD1/SD2 z wykresu Poincaré: stosunek zmienności szybkiej do wolnej. Regularny rytm z oddechową "
               "modulacją ma SD1/SD2 wyraźnie < 1; chaotyczne odstępy (np. pobudzenia dodatkowe, migotanie "
               "przedsionków) – bliskie 1 (Brennan i in. 2001). Próg ostrzegawczy: 0,75.")
        T.h2("Wyniki")
        T.table([["wskaźnik", "rho z wiekiem", "rho <60 l.", "p (<60)", "rho z aktywnością", "p"]] + [
            [V.loc[c, "index"][:26], fr(V.loc[c, "rho_age"]), fr(V.loc[c, "rho_age_lt60"]), fp(V.loc[c, "p_age_lt60"]),
             fr(V.loc[c, "rho_act"]), fp(V.loc[c, "p_act"])] for c in ("hr_bpm", "rmssd_ms", "sdnn_ms", "sd1_sd2")],
            [0.08, 0.38, 0.52, 0.64, 0.75, 0.9])
        T.b(f"Osoby z SD1/SD2 > 0,75: " + ", ".join(f"#{int(r.subject)} ({int(r.age)} l., RMSSD "
                                                    f"{r.rmssd_ms:.0f} ms)" for r in irr.itertuples()) + ".")
        T.b("Uwaga: u młodych, aktywnych osób wysokie SD1/SD2 zwykle oznacza silną modulację oddechową – "
            "w grupie <60 l. SD1/SD2 rośnie z aktywnością. Bez EKG nie da się rozstrzygnąć, który przypadek "
            "zachodzi; dlatego taka osoba dostaje kategorię 'do weryfikacji', a nie 'optymalny'.")
        T.b("RMSSD jest trochę wyższe u kobiet niż u mężczyzn w tej grupie (mediana "
            f"{cohort[cohort['sex'] == 'Kobieta']['rmssd_ms'].median():.0f} vs "
            f"{cohort[cohort['sex'] == 'Mężczyzna']['rmssd_ms'].median():.0f} ms). Przy większej grupie "
            "warto mieć osobne normy dla płci.")
        T.flush()
        fig_module2(cohort, pdf)

        T = TextFlow(pdf, "Profil krążenia z fNIRS – metody, wyniki, wnioski")
        T.h1("Moduł 3 – skóra (10 mm) vs warstwa głębsza (30 mm)")
        T.h2("Zjawisko")
        T.p("Światło w fNIRS przechodzi przez skórę, kość i dopiero potem dociera do kory. Kanał 10 mm widzi "
            "praktycznie tylko skórę, kanał 30 mm – skórę + głębsze warstwy. Odejmowanie sygnału z kanału "
            "krótkiego to standardowa metoda w fNIRS (Saager i Berger 2005; Brigadoi i Cooper 2015), "
            "ale zwykle stosuje się ją do usuwania szumu z odpowiedzi mózgowej – tu używamy jej do rozdzielenia "
            "KSZTAŁTU pulsu na część skórną i głębszą. Naczynia skóry czoła reagują na chłód i emocje "
            "(skurcz), a krążenie mózgowe jest chronione autoregulacją.")
        T.h2("Co liczymy")
        T.n(1, "Puls skóry: ta sama procedura (kroki 5–8) na kanałach 10 mm → kształt, relax_area skóry "
               "i amplituda pulsu skóry [µM] (miara ukrwienia skóry – mała przy skurczu naczyń).")
        T.n(2, "Dla każdego kanału 30 mm: najbliższy kanał 10 mm, współczynnik β metodą najmniejszych kwadratów, "
               "reszta = 30 mm − β·10 mm. R² = jaka część pulsu 30 mm to skóra.")
        T.n(3, "Puls głęboki: reszta pocięta w tych samych momentach co kanał 30 mm, uśredniona "
               "(pojedyncze uderzenia reszty są zbyt zaszumione, ale średnia z ~650 – już nie), wyrównana "
               "liniowo i znormalizowana do 0–1.")
        T.h2("Wyniki")
        T.b(f"Skóra tłumaczy medianowo {100 * cohort['skin_share'].median():.0f}% pulsu w kanałach 30 mm. "
            "To dużo – bez separacji kształt 'pulsu z głowy' jest głównie kształtem pulsu skóry czoła.")
        T.b(f"Puls głęboki jest powtarzalny (rho połówek = {R.loc['deep_relax_area', 'rho_halves']:.2f}) i najsilniej "
            f"związany z wiekiem ({fr(V.loc['deep_relax_area', 'rho_age'])}, p = {fp(V.loc['deep_relax_area', 'p_age'])}); "
            f"puls skóry – słabiej i nieistotnie ({fr(V.loc['skin_relax_area', 'rho_age'])}, "
            f"p = {fp(V.loc['skin_relax_area', 'p_age'])}).")
        T.b("Związek wieku z pulsem głębokim zostaje po kontroli udziału skóry (R²), tętna oraz po wyłączeniu "
            "osób przyjmujących leki – nie wynika więc z tego, ile skóry odjęto.")
        T.b("Czy wynik głęboki to tylko przesunięcie w czasie między pulsem skóry i pulsem 30 mm? Odejmowanie "
            "dwóch prawie takich samych, lekko przesuniętych fal daje kształt podobny do pochodnej. Sprawdziliśmy: "
            "opóźnienie (korelacja wzajemna) wynosi medianowo ok. 5 ms, nie zależy od wieku, nie koreluje ze "
            "wskaźnikiem głębokim, a jego uwzględnienie nie zmienia związku z wiekiem (r = −0,39).")
        T.b("Kształt pulsu głębokiego jest mniej 'książkowy' niż pulsu 30 mm (szeroki rozrzut w grupie, Rys. 2). "
            "relax_area traktujemy tu jako powtarzalną cechę kształtu, która wiąże się z wiekiem – nie jako "
            "bezpośredni obraz fali odbitej.")
        T.b("Amplituda pulsu skóry nie zależy od wieku – zależy od chwilowego stanu (temperatura, emocje) "
            "i od optyki (kolor skóry, włosy). Dlatego używamy jej tylko jako wskazówki 'chwilowej reakcji skórnej'.")
        T.flush()
        fig_module3(cohort, val, pdf)

        T = TextFlow(pdf, "Profil krążenia z fNIRS – metody, wyniki, wnioski")
        T.h1("Kategorie przesiewowe")
        T.p("Każdy moduł daje percentyl w grupie. 'Obniżony' = najniższa ćwiartka (percentyl < 25). "
            "Kategorię wybiera prosta, jawna reguła:")
        T.table([["elastyczność", "RMSSD", "skóra", "rytm", "kategoria"],
                 ["obniżona", "niskie", "–", "regularny", cp.CATEGORIES["overload"][0]],
                 ["obniżona", "–", "–", "dowolny", cp.CATEGORIES["eva"][0]],
                 ["prawidłowa", "niskie", "obniżona", "regularny", cp.CATEGORIES["skin"][0]],
                 ["prawidłowa", "niskie", "prawidłowa", "regularny", cp.CATEGORIES["autonomic"][0]],
                 ["prawidłowa", "–", "–", "SD1/SD2 > 0,75", cp.CATEGORIES["irregular"][0]],
                 ["prawidłowa", "prawidłowe", "–", "regularny", cp.CATEGORIES["optimal"][0]]],
                [0.08, 0.2, 0.31, 0.43, 0.57], size=8)
        for k, (name, msg) in cp.CATEGORIES.items():
            T.p(f"{name} (n = {int((cohort['category_key'] == k).sum())}): {msg}", size=8.6, width=108)
        T.h2("Powtarzalność")
        T.p("Każdą osobę policzyliśmy osobno na pierwszych i ostatnich 5 minutach nagrania. Wszystkie wskaźniki "
            "dają spójne wyniki między połówkami (Rys. 5). Najmniej stabilne są surowy relax_area 30 mm i wysokość "
            "garbu; najbardziej – tętno i cechy skóry.")
        T.flush()
        fig_reliability(rel, cohort, pdf)

        T = TextFlow(pdf, "Profil krążenia z fNIRS – metody, wyniki, wnioski")
        T.h1("Wnioski")
        T.n(1, "Z jednego, 10-minutowego spoczynkowego nagrania fNIRS czoła da się wyciągnąć trzy różne "
               "i powtarzalne informacje o krążeniu: mechanikę naczyń (kształt pulsu), regulację nerwową "
               "(zmienność rytmu) i podział na krążenie skóry i głębszych warstw.")
        T.n(2, "Kluczowa obserwacja, która wydaje się nowa: około 4/5 pulsu w kanałach 30 mm pochodzi ze skóry. "
               "Dopiero po jej odjęciu kształt pulsu wyraźnie i powtarzalnie wiąże się z wiekiem – także "
               "w grupie 18–42 lat i bez osób przyjmujących leki. Wcześniejsze prace o elastyczności z pulsu "
               "fNIRS (PReFx) nie rozdzielały warstw w ten sposób.")
        T.n(3, "Daje to skalę populacyjną: każdą osobę można umieścić na tle grupy (percentyl, T-score względem "
               "młodych dorosłych) i śledzić tę samą osobę w czasie – np. przed i po treningu.")
        T.n(4, "Zmienność rytmu zachowuje się zgodnie z literaturą (rośnie z aktywnością, maleje z wiekiem), "
               "więc dobrze nadaje się na kontekst dla modułu 1.")
        T.n(5, "Wysokie RMSSD u osób starszych może oznaczać nieregularny rytm, a nie dobrą regulację. Wskaźnik "
               "SD1/SD2 to wychwytuje i chroni przed fałszywie dobrym wynikiem.")
        T.h1("Ograniczenia – trzeba je powiedzieć głośno")
        T.b("To skala względna wobec tej grupy, nie kalibrowany pomiar sztywności (brak referencyjnej PWV, "
            "ciśnienia krwi ani EKG). Żaden wynik nie jest diagnozą.")
        T.b("Grupa jest młoda: 55 osób w wieku 18–42 l. i 4 osoby w wieku 68–78 l. Środek przedziału wieku "
            "jest pusty; wszyscy starsi biorą leki na nadciśnienie, więc wieku nie da się oddzielić od leczenia.")
        T.b("Korelacje rzędu 0,3–0,4 oznaczają tendencję w grupie, nie dokładny pomiar u jednej osoby. "
            f"Policzyliśmy {len(V)} korelacji z wiekiem bez poprawki na wielokrotne testowanie. Najsilniejszy wynik "
            f"przetrwa poprawkę Bonferroniego (p·{len(V)} = {fp(min(1, V.loc['deep_relax_area', 'p_age'] * len(V)))})"
            ", ale wybór tego wskaźnika nastąpił po obejrzeniu danych – potrzebna replikacja na nowej grupie.")
        T.b("'Głębiej' nie znaczy 'mózg': po odjęciu skóry zostaje m.in. kość, opony i płyn mózgowo-rdzeniowy. "
            "Regresja usuwa tylko tę część skóry, która jest liniowo podobna do kanału 10 mm.")
        T.b("Progi kategorii (ćwiartka grupy, SD1/SD2 > 0,75) są umowne i nie były walidowane klinicznie.")
        T.h2("Co dalej")
        T.b("Walidacja z referencją: tonometria (cfPWV), mankiet ciśnieniowy i EKG podczas tego samego nagrania.")
        T.b("Zbiór BH (wstrzymanie oddechu) z tej samej sesji: czy puls głęboki reaguje na CO₂ inaczej niż skóra "
            "(reaktywność naczyń mózgowych)?")
        T.b("Więcej osób w wieku 40–65 l. i osobne normy dla płci.")
        T.h2("Literatura (podana z pamięci – przed cytowaniem sprawdź dane bibliograficzne)")
        for ref_ in (
            "Fabiani M i in. (2014). Taking the pulse of aging: mapping pulse pressure and elasticity in cerebral "
            "arteries with optical methods. Psychophysiology 51:1072–1088.",
            "Chiarelli AM i in. (2017). Individual differences in regional cortical volumes across the life span "
            "are associated with regional optical measures of arterial elasticity. NeuroImage 162:199–213.",
            "Tan CH i in. (2017). Mapping cerebral pulse pressure and arterial compliance over the adult lifespan "
            "with optical imaging. PLoS ONE 12:e0171305.",
            "Laurent S i in. (2006). Expert consensus document on arterial stiffness. Eur Heart J 27:2588–2605.",
            "Vlachopoulos C i in. (2010). Prediction of cardiovascular events and all-cause mortality with arterial "
            "stiffness: a systematic review and meta-analysis. J Am Coll Cardiol 55:1318–1327.",
            "Millasseau SC i in. (2002). Determination of age-related increases in large artery stiffness by "
            "digital pulse contour analysis. Clin Sci 103:371–377.",
            "Charlton PH i in. (2022). Assessing hemodynamics from the photoplethysmogram to gain insights into "
            "vascular age: a review from VascAgeNet. Am J Physiol Heart Circ Physiol 322:H493–H522.",
            "Task Force of ESC and NASPE (1996). Heart rate variability: standards of measurement. Circulation "
            "93:1043–1065.",
            "Shaffer F, Ginsberg JP (2017). An overview of heart rate variability metrics and norms. Front Public "
            "Health 5:258.",
            "Schäfer A, Vagedes J (2013). How accurate is pulse rate variability as an estimate of heart rate "
            "variability? Int J Cardiol 166:15–29.",
            "Brennan M, Palaniswami M, Kamen P (2001). Do existing measures of Poincaré plot geometry reflect "
            "nonlinear features of heart rate variability? IEEE Trans Biomed Eng 48:1342–1347.",
            "Saager RB, Berger AJ (2005). Direct characterization and removal of interfering absorption trends in "
            "two-layer turbid media. J Opt Soc Am A 22:1874–1882.",
            "Brigadoi S, Cooper RJ (2015). How short is short? Optimum source–detector distance for short-separation "
            "channels in fNIRS. Neurophotonics 2:025005.",
            "Fishburn FA i in. (2019). Temporal Derivative Distribution Repair (TDDR): a motion correction method "
            "for fNIRS. NeuroImage 184:171–179.",
        ):
            T.p(ref_, size=8, width=115, color=MUTED)
        T.flush()


# ---------------------------------------------------------------------------
# Person card
# ---------------------------------------------------------------------------
def _scale(ax, pct_value, title, left, right, value_txt, warn_low=True):
    """Horizontal 0-100 percentile bar, red->green, with the person's marker."""
    grad = np.linspace(0, 1, 256)[None, :]
    cmap = plt.cm.RdYlGn if warn_low else plt.cm.Greys
    ax.imshow(grad, extent=(0, 100, 0, 1), aspect="auto", cmap=cmap, alpha=0.75)
    ax.axvline(cp.LOW_PCT, color=INK, lw=0.8, ls=":")
    if np.isfinite(pct_value):
        ax.plot([pct_value], [0.5], marker="v", ms=14, color=INK, transform=ax.transData, clip_on=False)
        title = f"{title}  –  {pct_value:.0f}. percentyl grupy"
    ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=7)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, weight="bold", pad=17)
    ax.text(0, -0.9, left, fontsize=7, color=MUTED, transform=ax.transData)
    ax.text(100, -0.9, right, fontsize=7, color=MUTED, ha="right", transform=ax.transData)
    ax.text(0, 1.35, value_txt, fontsize=8, color=INK, transform=ax.transData)


def person_card(r, cohort, ref, path):
    fig = plt.figure(figsize=A4)
    sid = f"#{int(r['subject'])}" if np.isfinite(r.get("subject", np.nan)) else Path(r["file"]).stem
    age = f"{int(r['age'])} l." if np.isfinite(r.get("age", np.nan)) else "wiek: brak"
    sex = r.get("sex") if isinstance(r.get("sex"), str) else ""
    act = f"aktywność {r['activity']:.1f}".replace(".", ",") if np.isfinite(r.get("activity", np.nan)) else ""
    fig.text(0.07, 0.955, f"Profil krążenia – osoba {sid}", fontsize=17, weight="bold", color=INK)
    fig.text(0.07, 0.932, "  ·  ".join(x for x in (age, sex, act, f"plik {r['file']}") if x), fontsize=8,
             color=MUTED)
    key = r.get("category_key", "")
    name, msg = cp.CATEGORIES.get(key, (str(r.get("category", "")), ""))
    col = {"optimal": GOOD, "skin": WARN, "autonomic": WARN, "eva": BAD, "overload": BAD}.get(key, MUTED)
    box = fig.add_axes([0.07, 0.835, 0.86, 0.075])
    box.set_xticks([]); box.set_yticks([])
    box.add_patch(plt.Rectangle((0, 0), 1, 1, color=col, alpha=0.12, transform=box.transAxes))
    for s in box.spines.values():
        s.set_color(col)
    box.text(0.02, 0.72, name, fontsize=12, weight="bold", color=col, transform=box.transAxes, va="center")
    box.text(0.02, 0.32, "\n".join(textwrap.wrap(msg, 120)), fontsize=7.8, color=INK, transform=box.transAxes,
             va="center")

    a1 = fig.add_axes([0.07, 0.73, 0.86, 0.025])
    _scale(a1, r["elast_pct"], "1. Elastyczność naczyń", "sztywniej niż grupa", "elastyczniej niż grupa",
           f"T-score {f2(r['elast_T'], 1)} vs 18–30 l.  ·  drugi garb: "
           f"{'tak' if r.get('hump_present') else 'nie'}".replace(".", ","))
    a2 = fig.add_axes([0.07, 0.635, 0.86, 0.025])
    _scale(a2, r["rmssd_pct"], "2. Balans autonomiczny (RMSSD)", "napięcie / stres", "spokój / regeneracja",
           f"tętno {f2(r['hr_bpm'], 0)} bpm  ·  RMSSD {f2(r['rmssd_ms'], 0)} ms  ·  SD1/SD2 "
           f"{f2(r['sd1_sd2'])}{'  ⚠ nieregularny' if r.get('irregular') else ''}".replace(".", ","))
    a3 = fig.add_axes([0.07, 0.54, 0.86, 0.025])
    _scale(a3, r["skin_pct"], "3. Ukrwienie skóry czoła (amplituda pulsu 10 mm)", "słabe (skurcz?)", "silne",
           f"amplituda {f2(r['skin_amp_uM'])} µM  ·  skóra = {f2(100 * r['skin_share'], 0)}% pulsu 30 mm"
           .replace(".", ","))

    ph = np.linspace(0, 1, cp.N_POINTS)
    b1 = fig.add_axes([0.08, 0.27, 0.38, 0.2])
    m, lo, hi = ref_wave(cohort, "_deep_wave")
    b1.fill_between(ph, lo, hi, color=ACCENT, alpha=0.15)
    b1.plot(ph, m, color=ACCENT, lw=1.5, label="młodzi 18–30 l.")
    if isinstance(r.get("_deep_wave"), np.ndarray):
        b1.plot(ph, r["_deep_wave"], color=INK, lw=2, label=f"osoba {sid}")
    b1.set_title("puls głęboki (30 mm − skóra)", fontsize=9); b1.set_xlabel("faza uderzenia")
    b1.legend(fontsize=7)
    b2 = fig.add_axes([0.57, 0.27, 0.36, 0.2])
    ibi = r.get("_ibi")
    if isinstance(ibi, np.ndarray) and len(ibi) > 2:
        b2.plot(1000 * ibi[:-1], 1000 * ibi[1:], ".", ms=2.5, color=INK, alpha=0.5)
        lim = np.percentile(1000 * ibi, [1, 99]) + np.array([-80, 80])
        b2.plot(lim, lim, color=MUTED, lw=0.6)
        b2.set_xlim(lim); b2.set_ylim(lim)
    b2.set_title("rytm: odstęp n vs n+1 [ms]", fontsize=9)
    b2.set_xlabel("odstęp n [ms]")

    lines = [
        "Jak czytać: każda skala pokazuje miejsce osoby w grupie 59 osób (percentyl). Kropkowana linia = 25. percentyl,",
        "poniżej którego wynik uznajemy za obniżony. Lewy wykres: kształt pulsu osoby na tle średniej młodych dorosłych",
        "(wybrzuszone opadanie = elastyczne naczynia). Prawy: wydłużona chmura wzdłuż przekątnej = regularny rytm;",
        "okrągła chmura = chaotyczne odstępy między uderzeniami.",
        "",
        "To narzędzie przesiewowe i porównawcze z hackathonu, nie badanie medyczne. Wynik odnosi się do tej konkretnej",
        "grupy (głównie zdrowi 18–42 l.) i jednego 10-minutowego nagrania. Przy niepokojącym wyniku: pomiar ciśnienia,",
        "EKG i konsultacja lekarska.",
    ]
    for i, t in enumerate(lines):
        fig.text(0.07, 0.19 - i * 0.016, t, fontsize=7.8, color=MUTED if i >= 5 else INK)
    fig.savefig(path)
    plt.close(fig)
