"""page_predict.py - "how much does this person exercise?" from one resting recording.

Model: logistic regression on the slow vasomotion band (the only input that helps; richer models are compared on the
page and do worse at n = 52). Training group = the healthy reference group of the module (18-42 y, no medication).
A person from the dataset is predicted by a model that has not seen them (leave-one-out); the track record and the
calibration ("people with a result like this one: X % really train > 3 h/week") come from the same leave-one-out run.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin"))
sys.path.insert(0, str(ROOT / "src"))
import fnirs_vascular as fv  # noqa: E402
import vascular as V  # noqa: E402
from features import load_meta  # noqa: E402

SETS = {
    "pasmo naczyniowe z fNIRS (sam)": ["band"],
    "wiek, płeć, BMI": ["age", "sex_male", "bmi"],
    "wiek, płeć, BMI + tętno spoczynkowe": ["age", "sex_male", "bmi", "hr"],
    "pasmo + tętno": ["band", "hr"],
    "pasmo + tętno + TI": ["band", "hr", "TI"],
    "wszystko (pasmo, tętno, TI, wiek, płeć, BMI)": ["band", "hr", "TI", "age", "sex_male", "bmi"],
}
LEVELS = ["niski", "średni", "wysoki"]
LEVELS_WITH = ["niskim", "średnim", "wysokim"]   # "z ... wynikiem"
LV3 = ["poniżej 1 h/tydz", "1–3 h/tydz", "ponad 3 h/tydz"]   # questionnaire 0-1.06 / 1.76 / 3.03
RAMP3 = ["#9ec5f4", "#3987e5", "#184f95"]                   # ordinal: light -> dark = less -> more training


def pl(v, f=".2f") -> str:
    return format(v, f).replace(".", ",")


def model():
    return make_pipeline(StandardScaler(), LogisticRegression())


@st.cache_data(show_spinner=False, persist="disk")
def run_many(paths: tuple[str, ...]) -> list[dict]:
    return fv.analyze_many(list(paths))


@st.cache_data(show_spinner=False)
def training_table() -> pd.DataFrame:
    res = run_many(tuple(sorted(str(p) for p in V.BASE.glob("*.snirf"))))
    rows = [dict(file=r["file"], subject=int(r["file"].split("_")[0]), hr=r["heart_rate_bpm"],
                 band=r["vascular_regulation"]["endothelial_band_share"], TI=r["arterial_timing"]["timing_index_per_s"])
            for r in res if "error" not in r]
    d = pd.DataFrame(rows).merge(load_meta(), on="subject")
    d = d[(d.age < 60) & (d.medication_any == 0)].dropna(subset=["band", "hr", "TI", "activity", "age", "sex_male", "bmi"])
    d["active"] = (d.activity == d.activity.max()).astype(int)
    return d.sort_values("subject").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def evaluate(d: pd.DataFrame):
    y = d.active.to_numpy()
    rows, loo_band = [], None
    for name, cols in SETS.items():
        X = d[cols].to_numpy(float)
        loo = cross_val_predict(model(), X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
        kf = [roc_auc_score(y, cross_val_predict(model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                                                 method="predict_proba")[:, 1]) for s in range(20)]
        rows.append({"dane wejściowe": name, "AUC (leave-one-out)": round(roc_auc_score(y, loo), 2),
                     "AUC (5-fold, 20 powtórzeń)": round(float(np.mean(kf)), 2),
                     "trafienia": f"{int(((loo >= 0.5) == y).sum())}/{len(y)}"})
        if cols == ["band"]:
            loo_band = loo
    edges = np.quantile(loo_band, [1 / 3, 2 / 3])
    lvl = np.digitize(loo_band, edges)
    calib = [(int((lvl == k).sum()), float(y[lvl == k].mean())) for k in range(3)]
    return pd.DataFrame(rows), loo_band, edges, calib


def ord_probs(params, z):
    """P(level 0, 1, 2) of an ordinal logit with one predictor; statsmodels params = [beta, cut1, log(cut2 - cut1)]."""
    b, c1, c2 = params[0], params[1], params[1] + np.exp(params[2])
    lo, hi = expit(c1 - b * z), expit(c2 - b * z)
    return np.array([lo, hi - lo, 1 - hi])


@st.cache_data(show_spinner=False)
def ordinal(d: pd.DataFrame):
    """Three activity levels from the band: leave-one-out probabilities, full fit, calibration by model tertile."""
    y = np.where(d.activity >= 3.03, 2, np.where(d.activity >= 1.76, 1, 0))
    x = d.band.to_numpy(float)
    n = len(x)
    P = np.zeros((n, 3))
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        mu, sd = x[m].mean(), x[m].std()
        r = OrderedModel(y[m], ((x[m] - mu) / sd)[:, None], distr="logit").fit(method="bfgs", disp=False)
        P[i] = ord_probs(np.asarray(r.params), (x[i] - mu) / sd)
    mu, sd = x.mean(), x.std()
    full = OrderedModel(y, ((x - mu) / sd)[:, None], distr="logit").fit(method="bfgs", disp=False)
    ev = P @ np.arange(3)
    edges = np.quantile(ev, [1 / 3, 2 / 3])
    lvl = np.digitize(ev, edges)
    table = np.array([np.bincount(y[lvl == k], minlength=3) / max(int((lvl == k).sum()), 1) for k in range(3)])
    rho, p = stats.spearmanr(ev, y)
    summary = dict(rho=float(rho), p=float(p), acc=float((P.argmax(1) == y).mean()), base=float((y == 2).mean()),
                   counts=[int((y == k).sum()) for k in range(3)], tert=[int((lvl == k).sum()) for k in range(3)],
                   edges=[float(e) for e in edges])
    return y, P, (float(mu), float(sd), np.asarray(full.params)), table, summary


# ---------------------------------------------------------------- data + model
st.title("Przewidywanie aktywności z jednego nagrania")
st.markdown("Z 10 minut spoczynku model ocenia, czy osoba **trenuje aerobowo ponad 3 h tygodniowo**. "
            "Wejście: jeden parametr z fNIRS (wolny rytm naczyń). Obok przewidywania pokazujemy, "
            "**jak często model ma rację** i co wynik znaczy dla osób z podobnym wynikiem.")
with st.spinner("Przygotowuję model (pierwsze uruchomienie liczy 59 nagrań)…"):
    d = training_table()
    table, loo_band, edges, calib = evaluate(d)
y = d.active.to_numpy()
full = model().fit(d[["band"]].to_numpy(float), y)
hits = int(((loo_band >= 0.5) == y).sum())
guess = int(max(y.sum(), len(y) - y.sum()))  # always giving the more common answer

src = st.sidebar.radio("Nagranie", ["Osoba z zestawu danych", "Wgraj plik .snirf"])
truth = None
if src.startswith("Osoba"):
    sid = st.sidebar.selectbox("Osoba", list(d.subject), format_func=lambda s: f"Osoba {s}",
                               index=int(np.flatnonzero(d.subject.to_numpy() == 17)[0]) if 17 in set(d.subject) else 0)
    i = int(np.flatnonzero(d.subject.to_numpy() == sid)[0])
    band, p = float(d.band[i]), float(loo_band[i])
    truth = d.iloc[i]
    st.sidebar.caption("Przewiduje model wyuczony bez tej osoby (leave-one-out).")
else:
    up = st.sidebar.file_uploader("Plik .snirf: 10 min spoczynku, Cortivision C23", type=["snirf"])
    if up is None:
        st.info("Wgraj nagranie .snirf w panelu po lewej.")
        st.stop()
    tmp = Path(tempfile.mkdtemp(prefix="fnirs_pred_")) / up.name
    tmp.write_bytes(up.getvalue())
    r = fv.analyze(tmp)
    if not r["recording"]["reference_montage"]:
        st.error("Inny montaż niż grupa, na której uczono model: przewidywanie byłoby niemiarodajne.")
        st.json(r)
        st.stop()
    band = r["vascular_regulation"]["endothelial_band_share"]
    p = float(full.predict_proba([[band]])[0, 1])
    for wmsg in r["warnings"]:
        st.warning(wmsg)

level = int(np.digitize([p], edges)[0])
n_lvl, share = calib[level]

# ---------------------------------------------------------------- answer
c1, c2, c3 = st.columns(3)
c1.metric("Przewidywanie", "trenuje > 3 h/tydz" if p >= 0.5 else "mniej niż 3 h/tydz")
c2.metric("Prawdopodobieństwo", f"{100 * p:.0f}%")
c3.metric("Trafność modelu", f"{hits}/{len(y)} osób",
          help=f"Każda osoba przewidziana modelem, który jej nie widział. Dla porównania: zgadywanie zawsze "
               f"częstszej odpowiedzi trafia {guess}/{len(y)}.")
st.markdown(f"**Co to znaczy:** wynik tej osoby jest **{LEVELS[level]}** na tle grupy. Spośród {n_lvl} osób z "
            f"podobnym wynikiem **{100 * share:.0f}% faktycznie trenuje** ponad 3 h tygodniowo "
            f"(w całej grupie {100 * y.mean():.0f}%).")
if truth is not None:
    act = V.ACT.get(truth.activity, "?")
    ok = (p >= 0.5) == bool(truth.active)
    st.markdown(f"**Ankieta tej osoby:** {act} (wiek {int(truth.age)} l.) → model "
                f"{'**trafił**' if ok else '**nie trafił**'}.")

a, b = st.columns([1.35, 1])
with a:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rng = np.random.default_rng(1)
    for row, (lab, m, col) in enumerate((("mniej niż 3 h/tydz", 0, V.GREY), ("trenuje > 3 h/tydz", 1, V.ORANGE))):
        x = 100 * d.band[d.active == m]
        ax.scatter(x, row + rng.uniform(-0.12, 0.12, len(x)), s=34, color=col, edgecolor="white", label=lab)
    sc, lr = full.named_steps["standardscaler"], full.named_steps["logisticregression"]
    cut = 100 * (sc.mean_[0] - sc.scale_[0] * lr.intercept_[0] / lr.coef_[0, 0])
    ax.axvline(cut, color=V.MUTED, ls="--", lw=1)
    ax.text(cut, 1.45, " granica decyzji modelu", fontsize=8, color=V.MUTED, va="center")
    ax.scatter([100 * band], [0.5], s=360, marker="*", color=V.INK, zorder=5, label="ta osoba")
    ax.set_yticks([0, 1], ["mniej niż\n3 h/tydz", "trenuje\n> 3 h/tydz"])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("wolny rytm naczyń [% sygnału]")
    ax.set_title(f"Grupa ucząca (zdrowi 18–42 l., n = {len(d)}) i miejsce tej osoby", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
with b:
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.bar(LEVELS, [100 * s for _, s in calib], color=[V.GREY, V.BLUE, V.ORANGE])
    ax.axhline(100 * y.mean(), color=V.MUTED, ls=":", lw=1)
    for k, (n, s) in enumerate(calib):
        ax.text(k, 100 * s + 2, f"{100 * s:.0f}%\n(n = {n})", ha="center", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("faktycznie trenuje > 3 h/tydz [%]")
    ax.set_xlabel("wynik modelu")
    ax.set_title("Czy wynik modelu się sprawdza?", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------- finer: three levels
st.subheader("Dokładniej: ile godzin tygodniowo?")
y3, P3, (mu3, sd3, par3), table3, s3 = ordinal(d)
p3 = P3[i] if truth is not None else ord_probs(par3, (band - mu3) / sd3)
m1, m2, m3 = st.columns(3)
for col, lab, v in zip((m1, m2, m3), LV3, p3):
    col.metric(lab, f"{100 * v:.0f}%")
t3 = int(np.digitize([p3 @ np.arange(3)], s3["edges"])[0])
st.markdown(f"**Szacunek modelu dla tej osoby:** {100 * p3[0]:.0f}% poniżej 1 h, {100 * p3[1]:.0f}% 1–3 h, "
            f"{100 * p3[2]:.0f}% ponad 3 h tygodniowo. **W danych, wśród osób z {LEVELS_WITH[t3]} wynikiem (n = "
            f"{s3['tert'][t3]}), faktycznie:** {100 * table3[t3, 0]:.0f}% / {100 * table3[t3, 1]:.0f}% / "
            f"{100 * table3[t3, 2]:.0f}%. Przy skrajnych wynikach model jest pewniejszy, niż pozwalają dane; "
            "bardziej wiarygodne są liczby z danych.")
a, b = st.columns([1.35, 1])
with a:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    left = np.zeros(3)
    for k in range(3):
        ax.barh(LEVELS, 100 * table3[:, k], left=left, color=RAMP3[k], edgecolor="white", linewidth=2, label=LV3[k])
        for row in range(3):
            v = 100 * table3[row, k]
            if v >= 5:
                ax.text(left[row] + v / 2, row, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                        color=V.INK if k == 0 else "white")
        left += 100 * table3[:, k]
    ax.set_xlim(0, 100)
    ax.set_xlabel("faktyczna odpowiedź w ankiecie [% osób]")
    ax.set_ylabel("wynik modelu")
    ax.set_title("Czy szacowany poziom się sprawdza? (osoby przewidziane bez siebie)", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
with b:
    st.markdown(
        f"- **Kolejność się zgadza:** szacowany poziom rośnie razem z prawdziwym (ρ = {pl(s3['rho'])}, "
        f"p = {pl(s3['p'], '.3f')}). Przy niskim wyniku modelu {100 * table3[0, 0]:.0f}% osób trenuje poniżej 1 h, "
        f"przy wysokim {100 * table3[2, 0]:.0f}%.\n"
        f"- **Dokładnej kategorii jeszcze nie:** trafia w {100 * s3['acc']:.0f}%, tyle co zgadywanie zawsze "
        f"„ponad 3 h” ({100 * s3['base']:.0f}%).\n"
        f"- **Dlaczego:** ankieta ma 5 przedziałów, w grupie {s3['counts'][0]} / {s3['counts'][1]} / "
        f"{s3['counts'][2]} osób na trzech poziomach. Do godzin co do jednej potrzebny jest pomiar ciągły "
        "(np. tydzień z akcelerometrem) i więcej osób; to proof of concept.")

# ---------------------------------------------------------------- honesty
st.subheader("Czy więcej danych pomaga przewidywać?")
st.dataframe(table, hide_index=True, width="stretch")
t = table.set_index("dane wejściowe")
best = t["AUC (leave-one-out)"].idxmax()
dem = t.loc[["wiek, płeć, BMI", "wiek, płeć, BMI + tętno spoczynkowe"]]
st.markdown(
    f"- Najlepiej przewiduje: **{best}**: {t.loc[best, 'trafienia']} trafień, AUC "
    f"{pl(t.loc[best, 'AUC (leave-one-out)'])}.\n"
    f"- Dane osobowe (wiek, płeć, BMI, tętno): {' i '.join(dem['trafienia'])} trafień, AUC "
    f"{pl(dem['AUC (leave-one-out)'].min())}–{pl(dem['AUC (leave-one-out)'].max())}. Zgadywanie zawsze częstszej "
    f"odpowiedzi: {guess}/{len(y)}.\n"
    "- **Wniosek:** nagranie fNIRS niesie informację o treningu ponad dane osobowe, ale przewaga jest umiarkowana. "
    f"Dokładanie kolejnych cech przy {len(d)} osobach pogarsza wynik (model się przeucza), więc aplikacja używa "
    "jednego parametru.")
st.caption("Uczciwie: pasmo wybrano w przesiewie na tych samych danych, więc na nowych osobach trafność będzie raczej "
           "niższa; aktywność pochodzi z ankiety; jedno nagranie ma ograniczoną powtarzalność. To wskazówka dla "
           "grupy osób z podobnym wynikiem, nie pomiar aktywności tej osoby.")
