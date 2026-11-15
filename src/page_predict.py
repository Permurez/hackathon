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
import style  # noqa: E402
from features import load_meta  # noqa: E402

SETS = {
    "pasmo naczyniowe z fNIRS": ["band"],
    "wiek, płeć, BMI": ["age", "sex_male", "bmi"],
    "wiek, płeć, BMI + tętno spoczynkowe": ["age", "sex_male", "bmi", "hr"],
    "pasmo + tętno": ["band", "hr"],
    "pasmo + tętno + TI": ["band", "hr", "TI"],
    "wszystko (pasmo, tętno, TI, wiek, płeć, BMI)": ["band", "hr", "TI", "age", "sex_male", "bmi"],
}
LEVELS = ["niski", "średni", "wysoki"]
LEVELS_WITH = ["niskim", "średnim", "wysokim"]   # "z ... wynikiem"
LV3 = ["poniżej 1 h/tydz", "1–3 h/tydz", "ponad 3 h/tydz"]   # questionnaire 0-1.06 / 1.76 / 3.03


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


@st.cache_data(show_spinner=False)
def module_result(path: str) -> dict:
    return fv.analyze(path)


@st.cache_data(show_spinner=False)
def breath_hold() -> pd.DataFrame:
    """Per person, from the breath-hold recording of the same session (src/bh_reflex.py, src/bh_co2.py)."""
    r = pd.read_csv(ROOT / "bh_reflex.csv").groupby("subject").agg(
        n_holds=("duration", "size"), longest_s=("duration", "max"), hr_change=("hr_change_pct", "mean"))
    return r.join(pd.read_csv(ROOT / "bh_co2.csv").groupby("subject").long_rise.mean().rename("hbo_rise"))


# ---------------------------------------------------------------- data + model
st.title("Przewidywanie aktywności z jednego nagrania")
style.hero("Jedno 10-minutowe nagranie, jeden profil: tętno, praca naczyń, reakcja na wstrzymanie oddechu i "
           "szacunek, ile osoba trenuje. Przy każdej liczbie widać, czy to pomiar, porównanie z grupą, czy tylko "
           "wskazówka, a przy przewidywaniu: jak często model ma rację.",
           ["Wybierz osobę z zestawu albo wgraj plik .snirf", "Przeczytaj profil na górze",
            "Niżej: czy trenuje ponad 3 h tygodniowo i dokładniej, ile"])
with st.spinner("Przygotowuję model (pierwsze uruchomienie liczy 59 nagrań)…"):
    d = training_table()
    table, loo_band, edges, calib = evaluate(d)
y = d.active.to_numpy()
full = model().fit(d[["band"]].to_numpy(float), y)

src = st.sidebar.radio("Nagranie", ["Osoba z zestawu danych", "Wgraj plik .snirf"])
truth = None
if src.startswith("Osoba"):
    sid = st.sidebar.selectbox("Osoba", list(d.subject), format_func=lambda s: f"Osoba {s}",
                               index=int(np.flatnonzero(d.subject.to_numpy() == 17)[0]) if 17 in set(d.subject) else 0)
    i = int(np.flatnonzero(d.subject.to_numpy() == sid)[0])
    band, p = float(d.band[i]), float(loo_band[i])
    truth = d.iloc[i]
    st.sidebar.caption("Przewiduje model, który nie widział tej osoby.")
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
y3, P3, (mu3, sd3, par3), table3, s3 = ordinal(d)
p3 = P3[i] if truth is not None else ord_probs(par3, (band - mu3) / sd3)
t3 = int(np.digitize([p3 @ np.arange(3)], s3["edges"])[0])
mod = module_result(str(V.BASE / d.file[i])) if truth is not None else r
bhs = breath_hold()
bh = bhs.loc[int(truth.subject)] if truth is not None and int(truth.subject) in bhs.index else None

# ---------------------------------------------------------------- profile: everything one session says
st.subheader("Wszystko, co mówi to jedno badanie")
style.legend()
vr, tm, q = mod["vascular_regulation"], mod["arterial_timing"], mod["recording"]
row1, row2 = st.columns(3), st.columns(3)
B = style.badge
row1[0].metric("Tętno spoczynkowe", f"{mod['heart_rate_bpm']:.0f} /min")
row1[0].caption(f"{B('pomiar')}u dorosłych w spoczynku zwykle 60–100/min", unsafe_allow_html=True)
row1[1].metric("Aktywność", f"{100 * table3[t3, 2]:.0f}% trenuje > 3 h/tydz")
row1[1].caption(f"{B('wskazowka')}tyle osób z {LEVELS_WITH[t3]} wynikiem faktycznie trenuje ponad 3 h tygodniowo "
                "(szczegóły niżej)", unsafe_allow_html=True)
row1[2].metric("Jakość nagrania", "bez ostrzeżeń" if not mod["warnings"] else f"ostrzeżenia: {len(mod['warnings'])}")
row1[2].caption(f"{B('pomiar')}{pl(q['duration_min'], '.1f')} min, {mod['beats_used']} uderzeń serca, braki "
                f"{pl(q['missing_samples_pct'], '.1f')}% próbek", unsafe_allow_html=True)
pe, pt = vr["percentile_vs_reference"], tm["percentile_vs_reference"]
row2[0].metric("Regulacja naczyń", "–" if pe is None else f"wyżej niż {pe:.0f}% osób")
row2[0].caption(f"{B('grupa')}jedno nagranie ma dużą niepewność; przy powtórce wynik może się wyraźnie przesunąć",
                unsafe_allow_html=True)
row2[1].metric("Fala odbita (TI)", "–" if pt is None else f"wyżej niż {pt:.0f}% osób")
row2[1].caption(f"{B('grupa')}wyżej = fala tętna wraca szybciej; w grupie rośnie z wiekiem; pomiar stabilny",
                unsafe_allow_html=True)
if bh is not None and np.isfinite(bh.hr_change):
    rise = f"; HbO pod koniec {pl(bh.hbo_rise, '+.2f')} µM" if np.isfinite(bh.hbo_rise) else ""
    row2[2].metric("Wstrzymanie oddechu", f"tętno {bh.hr_change:+.0f}%")
    row2[2].caption(f"{B('pomiar')}to samo badanie, bez normy; analizowane bezdechy: {int(bh.n_holds)}, "
                    f"najdłuższy z nich {bh.longest_s:.0f} s{rise}", unsafe_allow_html=True)
else:
    row2[2].metric("Wstrzymanie oddechu", "brak nagrania")
    row2[2].caption("nagranie bezdechu z tego badania nie zostało wczytane")
st.caption("Czego nie pokazujemy: częstości oddechu. Z sygnału spoczynkowego na leżąco nie dało się jej tu "
           "wiarygodnie wyznaczyć (sprawdziliśmy), więc jej nie podajemy.")

# ---------------------------------------------------------------- answer
st.subheader("Czy ta osoba trenuje ponad 3 h tygodniowo?")
c1, c2 = st.columns(2)
c1.metric("Przewidywanie", "trenuje > 3 h/tydz" if p >= 0.5 else "mniej niż 3 h/tydz")
c2.metric("Prawdopodobieństwo", f"{100 * p:.0f}%")
st.markdown(f"**Co to znaczy:** wynik tej osoby jest **{LEVELS[level]}** na tle grupy. Spośród {n_lvl} osób z "
            f"podobnym wynikiem **{100 * share:.0f}% faktycznie trenuje** ponad 3 h tygodniowo "
            f"(w całej grupie {100 * y.mean():.0f}%).")
if truth is not None:
    act = V.ACT.get(truth.activity, "?")
    ok = (p >= 0.5) == bool(truth.active)
    st.markdown(f"**Ankieta tej osoby:** {act} (wiek {int(truth.age)} l.) → model "
                f"{'**trafił**' if ok else '**nie trafił**'}.")

# ---------------------------------------------------------------- finer: three levels
st.subheader("Dokładniej: ile godzin tygodniowo?")
m1, m2, m3 = st.columns(3)
for col, lab, v in zip((m1, m2, m3), LV3, p3):
    col.metric(lab, f"{100 * v:.0f}%")
st.markdown(f"**Szacunek modelu dla tej osoby:** {100 * p3[0]:.0f}% poniżej 1 h, {100 * p3[1]:.0f}% 1–3 h, "
            f"{100 * p3[2]:.0f}% ponad 3 h tygodniowo. **W danych, wśród osób z {LEVELS_WITH[t3]} wynikiem (n = "
            f"{s3['tert'][t3]}), faktycznie:** {100 * table3[t3, 0]:.0f}% / {100 * table3[t3, 1]:.0f}% / "
            f"{100 * table3[t3, 2]:.0f}%. Przy skrajnych wynikach model jest pewniejszy, niż pozwalają dane; "
            "bardziej wiarygodne są liczby z danych.")
st.markdown(
        f"- **Kolejność się zgadza:** szacowany poziom rośnie razem z prawdziwym (ρ = {pl(s3['rho'])}, "
        f"p = {pl(s3['p'], '.3f')}). Przy niskim wyniku modelu {100 * table3[0, 0]:.0f}% osób trenuje poniżej 1 h, "
        f"przy wysokim {100 * table3[2, 0]:.0f}%.\n"
        f"- **Dokładnej kategorii jeszcze nie:** trafia w {100 * s3['acc']:.0f}%, tyle co zgadywanie zawsze "
        f"„ponad 3 h” ({100 * s3['base']:.0f}%).\n"
        f"- **Dlaczego:** ankieta ma 5 przedziałów, w grupie {s3['counts'][0]} / {s3['counts'][1]} / "
        f"{s3['counts'][2]} osób na trzech poziomach. To proof of concept: dokładniejszy model wymaga więcej osób "
        "w danych uczących; użytkownik nadal robi tylko to jedno nagranie.")

# ---------------------------------------------------------------- honesty
st.subheader("Które dane najlepiej przewidują trening?")
show = table.drop(columns=["AUC (5-fold, 20 powtórzeń)", "trafienia"]).rename(
    columns={"AUC (leave-one-out)": "trafność (AUC)"}).sort_values("trafność (AUC)", ascending=False)
st.dataframe(show, hide_index=True, width="stretch")
st.caption("Trafność (AUC): 0,5 to zgadywanie, 1,0 to bezbłędne rozpoznanie. Liczona na osobach, których model "
           "nie widział.")
t = table.set_index("dane wejściowe")
best = t["AUC (leave-one-out)"].idxmax()
dem = t.loc[["wiek, płeć, BMI", "wiek, płeć, BMI + tętno spoczynkowe"]]
st.markdown(
    f"- Najlepiej przewiduje **{best}** (AUC {pl(t.loc[best, 'AUC (leave-one-out)'])}).\n"
    f"- Same dane osobowe (wiek, płeć, BMI, tętno): AUC {pl(dem['AUC (leave-one-out)'].min())}–"
    f"{pl(dem['AUC (leave-one-out)'].max())}.\n"
    "- **Wniosek:** nagranie fNIRS niesie informację o treningu ponad dane osobowe, ale przewaga jest umiarkowana.")
st.caption("Uczciwie: pasmo wybrano w przesiewie na tych samych danych, więc na nowych osobach trafność będzie raczej "
           "niższa; aktywność pochodzi z ankiety; jedno nagranie ma ograniczoną powtarzalność. To wskazówka dla "
           "grupy osób z podobnym wynikiem, nie pomiar aktywności tej osoby.")
