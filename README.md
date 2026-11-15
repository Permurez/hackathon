# Puls pod czepkiem: ślad treningu w spoczynkowym fNIRS

Heroes of the Brain 2026 · wyzwanie Cortivision *Breath, Brain & Body*.

**Wynik:** w 10-minutowym spoczynkowym nagraniu fNIRS z czoła wolny rytm naczyń (0,0095–0,021 Hz,
pasmo przypisywane śródbłonkowi) jest silniejszy u osób bardziej aktywnych fizycznie:
ρ = +0,46 (p = 0,0004, ages 18–42, n = 55), 1,49× w grupie trenującej > 3 h/tydz.
(literatura u sportowców, laser doppler w skórze: 1,5×; Kvernmo i in. 2003). Wynik niezależny od wieku,
płci i BMI, specyficzny dla tego pasma i kanałów na czole, potwierdzony ślepym przeszukaniem 5000 hipotez.
Wynik pomocniczy: indeks fali odbitej (TI, replikacja Afkhami i in. 2021) rośnie z wiekiem (ρ = +0,38).
To wskaźnik grupowy, nie diagnoza pojedynczej osoby.

## Szybki start

```bash
./download_data.sh                                   # dane hackathonu (~5 GB) -> dane/
source ~/.venvs/fnirs/bin/activate                   # Python 3.14
pip install -r requirements.txt
streamlit run src/app.py                             # interaktywne demo w przeglądarce
python src/demo.py dane/infopack_and_data/fnirs_high_freq_supine/data/baseline/17_PK_BS_2025-08-11-12-19-59.snirf --show
```
`app.py` (Streamlit) i `demo.py` (terminal) robią to samo: jedno nagranie → HbO → widmo → wolny rytm naczyń
→ uśredniona fala tętna i fala odbita (TI) → pozycja na mapie „dwóch osi” na tle zdrowej grupy
(18–42 l. bez leków, n = 52) + model aktywności (regresja logistyczna, AUC ≈ 0,70). Rdzeń: `src/vascular.py`.

**Dwie osie są niezależne** (korelacja ρ = −0,05): regulacja naczyń śledzi trening, indeks fali odbitej śledzi wiek
(r = +0,36, p = 0,009, bez leków; powtarzalność 0,95).

## Odtworzenie wyników

| Krok | Polecenie | Wynik |
|---|---|---|
| konwersja światło → HbO/HbR (test vs MNE) | `python src/test_haemo.py` | ALL OK |
| parametry wszystkich osób | `python src/features.py` | `features.csv`, `correlations.csv` |
| pasma wazomocji | `python src/bands.py` | `bands_supine.csv` |
| indeks fali odbitej | `python src/timing_index.py` | `timing_index.csv` |
| 17 testów statystycznych | `python src/full_stats.py` | wydruk |
| test hipotezy (trend, regresja porządkowa, AUC, falsyfikacja) | `python src/hypothesis_endo.py` | wydruk |
| odporność: multiverse, szum, 5000 losowych hipotez | `python src/multiverse.py`, `src/sensitivity.py`, `src/random_hypotheses.py` | csv + wydruk |
| wykresy na slajdy | `python src/figures.py` | `fig_*.png` |

## Pliki

| Plik | Co robi |
|---|---|
| `src/haemo.py` | SNIRF → gęstość optyczna → HbO/HbR (Beer–Lambert, DPF Scholkmann & Wolf 2013), bez MNE |
| `src/features.py` | ~90 parametrów na osobę + przegląd korelacji z FDR |
| `src/bands.py` | moc pasm Stefanovskiej (śródbłonkowe, neurogenne, miogenne, oddechowe, sercowe) |
| `src/timing_index.py` | replikacja TI_NIRS (Afkhami i in. 2021) |
| `src/full_stats.py`, `src/hypothesis_endo.py` | testy statystyczne głównego wyniku |
| `src/multiverse.py`, `src/sensitivity.py`, `src/random_hypotheses.py`, `src/deep_screen.py` | odporność wyniku |
| `src/subgroup.py`, `src/plot_correlations.py` | mapy korelacji (grupa 18–42) |
| `src/endo_followup.py` | stabilność pasma w ciągu dnia (dane DOT) |
| `src/bh_reflex.py`, `src/bh_co2.py`, `src/spiroximetry.py` | oddech i bezdech: reakcja HbO na CO₂, tętno, saturacja żylna |
| `src/vascular.py` | rdzeń demo: oba wskaźniki dla jednego nagrania, grupa odniesienia, wykresy |
| `src/app.py`, `src/demo.py` | demo: aplikacja Streamlit i wersja terminalowa |
| `src/ti_followup.py` | indeks fali odbitej bez osób na lekach + powtarzalność (połowy nagrania) |

Poprzednia wersja projektu (przesiew krążenia, Streamlit) jest w historii gita, commit `113ec9d`.

## Ograniczenia

Aktywność z ankiety (Jurca 2005), badanie przekrojowe, 10 min nagrania, brak pomiaru odniesienia
(FMD, doppler, test wysiłkowy), grupa głównie 18–42 lata. Dane hackathonu: użycie tylko do 15.11.2026.
