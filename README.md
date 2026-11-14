# Puls pod czepkiem – przesiew krążenia i układu autonomicznego z fNIRS

Heroes of the Brain 2026 · wyzwanie Cortivision *Breath, Brain & Body*
(„ile informacji o stanie całego ciała da się wyciągnąć z sygnału fNIRS?”).

Z **jednego 10-minutowego nagrania spoczynkowego** czepkiem fNIRS (Cortivision Spectrum C23, 141,79 Hz,
czoło) wyciągamy trzy filary stanu krążenia, każdy z percentylem w grupie i 95% przedziałem ufności:

| Filar | Co mierzymy | Z czego |
|---|---|---|
| 1. Elastyczność naczyń | kształt fali tętna: pole relaksacji na stałym oknie fazy (podobne do PReFx), Crest Time, max. narost, garb dykrotyczny; korekta do 75 bpm | kanały 30 mm po odjęciu skóry (10 mm) |
| 2. Balans autonomiczny | tętno, RMSSD, SDNN, SD1/SD2 (Poincaré) ze zmienności tętna (PRV) po filtrze artefaktów ±20% | uderzenia z kanałów 30 mm |
| 3. Ukrwienie i udział skóry | amplituda pulsu skóry [µM], % pulsu 30 mm pochodzący ze skóry | 10 mm vs 30 mm |

To **wskaźnik przesiewowy**, porównawczy względem grupy 59 osób, a **nie diagnoza medyczna**.

## Szybki start

```bash
source ~/.venvs/fnirs/bin/activate          # Python 3.14, mne 1.13 (+ streamlit)
pip install -r requirements.txt
./download_data.sh                          # nagrania ~5 GB -> dane/infopack_and_data (nie są w repo)
cd kod_fnirs
python test_screening.py                    # testy na sygnałach syntetycznych -> ALL OK
python screening.py                         # 59 osób: cache + CSV + tabela Spearmana w terminalu (~1 min)
streamlit run app.py                        # dashboard: wybór osoby, 3 filary, krzywe, Poincaré, opis
```

Wyniki trafiają do `kod_fnirs/cardio_out/`: `screening_cohort.csv` (1 wiersz na osobę),
`screening_validation.csv` (Spearman), `screening_intra.csv` (pierwsze vs ostatnie 3 min).

## Metoda (`kod_fnirs/screening.py`)

1. **Stałe kanały dla wszystkich osób** (zamiast „top-3 na osobę”, które dobierało inne miejsca anatomiczne
   u różnych ludzi): 30 mm `S6_D4 S5_D3 S6_D5 S5_D4`, 10 mm `S7_D4 S8_D5`; kanały pod włosami (F1/F2) pominięte.
   Kanał odpada tylko przy `cardiac_quality < 4` (wykluczenie ustalone przed analizą wyników).
2. Intensywność → gęstość optyczna → TDDR → Beer-Lambert (HbO, µM) → Butterworth zero-fazowy 0,5–10 Hz
   (zostawia harmoniczne, czyli kształt fali).
3. **Trzy warstwy**: `Raw_30mm` (średnia 4 długich), `Scalp_10mm` (średnia 2 krótkich), `Brain_Isolated`
   (każdy długi minus β·najbliższy krótki, regresja OLS – short-separation regression).
4. **Uderzenia**: dołki `Raw_30mm`; uderzenie jest czyste, gdy jego długość mieści się w ±20% mediany
   i kształt koreluje r > 0,9 z medianowym. Te same uderzenia tną wszystkie trzy warstwy.
5. **Kształt (bez globalnego argmax)**: P1 = pierwsze lokalne maksimum w fazie 0,10–0,32 (gdy go nie ma:
   minimum 2. pochodnej). Crest Time i max. narost liczone w czasie rzeczywistym, przed rozciągnięciem do
   200 punktów; `fixed_relax_area` na stałym oknie 0,25–1,00; garb dykrotyczny w oknie 0,35–0,55 względem
   cięciwy P1→koniec. Liniowy wpływ tętna usunięty regresją w kohorcie (`*_75bpm`). 95% CI: bootstrap
   (300×) po czystych uderzeniach (zwykle 450–700).
6. **PRV**: znacznik = najstromszy punkt narostu (interpolacja paraboliczna). Każdy odstęp IBI różniący się
   o ponad ±20% od mediany danej osoby jest odrzucany; RMSSD i SD1/SD2 liczone tylko z par kolejnych
   czystych odstępów.
7. **Zmiana wewnątrzosobnicza**: pierwsze 3 min vs ostatnie 3 min tego samego nagrania + test Wilcoxona.

## Wyniki (59 osób, baseline na leżąco)

* Czyszczenie IBI działa: maksymalne RMSSD w kohorcie spadło z 900 do 154 ms (mediana 63 → 55 ms).
  Dopiero oczyszczone RMSSD wiąże się z aktywnością fizyczną: ρ = +0,32 (p = 0,014); przed czyszczeniem
  ρ = +0,16 (p = 0,22). Po kontroli wieku: ρ = +0,24 (p = 0,065).
* P1 jest prawdziwym lokalnym szczytem u 90% osób (30 mm), ale u 20% globalne maksimum wypada po fazie 0,32
  (późna fala odbita). Stary wskaźnik liczony od argmax mieszał więc u nich dwa różne zjawiska.
* Elastyczność (`Brain_Isolated_fixed_relax_area_75bpm`) maleje z wiekiem: ρ = −0,29 (p = 0,025, N = 59);
  w podgrupie < 60 lat już nieistotnie (ρ = −0,21, p = 0,13, n = 55). Wiąże się też z BMI: ρ = −0,27 (p = 0,04).
* Crest Time silnie zależy od tętna (ρ ≈ −0,6), dlatego korekta do 75 bpm jest konieczna. Pole relaksacji
  od tętna praktycznie nie zależy (nachylenie ≈ 0).
* Skóra odpowiada medianowo za **85%** pulsu w kanałach 30 mm, więc warstwa „mózgowa” to mała reszta.
* Na poziomie grupy nie widać wyciszania w trakcie 10 min: tętno −0,5 bpm (p = 0,053), RMSSD bez zmian.
  Ta analiza ma sens dla pojedynczej osoby, nie jako efekt grupowy.
* **Uczciwie**: 105 testów (21 metryk × 5 zmiennych), 15 z p < 0,05, ale **żaden nie przetrwa korekty
  FDR** (najniższe q = 0,07). To wyniki eksploracyjne do potwierdzenia na nowej próbie.

## Ograniczenia

* Brak złotego standardu: nie mamy EKG, ciśnienia ani PWV, więc to porównanie z wiekiem i ankietą, a nie walidacja kliniczna.
* Grupa: 55 osób w wieku 18–42 lat i tylko 4 w wieku 68–78 lat. Te same 4 osoby przyjmują leki na
  nadciśnienie, więc efektu leków nie da się oddzielić od wieku. Normy wiekowe w aplikacji: 18–24 (n = 44),
  25–59 (n = 11), 60+ (n = 4).
* PRV ≠ HRV: zmienność czasu przejścia fali tętna dokłada się do zmienności rytmu serca.
* Amplituda w µM zależy od stałego współczynnika PPF = 6, docisku optody, skóry i włosów, więc porównania między osobami są przybliżone.
* Regresja 10 mm zakłada ten sam przebieg fali w skórze i w korze (jedno β, bez przesunięcia w czasie).

## Struktura

```
README.md                 ten plik
download_data.sh          pobranie danych (Google Drive, gdown)
requirements.txt
kod_fnirs/
  screening.py            >>> główny pipeline (3 warstwy, 3 filary, CI, PRV, walidacja)
  app.py                  >>> dashboard Streamlit
  test_screening.py       testy: późne P2, monotoniczność z planted stiffness, reguła ±20% IBI
  stiffness.py            funkcje kształtu uderzenia (find_p1, fixed_relax_area, dicrotic_hump) + stary indeks
  fnirs_toolkit.py        ładowanie SNIRF, filtry, jakość kanału, oddech, bezdech, statystyka
  synthetic.py, test_*.py sygnały syntetyczne ze znaną prawdą
  cardio_profile.py, cardio_report.py   poprzednia wersja (karty PDF) – zastąpiona przez screening.py
  plot_pulse_age.py, shape_age.py, upstroke.py, pulse_gallery.py, heart_rate.py, view_snirf.py   eksploracja
  README.md               dokumentacja techniczna toolkitu
```

Dane są dostępne wyłącznie na czas hackathonu (13–15.11.2026); dalsze użycie wymaga zgody Cortivision.
