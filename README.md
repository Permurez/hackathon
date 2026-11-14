# Heroes of the Brain 2026 – Cortivision *Breath, Brain & Body*

Nowy start. Poprzednia wersja (przesiew krążenia: `kod_fnirs/`, `screening.py`, `app.py`) jest
w historii gita, w commicie `113ec9d`.

```bash
./download_data.sh          # nagrania ~5 GB -> dane/infopack_and_data (nie są w repo)
```

## Surowa intensywność → OD → HbO/HbR (`src/haemo.py`)

Dane w obu zbiorach to surowa intensywność światła (CW, 760 i 850 nm). Konwersja napisana od zera
(numpy + h5py, bez MNE), krok po kroku:

1. odczyt SNIRF: intensywność, `measurementList` (źródło, detektor, długość fali), pozycje 3D optod → odległość
   źródło–detektor; pary < 15 mm to kanały krótkie (skóra);
2. próbki z `metaDataTags/missing_sample` → interpolacja liniowa; próbki z intensywnością ≤ 0 są przycinane
   i wypisywane jako ostrzeżenie;
3. gęstość optyczna ΔOD = −log₁₀(I / średnia I);
4. zmodyfikowane prawo Beera-Lamberta dla każdej pary, obie długości fali naraz (układ 2×2):
   ΔOD(λ) = [ε_HbO(λ)·ΔHbO + ε_HbR(λ)·ΔHbR] · d · DPF(λ), wynik w µM.
   ε: tabela Prahla (omlc.org, dane Gratzera i Kolliasa). DPF zależny od wieku i długości fali
   (Scholkmann & Wolf 2013) albo stały (`--dpf`).

```bash
python src/haemo.py PLIK.snirf --age 25 --out wynik.npz
python src/test_haemo.py      # syntetyczny test w obie strony + zgodność z MNE (r = 1,000, skala ±0,02%)
```

Sprawdzone na wszystkich 288 plikach (118 na leżąco, 170 DOT): 0 błędów.
