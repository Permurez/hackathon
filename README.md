# Pulse Under the Cap

**Heroes of the Brain 2026 · Cortivision challenge *Breath, Brain & Body***

From **10 minutes of lying still in an fNIRS cap** we read how the blood vessels work and estimate roughly how
many hours a week a person trains. No extra equipment: one resting recording (`.snirf`) in, a profile out.

| What we read from one recording | How far to trust it |
|---|---|
| Estimated aerobic training, e.g. *about 4 h / week* | hint, typical error ±1.2 h |
| Vessel regulation (slow vessel wave, ~once a minute) | compared with healthy people aged 18–42 |
| Pulse echo index TI (rises with artery age) | compared with healthy people aged 18–42 |
| Resting heart rate, recording quality | measured |
| Heart-rate drop during breath holding (diving reflex) | measured (breath-hold recording of the same session) |

These are group-level hints, **not a medical diagnosis**.

## Main finding

The slow vessel wave (0.0095–0.021 Hz) in the forehead fNIRS signal is **1.49× stronger in people who do aerobic
exercise over 3 h a week** (Spearman ρ = +0.46, p = 0.0004, n = 55, ages 18–42; ρ = +0.49 after adjusting for age,
sex and BMI). Laser Doppler studies attribute this band to the endothelium, the inner lining of blood vessels
(Kvernmo et al. 1999), and found it **1.5× stronger in athletes** (Kvernmo et al. 2003).

Second, independent marker: the pulse echo index TI (Afkhami et al. 2021) rises with age (r = +0.36, p = 0.009,
n = 52 healthy, no medication); the two markers are unrelated (ρ = −0.05).

**Status: promising, not yet confirmed.** The band came out of a 1350-test screen (it survives FDR, q = 0.026,
and only 1.1% of 5000 random hypotheses are as strong), but it did not replicate in the small DOT dataset
(ρ = −0.06; 12 of 17 people there report the maximum activity). See *Limitations* and *Next studies*.

## From data to the training estimate

**Data.** `fnirs_high_freq_supine`: 59 resting recordings, 10 min, lying down, eyes open, Cortivision Spectrum C23,
141.79 Hz, 10 source–detector pairs × 2 wavelengths (760 / 850 nm), plus each person's age, sex, height, weight,
medication and a physical-activity questionnaire (Jurca et al. 2005 scale). The breath-hold recordings of the same
people feed the breath-hold results. The `dot_montage_sitting` dataset (17 people × 4 rests in one day) is used only
to check repeatability and as an independent check, not to train anything.

**Channels.** The vessel wave uses the **4 long (≈30 mm) forehead channels without hair**: `S5_D3 S5_D4 S6_D4 S6_D5`,
averaged. Heartbeats are found on the same average; TI is computed on every long (20–35 mm) channel with a clear
heartbeat and averaged over at least 3 channels. The two short 10 mm channels (`S7_D4 S8_D5`, skin only) are used in
the research scripts to compare skin with deeper signal.

**Steps** (`plugin/fnirs_vascular/core.py`, `analyze()`):

1. Read raw light intensity from the `.snirf` file; fill missing samples by interpolation.
2. Optical density: `−log10(I / mean I)`.
3. Modified Beer–Lambert law (DPF = 6, Prahl extinction coefficients) → oxygenated / deoxygenated haemoglobin
   (HbO / HbR, µM) for every pair. Written from scratch and checked against MNE (`src/test_haemo.py`).
4. Mean HbO of the 4 forehead channels → downsample to ~5 Hz → remove trend.
5. Welch spectrum with 300-s windows (the band has a ~1-minute period, so the rest must be ≥ 8 min).
6. **Vessel wave = share of power in 0.0095–0.021 Hz out of 0.0095–2 Hz** (one number; e.g. 0.77 for person 17).
7. **TI**: average hundreds of beats into one typical pulse per channel, find the wave from the heart and its echo
   from the second derivative, `TI = 1 / (t_echo − t_heart)`.
8. Compare with the **reference group**: healthy people aged 18–42 without medication (n = 52), same sensor layout.

**Training estimate** (`src/page_predict.py`):

9. The questionnaire has 5 answers, two of them with 1 and 3 people, so they are merged into **3 levels**:
   under 1 h a week (n = 10), 1–3 h (n = 14), over 3 h (n = 28).
10. An **ordinal logistic regression on the vessel wave alone** gives the probability of each level. Adding age, sex,
    BMI, heart rate or TI made predictions worse at this sample size, so the model uses one input.
11. **Estimate = probability-weighted mean of each level's typical hours** (0.2 / 2 / 4 h a week), rounded to
    half an hour. Person 17: probabilities 1% / 4% / 95% → *about 4 h a week* (questionnaire: over 3 h).
12. Checked leave-one-out (every person predicted by a model that has not seen them): off by **±1.2 h** on average,
    versus ±1.4 h for giving everyone the group mean; the ranking of people holds (ρ = 0.37).

## Quick start

```bash
./download_data.sh                      # hackathon data (~5 GB) -> dane/  (not in the repository)
python -m venv ~/.venvs/fnirs && source ~/.venvs/fnirs/bin/activate
pip install -r requirements.txt
streamlit run src/app.py                # the app, from the repository root
```

**The app** has four views: *Recording* (signal, spectrum, pulse wave, quality), *Activity estimate* (the profile
above and the training estimate), *Whole study* (a folder of recordings, two-group comparison, CSV export) and
*One-recording reliability* (upload one file: its value and how much a repeat could differ).

**The plug-in module** (`plugin/`, see [`plugin/README.md`](plugin/README.md)): `.snirf` in, JSON out, 0.2 s per
recording, only numpy / scipy / h5py, values identical to the analysis.

```bash
pip install ./plugin
fnirs-vascular recording.snirf                     # JSON
fnirs-vascular study_folder/ --out results.csv     # every .snirf in a folder (BIDS works)
```

## Reproducing the results

| Step | Command | Output |
|---|---|---|
| light → HbO/HbR, checked against MNE | `python src/test_haemo.py` | ALL OK |
| ~90 features per person + FDR screen | `python src/features.py` | `features.csv`, `correlations.csv` |
| vasomotion bands | `python src/bands.py` | `bands_supine.csv` |
| pulse echo index | `python src/timing_index.py`, `python src/ti_followup.py` | `timing_index.csv`, `ti_followup.csv` |
| statistics of the main finding | `python src/full_stats.py`, `python src/hypothesis_endo.py` | printout |
| robustness: 270 variants, noise, 5000 random hypotheses | `python src/multiverse.py`, `src/sensitivity.py`, `src/random_hypotheses.py` | csv + printout |
| repeatability and DOT check | `python src/endo_followup.py` | printout |
| breath holding and breathing | `python src/bh_reflex.py`, `src/bh_co2.py`, `src/spiroximetry.py` | csv + figures |
| what the next studies must show (mock data) | `python src/simulate_future.py` | printout + `fig_future_studies.png` |
| reference group for the module | `python src/plugin_reference.py` | `plugin/fnirs_vascular/reference.json` |
| figures | `python src/figures.py` | `fig_*.png` |

## Other results

* **Breath holding:** heart rate drops during the hold in 75% of people (median −3.1%, p = 0.006; vs the same
  windows at rest p = 0.035). HbO rises at the end of the hold (+0.55 µM in 71%, vs rest p = 0.002), but not after
  removing the skin signal, so we do not claim it is the brain.
* **Venous oxygen saturation** from the breathing oscillation during paced breathing (DOT): 70–80%, depends on DPF.
* **Not shown on purpose:** breathing rate. At supine rest the spectral-peak estimate sits at the edge of its range.

## Limitations

Activity comes from a questionnaire; the design is cross-sectional; mostly ages 18–42; no reference measurement
(flow-mediated dilation, laser Doppler, exercise test). One 10-min recording has limited repeatability for the
vessel wave (ICC 0.36–0.56 across four rests in one day). The forehead signal is largely skin, as in the laser
Doppler literature. Hackathon data may be used only until 15.11.2026; the reference group in the module is derived
from Cortivision data.

## Next studies

Simulations with our parameters (`src/simulate_future.py`): confirm on about 70 new people with the hypothesis
written down first; repeat recordings on different days; compare with laser Doppler or flow-mediated dilation
(r ≥ 0.6, about 40 people); a training study needs the band to rise by ≥ 6 percentage points (30 + 30 people,
3 recordings before and after). If confirmed: a vessel-health check without a cuff, drugs or an ultrasound
operator, during every fNIRS session.

## Files

| Path | What it does |
|---|---|
| `plugin/fnirs_vascular/` | the module: `analyze()` for one recording, CLI for folders, reference group |
| `src/app.py`, `src/page_*.py`, `src/style.py` | the app (four views, look) |
| `src/haemo.py` | SNIRF → optical density → HbO/HbR, step by step |
| `src/vascular.py` | per-recording markers and figures used by the app |
| `src/features.py`, `src/bands.py`, `src/timing_index.py` | features, vasomotion bands, pulse echo index |
| `src/full_stats.py`, `src/hypothesis_endo.py`, `src/hypotheses.py` | statistics; pre-specified hypotheses |
| `src/multiverse.py`, `src/sensitivity.py`, `src/random_hypotheses.py`, `src/deep_screen.py` | robustness |
| `src/endo_followup.py`, `src/ti_followup.py` | repeatability, DOT check, TI without medication |
| `src/bh_reflex.py`, `src/bh_co2.py`, `src/spiroximetry.py` | breath holding and breathing |
| `src/simulate_future.py` | what the next studies must show |

The earlier version of the project (a broad circulation screen) is in the git history, commit `113ec9d`.
