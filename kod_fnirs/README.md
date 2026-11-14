# fNIRS body-state toolkit (Heroes of the Brain 2026, Cortivision challenge)

Feature extraction for the *systemic physiology* hidden in fNIRS: heart rate, pulse shape,
breathing rate and phase, breath-hold detection and response, band powers, plus honest
small-N validation helpers.

## Setup (do it TONIGHT, not at 10:30)

```bash
python3 -m venv .venv            # isolated Python environment for this project
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python test_toolkit.py           # every line must say OK
python demo_figure.py            # writes demo_overview.png
```

## Files

| File | What it does |
|---|---|
| `heart_rate.py` | Heart rate over time from one `.snirf`: beat detection on the averaged pulse of the best channels, HR trend with markers, HRV, beats -> CSV |
| `view_snirf.py` | Look inside one `.snirf`: summary (markers, channels, SCI, best heartbeat channels) + figure: Hb traces, accelerometer, spectrum, averaged pulse wave |
| `fnirs_toolkit.py` | All feature functions (numpy arrays in, numbers out) + `load_snirf_haemo()` |
| `synthetic.py` | Fake recordings with known ground truth (HR, RR, breath holds, artery stiffness, cohort with age/activity) |
| `test_toolkit.py` | Checks each feature recovers the planted truth |
| `run_pipeline.py` | Folder of `.snirf`/`.npz` -> `features.csv` (one row per recording) |
| `demo_figure.py` | 4-panel overview figure; reuse the panels on real data on day 1 |
| `validate_real.py` | Compares our fNIRS-only breathing rate, heart rate and inhale/exhale against a REAL respiration belt + PPG stored as SNIRF aux channels (e.g. the open BU dataset below) |

## Validate on real data TONIGHT (open dataset with a respiration belt)

Open Access Multimodal fNIRS Resting State Dataset (von Luhmann et al. 2020, Front. Neurosci.,
doi:10.3389/fnins.2020.579353), BSD license, on NITRC: project `luhmann20synhrf`.
Dataset I: 14 people, 5 min rest, 50 Hz, SNIRF, short-separation channels, aux = PPG, BP, RESP belt.

```bash
python validate_real.py --data-dir BU_DATASET_I/ --out validation.csv
```
`--data-dir` = folder with the downloaded .snirf files, `--out` = per-window CSV,
`--win` = window length in seconds (default 60).

## Viewing a recording

```bash
python view_snirf.py FILE.snirf                     # summary + figure window
python view_snirf.py FILE.snirf --save out.png      # PNG instead (Colab / no display)
python view_snirf.py FILE.snirf --info              # summary only
python view_snirf.py FILE.snirf --ch S1_D1 S7_D4 --band cardiac --tmin 100 --tmax 130
```
`--band`: `raw`, `vlf`, `mayer`, `resp`, `cardiac` or `LO-HI` in Hz (e.g. `0.5-8`).

## Heart rate from one recording

```bash
python heart_rate.py FILE.snirf --save hr.png --csv hr.csv --zoom 500 520
```

## Day-1 quickstart on real data

```python
import fnirs_toolkit as fk
d = fk.load_snirf_haemo("sub01_breath_hold.snirf")   # intensity -> OD -> TDDR -> Beer-Lambert
best = fk.rank_channels(d["hbo"], d["fs"])[0]          # channel with the clearest heartbeat
x, y, fs = d["hbo"][best], d["hbr"][best], d["fs"]
print(fk.heart_rate(x, fs), fk.respiration_rate(x, fs), fk.respiration_rate_fused(x, fs))
print(fk.detect_apnea(x, fs))                          # unsupervised breath-hold detector
print(d["annotations"])                                # event markers, if the file has them
```

```bash
python run_pipeline.py --data-dir DATA/ --meta meta.csv --out features.csv
```

## Lessons already learned on synthetic data

- **Mayer-wave trap:** Mayer waves (~0.1 Hz) can be stronger than breathing in HbO. A naive
  peak search from 0.05 Hz returned 6/min instead of 15/min. Default `respiration_rate()` now
  starts at 0.13 Hz; for *paced slow breathing* pass `fmin=0.05` and check against the pace.
- **Garbage channels poison averages:** always filter by `cardiac_quality` (noise ~1.5-2,
  clean > 15) before aggregating channels.
- **Breathing phase sign must be calibrated** (belt, paced cue or RSA). Literature default
  used by `inhale_mask()`: head HbO/HbT FALL on inhale and RISE on exhale (Elwell et al. 1996).
- **Harmonics break beat detection:** a wide 0.7-2.5 Hz band gave false double beats (HR noise
  ~6 bpm). `beat_times()` now filters +-40% around the fundamental (noise ~1 bpm).
- **Paced breathing = a free physiological probe:** `paced_phasor()` (lock-in at the known pace,
  HbR-vs-HbO phase, idea from coherent hemodynamics spectroscopy) and `rsa_amplitude()`.

## Validation rules

- One subject never appears in both train and test (`evaluate_subject_level`, LOSO/LOO).
- Always report the dummy baseline and the permutation p-value next to the score.
- Activity correlates with age: report `partial_corr(feature, activity, age)`.
