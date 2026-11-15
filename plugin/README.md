# fnirs-vascular

Vascular markers from a 10-minute resting-state fNIRS recording. **SNIRF in, JSON out**, three dependencies
(numpy, scipy, h5py), about 0.2 s per recording.

```bash
pip install ./plugin                      # or: pip install "git+<repo url>#subdirectory=plugin"
fnirs-vascular rest.snirf                 # JSON to the terminal
fnirs-vascular study_folder/ --out res.csv  # every .snirf in a folder, recursively (works on a BIDS dataset)
```

```python
from fnirs_vascular import analyze
r = analyze("rest.snirf")                     # or analyze(mne.io.read_raw_snirf("rest.snirf"))
r["vascular_regulation"]["endothelial_band_share"], r["arterial_timing"]["timing_index_per_s"], r["warnings"]
```

## What it returns

| Key | Meaning |
|---|---|
| `heart_rate_bpm` | heart rate from the HbO pulsation |
| `vascular_regulation.endothelial_band_share` | share of the slow vasomotion band 0.0095–0.021 Hz in the forehead HbO spectrum (endothelial, NO-dependent band; Kvernmo et al. 1999, 2003) |
| `vascular_regulation.percentile_vs_reference` | position in the healthy reference group |
| `vascular_regulation.prob_aerobic_gt_3h_week` | logistic model on the band (leave-one-out AUC 0.67) |
| `arterial_timing.timing_index_per_s` | reflected-wave timing index of the averaged pulse (Afkhami et al. 2021); rises with age |
| `arterial_timing.percentile_vs_reference` | position in the healthy reference group |
| `recording.*` | duration, sampling, dead channels, missing samples, whether the montage matches the reference |
| `warnings` | everything that limits the result (other montage, short rest, low sampling rate, too few channels) |

## Built-in safeguards

* **Comparison only when comparable.** Percentiles and the probability are given only for the reference montage
  (same pair names and distances within 3 mm). Any other montage returns raw values plus a warning.
* **Sampling below 50 Hz**: no pulse timing index (the reflected wave comes ~100 ms after the systolic peak).
* **Rest shorter than 8 min**: the band value is flagged as unreliable (about one cycle per minute).
* **Batches never stop** on a bad file: it becomes `{"file", "error"}`.

## Reference group

Healthy adults 18–42 y without medication (n = 52), supine, 10-min eyes-open rest, Cortivision Spectrum C23, sparse
prefrontal montage (Heroes of the Brain 2026 dataset). `reference.json` holds the two distributions, each sorted on
its own (no subject ids), the model coefficients and the montage signature; rebuild it with
`python src/plugin_reference.py`. A recording that is itself part of the reference group is compared with a group
that includes it.

## Limits

Group-level markers, not a diagnosis. One 10-min recording measures the band with limited test-retest reliability
(ICC ≈ 0.36 across four rests on the same day); averaging several recordings helps. The activity link comes from a
questionnaire, a cross-sectional design and ages 18–42, and it has not yet been confirmed in an independent group.
