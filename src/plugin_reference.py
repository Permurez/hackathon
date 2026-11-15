"""plugin_reference.py - build plugin/fnirs_vascular/reference.json from the supine dataset.

The plugin ships only what it needs to place a new recording against the healthy reference group
(18-42 y, no medication): the two marker distributions, each sorted on its own (no subject ids, no link
between a person's two values), the activity model coefficients, and the montage signature
(pair names + distances) so that a recording from another montage is not compared with this group.

  python src/plugin_reference.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haemo  # noqa: E402
from vascular import BASE, cohort  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plugin/fnirs_vascular/reference.json"

if __name__ == "__main__":
    coh = cohort()
    ref = coh[coh.healthy_ref].dropna(subset=["endo", "TI", "activity"])
    y = (ref.activity == ref.activity.max()).astype(int).values
    mu, sd = float(ref.endo.mean()), float(ref.endo.std())
    z = ((ref.endo - mu) / sd).values[:, None]
    model = LogisticRegression().fit(z, y)
    loo = cross_val_predict(LogisticRegression(), z, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    auc_loo = float(roc_auc_score(y, loo))

    h = haemo.convert(next(BASE.glob("*.snirf")), dpf=6.0)
    montage = {n: round(float(d), 1) for n, d in zip(h["pairs"], h["distance_mm"])}

    out = {
        "description": "Healthy adults 18-42 y without medication, supine, 10-min eyes-open rest, "
                       "Cortivision Spectrum C23, sparse prefrontal montage (Heroes of the Brain 2026 dataset).",
        "n": int(len(ref)),
        "endothelial_band_share": sorted(float(v) for v in ref.endo),
        "timing_index_per_s": sorted(float(v) for v in ref.TI),
        "activity_model": {"target": "aerobic exercise > 3 h/week (questionnaire)", "endo_mean": mu, "endo_sd": sd,
                           "coef": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0]),
                           "auc_leave_one_out": auc_loo, "n_active": int(y.sum())},
        "montage": montage,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"reference n={out['n']} ({y.sum()} active), LOO AUC {auc_loo:.2f}, montage {len(montage)} pairs -> {OUT}")
