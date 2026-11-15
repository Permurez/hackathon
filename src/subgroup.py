"""subgroup.py - the full correlation map recomputed inside one pre-chosen age group (default 18-42 y: everyone
except the 4 people aged 68-78, who all take antihypertensives)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, false_discovery_control
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_correlations as pc
ROOT = Path(__file__).resolve().parents[1]
LO, HI = 18, 59
f = pd.read_csv(ROOT / "results/features.csv")
b = pd.read_csv(ROOT / "results/bands_supine.csv")[["subject", "l30_hbo_endothelial_rel", "l30_hbo_neurogenic_rel", "l30_hbo_myogenic_rel"]]
t = pd.read_csv(ROOT / "results/timing_index.csv")[["subject", "TI", "CRF"]]
df = f.merge(b, on="subject", how="left").merge(t, on="subject", how="left")
df = df[df.age.between(LO, HI)]
meta = ["sex_male", "age", "height", "weight", "bmi", "head_circ", "hair_darkness", "hair_curl", "hair_thickness",
        "fitzpatrick", "medication_any", "antihypertensive", "activity", "treadmill_kmh", "NEG_CONTROL_group", "CRF"]
feats = [c for c in df if c not in meta + ["subject"] and df[c].dtype != object]
rows = []
for tg in meta:
    for ft in feats:
        d = df[list(dict.fromkeys([ft, tg]))].dropna()
        if len(d) < 30 or d[tg].nunique() < 2 or d[ft].nunique() < 3:
            continue
        r, p = spearmanr(d[ft], d[tg]); rows.append(dict(feature=ft, target=tg, rho=r, p=p, n=len(d)))
R = pd.DataFrame(rows).dropna(subset=["p"]); R["q_fdr"] = false_discovery_control(R.p)
R.to_csv(ROOT / "results/correlations_18_42.csv", index=False)
print(f"age {LO}-{HI}: n = {len(df)}; {len(R)} tests; p<0.05 {(R.p < .05).sum()} (chance {0.05 * len(R):.0f}); q<0.10 {(R.q_fdr < .1).sum()}")
print(R[R.q_fdr < .1].round(3).to_string(index=False))
neg = R[R.target == "NEG_CONTROL_group"]; print(f"NEG control min q {neg.q_fdr.min():.2f}")
print("our two findings here:", R[(R.feature.isin(["TI", "l30_hbo_endothelial_rel"])) & R.target.isin(["age", "activity", "CRF"])].round(3).to_string(index=False))
ROWS = [("Naczynia\n(hipotezy z góry)", [("TI", "indeks fali odbitej (TI)"), ("l30_hbo_endothelial_rel", "pasmo śródbłonkowe"),
                                         ("l30_hbo_neurogenic_rel", "pasmo neurogenne"), ("l30_hbo_myogenic_rel", "pasmo miogenne")])] + pc.ROWS
pc.COLS = [(s, [(k, l) for k, l in items if k not in ("age_lt60", "antihypertensive")]) for s, items in pc.COLS]
pc.COLS[2] = ("Styl życia", pc.COLS[2][1] + [("CRF", "wydolność (wzór Jurca)")])
pc.draw(ROWS, "results/correlations_18_42.png", f"Grupa 18–42 lata (n = {len(df)}): parametry fNIRS × cechy osób", csv="results/correlations_18_42.csv")
