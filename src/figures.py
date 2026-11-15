"""figures.py - the two result figures for the slides (ages 18-42, n = 55)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
f = pd.read_csv(ROOT / "features.csv")[["subject", "age", "activity"]]
d = f.merge(pd.read_csv(ROOT / "bands_supine.csv")[["subject", "l30_hbo_endothelial_rel"]], on="subject") \
     .merge(pd.read_csv(ROOT / "timing_index.csv")[["subject", "TI"]], on="subject")
d = d[d.age < 60]
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#d97706"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
# 1. TI vs age: healthy people without any medication (TI from ti_followup.py)
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_meta
from scipy.stats import pearsonr
t = pd.read_csv(ROOT / "ti_followup.csv")[["subject", "TI_full"]].merge(load_meta(), on="subject")
t = t[(t.age < 60) & (t.medication_any == 0)].dropna(subset=["TI_full"])
fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor="white")
ax.scatter(t.age, t.TI_full, s=36, color=BLUE, alpha=0.8, edgecolor="white", lw=0.6)
k = np.polyfit(t.age, t.TI_full, 1); xs = np.linspace(t.age.min(), t.age.max(), 50); ax.plot(xs, np.polyval(k, xs), color=INK, lw=2)
r, p = pearsonr(t.age, t.TI_full)
ax.set_xlabel("wiek [lata]"); ax.set_ylabel("indeks fali odbitej TI [1/s]")
ax.set_title(f"Fala odbita wraca szybciej u starszych\nr = {r:+.2f}, p = {p:.3f}, n = {len(t)}, zdrowi bez leków", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(ROOT / "fig_ti_age.png", dpi=150)
# 2. endothelial band by activity level (Polish and English versions)
def endo_figure(lab, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor="white")
    lv = sorted(d.activity.unique())
    for i, a in enumerate(lv):
        y = 100 * d[d.activity == a].l30_hbo_endothelial_rel
        ax.scatter(np.full(len(y), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(y)), y, s=30, color=ORANGE if a == lv[-1] else MUTED, alpha=0.75)
        ax.plot([i - 0.25, i + 0.25], [y.median()] * 2, color=INK, lw=2.5)
        ax.text(i, y.max() + 2, f"n={len(y)}", ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(lv)), [lab.get(a, a) for a in lv], fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(ROOT / out, dpi=150)

r, p = spearmanr(d.activity, d.l30_hbo_endothelial_rel)
endo_figure({0.0: "brak", 0.32: "lekka\nregularnie", 1.06: "aeroby\n<1 h/tydz", 1.76: "aeroby\n1–3 h/tydz", 3.03: "aeroby\n>3 h/tydz"},
            "moc pasma śródbłonkowego [% mocy]", f"Silniejsze falowanie naczyń u trenujących (ρ = {r:+.2f}, p < 0,001)",
            "fig_endothelial_activity.png")
endo_figure({0.0: "none", 0.32: "light,\nregularly", 1.06: "aerobic\n<1 h/week", 1.76: "aerobic\n1–3 h/week", 3.03: "aerobic\n>3 h/week"},
            "slow vessel waves [% of signal]", f"Stronger vessel waves in people who train (ρ = {r:+.2f}, p < 0.001)",
            "fig_endothelial_activity_en.png")
print("-> fig_ti_age.png, fig_endothelial_activity.png, fig_endothelial_activity_en.png")
