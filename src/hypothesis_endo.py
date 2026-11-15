"""hypothesis_endo.py - formal test battery for H1: the endothelial band (0.0095-0.021 Hz) determines weekly activity."""
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
f = pd.read_csv("features.csv"); b = pd.read_csv("bands_supine.csv")
mv = pd.read_csv("multiverse_endothelial.csv")
df = f.merge(b, on="subject")
E = "l30_hbo_endothelial_rel"; rng = np.random.default_rng(0)

def jonckheere(x, g):
    lv = np.sort(np.unique(g)); S = 0
    for i in range(len(lv)):
        for j in range(i + 1, len(lv)):
            a, c = x[g == lv[i]], x[g == lv[j]]; S += np.sum(c[:, None] > a[None, :]) - np.sum(c[:, None] < a[None, :])
    null = []
    for _ in range(5000):
        gp = rng.permutation(g); s = 0
        for i in range(len(lv)):
            for j in range(i + 1, len(lv)):
                a, c = x[gp == lv[i]], x[gp == lv[j]]; s += np.sum(c[:, None] > a[None, :]) - np.sum(c[:, None] < a[None, :])
        null.append(s)
    return S, np.mean(np.array(null) >= S)

for label, d in (("18-42 (main)", df[df.age < 60]), ("all", df)):
    d = d.dropna(subset=[E, "activity", "age", "sex_male", "bmi"]).copy()
    x, a = d[E].values, d.activity.values; y = (a == a.max()).astype(int)
    print(f"\n================ {label}: n = {len(d)} ================")
    # 1. dose-response
    tau, pt = stats.kendalltau(x, a); S, pj = jonckheere(x, a)
    print(f"1. Kendall tau-b {tau:+.2f} (p {pt:.4f}); Jonckheere-Terpstra trend S = {S}, one-sided perm p {pj:.4f}")
    print("   medians by level:", {float(k): round(v, 3) for k, v in d.groupby('activity')[E].median().items()})
    # 2. ordinal regression, LR test
    lev = np.sort(np.unique(a)); yo = pd.Series(pd.Categorical(a, categories=lev, ordered=True))
    Xb = d[["age", "sex_male", "bmi"]].astype(float); Xb = (Xb - Xb.mean()) / Xb.std()
    Xf = Xb.assign(endo=(d[E] - d[E].mean()) / d[E].std())
    m0 = OrderedModel(yo.values, Xb.values, distr="logit").fit(method="bfgs", disp=False)
    m1 = OrderedModel(yo.values, Xf.values, distr="logit").fit(method="bfgs", disp=False)
    lr = 2 * (m1.llf - m0.llf); plr = stats.chi2.sf(lr, 1)
    or_ = np.exp(m1.params[3]); ci = np.exp(np.asarray(m1.conf_int())[3])
    print(f"2. ordinal logit: endo OR per 1 SD = {or_:.2f} [95% CI {ci[0]:.2f}-{ci[1]:.2f}]; LR test vs demographics chi2 {lr:.2f}, p {plr:.4f}")
    null_ll = OrderedModel(yo.values, np.zeros((len(d), 1)) + rng.normal(0, 1e-6, (len(d), 1)), distr="logit").fit(method="bfgs", disp=False).llf
    print(f"   McFadden pseudo-R2: demographics {1 - m0.llf/null_ll:.3f} -> + endo {1 - m1.llf/null_ll:.3f}")
    # 3. goodness of fit (binary model >3 h/week): Hosmer-Lemeshow
    lg = sm.Logit(y, sm.add_constant(Xf.values)).fit(disp=False); ph = lg.predict()
    grp = pd.qcut(ph, 5, labels=False, duplicates="drop"); hl = 0
    for g_ in np.unique(grp):
        m = grp == g_; o, e, n = y[m].sum(), ph[m].sum(), m.sum(); hl += (o - e) ** 2 / (e * (1 - e / n) + 1e-12)
    print(f"3. Hosmer-Lemeshow (5 groups) chi2 {hl:.2f}, p {stats.chi2.sf(hl, len(np.unique(grp)) - 2):.3f}  (p > 0.05 = no evidence of misfit)")
    # 4. out-of-sample prediction
    def loo(cols, yy=y):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return cross_val_predict(mdl, d[cols].values, yy, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    pb, pf, pe = loo(["age", "sex_male", "bmi"]), loo(["age", "sex_male", "bmi", E]), loo([E])
    def auc_ci(p):
        bs = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if 0 < y[i].sum() < len(i)]
        return roc_auc_score(y, p), np.percentile(bs, [2.5, 97.5])
    for nm, p in (("demographics", pb), ("demographics + endo", pf), ("endo alone", pe)):
        au, ci = auc_ci(p); print(f"4. LOO AUC {nm:<20} {au:.2f} [95% CI {ci[0]:.2f}-{ci[1]:.2f}]")
    fpr, tpr, thr = roc_curve(y, pf); k = np.argmax(tpr - fpr)
    print(f"   at best threshold: sensitivity {tpr[k]:.2f}, specificity {1-fpr[k]:.2f}; accuracy {np.mean((pf >= thr[k]) == y):.2f} vs guessing majority {max(y.mean(), 1-y.mean()):.2f}")
    # exact level prediction: ordinal model LOO, accuracy vs majority
    hits = []
    for i in range(len(d)):
        tr = np.arange(len(d)) != i
        try:
            mm = OrderedModel(yo.values[tr], Xf.values[tr], distr="logit").fit(method="bfgs", disp=False)
            hits.append(lev[np.argmax(mm.predict(Xf.values[[i]])[0])] == a[i])
        except Exception: hits.append(False)
    print(f"   exact activity LEVEL (5 levels), LOO accuracy {np.mean(hits):.2f} vs always guessing the commonest level {np.mean(a == pd.Series(a).mode()[0]):.2f}")
    # 5. falsification
    print("5. falsification (Spearman with activity; should be ~0 except the endothelial band):")
    for c in ["l30_hbo_respiratory_rel", "l30_hbo_cardiac_rel", "l30_hbo_myogenic_rel", "l30_hbo_neurogenic_rel", E]:
        r, pv = stats.spearmanr(d[c], a); print(f"   {c:<28} rho {r:+.2f} p {pv:.3f}")
    r, pv = stats.spearmanr(x, d.NEG_CONTROL_group); print(f"   endo vs NEG control (Grupa)    rho {r:+.2f} p {pv:.3f}")
hair = mv[(mv.ch.isin(["S1_D1", "S2_D1", "S3_D2", "S4_D2"])) & (mv.chrom == "hbo") & (mv.win == 300) & (mv.edge == "0.0095-0.021")]
print("\n5b. same band in channels UNDER HAIR (all 59; should fail if the signal is skin/forehead-specific):",
      ", ".join(f"{r.ch} rho {r.rho:+.2f} p {r.p:.2f}" for r in hair.itertuples()))
