"""full_stats.py - complete statistical battery for: endothelial band (forehead HbO) vs physical activity, ages 18-42."""
import numpy as np, pandas as pd
from scipy import stats, integrate
import statsmodels.api as sm
f = pd.read_csv("results/features.csv"); b = pd.read_csv("results/bands_supine.csv")
d = f.merge(b, on="subject"); d = d[d.age < 60].dropna(subset=["l30_hbo_endothelial_rel", "activity", "age", "sex_male", "bmi"])
x = d.l30_hbo_endothelial_rel.values; a = d.activity.values; n = len(x); top = a == a.max()
rng = np.random.default_rng(0); out = {}
# 0 descriptive + normality
out["shapiro_endo_p"] = stats.shapiro(x).pvalue; out["shapiro_log_endo_p"] = stats.shapiro(np.log(x)).pvalue
out["median_top"], out["median_rest"] = np.median(x[top]), np.median(x[~top])
# 1 rank correlations + bootstrap CI
rho, p = stats.spearmanr(x, a); out["spearman"] = (rho, p)
bs = [stats.spearmanr(x[i], a[i])[0] for i in (rng.integers(0, n, n) for _ in range(5000))]
out["spearman_ci"] = np.percentile(bs, [2.5, 97.5]); out["kendall"] = stats.kendalltau(x, a)
# 2 permutation
out["perm_p"] = np.mean([abs(stats.spearmanr(rng.permutation(x), a)[0]) >= abs(rho) for _ in range(10000)])
# 3 group comparison
U = stats.mannwhitneyu(x[top], x[~top]); out["mwu_p"] = U.pvalue
cliff = (np.sum(x[top][:, None] > x[~top][None, :]) - np.sum(x[top][:, None] < x[~top][None, :])) / (top.sum() * (~top).sum())
out["cliffs_delta"] = cliff
lt, lr_ = np.log(x[top]), np.log(x[~top]); sp = np.sqrt(((len(lt)-1)*lt.var(ddof=1) + (len(lr_)-1)*lr_.var(ddof=1)) / (len(lt)+len(lr_)-2))
out["cohens_d_log"] = (lt.mean() - lr_.mean()) / sp; out["welch_t_log_p"] = stats.ttest_ind(lt, lr_, equal_var=False).pvalue
out["ratio_geo_means"] = np.exp(lt.mean() - lr_.mean())
out["kruskal_p"] = stats.kruskal(*[x[a == v] for v in np.unique(a)]).pvalue
# 4 partial correlation (ranks)
R = d[["l30_hbo_endothelial_rel", "activity", "age", "sex_male", "bmi"]].rank(); X = np.c_[np.ones(n), R[["age", "sex_male", "bmi"]].values]
res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]; out["partial"] = stats.pearsonr(res(R.iloc[:, 0].values), res(R.iloc[:, 1].values))
# 5 jackknife (leave-one-out influence) and outlier trimming
jk = np.array([stats.spearmanr(np.delete(x, i), np.delete(a, i))[0] for i in range(n)]); out["jackknife_min_max"] = (jk.min(), jk.max())
keep = (x > np.percentile(x, 5)) & (x < np.percentile(x, 95)); out["trimmed"] = stats.spearmanr(x[keep], a[keep])
# 6 robust regression (Huber) of log endo on activity + covariates
Xr = sm.add_constant(d[["activity", "age", "sex_male", "bmi"]].astype(float)); hub = sm.RLM(np.log(x), Xr, M=sm.robust.norms.HuberT()).fit()
out["huber_activity"] = (hub.params["activity"], hub.pvalues["activity"])
ols = sm.OLS(np.log(x), Xr).fit(); infl = ols.get_influence().cooks_distance[0]; out["max_cooks"] = infl.max(); out["n_cooks_gt_4n"] = int(np.sum(infl > 4 / n))
out["ols_activity"] = (ols.params["activity"], ols.pvalues["activity"], ols.rsquared)
out["breusch_pagan_p"] = sm.stats.diagnostic.het_breuschpagan(ols.resid, ols.model.exog)[1]
# 7 repeated split-half replication
reps = []
for _ in range(2000):
    pm = rng.permutation(n); A, B = pm[: n // 2], pm[n // 2:]
    rA, pA = stats.spearmanr(x[A], a[A]); rB, pB = stats.spearmanr(x[B], a[B]); reps.append((rA > 0) and (rB > 0) and (pB < 0.05))
out["split_half_rep"] = np.mean(reps)
# 8 Bayes factor (Jeffreys, uniform prior on rho; on ranks -> approximate)
r = rho
bf = integrate.quad(lambda q: 0.5 * (1 - q**2) ** ((n - 1) / 2) * (1 - q * r) ** (-(n - 1.5)), -1, 1)[0]
out["BF10"] = bf
# 9 sensitivity: smallest |rho| detectable with 80% power, alpha 0.05 (Fisher z)
out["min_detectable_rho"] = np.tanh((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / np.sqrt(n - 3))
for k, v in out.items(): print(f"{k:<22} {np.round(v, 4) if not isinstance(v, tuple) else tuple(np.round(np.asarray(v, float), 4))}")
