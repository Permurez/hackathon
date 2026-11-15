"""simulate_future.py - what would the next studies have to show for the band to matter?
Simulations on synthetic (mock) data whose parameters come from our data:
  observed Spearman rho band-activity 0.46 (n = 55), questionnaire answer proportions (ceiling: 54 % at the top),
  one-recording reliability ICC 0.56 (module, DOT) / 0.36 (all long channels), within-person SD 10.3 pp per recording,
  median gap trained vs untrained 12.5 pp.

  S0 null check     how often does |rho| >= 0.46 appear at n = 55 with no effect at all (x 1350 screened tests)
  S1 confirmation   power to confirm the link on new people vs n, if the true effect is ours / 2/3 / 1/2 of it,
                    with the same questionnaire and with an objective measure (VO2max-like, reliability 0.9)
  S2 one person     best possible AUC for "trains > 3 h/week" (huge mock sample, perfect model) vs number of
                    averaged recordings, for several true-score correlations
  S3 training       power to show that the band rises after training vs n per group (3 recordings per time point)
  S4 validation     power to show agreement r > 0.3 with laser Doppler / FMD when the true r is 0.5-0.7

  python src/simulate_future.py      -> criteria printed, fig_future_studies.png for the slides
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(2026)
RHO_OBS, N_OBS = 0.46, 55
ICC_MODULE, ICC_ALL = 0.56, 0.36
SW, GAP = 10.3, 12.5                       # pp: within-person SD of one recording; trained vs untrained medians
BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal ramp, light -> dark = weak -> strong
INK, MUTED = "#14213d", "#5b6472"

meta = load_meta()
props = meta[meta.age < 60].activity.value_counts(normalize=True).sort_index().to_numpy()
CUTS = stats.norm.ppf(np.cumsum(props)[:-1])   # latent activity -> 5 questionnaire answers, same proportions as ours
TOP = 1 - props[-1]                            # share below the top answer


def spearman_rows(x, y):
    rx, ry = stats.rankdata(x, axis=1), stats.rankdata(y, axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def p_one_sided(r, n):
    t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    return stats.t.sf(t, n - 2)


def mock(n, rho, reps, measure="questionnaire"):
    """band and activity for reps x n mock people; latent correlation rho."""
    a = rng.standard_normal((reps, n))
    x = rho * a + np.sqrt(1 - rho ** 2) * rng.standard_normal((reps, n))
    if measure == "questionnaire":
        y = np.digitize(a, CUTS).astype(float)
    else:                                       # objective measure with reliability 0.9
        y = np.sqrt(0.9) * a + np.sqrt(0.1) * rng.standard_normal((reps, n))
    return x, y


# ---------------------------------------------------------------- S0
r0 = spearman_rows(*mock(N_OBS, 0.0, 100_000))
p0 = float(np.mean(np.abs(r0) >= RHO_OBS))
print(f"S0  no effect, n = {N_OBS}: P(|rho| >= {RHO_OBS}) = {p0:.5f} -> about {1350 * p0:.1f} such hits expected "
      "among 1350 independent screened tests (ours: 1, and it matches the literature band)")

# ---------------------------------------------------------------- calibrate: which latent rho gives ours?
grid = np.round(np.arange(0.40, 0.71, 0.01), 2)
obs = [spearman_rows(*mock(N_OBS, g, 3000)).mean() for g in grid]
RHO_LAT = float(grid[np.argmin(np.abs(np.array(obs) - RHO_OBS))])
print(f"\ncalibration: latent rho {RHO_LAT:.2f} reproduces our questionnaire rho {RHO_OBS} at n = {N_OBS}")

# ---------------------------------------------------------------- S1
NS = np.array([20, 30, 40, 55, 70, 85, 100, 130, 160, 200])
scen1 = {"jak u nas": RHO_LAT, "o 1/3 słabszy": RHO_LAT * 2 / 3, "o połowę słabszy": RHO_LAT / 2}
pow1 = {}
print("\nS1  confirmation on new people: power (one-sided p < 0.05)")
print(f"    {'true effect':<18}{'measure':<15}" + "".join(f"{n:>6}" for n in NS))
for lab, rho in scen1.items():
    for meas in ("questionnaire", "objective"):
        pw = [np.mean(p_one_sided(spearman_rows(*mock(n, rho, 2000, meas)), n) < 0.05) for n in NS]
        pow1[(lab, meas)] = np.array(pw)
        print(f"    {lab:<18}{meas:<15}" + "".join(f"{p:6.2f}" for p in pw))


def n_for(pw, target=0.8):
    ok = np.flatnonzero(pw >= target)
    return int(NS[ok[0]]) if len(ok) else None


# ---------------------------------------------------------------- S2
KS = np.arange(1, 9)
scen2 = [0.4, 0.55, 0.7, 0.85]
N_BIG = 200_000
auc2 = {}
print("\nS2  best possible AUC for 'trains > 3 h/week' (perfect model, huge sample) vs averaged recordings, ICC 0.56")
print(f"    {'true-score rho':<16}" + "".join(f"{k:>6}" for k in KS))
a = rng.standard_normal(N_BIG)
active = a > stats.norm.ppf(TOP)
for rt in scen2:
    t = rt * a + np.sqrt(1 - rt ** 2) * rng.standard_normal(N_BIG)      # the person's stable band value
    row = []
    for k in KS:
        noise = rng.standard_normal((k, N_BIG)).mean(0) * np.sqrt((1 - ICC_MODULE) / ICC_MODULE)
        x = t + noise
        row.append(stats.mannwhitneyu(x[active], x[~active]).statistic / (active.sum() * (~active).sum()))
    auc2[rt] = np.array(row)
    print(f"    {rt:<16.2f}" + "".join(f"{v:6.2f}" for v in row))
rt_ours = min(RHO_LAT / np.sqrt(ICC_MODULE), 0.99)
print(f"    our data imply a true-score rho of about {rt_ours:.2f} (latent {RHO_LAT:.2f} / sqrt(ICC {ICC_MODULE}))")

# ---------------------------------------------------------------- S3
NG = np.array([10, 15, 20, 30, 40, 50, 60, 80])
scen3 = [3.0, 6.0, 9.0, 12.0]
TAU, K3 = 5.0, 3
pow3 = {}
print(f"\nS3  training vs control: power (one-sided p < 0.05) vs n per group; {K3} recordings before and after, "
      f"within-person SD {SW} pp, individual response SD {TAU} pp")
print(f"    {'true rise [pp]':<16}" + "".join(f"{n:>6}" for n in NG))
for d in scen3:
    pw = []
    for n in NG:
        se = SW * np.sqrt(2 / K3)
        tr = d + TAU * rng.standard_normal((2000, n)) + se * rng.standard_normal((2000, n))
        co = se * rng.standard_normal((2000, n))
        pw.append(np.mean(stats.ttest_ind(tr, co, axis=1, equal_var=False, alternative="greater").pvalue < 0.05))
    pow3[d] = np.array(pw)
    print(f"    {d:<16.0f}" + "".join(f"{p:6.2f}" for p in pw))
for k in (1, 3, 6):
    print(f"    one person, {k} recording(s) before and after: change must exceed {1.96 * SW * np.sqrt(2 / k):.1f} pp "
          "to be more than noise")

# ---------------------------------------------------------------- S4
NV = np.array([15, 20, 30, 40, 60, 80])
print("\nS4  validation vs laser Doppler / FMD: power to show r > 0.3 (Fisher z, one-sided 0.05)")
print(f"    {'true r':<10}" + "".join(f"{n:>6}" for n in NV))
for r in (0.5, 0.6, 0.7):
    pw = []
    for n in NV:
        z = rng.standard_normal((4000, n))
        x = r * z + np.sqrt(1 - r ** 2) * rng.standard_normal((4000, n))
        rs = np.array([np.corrcoef(z[i], x[i])[0, 1] for i in range(4000)])
        pw.append(np.mean((np.arctanh(rs) - np.arctanh(0.3)) * np.sqrt(n - 3) > 1.645))
    print(f"    {r:<10.1f}" + "".join(f"{p:6.2f}" for p in pw))

# ---------------------------------------------------------------- figure for the slides
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for i, (lab, _) in enumerate(scen1.items()):
    pw = pow1[(lab, "questionnaire")]
    ax[0].plot(NS, pw, "o-", color=BLUES[-1 - i], lw=2, ms=5, label=f"efekt {lab}")
ax[0].axhline(0.8, color=MUTED, ls="--", lw=1)
ax[0].set(xlabel="liczba nowych osób", ylabel="szansa potwierdzenia (moc)", ylim=(0, 1.02),
          title="1. Potwierdzenie na nowych osobach")
for i, rt in enumerate(scen2):
    ax[1].plot(KS, auc2[rt], "o-", color=BLUES[i], lw=2, ms=5, label=f"prawdziwy związek {rt:.2f}".replace(".", ","))
ax[1].axhline(0.8, color=MUTED, ls="--", lw=1)
ax[1].text(8, 0.792, "dobra trafność dla jednej osoby (0,8)", ha="right", va="top", fontsize=8, color=MUTED)
ax[1].plot([1], [0.67], "D", color=INK, ms=7, zorder=5)
ax[1].annotate("my dziś: 1 nagranie, AUC 0,67", (1, 0.67), xytext=(1.6, 0.60), fontsize=8, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
ax[1].set(xlabel="liczba uśrednionych nagrań jednej osoby", ylabel="najlepsze możliwe AUC", ylim=(0.5, 1.0),
          title="2. Czy da się ocenić jedną osobę?")
for i, d in enumerate(scen3):
    ax[2].plot(NG, pow3[d], "o-", color=BLUES[i], lw=2, ms=5, label=f"wzrost o {d:.0f} pp")
ax[2].axhline(0.8, color=MUTED, ls="--", lw=1)
ax[2].set(xlabel="osób w grupie (trening i kontrola)", ylabel="szansa wykazania wzrostu (moc)", ylim=(0, 1.02),
          title=f"3. Badanie treningowe ({K3} nagrania przed i po)")
for a_ in ax:
    a_.legend(fontsize=8, frameon=False, loc="lower right")
    a_.title.set_color(INK)
fig.suptitle("Symulacje na danych syntetycznych z parametrami z naszych danych: co musi wyjść w kolejnych badaniach",
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(ROOT / "results/fig_future_studies.png", dpi=150)
print(f"\n-> {ROOT / 'results/fig_future_studies.png'}")
print(f"\nsummary: n for 80% power (questionnaire / objective): " + "; ".join(
    f"{lab}: {n_for(pow1[(lab, 'questionnaire')])} / {n_for(pow1[(lab, 'objective')])}" for lab in scen1))
