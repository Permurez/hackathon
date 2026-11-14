"""
shape_age.py - how the averaged pulse SHAPE changes with age (supine 141.79 Hz baseline).

Usage:
  python shape_age.py --save shape_age.png [--cache pulse_cache.pkl]

Shape = per-person ensemble-averaged beat (stiffness.mean_pulse: trough -> trough, time scaled to
one cycle, amplitude scaled 0..1), so heart rate and signal strength are removed; only shape stays.

Panels:
  A  mean shape per age group (+- IQR)
  B  Spearman rho(age, shape value) at every point of the cycle: WHERE in the beat age matters
  C  PCA of all shapes: PC1/PC2 drawn as mean +- 2 SD (what each component changes)
  D  PC1 and PC2 score vs age
  E  Fourier: harmonic k amplitude / fundamental (H2..H6) and relative phase, rho with age
  F  beat-to-beat similarity: median correlation of single beats with the person's own mean beat
Every rho is given for all people and for age < 60 only (the 4 people aged 68-78 all take
antihypertensives, so 'old' and 'medicated' cannot be separated).
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fnirs_toolkit as fk  # noqa: E402
import stiffness as st  # noqa: E402

GROUPS = [(18, 20, "18-20"), (21, 23, "21-23"), (24, 42, "24-42"), (60, 90, "68-78")]
N_HARM = 6


def beat_similarity(path):
    """Median correlation of every single beat with the person's mean beat (best channel)."""
    try:
        d = fk.load_snirf_haemo(str(path))
        fs = d["fs"]
        long_idx = np.flatnonzero(~d["short_mask"])
        x = d["hbo"][long_idx[fk.rank_channels(d["hbo"][long_idx], fs)[0]]]
        xc = fk.bandpass(x, fs, 0.5, 10.0, order=3)
        period = 60.0 / fk.heart_rate(x, fs)
        tr, _ = signal.find_peaks(-xc, distance=int(0.6 * period * fs), prominence=0.3 * np.std(xc))
        B = []
        for a, b in zip(tr[:-1], tr[1:]):
            if 0.7 * period < (b - a) / fs < 1.3 * period:
                s = np.interp(np.linspace(a, b, 200), np.arange(a, b + 1), xc[a:b + 1])
                B.append((s - s.min()) / (np.ptp(s) + 1e-12))
        B = np.array(B)
        mean = B.mean(axis=0)
        return st.subject_id(path), float(np.median([np.corrcoef(b, mean)[0, 1] for b in B]))
    except Exception:
        return st.subject_id(path), np.nan


def harmonics(w: np.ndarray):
    """Amplitude ratio H_k/H_1 and shift-free relative phase phi_k - k*phi_1 for k = 2..N_HARM."""
    w = w - np.linspace(w[0], w[-1], len(w))  # make the cycle periodic
    X = np.fft.rfft(w[:-1])
    amp, ph = np.abs(X[1:N_HARM + 1]), np.angle(X[1:N_HARM + 1])
    k = np.arange(1, N_HARM + 1)
    rel = np.angle(np.exp(1j * (ph - k * ph[0])))
    return amp[1:] / amp[0], rel[1:]


def rho2(df, col):
    young = df["age"] < 60
    r, p = spearmanr(df["age"], df[col], nan_policy="omit")
    ry, py = spearmanr(df.loc[young, "age"], df.loc[young, col], nan_policy="omit")
    return r, p, ry, py


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(st.DEFAULT_DIR))
    ap.add_argument("--meta", default=str(st.DEFAULT_META))
    ap.add_argument("--cache", default="pulse_cache.pkl", help="per-person mean beats (shared with plot_pulse_age.py)")
    ap.add_argument("--save", default="shape_age.png")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(Path(args.data_dir).glob("*.snirf"))
    cache = Path(args.cache)
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        with Pool(args.jobs) as pool:
            df = pd.DataFrame([r for r in pool.map(st._safe, [(f, 3) for f in files]) if "_wave" in r])
        df.to_pickle(cache)
    sim_cache = cache.with_name(cache.stem + "_similarity.pkl")
    if sim_cache.exists():
        sim = pd.read_pickle(sim_cache)
    else:
        with Pool(args.jobs) as pool:
            sim = pd.DataFrame(pool.map(beat_similarity, files), columns=["subject", "beat_similarity"])
        sim.to_pickle(sim_cache)
    df = df.merge(st.load_meta(Path(args.meta)), on="subject").merge(sim, on="subject").dropna(subset=["age"])
    df = df.sort_values("age").reset_index(drop=True)
    W = np.stack(df["_wave"].to_numpy())
    ph = np.linspace(0, 1, W.shape[1])
    age = df["age"].to_numpy()
    young = age < 60

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(GROUPS)))
    cols[-1] = (0.85, 0.1, 0.1, 1)

    # A: group mean shapes
    for (lo, hi, lab), c in zip(GROUPS, cols):
        g = W[(age >= lo) & (age <= hi)]
        ax[0, 0].plot(ph, np.median(g, axis=0), color=c, lw=2, label=f"{lab} y (n={len(g)})")
        ax[0, 0].fill_between(ph, *np.percentile(g, [25, 75], axis=0), color=c, alpha=0.15)
    ax[0, 0].set(title="A  median beat shape per age group (band = IQR)", xlabel="phase of heartbeat",
                 ylabel="normalised pulse")
    ax[0, 0].legend(fontsize=8)

    # B: pointwise correlation with age
    r_all = np.array([spearmanr(age, W[:, i])[0] for i in range(W.shape[1])])
    r_y = np.array([spearmanr(age[young], W[young, i])[0] for i in range(W.shape[1])])
    ax[0, 1].plot(ph, r_all, "k", lw=2, label="all (n=%d)" % len(age))
    ax[0, 1].plot(ph, r_y, "C0", lw=1.5, label="age < 60 (n=%d)" % young.sum())
    ax[0, 1].axhline(0, color="0.5", lw=0.8)
    crit = 0.26  # |rho| for p~0.05 at n~56
    ax[0, 1].axhspan(-crit, crit, color="0.85", zorder=0, label="|rho| < ~0.26: not significant")
    ax2 = ax[0, 1].twinx()
    ax2.plot(ph, np.median(W, axis=0), color="0.6", ls=":", lw=1)
    ax2.set_yticks([])
    ax[0, 1].set(title="B  where in the beat shape follows age (rho > 0: higher with age)",
                 xlabel="phase of heartbeat (dotted = median shape)", ylabel="Spearman rho with age")
    ax[0, 1].legend(fontsize=8)

    # C/D: PCA
    Wc = W - W.mean(axis=0)
    U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    scores = U * S
    for j, c in zip(range(2), ("C1", "C2")):
        sd = scores[:, j].std()
        r, p = spearmanr(age, scores[:, j])
        sign = np.sign(r) if r != 0 else 1  # draw '+' in the direction of older age
        ax[0, 2].plot(ph, W.mean(axis=0) + sign * 2 * sd * Vt[j], color=c, lw=2,
                      label=f"PC{j + 1} ({100 * var[j]:.0f}% var) toward older")
        ax[0, 2].plot(ph, W.mean(axis=0) - sign * 2 * sd * Vt[j], color=c, lw=1, ls="--",
                      label=f"PC{j + 1} toward younger")
        df[f"pc{j + 1}"] = scores[:, j] * sign
        ra, pa, ry, py = rho2(df, f"pc{j + 1}")
        ax[1, 0].scatter(age, df[f"pc{j + 1}"] / sd, color=c, s=25,
                         label=f"PC{j + 1}: rho {ra:+.2f} (p {pa:.3f}); <60: {ry:+.2f} (p {py:.3f})")
    ax[0, 2].plot(ph, W.mean(axis=0), "k", lw=1, label="mean shape")
    ax[0, 2].set(title="C  main shape variations across people (mean +- 2 SD)", xlabel="phase of heartbeat")
    ax[0, 2].legend(fontsize=7)
    ax[1, 0].set(title="D  PC scores vs age (oriented: + = older-like)", xlabel="age [years]",
                 ylabel="score [SD]")
    ax[1, 0].legend(fontsize=7)

    # E: Fourier harmonics
    H = np.array([harmonics(w) for w in W])
    labels, rows = [], []
    for k in range(N_HARM - 1):
        df[f"amp{k + 2}"] = H[:, 0, k]
        df[f"phs{k + 2}"] = H[:, 1, k]
    for kind in ("amp", "phs"):
        for k in range(2, N_HARM + 1):
            ra, pa, ry, py = rho2(df, f"{kind}{k}")
            labels.append(f"H{k}/H1" if kind == "amp" else f"phase H{k}")
            rows.append((ra, ry, pa))
    rows = np.array(rows)
    xx = np.arange(len(labels))
    ax[1, 1].bar(xx - 0.2, rows[:, 0], 0.4, color="k", label="all")
    ax[1, 1].bar(xx + 0.2, rows[:, 1], 0.4, color="C0", label="age < 60")
    for i, p in enumerate(rows[:, 2]):
        if p < 0.05:
            ax[1, 1].text(xx[i] - 0.2, rows[i, 0], "*", ha="center", va="bottom" if rows[i, 0] > 0 else "top")
    ax[1, 1].axhspan(-crit, crit, color="0.85", zorder=0)
    ax[1, 1].set_xticks(xx, labels, rotation=45, fontsize=8)
    ax[1, 1].set(title="E  Fourier harmonics vs age (* p<0.05, all)", ylabel="Spearman rho with age")
    ax[1, 1].legend(fontsize=8)

    # F: beat-to-beat similarity
    ra, pa, ry, py = rho2(df, "beat_similarity")
    ax[1, 2].scatter(age, df["beat_similarity"], c=df["activity"], cmap="viridis", s=30)
    ax[1, 2].set(title=f"F  how alike single beats are (rho {ra:+.2f} p {pa:.3f}; <60: {ry:+.2f} p {py:.3f})",
                 xlabel="age [years]", ylabel="median r(single beat, own mean beat)")

    fig.tight_layout()
    fig.savefig(args.save, dpi=100)
    print(f"figure -> {args.save}")
    print("\nSpearman rho with age      all (p)          age<60 (p)")
    for col in ["pc1", "pc2"] + [f"amp{k}" for k in range(2, N_HARM + 1)] + \
               [f"phs{k}" for k in range(2, N_HARM + 1)] + ["beat_similarity"]:
        ra, pa, ry, py = rho2(df, col)
        print(f"  {col:<16} {ra:+.2f} ({pa:.3f})    {ry:+.2f} ({py:.3f})")
    print("PCA variance explained: " + ", ".join(f"PC{i + 1} {100 * v:.0f}%" for i, v in enumerate(var[:4])))
    i_max = int(np.argmax(np.abs(r_all)))
    print(f"strongest pointwise age effect at phase {ph[i_max]:.2f}: rho {r_all[i_max]:+.2f}")


if __name__ == "__main__":
    main()
