"""
haemo.py - raw CW light intensity (SNIRF, Cortivision Spectrum C23) -> optical density -> HbO / HbR.

Only numpy + h5py; every step is written out (no MNE), so it can be checked line by line.

  python src/haemo.py FILE.snirf                     # summary
  python src/haemo.py FILE.snirf --age 25 --out x.npz
  python src/haemo.py FILE.snirf --dpf 6             # fixed DPF instead of the age formula

Steps
  1. read intensity I(t) [n_samples, n_channels], measurementList (source, detector, wavelength),
     probe 3D positions -> source-detector distance per pair
  2. samples flagged in metaDataTags/missing_sample -> linear interpolation per channel
  3. optical density change   dOD(t) = -log10( I(t) / mean(I) )
  4. modified Beer-Lambert law per source-detector pair, both wavelengths at once:
        dOD(lambda) = [eps_HbO(lambda) * dHbO + eps_HbR(lambda) * dHbR] * d * DPF(lambda)
     eps: molar extinction coefficients (Prahl, decadic, cm^-1 / M),
     d:   source-detector distance [cm],
     DPF: differential pathlength factor, age- and wavelength-dependent
          (Scholkmann & Wolf 2013, J Biomed Opt 18(10):105004) or a fixed number
     -> 2x2 linear system solved for dHbO, dHbR [uM] at every sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

# Molar extinction coefficient [cm^-1 / M], decadic: (HbO2, HbR). Table of S. Prahl, omlc.org/spectra/hemoglobin
# (data of W. B. Gratzer and N. Kollias) - the same table MNE ships in mne/data/extinction_coef.mat.
EXTINCTION = {760: (586.0, 1548.52), 850: (1058.0, 691.32)}
SHORT_MM = 15.0  # pairs closer than this are short (skin) channels


def _str(x):
    x = x[()] if isinstance(x, h5py.Dataset) else x
    if isinstance(x, np.ndarray):
        x = x.item() if x.size == 1 else x
    return x.decode() if isinstance(x, bytes) else x


def read_snirf(path) -> dict:
    """Raw content of a SNIRF 1.1 file (first nirs/data block)."""
    with h5py.File(path, "r") as f:
        n = f["nirs"]
        d = n["data1"]
        I = d["dataTimeSeries"][()]
        t = d["time"][()]
        ml = sorted((k for k in d if k.startswith("measurementList")), key=lambda k: int(k[15:]))
        src = np.array([int(d[k]["sourceIndex"][()]) for k in ml])
        det = np.array([int(d[k]["detectorIndex"][()]) for k in ml])
        wli = np.array([int(d[k]["wavelengthIndex"][()]) for k in ml])
        p = n["probe"]
        unit = _str(n["metaDataTags"]["LengthUnit"]) if "LengthUnit" in n["metaDataTags"] else "mm"
        scale = {"mm": 1.0, "cm": 10.0, "m": 1000.0}[unit]
        tags = {k: n["metaDataTags"][k][()] for k in n["metaDataTags"]}
        aux = {_str(n[k]["name"]): n[k]["dataTimeSeries"][()].ravel() for k in n if k.startswith("aux")}
        markers = []
        for k in (k for k in n if k.startswith("stim")):
            s = n[k]["data"][()]
            if s.size:
                markers += [(float(row[0]), _str(n[k]["name"])) for row in np.atleast_2d(s)]
        return {
            "intensity": I, "time": t, "source": src, "detector": det, "wavelength_index": wli,
            "wavelengths": p["wavelengths"][()], "source_pos_mm": p["sourcePos3D"][()] * scale,
            "detector_pos_mm": p["detectorPos3D"][()] * scale,
            "source_labels": [_str(x) for x in p["sourceLabels"][()]] if "sourceLabels" in p else None,
            "detector_labels": [_str(x) for x in p["detectorLabels"][()]] if "detectorLabels" in p else None,
            "missing": tags.get("missing_sample", np.zeros(len(t))).astype(bool), "aux": aux,
            "markers": sorted(markers), "tags": tags,
        }


def fill_missing(I, missing):
    """Linear interpolation over flagged samples, channel by channel."""
    I = I.copy()
    if missing.any() and not missing.all():
        k = np.arange(len(I))
        for c in range(I.shape[1]):
            I[missing, c] = np.interp(k[missing], k[~missing], I[~missing, c])
    return I


def optical_density(I):
    """dOD = -log10(I / mean I). Non-positive intensities are clipped to the smallest positive value."""
    I = np.asarray(I, float)
    pos = I[I > 0]
    I = np.where(I > 0, I, pos.min() if pos.size else 1e-12)
    return -np.log10(I / I.mean(axis=0, keepdims=True))


def dpf_scholkmann(wavelength_nm, age_years):
    """General DPF(lambda, age) for the adult head, Scholkmann & Wolf (2013)."""
    lam, a = np.asarray(wavelength_nm, float), float(age_years)
    return 223.3 + 0.05624 * a ** 0.8493 - 5.723e-7 * lam ** 3 + 0.001245 * lam ** 2 - 0.9025 * lam


def extinction(wavelength_nm):
    w = int(round(float(wavelength_nm)))
    if w not in EXTINCTION:
        raise ValueError(f"no extinction coefficients for {w} nm (have {sorted(EXTINCTION)})")
    return EXTINCTION[w]


def convert(path, age=None, dpf=None) -> dict:
    """SNIRF -> dict with time, fs, od [n_channels, T], hbo/hbr [n_pairs, T] in uM, pair names,
    distance_mm, short (bool), dpf per wavelength, aux, markers, missing mask.
    DPF: `dpf` number if given, else the age formula (age default 25 y if unknown)."""
    r = read_snirf(path)
    I = fill_missing(r["intensity"], r["missing"])
    od = optical_density(I)
    wls = r["wavelengths"]
    if dpf is not None:
        dpfs = np.full(len(wls), float(dpf))
    else:
        dpfs = dpf_scholkmann(wls, 25.0 if age is None else age)
    pairs = sorted(set(zip(r["source"], r["detector"])))
    hbo, hbr, dist, names = [], [], [], []
    for s, d in pairs:
        idx = [np.flatnonzero((r["source"] == s) & (r["detector"] == d) & (r["wavelength_index"] == w + 1))
               for w in range(len(wls))]
        if any(len(i) != 1 for i in idx):
            continue  # pair without every wavelength
        L_mm = float(np.linalg.norm(r["source_pos_mm"][s - 1] - r["detector_pos_mm"][d - 1]))
        L_cm = L_mm / 10.0
        E = np.array([np.array(extinction(wls[w])) * L_cm * dpfs[w] for w in range(len(wls))])  # [n_wl, 2]
        y = od[:, [i[0] for i in idx]].T                                                       # [n_wl, T]
        c = np.linalg.lstsq(E, y, rcond=None)[0] * 1e6                                         # M -> uM
        hbo.append(c[0]); hbr.append(c[1]); dist.append(L_mm)
        sl = r["source_labels"][s - 1] if r["source_labels"] else f"S{s}"
        dl = r["detector_labels"][d - 1] if r["detector_labels"] else f"D{d}"
        names.append(f"{sl}_{dl}")
    t = r["time"]
    dist = np.array(dist)
    ch = [f"S{s}_D{d} {int(wls[w - 1])}" for s, d, w in zip(r["source"], r["detector"], r["wavelength_index"])]
    return {"time": t, "fs": float(1.0 / np.median(np.diff(t))), "od": od.T, "od_channels": ch,
            "hbo": np.array(hbo), "hbr": np.array(hbr), "pairs": names, "distance_mm": dist,
            "short": dist < SHORT_MM, "wavelengths": wls, "dpf": dpfs, "aux": r["aux"],
            "markers": r["markers"], "missing": r["missing"],
            "nonpositive": (r["intensity"] <= 0).sum(axis=0)}  # per OD channel: dead samples (I <= 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--age", type=float, help="age [years] for the DPF formula (default 25)")
    ap.add_argument("--dpf", type=float, help="fixed DPF for both wavelengths (overrides --age)")
    ap.add_argument("--out", help="save everything to this .npz")
    a = ap.parse_args()
    h = convert(a.file, a.age, a.dpf)
    T = h["time"][-1]
    print(f"{Path(a.file).name}: {h['hbo'].shape[1]} samples, {h['fs']:.2f} Hz, {T:.1f} s, "
          f"missing samples {100 * h['missing'].mean():.2f} % (interpolated)")
    print("DPF " + ", ".join(f"{w:.0f} nm = {d:.2f}" for w, d in zip(h["wavelengths"], h["dpf"])))
    print(f"{'pair':<8} {'dist mm':>8} {'type':>6} {'HbO sd uM':>10} {'HbR sd uM':>10}")
    for n, d, s, o, rr in zip(h["pairs"], h["distance_mm"], h["short"], h["hbo"], h["hbr"]):
        print(f"{n:<8} {d:8.1f} {'short' if s else 'long':>6} {o.std():10.3f} {rr.std():10.3f}")
    bad = [f"{c}: {n}" for c, n in zip(h["od_channels"], h["nonpositive"]) if n]
    if bad:
        print("WARNING samples with intensity <= 0 (clipped): " + ", ".join(bad))
    print(f"aux: {', '.join(h['aux'])}   markers: {len(h['markers'])}")
    if a.out:
        np.savez_compressed(a.out, **{k: v for k, v in h.items() if k not in ("aux", "markers")},
                            **{f"aux_{k}": v for k, v in h["aux"].items()},
                            markers=np.array(h["markers"], dtype=object))
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
