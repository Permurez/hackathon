"""
run_pipeline.py - whole folder of recordings -> one features table (CSV), one row per file.

Usage:
  python run_pipeline.py --data-dir DATA/ --out features.csv [--meta meta.csv] [--top-k 3]

Supported inputs: .snirf (via MNE) and .npz (keys: hbo, hbr, fs; e.g. from synthetic.py).
For each file:
  1. drop short channels (< 1.5 cm: scalp only)
  2. rank long channels by heartbeat visibility (cardiac_quality), keep the best top-k
     that pass --min-quality (pure noise scores ~1.5-2, clean channels > 15)
  3. compute subject_features() per channel and take the median across the kept channels
If --meta is given (CSV with a 'file' or 'subject' column) it is merged in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import fnirs_toolkit as fk


def load_any(path: Path) -> dict:
    if path.suffix == ".snirf":
        return fk.load_snirf_haemo(str(path))
    if path.suffix == ".npz":
        z = np.load(path)
        hbo, hbr = np.atleast_2d(z["hbo"]), np.atleast_2d(z["hbr"])
        return {"hbo": hbo, "hbr": hbr, "fs": float(z["fs"]), "short_mask": np.zeros(len(hbo), bool)}
    raise ValueError(f"unsupported file type: {path}")


def features_for_file(path: Path, top_k: int = 3, min_quality: float = 4.0) -> dict:
    d = load_any(path)
    long_idx = np.flatnonzero(~d["short_mask"])
    hbo, hbr, fs = d["hbo"][long_idx], d["hbr"][long_idx], d["fs"]
    q = np.array([fk.cardiac_quality(ch, fs) for ch in hbo])
    order = [i for i in fk.rank_channels(hbo, fs)[:top_k] if q[i] >= min_quality]
    if not order:  # nothing passes: keep the single best channel, flag it
        order = [int(np.nanargmax(q))]
    rows = [fk.subject_features(hbo[i], hbr[i], fs) for i in order]
    feats = pd.DataFrame(rows).median(numeric_only=True).to_dict()
    feats["file"] = path.name
    feats["subject"] = path.stem
    feats["n_channels_used"] = len(order)
    feats["fs"] = fs
    feats["duration_s"] = hbo.shape[1] / fs
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder with .snirf / .npz files (searched recursively)")
    ap.add_argument("--out", default="features.csv", help="output CSV path")
    ap.add_argument("--meta", default=None, help="optional CSV with age / activity per file or subject")
    ap.add_argument("--top-k", type=int, default=3, help="how many best channels to average per file")
    ap.add_argument("--min-quality", type=float, default=4.0,
                    help="cardiac_quality threshold; pure noise ~1.5-2, usable >~5, clean >~15")
    args = ap.parse_args()

    files = sorted([p for ext in ("*.snirf", "*.npz") for p in Path(args.data_dir).rglob(ext)])
    if not files:
        raise SystemExit(f"no .snirf/.npz files in {args.data_dir}")
    rows = []
    for p in files:
        try:
            rows.append(features_for_file(p, args.top_k, args.min_quality))
            print(f"ok    {p.name}")
        except Exception as e:  # keep going, report at the end
            print(f"FAIL  {p.name}: {e}")
    df = pd.DataFrame(rows)
    if args.meta:
        meta = pd.read_csv(args.meta)
        key = "file" if "file" in meta.columns else "subject"
        df = df.merge(meta, on=key, how="left")
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} rows x {df.shape[1]} columns -> {args.out}")


if __name__ == "__main__":
    main()
