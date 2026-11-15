"""Command line: python -m fnirs_vascular FILE_OR_FOLDER [...] [--out results.json|results.csv] [--channels A,B]"""
import argparse
import csv
import json
import sys

from .core import analyze_many


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, list):
            out[key] = "; ".join(map(str, v))
        else:
            out[key] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fnirs_vascular", description="Vascular markers from resting-state fNIRS.")
    ap.add_argument("paths", nargs="+", help=".snirf files and/or folders (searched recursively)")
    ap.add_argument("--out", help="write results to .json or .csv")
    ap.add_argument("--channels", help="comma-separated pair names for the band, e.g. S5_D3,S5_D4")
    a = ap.parse_args(argv)
    res = analyze_many(a.paths, a.channels.split(",") if a.channels else None)
    if not res:
        sys.exit("no .snirf files found")
    if a.out and a.out.endswith(".csv"):
        rows = [_flat(r) for r in res]
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    elif a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    if len(res) == 1 and not a.out:
        print(json.dumps(res[0], indent=1, ensure_ascii=False))
        return
    print(f"{'file':<40} {'HR':>5} {'band':>6} {'pct':>5} {'P(>3h)':>7} {'TI':>6} {'pct':>5}  warnings")
    for r in res:
        if "error" in r:
            print(f"{r['file']:<40} ERROR {r['error']}")
            continue
        v, t = r["vascular_regulation"], r["arterial_timing"]
        s = lambda x, f: "-" if x is None else format(x, f)  # noqa: E731
        print(f"{r['file']:<40} {r['heart_rate_bpm']:5.0f} {v['endothelial_band_share']:6.3f} "
              f"{s(v['percentile_vs_reference'], '5.0f')} {s(v['prob_aerobic_gt_3h_week'], '7.2f')} "
              f"{s(t['timing_index_per_s'], '6.2f')} {s(t['percentile_vs_reference'], '5.0f')}  {len(r['warnings'])}")
    if a.out:
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
