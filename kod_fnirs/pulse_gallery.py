"""
pulse_gallery.py - one pulse-shape figure per person + their questionnaire data, in one HTML page.

Usage:
  python pulse_gallery.py --out ~/Pictures/pulse_gallery          # then open index.html
  python pulse_gallery.py --window 120 130 --data-dir DIR --meta CSV

Each figure (same scale and time window for everyone, so shapes can be compared by eye):
  left : --window seconds of the pulse (best channels, 0.5-10 Hz), upstrokes (foot -> 90%) in red
  right: ~60 beats from that person overlaid, aligned at the foot of the wave, mean in black
The page lists people with age, sex, height, weight, activity, medication, skin and hair next to
the figure and can be sorted by any of them.
"""

from __future__ import annotations

import argparse
import html
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fnirs_toolkit as fk  # noqa: E402
import stiffness as st  # noqa: E402
import upstroke as up  # noqa: E402

COLS = {"Nr osoby badanej": "subject", "Wiek": "age", "Płeć": "sex", "Wzrost": "height_cm", "Waga": "weight_kg",
        "Ocena aktywności fizycznej": "activity", "Przyjmowane leki": "meds", "Fitzpatrick": "fitzpatrick",
        "Kolor włosów": "hair_colour", "Typ włosów": "hair_type"}
OVERLAY_MS = 1300


def plot_one(job):
    path, out_png, title, (w0, w1) = job
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        d = fk.load_snirf_haemo(str(path))
        fs = d["fs"]
        x, chans = up.pulse(d, 3)
        B = up.beats(x, fs)
    except Exception as e:
        return str(path), f"error: {e}"
    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots(1, 2, figsize=(15, 3.2), gridspec_kw={"width_ratios": [3, 1]})
    m = (t >= w0) & (t <= w1)
    ax[0].plot(t[m], x[m], "k", lw=1)
    for a, k, _, _ in B:
        a, k = int(a), int(k)
        if w0 <= a / fs <= w1:
            ax[0].plot(t[a:k + 1], x[a:k + 1], "C3", lw=2.5)
    ax[0].set_ylim(-3, 2.5)
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("pulse (z)")
    n = int(OVERLAY_MS / 1000 * fs)
    sel = [int(a) for a, *_ in B if int(a) + n < len(x)]
    sel = sel[:: max(1, len(sel) // 60)]
    tt = np.arange(n) / fs * 1000
    segs = np.array([x[a:a + n] - x[a] for a in sel])
    for s in segs:
        ax[1].plot(tt, s, color="0.6", lw=0.5, alpha=0.5)
    if len(segs):
        ax[1].plot(tt, np.median(segs, axis=0), "k", lw=2)
    ax[1].set_ylim(-1, 4.5)
    ax[1].set_xlabel("ms from foot of the wave")
    ax[1].set_title("beats overlaid", fontsize=9)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    return str(path), "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(st.DEFAULT_DIR))
    ap.add_argument("--meta", default=str(st.DEFAULT_META))
    ap.add_argument("--out", default="pulse_gallery")
    ap.add_argument("--window", type=float, nargs=2, default=(120, 130), metavar=("T0", "T1"))
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.meta, decimal=",").rename(columns=COLS)[list(COLS.values())]
    files = sorted(Path(args.data_dir).glob("*.snirf"))
    rows, jobs = [], []
    for f in files:
        sid = st.subject_id(f)
        r = meta[meta["subject"] == sid]
        r = r.iloc[0].to_dict() if len(r) else {"subject": sid}
        png = f"{sid:02d}.png"
        title = (f"#{sid}  |  {r.get('age', '?')} y  |  {r.get('sex', '?')}  |  activity {r.get('activity', '?')}  |  "
                 f"meds: {r.get('meds', '?')}   ({f.name})")
        jobs.append((f, out / png, title, tuple(args.window)))
        rows.append({**r, "png": png, "file": f.name})
    with Pool(args.jobs) as pool:
        for path, status in pool.imap_unordered(plot_one, jobs):
            if status != "ok":
                print(f"{Path(path).name}: {status}")
    write_html(out / "index.html", rows, args.window)
    print(f"{len(rows)} people -> {out / 'index.html'}")


def write_html(path: Path, rows: list[dict], window) -> None:
    fields = [c for c in COLS.values() if c != "subject"]
    items = []
    for r in rows:
        attrs = " ".join(f'data-{k}="{html.escape(str(r.get(k, "")))}"' for k in ["subject"] + fields)
        cells = "".join(f"<tr><th>{k}</th><td>{html.escape(str(r.get(k, '')))}</td></tr>" for k in fields)
        items.append(f'<div class="p" {attrs}><table><tr><th>subject</th><td><b>#{r["subject"]}</b></td></tr>'
                     f'{cells}</table><img src="{r["png"]}" loading="lazy"></div>')
    buttons = "".join(f'<button onclick="srt(\'{k}\')">{k}</button>' for k in ["subject"] + fields)
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>Pulse gallery</title>
<style>body{{font:13px sans-serif;margin:12px;background:#fafafa}}
.p{{display:flex;gap:10px;align-items:center;background:#fff;border:1px solid #ddd;margin:6px 0;padding:6px}}
table{{min-width:190px;border-collapse:collapse}}th{{text-align:left;color:#666;padding-right:8px;font-weight:normal}}
img{{max-width:calc(100% - 210px)}}button{{margin:2px}}</style>
<h3>Pulse shape per person (supine baseline, {window[0]:.0f}-{window[1]:.0f} s, red = upstroke foot to 90%)</h3>
<div>sort by: {buttons} <label><input type="checkbox" id="desc"> descending</label></div>
<div id="list">{''.join(items)}</div>
<script>
function srt(k){{const L=document.getElementById('list'),d=document.getElementById('desc').checked?-1:1;
[...L.children].sort((a,b)=>{{const x=a.dataset[k],y=b.dataset[k],nx=parseFloat(x),ny=parseFloat(y);
return d*(isNaN(nx)||isNaN(ny)?x.localeCompare(y):nx-ny)}}).forEach(e=>L.appendChild(e))}}
srt('age');
</script>""", encoding="utf-8")


if __name__ == "__main__":
    main()
