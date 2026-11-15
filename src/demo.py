"""
demo.py - the same analysis as the Streamlit app, from the terminal.

  python src/demo.py dane/infopack_and_data/fnirs_high_freq_supine/data/baseline/17_PK_BS_2025-08-11-12-19-59.snirf --show

Prints both vascular markers and saves a one-page card (demo_<file>.png); --show opens it in the image viewer.
Group-level markers, not a diagnosis.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vascular as V  # noqa: E402


def main():
    path = Path(sys.argv[1])
    print(f"[1/3] {path.name}: raw light -> HbO ...")
    res = V.analyse(path)
    print(f"[2/3] spectrum + {res['n_beats']} averaged heartbeats ...")
    coh = V.cohort()
    sc = V.score(res, coh)
    print("[3/3] compared with the healthy reference group (18-42 y, no medication)\n")
    for line in V.interpret(res, sc):
        print(" -", line)
    print(f"\n {V.DISCLAIMER}")
    out = V.ROOT / f"demo_{path.stem}.png"
    V.fig_card(res, coh, sc).savefig(out, dpi=120)
    print(f" figure -> {out}")
    if "--show" in sys.argv:
        subprocess.Popen(["xdg-open", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
