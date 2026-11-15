"""fnirs_vascular - vascular markers from a 10-min resting-state fNIRS recording (SNIRF in, JSON-ready dict out).

    from fnirs_vascular import analyze
    result = analyze("rest.snirf")          # or analyze(mne.io.read_raw_snirf("rest.snirf"))

    python -m fnirs_vascular rest.snirf                 # JSON to the terminal
    python -m fnirs_vascular data_folder/ --out res.csv # every .snirf in a folder (recursive) -> CSV
"""
from .core import BAND_HZ, DISCLAIMER, __version__, analyze, analyze_many, find_recordings, reference

__all__ = ["analyze", "analyze_many", "find_recordings", "reference", "BAND_HZ", "DISCLAIMER", "__version__"]
