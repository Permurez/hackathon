#!/usr/bin/env bash
# Pobiera nagrania fNIRS Cortivision (~5 GB) z Google Drive do data/cortivision.
# Nagrania nie są w repo — każdy pobiera je lokalnie tym skryptem.
set -euo pipefail
cd "$(dirname "$0")"
command -v gdown >/dev/null || pip install --user gdown
gdown --folder --continue --retries 5 -O data/cortivision 1996bVSior9vFjLcnzwqsx6tNbm0gAzYm
