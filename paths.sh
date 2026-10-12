# Where this machine keeps its python, its models and its image sets.
#
# The scripts here were written against one machine and carried its absolute paths, which made them
# unreadable to anyone else and published a drive layout besides. Everything now comes from here,
# and every value can be overridden from the environment:
#
#   LAB_PY=/usr/bin/python3 LAB_MODELS=~/models LAB_SETS=~/sets ./night.sh
#
# Copy paths.local.sh.example to paths.local.sh to set them once; that file is not tracked.

PY="${LAB_PY:-python}"
OUT="${LAB_MODELS:-$HOME/latent-lab/models}"
SETS="${LAB_SETS:-$HOME/latent-lab/sets}"
LAB="${LAB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

[ -f "$LAB/paths.local.sh" ] && . "$LAB/paths.local.sh"
mkdir -p "$OUT"
