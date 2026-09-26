#!/usr/bin/env bash
#
# Run a PythonVision algorithm over the captures in its own input folder.
#
# Each algorithm is a folder with the same shape:
#
#   PythonVision/
#       common/         shared geometry, image io and drawing
#       script.py       the shared harness
#       run.sh          this file
#       aruco/features/gradient/
#           input/      FieldBW.png + capture_*.png
#           out/        results are written here
#           <name>.py   the algorithm
#
# Usage:
#   ./run.sh aruco                    # every capture in aruco/input
#   ./run.sh features --write-steps    # algorithm-specific flags pass through
#   ./run.sh gradient capture_x.png    # one specific capture
#   ./run.sh aruco --list              # list the captures it would process
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
VENV_PY="$PROJECT/.venv/bin/python"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

ALGO="$1"
shopt -s nullglob
ALGO_DIRS=("$HERE"/*/)
shopt -u nullglob
if [[ ! -d "$HERE/$ALGO" ]]; then
    echo "error: no algorithm folder '$ALGO' in $HERE" >&2
    echo "available: $(for d in "${ALGO_DIRS[@]}"; do
        b="$(basename "$d")"; [[ -f "$d/$b.py" ]] && printf '%s ' "$b"; done)" >&2
    exit 2
fi

if [[ ! -x "$VENV_PY" ]]; then
    echo "error: no Python environment at $VENV_PY" >&2
    echo "create one with:" >&2
    echo "  python3 -m venv \"$PROJECT/.venv\"" >&2
    echo "  \"$PROJECT/.venv/bin/pip\" install opencv-contrib-python numpy matplotlib" >&2
    exit 1
fi

# The harness resolves input/ and out/ relative to its own folder, so it can be
# run from anywhere; keeping the caller's cwd is only for relative --map paths.
exec "$VENV_PY" "$HERE/script.py" "$@"
