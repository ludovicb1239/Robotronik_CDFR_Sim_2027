"""Where an algorithm's files live.

Every algorithm is a folder beside this package with the same shape:

    PythonVision/
        common/                 shared library (geometry, image io, ...)
        script.py               the shared harness that runs any algorithm
        aruco/input/            FieldBW.png + capture_*.png
        aruco/out/              outputs
        aruco/aruco.py          the algorithm itself

The capture set and the field map used to sit in `Captures/` beside the scripts
and were found by looking next to `__file__`. They now live under
`<algorithm>/input/`, so that every algorithm keeps its own copy of the images
it was tuned on. Nothing else about the pipeline changed: the same constants,
transformations and thresholds are applied, only the paths are resolved here
instead of in each script.
"""

from __future__ import annotations

import glob
import os

# Directory names, fixed rather than configurable, so every algorithm folder has
# the same shape and the harness can be told just the algorithm's name.
INPUT_DIRNAME = "input"
OUTPUT_DIRNAME = "out"

# The capture set and the field map, by the names the scripts already used.
CAPTURE_GLOB = "capture_*.png"
MAP_NAME = "FieldBW.png"

# PythonVision/ itself, i.e. the parent of this package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def algorithm_dir(name: str) -> str:
    """The folder holding algorithm `name`, e.g. `name="aruco"`."""
    return os.path.join(ROOT, name)


def input_dir(name: str) -> str:
    """The folder holding algorithm `name`'s `FieldBW.png` and captures."""
    return os.path.join(algorithm_dir(name), INPUT_DIRNAME)


def output_dir(name: str) -> str:
    """The folder algorithm `name` writes its results into."""
    return os.path.join(algorithm_dir(name), OUTPUT_DIRNAME)


def map_path(name: str, map_name: str = MAP_NAME) -> str:
    """The field map an algorithm matches against, in its own input folder."""
    return os.path.join(input_dir(name), map_name)


def captures(name: str, pattern: str = CAPTURE_GLOB) -> list[str]:
    """Every capture in `name`'s input folder, sorted.

    Sorted by name, which is what the old scripts did; `opencv_features` then
    re-sorts by the id in the filename, because string order puts
    `capture_100` before `capture_2`. That re-sort stays with the algorithm, not
    here, so a caller that wants plain name order still gets it.
    """
    return sorted(glob.glob(os.path.join(input_dir(name), pattern)))


def resolve_capture(name: str, reference: str) -> str:
    """A capture named on the command line, relative to `name`'s input folder.

    An absolute path is taken as-is, so a run can point at a capture outside the
    input folder without copying it in.
    """
    if os.path.isabs(reference):
        return reference
    candidate = os.path.join(input_dir(name), reference)
    return candidate if os.path.exists(candidate) else reference
