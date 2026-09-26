"""The pose a capture was taken at, read back out of its filename.

`OnboardCamScript` writes the true pose into each capture's name, e.g.

    capture_20260923_143409_x275.1_y985.5_yaw2.4_pitch45.0.png

so every algorithm can report its own error without a ground-truth file. The
name is also the only place the capture's order on the field is recorded, which
is what orders a video built from a run's frames.
"""

from __future__ import annotations

import os
import re

# x, y and yaw anywhere in the stem, each preceded by `_` (or the start) so the
# digits of a timestamp cannot be mistaken for a value.
POSE_IN_NAME = re.compile(r"(?:^|_)x(-?[\d.]+)_y(-?[\d.]+)_yaw(-?[\d.]+)")

# The capture's ordinal on the field. NOT the order `sorted()` puts the names
# in: string sort ranks `capture_100` before `capture_2` and puts `capture_9`
# last, so numbering frames by list position scrambles the sequence.
CAPTURE_ID = re.compile(r"^capture_(\d+)")


def parse_pose_filename(path: str) -> tuple[float, float, float] | None:
    """The `(x_mm, y_mm, yaw_deg)` recorded in a name, or None if absent.

    Field millimetres from the field centre, and yaw in degrees, referring to
    the camera at the centre of the image - not to the patch centre, which sits
    `CAMERA_PATCH_CENTRE_MM` further along the heading.
    """
    match = POSE_IN_NAME.search(os.path.splitext(os.path.basename(path))[0])
    if match is None:
        return None
    return tuple(float(v) for v in match.groups())


def capture_sort_key(path: str) -> tuple[int, str]:
    """Sort captures by their numeric id, then by name for non-matching files.

    A name with no id sorts after every name that has one, rather than being
    dropped: an unexpected file should still appear, just at the end where it is
    obvious.
    """
    match = CAPTURE_ID.match(os.path.basename(path))
    if match is None:
        return (1 << 62), os.path.basename(path)
    return int(match.group(1)), os.path.basename(path)


def wrap_deg(angle: float) -> float:
    """An angle in degrees folded into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0
