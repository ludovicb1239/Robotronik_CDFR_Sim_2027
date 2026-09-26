"""Drawing poses, labels and quads onto an image.

The three algorithms each report a pose on the field and each wants to show it,
but they were written independently: `opencv.py` draws a red quad and labelled
box, `opencv_aruco.py` draws a green quad with the first corner marked, and
`opencv_features.py` draws solved and true poses as a cross plus a heading
arrow. Only the last of those is genuinely shared - the other two are each
algorithm's own report format - so the map/pose drawing lives here and the
per-detection annotations stay in their own modules.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

# A solution is only worth drawing if the robot's own marker is legible; the map
# overlays are downscaled so a contact sheet of a whole run stays readable.
OVERLAY_MAX_SIDE = 900
OVERLAY_COLOR = (0, 0, 255)          # solved pose, red
OVERLAY_TRUTH_COLOR = (0, 200, 0)    # truth from the filename, green

# Drawn as a heading arrow rather than a bare dot: without it a 180 deg error
# looks like a correct fix.
POSE_ARROW_MM = 300.0
POSE_ARROW_TIP = 0.25


def heading_marker_mm(x_mm: float, y_mm: float, yaw_deg: float,
                      length_mm: float) -> np.ndarray:
    """Two ground points along a heading, as field millimetres.

    Field yaw turns anticlockwise from +x, so the forward unit vector is
    `(cos, sin)` in field millimetres; the map's rows then flip the y term when
    the points are converted to pixels.
    """
    a = np.radians(yaw_deg)
    tip = (x_mm + np.cos(a) * length_mm, y_mm + np.sin(a) * length_mm)
    return np.array([[x_mm, y_mm], tip], np.float64)


def draw_pose(frame: np.ndarray, to_px, x_mm: float, y_mm: float, yaw_deg: float,
              colour: tuple[int, int, int], label: str,
              length_mm: float = POSE_ARROW_MM, thickness: int = 3) -> np.ndarray:
    """Mark a ground pose on a map-sized image, in place: a cross and an arrow.

    `to_px` converts field millimetres to pixels in `frame`'s own convention,
    which differs between the renderer (padded map) and the feature matcher
    (centred map with y up), so it is passed in rather than assumed.

    The arrow shows which way the robot faces, which a bare dot cannot; without
    it a heading error of 180 deg would look like a correct fix.
    """
    h, w = frame.shape[:2]
    px = np.array([to_px(p[0], p[1])
                   for p in heading_marker_mm(x_mm, y_mm, yaw_deg, length_mm)])
    base = tuple(np.rint(px[0]).astype(int))
    tip = tuple(np.rint(px[1]).astype(int))
    cv.drawMarker(frame, base, colour, cv.MARKER_CROSS, 26, thickness, cv.LINE_AA)
    cv.arrowedLine(frame, base, tip, colour, thickness, cv.LINE_AA,
                   tipLength=POSE_ARROW_TIP)

    org = (int(np.clip(base[0] + 10, 0, max(w - 1, 0))),
           int(np.clip(base[1] - 10, 18, max(h - 1, 0))))
    cv.putText(frame, label, org, cv.FONT_HERSHEY_SIMPLEX, 0.7, colour,
               thickness, cv.LINE_AA)
    return frame


def draw_quad(img: np.ndarray, corners_px: np.ndarray, colour=(0, 255, 0),
              thickness: int = 2) -> None:
    """Outline a detected quad, in place."""
    quad = np.int32(np.rint(corners_px)).reshape(-1, 1, 2)
    cv.polylines(img, [quad], True, colour, thickness, cv.LINE_AA)
