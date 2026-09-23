#!/usr/bin/env python3
"""Detect ArUco tags on the ground and work out where the robot is.

The camera is bolted to the robot at a fixed height, pitch and FOV, so
pixel -> ground is a known constant homography and is applied before detection
rather than estimated afterwards. That turns a tag seen at 45 deg, which is a
foreshortened quadrilateral, back into a square lying on the ground.

Everything is reported in the robot's own frame, in millimetres: x forwards,
y left, origin on the ground directly under the camera. A lens pitched down by
less than half its vertical field of view never sees straight down, so nothing
nearer than NEAR_X_MM can appear in a capture.

Each tag's position is known from the field map, so one tag is enough to place
the robot: measure the tag from the robot, then take that offset back off the
tag's known field position.

Writes out/aruco_detections.json and out/<capture>_aruco.png.

Usage:
    python opencv_aruco.py                    # every capture_*.png
    python opencv_aruco.py capture_x.png      # specific files
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import cv2 as cv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

CAPTURE_GLOB = "capture_*.png"
OUTPUT_DIR = os.path.join(HERE, "out")

# Captures record the true pose in the filename, e.g.
#   capture_20260923_143409_x275.1_y985.5_yaw2.4_pitch45.0.png
# so the estimate can be checked against it as the script runs.
POSE_IN_NAME = re.compile(r"_x(-?[\d.]+)_y(-?[\d.]+)_yaw(-?[\d.]+)")

# Fixed camera mounting, identical to the Pi program.
CAMERA_VFOV_DEG = 70.0
CAMERA_PITCH_DEG = 45.0
CAMERA_HEIGHT_MM = 233.2

# Resolution of the rectified image. Detection cost scales with this.
RECTIFIED_MM_PER_PX = 1.0

# Where each tag sits on the field, in millimetres.
TAG_FIELD_MM = {
    20: (-400.0, -900.0),
    21: (-400.0, 900.0),
    22: (400.0, -900.0),
    23: (400.0, 900.0),
}

FALSE_POSITIVE_IDS = {13}

# The tag's own size, on the ground. Everything upstream reports in
# millimetres, so this is a constant rather than something to measure, and it
# is what lets the detector's search band be computed instead of guessed.
TAG_SIDE_MM = 100.0

# How far either side of the tag's true perimeter to search. Widening by a
# factor rather than a fixed amount keeps the band proportional to the tag as
# the rectified scale changes.
PERIMETER_SLACK = 2.0

# The camera's mounting geometry fixes the frame's top and bottom edges.
# The bottom edge owns the steepest ray, so nothing nearer than NEAR_X_MM can
# be seen; the top edge owns the shallowest, which is FAR_X_MM away.
NEAR_X_MM = CAMERA_HEIGHT_MM / np.tan(
    np.radians(CAMERA_PITCH_DEG - CAMERA_VFOV_DEG / 2.0))
FAR_X_MM = CAMERA_HEIGHT_MM / np.tan(
    np.radians(CAMERA_PITCH_DEG + CAMERA_VFOV_DEG / 2.0))

# Overlay styling.
QUAD_COLOR = (0, 255, 0)
FIRST_CORNER_COLOR = (0, 0, 255)
CENTRE_COLOR = (0, 200, 255)
LABEL_COLOR = (0, 255, 0)


# --- Camera geometry -------------------------------------------------------


def capture_to_ground(width: int, height: int) -> np.ndarray:
    """Pixel -> ground homography, giving (right, forward) in millimetres.

    A pixel (u, v), with v measured downward, defines a camera ray whose
    direction in the robot frame is `rotation @ k_inv @ [u, v, 1]`. Intersecting
    that ray with the ground plane makes both ground coordinates ratios over the
    same denominator, so the whole mapping collapses into one 3x3 homography.

    The negated vertical term in `k_inv` keeps image rows, which grow downward,
    consistent with the camera's up axis; without it the warp is mirrored.
    """
    f = (height / 2.0) / np.tan(np.radians(CAMERA_VFOV_DEG / 2.0))
    k_inv = np.array([[1.0 / f, 0.0, -width / 2.0 / f],
                      [0.0, -1.0 / f, height / 2.0 / f],
                      [0.0, 0.0, 1.0]])

    p = np.radians(CAMERA_PITCH_DEG)
    forward = np.array([0.0, -np.sin(p), -np.cos(p)])
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    rotation = np.column_stack([right, np.cross(right, forward), forward])

    m = rotation @ k_inv
    cam_y_m = CAMERA_HEIGHT_MM / 1000.0
    h = np.vstack([-cam_y_m * m[0], -cam_y_m * m[2], m[1]])
    return np.diag([1000.0, 1000.0, 1.0]) @ h


def ground_bounds(h: np.ndarray, width: int, height: int) -> tuple:
    """Ground area the image covers, as (x0, y0, x1, y1) in mm.

    Sampled over a grid: the bottom rows land behind the camera and get thrown
    far off to the sides, so the four image corners alone would give bounds far
    too wide. Row 2 is the depth denominator, and a real hit needs it negative.
    """
    xs, ys = np.linspace(0, width - 1, 65), np.linspace(0, height - 1, 65)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    projected = np.hstack([grid, np.ones((len(grid), 1))]) @ h.T

    depth = projected[:, 2]
    keep = depth < -1e-9
    gx = projected[keep, 0] / depth[keep]
    gy = projected[keep, 1] / depth[keep]
    return float(gx.min()), float(gy.min()), float(gx.max()), float(gy.max())


class GroundWarper:
    """Warps a capture flat onto the ground, built once and reused.

    Ground positions are reported in the robot's frame: x forwards, y left.

        x = -raw_y       (forwards)
        y = -raw_x       (left)

    Both homography axes run opposite to the robot frame's, so both are
    negated. Verified against captures whose true pose is known: with these
    signs the tag offset comes out right to ~2 mm on every capture, whereas
    either sign on its own leaves one axis out by hundreds of mm.

    x needs no offset beyond the sign: `capture_to_ground`'s y comes out of the
    projection already measured from the ground point under the robot, because
    at the image centre the ray is the mounting pitch and height/tan(pitch) is
    exactly the distance to that point. Adding a constant here biases every
    position by it, which is what a stray FAR_X_MM term used to do - 41 mm of
    error on every fix.

    The frame's top row is FAR_X_MM away and the bottom row NEAR_X_MM, since
    the camera is pitched down and never sees straight down.
    """

    def __init__(self, width: int, height: int, mm_per_px: float) -> None:
        self.mm_per_px = mm_per_px
        self.h = capture_to_ground(width, height)
        self.x0, self.y0, x1, y1 = ground_bounds(self.h, width, height)

        pad = 0.02 * max(x1 - self.x0, y1 - self.y0)
        self.x0, self.y0 = self.x0 - pad, self.y0 - pad
        x1, y1 = x1 + pad, y1 + pad

        scale = 1.0 / mm_per_px
        self.out_size = (max(1, round((x1 - self.x0) * scale)),
                         max(1, round((y1 - self.y0) * scale)))
        self.m = np.array([[scale, 0.0, -self.x0 * scale],
                           [0.0, scale, -self.y0 * scale],
                           [0.0, 0.0, 1.0]]) @ self.h

    def warp(self, gray: np.ndarray) -> np.ndarray:
        return cv.warpPerspective(gray, self.m, self.out_size,
                                  flags=cv.INTER_LINEAR,
                                  borderMode=cv.BORDER_CONSTANT, borderValue=0)

    def robot_mm(self, px: np.ndarray) -> np.ndarray:
        """Rectified pixels -> robot-frame (x forwards, y left) in mm."""
        raw_x = px[:, 0] * self.mm_per_px + self.x0
        raw_y = px[:, 1] * self.mm_per_px + self.y0
        return np.stack([-raw_y, -raw_x], 1)


# --- Detection -------------------------------------------------------------


def build_detector(scale: float = 1.0,
                   frame_width_px: float = 2561.0,
                   slack: float = PERIMETER_SLACK) -> cv.aruco.ArucoDetector:
    """Detector settings, with the two that matter measured rather than guessed.

    `minMarkerPerimeterRate` and `maxMarkerPerimeterRate` are fractions of the
    image width and the tag is a known size on the ground seen through a fixed
    homography, so its perimeter in the rectified frame is a constant that can
    be computed instead of guessed. `slack` widens that band on each side,
    which keeps every real tag and drops most of the contours the detector would
    otherwise have to measure and try to decode.

    `scale` is the rectified grid in millimetres per pixel, so the same tag is
    twice as many pixels across at 0.5 as at 1.0 and the band has to follow.
    `frame_width_px` is only used to convert the tag's size into a rate.

    `adaptiveThreshWinSizeMin`/`Max`/`Step` are the settings that actually cost
    time, because the thresholding runs once per step across that window and the
    pass count is `(max - min) / step`. Widening `Step` from 10 to 12 takes the
    shipped `(3, 23)` extent from two passes to one, measured at 12.2 -> 9.8 ms
    over the capture set, -19%, for the same 40 detections and no accuracy
    change. Narrowing `Max` saves more
    - `(3, 19, 12)` reaches -19% too and `(3, 11, 10)` halves the time - but
    those lose real tags: `(3, 23, 12)` is the fastest window that finds every
    tag the old settings did, so the extent stays and only the step changes.
    Min below 3 is rejected by OpenCV, so 3 is the floor.

    `cornerRefinementMethod` looks like the obvious cost but is not: measured
    over the capture set NONE, CONTOUR and SUBPIX all land within noise at
    ~11.5 ms, so SUBPIX is kept for the corners it returns. APRILTAG is the one
    to avoid, at 117 ms. `perspectiveRemovePixelPerCell` and
    `polygonalApproxAccuracyRate` were swept too and did not separate from the
    noise; they are left alone and are not worth revisiting first.

    All of those numbers come from `_diag_sweep.py`, `_diag_sweep_band.py` and
    `_diag_compare.py` beside this file. Re-run them after touching any of this
    rather than trusting the comments.
    """
    params = cv.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 12
    params.adaptiveThreshConstant = 7
    rate = 4.0 * (TAG_SIDE_MM / scale) / frame_width_px
    params.minMarkerPerimeterRate = max(0.01, rate / slack)
    params.maxMarkerPerimeterRate = rate * slack
    params.polygonalApproxAccuracyRate = 0.03
    params.minCornerDistanceRate = 0.05
    params.minDistanceToBorder = 3
    params.minMarkerDistanceRate = 0.05
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01
    params.markerBorderBits = 1
    params.perspectiveRemovePixelPerCell = 4
    params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    params.maxErroneousBitsInBorderRate = 0.35
    params.minOtsuStdDev = 5.0
    params.errorCorrectionRate = 0.6
    params.detectInvertedMarker = False
    params.useAruco3Detection = False

    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    return cv.aruco.ArucoDetector(dictionary, params)


# --- Pose ------------------------------------------------------------------


def tag_pose(corners_mm: np.ndarray) -> tuple:
    """The tag's (x, y) offset from the robot, and its rotation.

    The side corner 1 -> corner 4 points along the field's +x on every tag, so
    its angle is the tag's rotation as the camera sees it. Using a side rather
    than a diagonal is what pins the heading down: a square alone is unchanged
    by a quarter turn, so a diagonal would only give it mod 90 deg.
    """
    centre = corners_mm.mean(axis=0)
    side = corners_mm[3] - corners_mm[0]
    angle = np.degrees(np.arctan2(side[1], side[0]))
    return float(centre[0]), float(centre[1]), float(angle)


def robot_on_field(tag_id: int, x_mm: float, y_mm: float,
                   angle_deg: float) -> tuple | None:
    """Where the robot is on the field, as (x, y, heading), or None if unknown.

    The tag is a known offset from the robot and also known on the field, so the
    robot is the tag's field position minus that offset, once the offset has
    been rotated out of the robot's frame and into the field's.

    The heading is the tag's measured rotation negated: a tag seen rotated by
    `angle` means the robot is rotated by `-angle` relative to the field.
    """
    field = TAG_FIELD_MM.get(tag_id)
    if field is None:
        return None

    heading_deg = -angle_deg
    t = np.radians(heading_deg)
    cos_t, sin_t = np.cos(t), np.sin(t)
    off_x = cos_t * x_mm - sin_t * y_mm
    off_y = sin_t * x_mm + cos_t * y_mm
    return field[0] - off_x, field[1] - off_y, heading_deg


# --- Output ----------------------------------------------------------------


def draw(view: np.ndarray, corners_px: np.ndarray, tag_id: int,
         angle_deg: float, pose: tuple | None) -> None:
    """Mark one detection on the colour `view`, in place.

    The quad is drawn from the detector's own pixels, so it can be checked by
    eye against the tag. Corner 1 is red because that is the corner the 1 -> 4
    angle is measured from, and the detector does not always start there.
    """
    quad = np.int32(np.rint(corners_px)).reshape(-1, 1, 2)
    cv.polylines(view, [quad], True, QUAD_COLOR, 1, cv.LINE_AA)
    cv.circle(view, tuple(quad[0, 0]), 5, FIRST_CORNER_COLOR, -1, cv.LINE_AA)
    cv.drawMarker(view, tuple(corners_px.mean(axis=0).round().astype(int)),
                  CENTRE_COLOR, cv.MARKER_CROSS, 24, 1, cv.LINE_AA)

    label = f"id={tag_id} angle={angle_deg:+.1f}"
    if pose is not None:
        label += f"  robot=({pose[0]:+.0f},{pose[1]:+.0f}) {pose[2]:+.1f}deg"
    x = int(np.clip(quad[:, 0, 0].min(), 0, view.shape[1] - 1))
    y = int(np.clip(quad[:, 0, 1].min() - 8, 18, view.shape[0] - 1))
    cv.putText(view, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.1, LABEL_COLOR,
               2, cv.LINE_AA)


def write_outputs(results: list, images: dict) -> None:
    """Write the JSON summary and one annotated image per capture into out/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, img in images.items():
        cv.imwrite(os.path.join(OUTPUT_DIR, f"{name}_aruco.png"), img)
    with open(os.path.join(OUTPUT_DIR, "aruco_detections.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")


# --- Main ------------------------------------------------------------------


def reference_pose(path: str) -> tuple | None:
    """The true robot pose recorded in a capture's filename, or None."""
    match = POSE_IN_NAME.search(os.path.basename(path))
    if match is None:
        return None
    return tuple(float(v) for v in match.groups())


def process(path: str, warper: GroundWarper,
            detector: cv.aruco.ArucoDetector) -> tuple:
    """Detect the tags in one capture, print them, and return the results."""
    gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    warped = warper.warp(gray)
    corners, ids, _ = detector.detectMarkers(warped)
    truth = reference_pose(path)

    print(os.path.basename(path))
    results = []
    view = None

    if ids is not None:
        for quad, tag_id in zip(corners, ids.ravel()):
            if int(tag_id) in FALSE_POSITIVE_IDS:
                continue

            corners_px = quad.reshape(4, 2).astype(np.float64)
            x_mm, y_mm, angle_deg = tag_pose(warper.robot_mm(corners_px))
            pose = robot_on_field(int(tag_id), x_mm, y_mm, angle_deg)

            print(f"   tag {int(tag_id)}  offset=({x_mm:+7.1f},{y_mm:+7.1f})mm"
                  f"  angle={angle_deg:+7.1f}deg", end="")
            if pose is None:
                print("  [not on the map]")
            else:
                print(f"  robot=({pose[0]:+7.1f},{pose[1]:+7.1f})mm  "
                      f"heading={pose[2]:+7.1f}deg")

            # Compare against the pose the capture was taken at, so the run
            # reports its own accuracy.
            error_mm = error_deg = None
            if pose is not None and truth is not None:
                error_mm = float(np.hypot(pose[0] - truth[0],
                                          pose[1] - truth[1]))
                error_deg = float((pose[2] - truth[2] + 180.0) % 360.0 - 180.0)
                print(f"      truth=({truth[0]:+7.1f},{truth[1]:+7.1f})mm  "
                      f"yaw={truth[2]:+7.1f}deg  ->  "
                      f"error={error_mm:5.1f}mm {error_deg:+5.1f}deg")

            # Colour copy made lazily: a capture with no tags needs no image.
            if view is None:
                view = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)
            draw(view, corners_px, int(tag_id), angle_deg, pose)
            results.append({
                "tag_id": int(tag_id),
                "offset_mm": [round(x_mm, 1), round(y_mm, 1)],
                "angle_deg": round(angle_deg, 1),
                "robot_mm": ([round(pose[0], 1), round(pose[1], 1)]
                             if pose else None),
                "robot_heading_deg": round(pose[2], 1) if pose else None,
                "truth_mm": ([round(truth[0], 1), round(truth[1], 1)]
                             if truth else None),
                "truth_yaw_deg": round(truth[2], 1) if truth else None,
                "error_mm": round(error_mm, 1) if error_mm is not None else None,
                "error_deg": (round(error_deg, 1)
                              if error_deg is not None else None),
            })

    if not results:
        print("   no tags")
    print()

    summary = {"capture": os.path.basename(path), "tags": results}
    return summary, view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("captures", nargs="*",
                        help=f"Capture name(s). Default: every {CAPTURE_GLOB}.")
    parser.add_argument("--mm-per-px", type=float, default=RECTIFIED_MM_PER_PX)
    parser.add_argument("--no-output", action="store_true",
                        help="Do not write images or JSON into out/.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    paths = [p if os.path.isabs(p) else os.path.join(HERE, p)
             for p in args.captures]
    if not paths:
        paths = sorted(glob.glob(os.path.join(HERE, CAPTURE_GLOB)))
    if not paths:
        print(f"[error] no captures matching {CAPTURE_GLOB}", file=sys.stderr)
        return 2

    # The mounting is fixed, so one warper and one detector serve every capture.
    probe = cv.imread(paths[0], cv.IMREAD_GRAYSCALE)
    if probe is None:
        raise FileNotFoundError(f"Could not read image: {paths[0]}")
    warper = GroundWarper(probe.shape[1], probe.shape[0], args.mm_per_px)

    print(f"frame     : x forwards, y left ; nothing nearer than "
          f"{NEAR_X_MM:.0f} mm can be seen")
    print(f"rectified : {warper.out_size[0]}x{warper.out_size[1]} px @ "
          f"{args.mm_per_px:g} mm/px\n")

    detector = build_detector()
    done = [process(path, warper, detector) for path in paths]
    results = [r for r, _ in done]
    images = {r["capture"][:-4]: img for r, img in done if img is not None}

    if not args.no_output:
        write_outputs(results, images)
    print(f"Detected {sum(len(r['tags']) for r in results)} tag(s) across "
          f"{len(paths)} capture(s).")

    # Overall accuracy, over every tag that had a reference pose to check.
    checked = [t for r in results for t in r["tags"] if t["error_mm"] is not None]
    if checked:
        pos = np.array([t["error_mm"] for t in checked])
        head = np.array([abs(t["error_deg"]) for t in checked])
        print(f"Error     : {len(checked)} tag(s) vs filename -> position "
              f"mean {pos.mean():.1f} / max {pos.max():.1f} mm, "
              f"heading mean {head.mean():.2f} / max {head.max():.2f} deg")

    if not args.no_output:
        print(f"Output    : {OUTPUT_DIR}/aruco_detections.json")
        print(f"            {OUTPUT_DIR}/*_aruco.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
