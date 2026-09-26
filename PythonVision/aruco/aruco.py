#!/usr/bin/env python3
"""Detect ArUco tags on the ground and work out where the robot is.

The camera is bolted to the robot at a fixed height, pitch and FOV, so
pixel -> ground is a known constant homography and is applied before detection
rather than estimated afterwards. That turns a tag seen at 45 deg, which is a
foreshortened quadrilateral, back into a square lying on the ground.

Everything is reported in the robot's own frame, in millimetres: x forwards,
y left, origin on the ground directly under the camera. A lens pitched down by
less than half its vertical field of view never sees straight down, so nothing
nearer than `near_x_mm` can appear in a capture.

Each tag's position is known from the field map, so one tag is enough to place
the robot: measure the tag from the robot, then take that offset back off the
tag's known field position.

Writes `out/aruco_detections.json` and `out/<capture>_aruco.png`.

Run with the shared harness:

    ../.venv/bin/python ../script.py aruco
    ../.venv/bin/python ../script.py aruco capture_x.png
"""

from __future__ import annotations

import json
import os
import sys

import cv2 as cv
import numpy as np

# Both the harness (which imports this module by path) and a direct
# `python aruco.py` need PythonVision/ importable for `common`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import geometry, imageio, paths, pose, warp  # noqa: E402
from common.geometry import TAG_FIELD_MM  # noqa: E402
from common.viz import draw_quad  # noqa: E402

# Resolution of the rectified image. Detection cost scales with this.
RECTIFIED_MM_PER_PX = 1.0

# The id the detector reports on the field every so often and which is not a
# real tag; dropped rather than treated as a robot position.
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
NEAR_X_MM = warp.near_x_mm()
FAR_X_MM = warp.far_x_mm()

# Overlay styling.
QUAD_COLOR = (0, 255, 0)
FIRST_CORNER_COLOR = (0, 0, 255)
CENTRE_COLOR = (0, 200, 255)
LABEL_COLOR = (0, 255, 0)


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

    All of those numbers came from a parameter sweep that lived beside the old
    `Captures/` scripts; re-measure rather than trusting the comments after
    touching any of this.
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


def tag_pose(corners_mm: np.ndarray) -> tuple[float, float, float]:
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
                   angle_deg: float) -> tuple[float, float, float] | None:
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
         angle_deg: float, robot_pose: tuple[float, float, float] | None) -> None:
    """Mark one detection on the colour `view`, in place.

    The quad is drawn from the detector's own pixels, so it can be checked by
    eye against the tag. Corner 1 is red because that is the corner the 1 -> 4
    angle is measured from, and the detector does not always start there.
    """
    draw_quad(view, corners_px, QUAD_COLOR, 1)
    first = tuple(np.int32(np.rint(corners_px))[0])
    cv.circle(view, first, 5, FIRST_CORNER_COLOR, -1, cv.LINE_AA)
    cv.drawMarker(view, tuple(corners_px.mean(axis=0).round().astype(int)),
                  CENTRE_COLOR, cv.MARKER_CROSS, 24, 1, cv.LINE_AA)

    label = f"id={tag_id} angle={angle_deg:+.1f}"
    if robot_pose is not None:
        label += (f"  robot=({robot_pose[0]:+.0f},{robot_pose[1]:+.0f}) "
                  f"{robot_pose[2]:+.1f}deg")
    quad = np.int32(np.rint(corners_px)).reshape(-1, 1, 2)
    x = int(np.clip(quad[:, 0, 0].min(), 0, view.shape[1] - 1))
    y = int(np.clip(quad[:, 0, 1].min() - 8, 18, view.shape[0] - 1))
    cv.putText(view, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.1, LABEL_COLOR,
               2, cv.LINE_AA)


def process(path: str, warper: warp.GroundWarper,
            detector: cv.aruco.ArucoDetector, quiet: bool = False) -> tuple[dict, np.ndarray | None]:
    """Detect the tags in one capture, print them, and return the results."""
    gray = imageio.imread_gray(path)
    warped = warper.warp(gray)
    corners, ids, _ = detector.detectMarkers(warped)
    truth = pose.parse_pose_filename(path)

    if not quiet:
        print(os.path.basename(path))
    results = []
    view = None

    if ids is not None:
        for quad, tag_id in zip(corners, ids.ravel()):
            if int(tag_id) in FALSE_POSITIVE_IDS:
                continue

            corners_px = quad.reshape(4, 2).astype(np.float64)
            x_mm, y_mm, angle_deg = tag_pose(warper.robot_mm(corners_px))
            robot = robot_on_field(int(tag_id), x_mm, y_mm, angle_deg)

            if not quiet:
                print(f"   tag {int(tag_id)}  "
                      f"offset=({x_mm:+7.1f},{y_mm:+7.1f})mm"
                      f"  angle={angle_deg:+7.1f}deg", end="")
                if robot is None:
                    print("  [not on the map]")
                else:
                    print(f"  robot=({robot[0]:+7.1f},{robot[1]:+7.1f})mm  "
                          f"heading={robot[2]:+7.1f}deg")

            # Compare against the pose the capture was taken at, so the run
            # reports its own accuracy.
            error_mm = error_deg = None
            if robot is not None and truth is not None:
                error_mm = float(np.hypot(robot[0] - truth[0],
                                          robot[1] - truth[1]))
                error_deg = pose.wrap_deg(robot[2] - truth[2])
                if not quiet:
                    print(f"      truth=({truth[0]:+7.1f},{truth[1]:+7.1f})mm  "
                          f"yaw={truth[2]:+7.1f}deg  ->  "
                          f"error={error_mm:5.1f}mm {error_deg:+5.1f}deg")

            # Colour copy made lazily: a capture with no tags needs no image.
            if view is None:
                view = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)
            draw(view, corners_px, int(tag_id), angle_deg, robot)
            results.append({
                "tag_id": int(tag_id),
                "offset_mm": [round(x_mm, 1), round(y_mm, 1)],
                "angle_deg": round(angle_deg, 1),
                "robot_mm": ([round(robot[0], 1), round(robot[1], 1)]
                             if robot else None),
                "robot_heading_deg": round(robot[2], 1) if robot else None,
                "truth_mm": ([round(truth[0], 1), round(truth[1], 1)]
                             if truth else None),
                "truth_yaw_deg": round(truth[2], 1) if truth else None,
                "error_mm": round(error_mm, 1) if error_mm is not None else None,
                "error_deg": (round(error_deg, 1)
                              if error_deg is not None else None),
            })

    if not results and not quiet:
        print("   no tags")
    if not quiet:
        print()

    return {"capture": os.path.basename(path), "tags": results}, view


# --- Main ------------------------------------------------------------------


def build_parser(parser) -> None:
    parser.add_argument("--mm-per-px", type=float, default=RECTIFIED_MM_PER_PX,
                        help="Resolution of the rectified image detection runs on.")


def run(captures: list[str], out_dir: str, args) -> int:
    # The mounting is fixed, so one warper and one detector serve every capture.
    probe = imageio.imread_gray(captures[0])
    warper = warp.GroundWarper(probe.shape[1], probe.shape[0], args.mm_per_px)

    if not args.quiet:
        print(f"frame     : x forwards, y left ; nothing nearer than "
              f"{NEAR_X_MM:.0f} mm can be seen")
        print(f"rectified : {warper.out_size[0]}x{warper.out_size[1]} px @ "
              f"{args.mm_per_px:g} mm/px\n")

    detector = build_detector()
    done = [process(path, warper, detector, args.quiet) for path in captures]
    results = [r for r, _ in done]
    images = {r["capture"][:-4]: img for r, img in done if img is not None}

    if not args.no_output:
        os.makedirs(out_dir, exist_ok=True)
        for name, img in images.items():
            cv.imwrite(os.path.join(out_dir, f"{name}_aruco.png"), img)
        with open(os.path.join(out_dir, "aruco_detections.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
            fh.write("\n")

    print(f"Detected {sum(len(r['tags']) for r in results)} tag(s) across "
          f"{len(captures)} capture(s).")

    # Overall accuracy, over every tag that had a reference pose to check.
    checked = [t for r in results for t in r["tags"] if t["error_mm"] is not None]
    if checked:
        pos = np.array([t["error_mm"] for t in checked])
        head = np.array([abs(t["error_deg"]) for t in checked])
        print(f"Error     : {len(checked)} tag(s) vs filename -> position "
              f"mean {pos.mean():.1f} / max {pos.max():.1f} mm, "
              f"heading mean {head.mean():.2f} / max {head.max():.2f} deg")

    if not args.no_output:
        print(f"Output    : {os.path.join(out_dir, 'aruco_detections.json')}")
        print(f"            {out_dir}/*_aruco.png")
    return 0


if __name__ == "__main__":
    # Run standalone with the same flags the harness would pass along.
    import argparse

    _parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _parser.add_argument("captures", nargs="*")
    _parser.add_argument("--no-output", action="store_true")
    _parser.add_argument("--quiet", action="store_true")
    build_parser(_parser)
    _args = _parser.parse_args(sys.argv[1:])
    _args.mm_per_px = getattr(_args, "mm_per_px", RECTIFIED_MM_PER_PX)
    _paths = ([paths.resolve_capture("aruco", n) for n in _args.captures]
              or paths.captures("aruco"))
    raise SystemExit(run(_paths, paths.output_dir("aruco"), _args))
