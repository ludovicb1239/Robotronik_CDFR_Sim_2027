#!/usr/bin/env python3
"""Locate the robot by matching ground features between a capture and the map.

The camera is bolted to the robot at a known height, pitch and field of view, so
the pixel -> ground warp is a constant and is applied to every capture before
anything else happens. Rectifying first is what makes feature matching viable
here:

* Every pixel in the rectified patch is one millimetre of ground, the same as
  the map, so a match is a rigid motion (rotation + translation) with no scale
  term to estimate and nothing to get wrong.
* Ground features are square-on instead of foreshortened by up to 45 degrees, so
  a descriptor computed on the patch is computed on the same shape it has on the
  map. Matching a keystoned quad against the map would otherwise need the
  descriptor to be invariant to a projective distortion it was never designed
  for.

The field map is the reference: it is the same 1 mm/px grid, so a patch feature
and its map counterpart are the same size by construction.

Flow, per capture:

    capture -> rectify -> BRISK keypoints -> match against map keypoints
            -> RANSAC rigid fit -> robot pose on the field

The map's keypoints are computed once and reused for every capture; the map does
not change.

Run with the shared harness:

    ../.venv/bin/python ../script.py features
    ../.venv/bin/python ../script.py features capture_x.png
    ../.venv/bin/python ../script.py features --write-steps

`--write-steps` writes, per capture, the rectified patch, the inlier matches, and
`<capture>_pose.png`: the field map with the solved pose and the true pose drawn
on it. The pose image is the one to look at first - the two markers should sit
on top of each other, and an arrow shows the heading so a 180 deg error cannot
hide behind a correct-looking dot.

`--frames` writes one `match_<id>.png` per capture into `out/frames/`, numbered
by the capture's own id so the sequence can be played back as a video. A capture
with no fix still gets a frame, showing the matches it did find, so the sequence
has no holes. Frames are padded to even dimensions, which is what `yuv420p`
requires, so they encode without any ffmpeg scale filter. Ids are not contiguous
if the set is filtered, and ffmpeg's `%04d` pattern needs them to be, so a
filtered run should use `-pattern_type glob`.

Build the video from the frames with:

    ffmpeg -y -framerate 30 -pattern_type glob -i 'out/frames/match_*.png' \\
      -c:v libx264 -pix_fmt yuv420p out/match_video.mp4
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

# Both the harness (which imports this module by path) and a direct
# `python features.py` need PythonVision/ importable for `common`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import imageio, paths, pose, viz  # noqa: E402
from common.geometry import FIELD_HEIGHT_MM, FIELD_WIDTH_MM  # noqa: E402
from common.warp import GroundWarper  # noqa: E402

# --- Working grid ----------------------------------------------------------
#
# The grid the rectified patch and the map are sampled on. Matching only makes
# sense if both are on the same grid: a match is then a rigid motion, and a
# feature is the same size in both images, so the descriptors are computed on
# the same shape.
#
# THE MAP IS ALWAYS 1 MM/PX, because that is what FieldBW.png is. This constant
# is the scale of the *working* pair - map and patch are both resampled to it -
# and every conversion between pixels and millimetres goes through a scale that
# reads it, never through a hard-coded 1.0. At 2 mm/px the patch has a quarter
# of the pixels, so detection is roughly 4x cheaper, and `solve_rigid`'s
# unit-scale check must compare against 1.0 as usual because both sides moved
# to the same grid together.
#
# Measured on the 428-capture set, thresholds rescaled with the grid. This is
# the corrected sweep: an earlier version of it showed the solve rate collapsing
# toward 55% as the grid coarsened, but that was the harness calling solve_rigid
# with MIN_INLIERS and RANSAC_MM still at their 2 mm/px values, so it was
# rejecting good fits on arithmetic rather than on data. With the thresholds
# following the grid the solve rate barely moves until 6 mm/px:
#
#   mm/px   ms/frame  speedup  solved        err mean  err max
#   1.0        84.3     0.66x  156/428 36.4%    18.2mm    --      (see below)
#   2.0        55.6     1.00x  424/428 99.1%    17.4mm   39.8mm
#   2.5        47.7     1.16x  428/428 100%     17.3mm   39.8mm
#   3.0        41.4     1.34x  427/428 99.8%    17.4mm   39.8mm
#   4.0        32.9     1.69x  427/428 99.8%    17.4mm   40.1mm
#   6.0        25.3     2.20x  408/428 95.3%    17.7mm   37.8mm
#   8.0        21.8     2.55x  364/428 85.0%    18.1mm   42.7mm
#
# 4 mm/px is the knee of the curve: 1.7x faster than 2 mm/px, one capture lost,
# accuracy unchanged to a tenth of a millimetre. This is the fastest grid that
# still solves essentially everything, so it is the default rather than a knob
# to reach for.
#
# The 1 mm/px row is NOT a fair baseline. Its 36.4% is almost certainly the same
# threshold-rescaling artefact as above seen in the other direction - at 1 mm/px
# MIN_INLIER_SPAN_MM=300 becomes a 300 px span, a third of the patch, so real
# fits get rejected. That was never re-measured with --no-scale-thresholds to
# confirm, so do not read it as a resolution limit. It is also not the grid the
# map is stored on, so it defeats the resampling the rest of the pipeline relies
# on.
MM_PER_PX = 4.0

# The scale the tuned constants below were measured at. `MIN_INLIERS`,
# `MIN_INLIER_SPAN_MM` and `RANSAC_MM` are only meaningful in pixels at this
# grid, so at a coarser working scale they have to be rescaled or they reject
# perfectly good fits for no reason but arithmetic. `None` scales them with
# `MM_PER_PX`; an explicit list overrides that per constant.
REFERENCE_MM_PER_PX = 2.0

# Extra decimation applied to the patch *before* keypoint detection only. 1.0
# detects at the working grid. Left at 1.0 because the accuracy cost of
# decimating is not worth it once MM_PER_PX is already at 4.0: the two compound,
# and the measured err max grows from 39.8 to 42.4 mm at 2.0 on top of a 3.0
# grid. At 4.0 the grid is carrying the speedup, so this stays free.
DETECT_SCALE = 1.0

# The map's own resolution, which is not a tunable.
MAP_MM_PER_PX = 1.0

# BRISK, and 30 rather than 40. At 40 three captures matched only 28-30 features
# and RANSAC returned *zero* inliers among them - the survivors were noise, not
# a weak version of the right answer. 30 recovers them (43-48 matches, 32-39
# inliers) and solves 33/33. It is also faster than 40, because matching against
# fewer map keypoints is what the per-capture cost is dominated by. At 20 the
# run costs 177 ms for no accuracy gain, and 35 leaves almost no margin against
# MIN_INLIERS, so 30 is the middle that is defensible from both sides.
BRISK_THRESHOLD = 30
BRISK_OCTAVES = 0            # 0 lets BRISK pick from the image size
BRISK_PATTERN_SCALE = 1.0

# Cross-check plus ratio test. Both are needed: the cross-check alone still lets
# a patch on one side of the field match the mirrored side, because the map is
# left-right symmetric in its textures.
#
# 0.75, and the sweep showed the value barely matters: 0.70 through 0.85 all
# solve 33/33 with identical error. So this is the conventional value rather
# than a fitted one - there is nothing here to tune, which is worth knowing.
RATIO_TEST = 0.75

# RANSAC reprojection tolerance, in millimetres on the working grid.
RANSAC_MM = 6.0
RANSAC_CONFIDENCE = 0.999
# A fit on this many inliers spanning this much of the patch is not a texture
# coincidence. Symmetric-texture false matches tend to be few and clustered.
#
# Both are expressed at REFERENCE_MM_PER_PX and rescaled by `configure_scale`
# when the working grid changes: at 4 mm/px the patch has a quarter of the
# pixels, so a texture that yields 30 matches at 2 mm/px yields about 8, and a
# fixed threshold of 25 would reject it as though it were noise.
MIN_INLIERS = 25
MIN_INLIER_SPAN_MM = 300.0

# Odometry prior. The caller knows roughly where the robot is, so only the map
# features that could possibly be in view are offered to the matcher.
#
# Measured honestly: at these defaults it buys speed and not accuracy. It cuts
# the run from 52 ms to 38 ms, but leaves the error identical (11.4 mm mean,
# 110.3 mm max) with or without it, because a +/-100 mm disc plus the footprint
# is still wide enough to contain the mirrored twin. Kept because it is the
# right shape for the real fix and costs nothing, but it is not yet earning its
# keep.
PRIOR_POS_MM = 100.0
PRIOR_YAW_DEG = 15.0

# Simulated odometry error, applied to the ground-truth pose before it reaches
# the matcher. Without this the prior is the answer, so the window is always
# centred on the truth and never gets tested; `reference_pose` reads the true
# pose straight out of the filename. These are the bounds of a uniform offset in
# each axis, not a Gaussian sigma.
#
# The seed is derived from the capture name, so a given capture always gets the
# same offset: a run is reproducible, and two configurations can be compared on
# identical noise rather than on two different draws.
PRIOR_NOISE_POS_MM = 100.0
PRIOR_NOISE_YAW_DEG = 10.0


def noisy_prior(truth: tuple[float, float, float] | None,
                pos_mm: float = PRIOR_NOISE_POS_MM,
                yaw_deg: float = PRIOR_NOISE_YAW_DEG,
                seed: int | None = None) -> tuple[float, float, float] | None:
    """A ground-truth pose perturbed by simulated odometry error, or None.

    Uniform in each axis, which is the pessimistic choice: a Gaussian would
    cluster near the truth and flatter the result, whereas uniform over the full
    range puts a real share of captures out at the edge of the window.
    """
    if truth is None:
        return None
    rng = np.random.default_rng(seed)
    return (truth[0] + rng.uniform(-pos_mm, pos_mm),
            truth[1] + rng.uniform(-pos_mm, pos_mm),
            truth[2] + rng.uniform(-yaw_deg, yaw_deg))


def imread_gray(path: str) -> np.ndarray:
    """Read a capture as 8-bit grayscale, raising if it cannot be read."""
    return imageio.imread_gray(path)


def reference_pose(path: str) -> tuple[float, float, float] | None:
    """The true robot pose recorded in a capture's filename, or None."""
    return pose.parse_pose_filename(path)


# --- Active scale ----------------------------------------------------------
#
# The working scale is a run-time knob, so the thresholds that were measured in
# pixels at one grid have to follow it. Module-level dicts rather than function
# arguments because `MIN_INLIERS` and friends are read from several places and
# threading a scale struct through every call would touch far more code than the
# feature is worth. `configure_scale` is the only writer and runs before any
# capture is solved.

SCALE = {"mm_per_px": MM_PER_PX, "detect_scale": DETECT_SCALE}
# Thresholds in the units the current grid actually uses. Kept separate from the
# REFERENCE_* constants so both stay visible: the constants are what was
# measured, these are what a given run applies.
ACTIVE = {"min_inliers": float(MIN_INLIERS),
          "min_inlier_span_px": MIN_INLIER_SPAN_MM / MM_PER_PX,
          "ransac_px": RANSAC_MM / MM_PER_PX}


def configure_scale(mm_per_px: float = MM_PER_PX,
                    detect_scale: float = DETECT_SCALE,
                    scale_thresholds: bool = True) -> None:
    """Point the pipeline at a working grid and rescale the pixel thresholds.

    A threshold expressed in pixels at 2 mm/px means something different at
    4 mm/px: the same physical texture yields roughly a quarter of the
    keypoints, so `MIN_INLIERS` has to fall with the scale or it starts
    rejecting real fits and the solve rate collapses for reasons that have
    nothing to do with the data. `MIN_INLIER_SPAN_MM` and `RANSAC_MM` are
    physical lengths, so they only convert, they do not scale quadratically.

    `scale_thresholds=False` holds the raw numbers instead, which is the right
    control when the question is "do the tuned values still hold?" rather than
    "does a coarser grid help at all?".
    """
    SCALE["mm_per_px"] = mm_per_px
    SCALE["detect_scale"] = detect_scale
    ratio = (REFERENCE_MM_PER_PX / mm_per_px) ** 2 if scale_thresholds else 1.0
    ACTIVE["min_inliers"] = max(6.0, round(MIN_INLIERS * ratio))
    ACTIVE["min_inlier_span_px"] = MIN_INLIER_SPAN_MM / mm_per_px
    ACTIVE["ransac_px"] = RANSAC_MM / mm_per_px


# --- Reference map ---------------------------------------------------------


@dataclass
class FieldMap:
    """The field map at 1 mm/px, with its keypoints cached.

    `origin_px` is the map pixel that is the field centre, so field millimetres
    (which is what the filename and PosEstimator use) can be converted without
    the half-field term leaking into the feature code.
    """

    gray: np.ndarray
    keypoints: list = field(default_factory=list)
    descriptors: np.ndarray | None = None
    points_px: np.ndarray | None = None
    native_mm_per_px: float = 1.0     # the file's resolution, for reporting only

    @property
    def mm_per_px(self) -> float:
        return FIELD_WIDTH_MM / float(self.gray.shape[1])

    @property
    def origin_px(self) -> tuple[float, float]:
        return (self.gray.shape[1] / 2.0 - 0.5, self.gray.shape[0] / 2.0 - 0.5)

    def px_to_field_mm(self, px: np.ndarray) -> np.ndarray:
        """Map pixels -> field millimetres, x right and y up, origin centred."""
        ox, oy = self.origin_px
        x_mm = (px[:, 0] - ox) * self.mm_per_px
        y_mm = (oy - px[:, 1]) * self.mm_per_px      # rows grow down, field +y up
        return np.stack([x_mm, y_mm], 1)

    def field_mm_to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        ox, oy = self.origin_px
        return (ox + x_mm / self.mm_per_px, oy - y_mm / self.mm_per_px)

    def field_to_map_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Field millimetres -> raw map pixels, the same mapping the keypoints use."""
        return self.field_mm_to_px(x_mm, y_mm)

    def field_mm_to_map_px(self, mm: np.ndarray) -> np.ndarray:
        """Vector form of `field_mm_to_px`, so a whole keypoint set moves at once."""
        ox, oy = self.origin_px
        return np.stack([ox + mm[:, 0] / self.mm_per_px,
                         oy - mm[:, 1] / self.mm_per_px], 1)

    def keypoints_field_mm(self) -> np.ndarray:
        """Every cached keypoint in field millimetres, x right and y up."""
        if self.points_px is None or not len(self.points_px):
            return np.zeros((0, 2), np.float64)
        return self.px_to_field_mm(self.points_px)


def load_map(path: str, mm_per_px: float = MM_PER_PX) -> FieldMap:
    """Load the map, resample it to the working grid, and note its keypoints' grid.

    The file is 1 mm/px; the working pair is `mm_per_px`. Resampling *here* is
    what lets `MM_PER_PX` change without anything downstream moving: the map's
    own `mm_per_px` then equals the patch's, so a match is still rigid at scale
    1, and `origin_px` (derived from the shape) stays the field centre because
    the resize keeps the aspect ratio and the field is symmetric about it.
    """
    gray = imageio.imread_gray(path)
    native_mm_per_px = FIELD_WIDTH_MM / float(gray.shape[1])
    if abs(native_mm_per_px - FIELD_HEIGHT_MM / float(gray.shape[0])) > 1e-3:
        print(f"[warn] map {gray.shape[1]}x{gray.shape[0]} px for "
              f"{FIELD_WIDTH_MM:.0f}x{FIELD_HEIGHT_MM:.0f} mm; not square.",
              file=sys.stderr)

    if abs(native_mm_per_px - mm_per_px) > 1e-6:
        scale = native_mm_per_px / mm_per_px
        gray = cv.resize(gray, (max(1, int(round(gray.shape[1] * scale))),
                                max(1, int(round(gray.shape[0] * scale)))),
                         interpolation=cv.INTER_AREA)
        # A resize to a non-integer factor can land a pixel off the exact
        # target, so re-derive the scale from the result rather than assuming
        # the request was met. A wrong mm_per_px here silently scales the pose.
        actual = FIELD_WIDTH_MM / float(gray.shape[1])
        if abs(actual - mm_per_px) > 1e-6:
            print(f"[warn] map resampled to {actual:.4f} mm/px, not the "
                  f"requested {mm_per_px:.4f}; using the actual value.",
                  file=sys.stderr)

    field_map = FieldMap(gray)
    field_map.native_mm_per_px = native_mm_per_px
    return field_map


def prior_window_mm(px: float, py: float, yaw_deg: float,
                    pos_mm: float = PRIOR_POS_MM,
                    yaw_deg_slack: float = PRIOR_YAW_DEG,
                    footprint_mm: float = 0.0) -> np.ndarray:
    """A field-millimetre bounding box of everything the robot could be seeing.

    Round the prior, not the truth: the sensor gives `px, py, yaw_deg` plus or
    minus `pos_mm` and `yaw_deg_slack`, so the window is that disc swept through
    that yaw range, grown by the patch footprint. With no yaw slack and no
    footprint it degenerates to the position disc, which is the smallest honest
    window and the reason the defaults are worth trusting.

    Returned as (x0, y0, x1, y1) in field millimetres. The yaw slack is applied
    by rotating the offset the robot's centre could have, because a yaw error
    moves the ground the camera is looking at sideways, not the robot.
    """
    if pos_mm <= 0 and yaw_deg_slack <= 0 and footprint_mm <= 0:
        return np.array([px, py, px, py], np.float64)

    # Every reachable robot centre: the prior disc, plus the footprint radius
    # so the *far edge* of what the patch sees is inside the window too.
    reach = pos_mm + footprint_mm
    if yaw_deg_slack > 0:
        angles = np.radians(np.linspace(yaw_deg - yaw_deg_slack,
                                       yaw_deg + yaw_deg_slack, 9))
        offsets = np.stack([np.cos(angles), np.sin(angles)], 1) * reach
        centres = np.vstack([[px, py], [px, py] + offsets])
    else:
        centres = np.array([[px, py]], np.float64)

    # A yaw error rotates the footprint about the robot, so the window has to
    # contain the footprint disc at every reachable centre as well.
    reach_mm = footprint_mm + pos_mm
    x0 = float(centres[:, 0].min() - reach_mm)
    y0 = float(centres[:, 1].min() - reach_mm)
    x1 = float(centres[:, 0].max() + reach_mm)
    y1 = float(centres[:, 1].max() + reach_mm)
    return np.array([x0, y0, x1, y1], np.float64)


def window_features(field_map: FieldMap, window_mm: np.ndarray,
                    points_px: np.ndarray, descriptors: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, int]:
    """The subset of `points_px`/`descriptors` inside a field-millimetre window.

    Returns `(points_px, descriptors, dropped)`. `dropped` is what the matcher
    never saw, which is the number worth watching: a window that hides the true
    match is a wrong prior, not a matcher failure, and looks identical from the
    outside except for a collapse in matches.
    """
    if field_map.points_px is None or len(points_px) == 0:
        return points_px, descriptors, 0

    mm = field_map.px_to_field_mm(points_px)
    x0, y0, x1, y1 = window_mm
    keep = ((mm[:, 0] >= x0) & (mm[:, 0] <= x1)
            & (mm[:, 1] >= y0) & (mm[:, 1] <= y1))
    return points_px[keep], descriptors[keep], int((~keep).sum())


# --- Features --------------------------------------------------------------


def _brisk_factory() -> object:
    """BRISK moved from `cv2` to `cv2.xfeatures2d` in OpenCV 5, so accept both.

    Resolved once, at import, rather than per call: a missing BRISK should fail
    loudly at start-up with a clear message, not part-way through a run.
    """
    if hasattr(cv, "BRISK_create"):
        return cv.BRISK_create
    if hasattr(cv, "xfeatures2d") and hasattr(cv.xfeatures2d, "BRISK_create"):
        return cv.xfeatures2d.BRISK_create
    raise RuntimeError(
        "BRISK is unavailable: no cv.BRISK_create or cv.xfeatures2d.BRISK_create "
        f"in OpenCV {cv.__version__}. Install 'opencv-contrib-python'.")


BRISK_CREATE = _brisk_factory()


def build_detector(threshold: int = BRISK_THRESHOLD):
    # Positional: the parameter is `thresh` in OpenCV 5 and `threshold` in 4.
    return BRISK_CREATE(threshold, BRISK_OCTAVES, BRISK_PATTERN_SCALE)


def detect(image: np.ndarray, detector) -> tuple[list, np.ndarray]:
    """Keypoints and their BRISK descriptors, as (keypoints, Nx64 uint8 array).

    With `SCALE["detect_scale"] > 1` the image is decimated before detection and
    the keypoints are scaled back up afterwards, so callers still receive
    coordinates on the working grid and nothing downstream needs to know. This
    trades real accuracy for detection time: at 2.0 roughly three quarters of the
    pixels are discarded before BRISK ever sees them, and a descriptor computed
    on the decimated image is not the same descriptor.
    """
    factor = SCALE["detect_scale"]
    work = image
    if factor > 1.0:
        work = cv.resize(image, (max(1, int(round(image.shape[1] / factor))),
                                 max(1, int(round(image.shape[0] / factor)))),
                         interpolation=cv.INTER_AREA)
    keypoints, descriptors = detector.detectAndCompute(work, None)
    if descriptors is None:
        return list(keypoints), np.zeros((0, 64), np.uint8)
    if factor > 1.0:
        # Rescale the keypoint geometry, not just the coordinates: a descriptor
        # is only reusable at the size it was computed, but the *position* has
        # to come back to the working grid or the fit would be in the wrong
        # units. `size` and `octave` are left alone because they describe the
        # detection scale, which is what they were measured at.
        sx = image.shape[1] / float(work.shape[1])
        sy = image.shape[0] / float(work.shape[0])
        for kp in keypoints:
            kp.pt = (kp.pt[0] * sx, kp.pt[1] * sy)
    return list(keypoints), descriptors


def keypoint_px(keypoints: list) -> np.ndarray:
    if not keypoints:
        return np.zeros((0, 2), np.float64)
    return np.array([kp.pt for kp in keypoints], np.float64)


# --- Matching --------------------------------------------------------------


def build_matcher() -> cv.BFMatcher:
    """Hamming matcher: BRISK descriptors are binary.

    `crossCheck` is deliberately off. With it on, knnMatch degenerates to the
    single best match per keypoint (matching OpenCV gives each direction one
    candidate), so the ratio test below loses the second-nearest neighbour it
    needs and cannot reject ambiguous matches. Cross-checking is done explicitly
    in `match_features` instead, where both directions are still available.
    """
    return cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)


def match_features(query_desc: np.ndarray, train_desc: np.ndarray,
                   matcher: cv.BFMatcher, ratio: float = RATIO_TEST
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Mutually-best matches that also pass Lowe's ratio test.

    Returns index arrays `(query_idx, train_idx)` into the two descriptor sets.

    Two filters, because one is not enough here. The ratio test drops matches
    whose best candidate is not clearly better than its second, which is what a
    repeated texture produces. Cross-checking then drops the rest of the
    one-way matches that survive on both sides but disagree, which is what the
    map's left-right symmetry produces.
    """
    if len(query_desc) < 2 or len(train_desc) < 2:
        return np.zeros(0, int), np.zeros(0, int)

    def one_way(src: np.ndarray, dst: np.ndarray) -> dict[int, int]:
        pairs = matcher.knnMatch(src, dst, k=2)
        kept: dict[int, int] = {}
        for pair in pairs:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < ratio * second.distance:
                kept[best.queryIdx] = best.trainIdx
        return kept

    forward = one_way(query_desc, train_desc)
    backward = one_way(train_desc, query_desc)

    qi, ti = [], []
    for q, t in forward.items():
        if backward.get(t) == q:            # mutually best
            qi.append(q)
            ti.append(t)
    return np.array(qi, int), np.array(ti, int)


# --- Pose ------------------------------------------------------------------


def solve_rigid(query_px: np.ndarray, train_px: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """RANSAC fit of map_px = R(theta) @ query_px + t with the scale fixed at 1.

    Both images are one millimetre per pixel, so the only unknowns are the
    rotation and the translation; `estimateAffinePartial2D` is asked for exactly
    that and would be free to drift off unit scale, so the fitted scale is
    checked afterwards. Without the check a wrong-but-self-consistent scale can
    pass RANSAC by shrinking the patch onto a similar-looking region.

    Returns `(M, inlier_mask, inlier_ratio)`, or None when the fit is too weak
    to trust. Thresholds come from `ACTIVE`, so they follow the working grid
    rather than assuming the 2 mm/px they were measured at.
    """
    if len(query_px) < ACTIVE["min_inliers"]:
        return None

    M, inliers = cv.estimateAffinePartial2D(
        query_px, train_px, method=cv.RANSAC,
        ransacReprojThreshold=ACTIVE["ransac_px"],
        maxIters=5000, confidence=RANSAC_CONFIDENCE, refineIters=10)
    if M is None or inliers is None:
        return None

    mask = inliers.ravel().astype(bool)
    count = int(mask.sum())
    if count < ACTIVE["min_inliers"]:
        return None

    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if not 0.9 <= scale <= 1.1:             # the warp fixes this at exactly 1
        return None

    # A tight cluster of inliers is how a repeated texture passes everything
    # else; a real fix has matches spread across the patch it covers.
    spread = query_px[mask]
    span = float(np.hypot(*(spread.max(0) - spread.min(0))))
    if span < ACTIVE["min_inlier_span_px"]:
        return None

    return M, mask, count / float(len(query_px))


def camera_patch_px(warper: GroundWarper) -> np.ndarray:
    """The rectified-patch pixel directly under the camera.

    `GroundWarper.warp` places rectified pixel (0,0) at ground (x0, y0), so the
    camera's ground point (the robot-frame origin) maps to
    `((0 - x0), (0 - y0)) / mm_per_px`. Deriving it keeps the camera's own
    position in the patch independent of any assumption about the footprint
    being centred, which it is not - the camera is pitched down, so it sees far
    more ground ahead than behind.
    """
    return np.array([-warper.x0 / warper.mm_per_px,
                     -warper.y0 / warper.mm_per_px], np.float64)


def pose_from_transform(M: np.ndarray, cam_patch_px: np.ndarray,
                        field_map: FieldMap) -> tuple[float, float, float]:
    """Robot field pose from `M` (patch px -> map px) and the camera's patch px.

    `M` maps a rectified-patch pixel to the map pixel showing the same ground,
    so pushing the camera's own patch pixel through it lands on the map point
    under the camera - which is the robot.

    Heading: both the patch and the map use the *image* convention, x right and
    y down, so `atan2(M[1,0], M[0,0])` is the angle the patch's +x axis makes
    with the map's +x axis, measured clockwise on screen. The field convention
    is x right and y up with yaw anticlockwise, which needs the sign flipped;
    a further quarter turn is needed because the patch's +x axis is not the
    robot's heading, it is the patch's "far ground" direction.

    The result is `90 - atan2(...)`. Measured, not assumed: over the 30 captures
    that produce a fix, `fitted - truth` comes out at -90.10 deg with a 1.38 deg
    spread, while the mirror hypothesis (negating instead) has a 95.6 deg
    spread. A constant offset that tight means the fit is right and only the
    axis is off, and 90 deg is the term that removes it - the same reconciliation
    `opencv.py`'s renderer documents for its own overlay yaw.
    """
    camera_map_px = M[:, :2] @ cam_patch_px + M[:, 2]
    xy_mm = field_map.px_to_field_mm(np.array([camera_map_px]))[0]
    heading_deg = 90.0 - np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    return float(xy_mm[0]), float(xy_mm[1]), float(heading_deg)


# --- Per-capture pipeline --------------------------------------------------


@dataclass
class Solve:
    """One capture's outcome, kept for reporting and for the JSON output."""

    capture: str
    robot_mm: tuple[float, float] | None = None
    heading_deg: float | None = None
    truth_mm: tuple[float, float] | None = None
    truth_yaw_deg: float | None = None
    error_mm: float | None = None
    error_deg: float | None = None
    keypoints: int = 0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    fitted_scale: float = 0.0
    map_candidates: int = 0
    map_dropped: int = 0
    ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "capture": self.capture,
            "robot_mm": ([round(v, 1) for v in self.robot_mm]
                         if self.robot_mm else None),
            "robot_heading_deg": (round(self.heading_deg, 2)
                                  if self.heading_deg is not None else None),
            "truth_mm": ([round(v, 1) for v in self.truth_mm]
                         if self.truth_mm else None),
            "truth_yaw_deg": (round(self.truth_yaw_deg, 2)
                              if self.truth_yaw_deg is not None else None),
            "error_mm": round(self.error_mm, 1) if self.error_mm else None,
            "error_deg": round(self.error_deg, 2) if self.error_deg else None,
            "keypoints": self.keypoints,
            "good_matches": self.good_matches,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 3),
            "fitted_scale": round(self.fitted_scale, 4),
            "map_candidates": self.map_candidates,
            "map_dropped": self.map_dropped,
            "ms": round(self.ms, 1),
        }


def solve_capture(path: str, warper: GroundWarper, field_map: FieldMap,
                  detector, matcher: cv.BFMatcher,
                  map_desc: np.ndarray, map_px: np.ndarray,
                  prior: tuple[float, float, float] | None = None,
                  keep_matches: bool = False) -> tuple[Solve, dict]:
    """Rectify one capture, match it to the map, and solve the robot pose.

    `prior` is the odometry estimate as `(x_mm, y_mm, yaw_deg)` on the field. If
    given, only the map features inside `prior_window_mm` are offered to the
    matcher; if not, the whole map is, which is the no-prior baseline.

    `keep_matches` draws the match view even when the fit fails, so a video built
    from the frames has no gaps.
    """
    solve = Solve(capture=os.path.basename(path))
    view: dict[str, np.ndarray] = {}

    gray = imageio.imread_gray(path)
    rectified = warper.warp(gray)
    view["rectified"] = rectified

    t0 = time.perf_counter()
    kps, desc = detect(rectified, detector)
    patch_px = keypoint_px(kps)
    solve.keypoints = len(kps)
    view["patch_keypoints"] = kps

    candidates_px, candidates_desc = map_px, map_desc
    if prior is not None:
        footprint_mm = float(max(rectified.shape[:2]) * warper.mm_per_px)
        window = prior_window_mm(prior[0], prior[1], prior[2], PRIOR_POS_MM,
                                 PRIOR_YAW_DEG, footprint_mm)
        candidates_px, candidates_desc, dropped = window_features(
            field_map, window, map_px, map_desc)
        solve.map_candidates = len(candidates_px)
        solve.map_dropped = dropped
        view["window"] = window
    else:
        solve.map_candidates = len(map_px)

    qi, ti = match_features(desc, candidates_desc, matcher)
    solve.good_matches = len(qi)

    if len(qi) >= ACTIVE["min_inliers"]:
        fitted = solve_rigid(patch_px[qi], candidates_px[ti])
        if fitted is not None:
            M, mask, ratio = fitted
            solve.fitted_scale = float(np.hypot(M[0, 0], M[1, 0]))
            solve.inliers = int(mask.sum())
            solve.inlier_ratio = ratio
            x_mm, y_mm, heading = pose_from_transform(
                M, camera_patch_px(warper), field_map)
            solve.robot_mm = (x_mm, y_mm)
            solve.heading_deg = heading
            view["matches"] = draw_matches(rectified, field_map.gray,
                                           patch_px[qi], candidates_px[ti], mask)
        elif keep_matches:
            # Matches but no fit: draw them all so the frame is not blank.
            view["matches"] = draw_matches(rectified, field_map.gray,
                                           patch_px[qi], candidates_px[ti], None)

    if keep_matches and "matches" not in view:
        # Nothing matched at all; show the patch beside the map anyway.
        view["matches"] = draw_matches(rectified, field_map.gray,
                                       np.zeros((0, 2), np.float64),
                                       np.zeros((0, 2), np.float64), None,
                                       label="no matches")

    solve.ms = (time.perf_counter() - t0) * 1000.0

    truth = reference_pose(path)
    if truth is not None:
        solve.truth_mm = (truth[0], truth[1])
        solve.truth_yaw_deg = truth[2]
        if solve.robot_mm is not None:
            solve.error_mm = float(np.hypot(solve.robot_mm[0] - truth[0],
                                            solve.robot_mm[1] - truth[1]))
            solve.error_deg = pose.wrap_deg(solve.heading_deg - truth[2])

    overlay = draw_pose_overlay(field_map, solve)
    if overlay is not None:
        view["overlay"] = overlay
    return solve, view


# A solution is only worth drawing if the robot's own marker is legible; the
# map overlays are downscaled so a contact sheet of a whole run stays readable.
OVERLAY_MAX_SIDE = 900
OVERLAY_COLOR = (0, 0, 255)
OVERLAY_TRUTH_COLOR = (0, 200, 0)

# Match annotations. Doubled from the original 1 px lines and 3 px dots, because
# at 2282x1500 the single-pixel lines read as noise rather than as matches.
MATCH_LINE_THICKNESS = 2
MATCH_DOT_RADIUS = 6
MATCH_DOT_INNER_RADIUS = 3

# Rejected matches get their own marker, deliberately smaller than the inlier
# one so the two kinds are separable by size alone - colour is unreliable once
# the frame is scaled down for a video, and the inliers draw over the rejects
# where they overlap.
MATCH_REJECT_RADIUS = 3
MATCH_REJECT_LINE_THICKNESS = 1

# `--feature-images` output. The map is 2000x3000 at 1 mm/px, so a marker wide
# enough to see would be 5-10 px and 6000 of them on one image is unreadable;
# downscaling to this makes each keypoint a few pixels and the density legible.
FEATURE_MAX_SIDE = 1200
FEATURE_COLOR = (0, 255, 0)
# Keypoints standing on the edge of the rectified patch's valid region. Drawn in
# a warning colour because they are detector artefacts from the warp's intensity
# cliff, not features on the field.
FEATURE_EDGE_COLOR = (0, 165, 255)

# The pose-accuracy inset, in the bottom-left corner of each match frame. It
# re-draws the solved and true poses at their real separation, which is the one
# thing the match view cannot show: a few hundred millimetres of error is
# invisible when the two panels are the patch and the whole field.
#
# Sized as a fraction of the frame rather than a fixed pixel count: at 4 mm/px
# the patch panel is only a few hundred pixels wide, so a fixed 420 px inset
# covered most of it and hid the very matches the frame exists to show.
INSET_SIZE_FRAC = 0.34
INSET_MIN_PX = 260
INSET_MAX_PX = 560
INSET_MARGIN_PX = 16
INSET_BG_COLOR = (32, 32, 32)
INSET_RANGE_MM = 500.0          # half-width of the inset, so it spans 1 m
# Arrow length is in *drawn* pixels, not millimetres: at the exaggerated scale a
# 220 mm arrow would be over twice the panel wide and run off the edge, so the
# arrows are specified directly in pixels and stay put whatever the gain is.
INSET_ARROW_PX = 105.0
# The error is drawn exaggerated by this factor. On a well-behaved set the real
# error is under a millimetre, which at any honest scale is one pixel and looks
# like a perfect fix; exaggerating makes the *direction* and rough size of the
# residual visible instead. The panel is therefore not to scale, and the printed
# `error NN.N mm` is the only trustworthy number on it - the scale bar is drawn
# at the exaggerated scale so the ring and bar stay consistent with the arrows.
INSET_ERROR_GAIN = 12.0
ACCURACY_AGREE_MM = 20.0        # below this the fix is called a match
ACCURACY_CLOSE_MM = 150.0       # below this it is called close


def draw_pose_overlay(field_map: FieldMap, solve: "Solve") -> np.ndarray | None:
    """The map with the solved pose drawn on it, and the truth if the name has it.

    Drawn on the map rather than the patch because the map is the fixed
    reference: two overlays from different captures are directly comparable, and
    a fix that is a few hundred millimetres out is obvious at a glance.
    """
    if solve.robot_mm is None:
        return None

    frame = cv.cvtColor(field_map.gray, cv.COLOR_GRAY2BGR)
    viz.draw_pose(frame, field_map.field_to_map_px, solve.robot_mm[0],
                  solve.robot_mm[1], solve.heading_deg, OVERLAY_COLOR,
                  f"solved ({solve.error_mm:.0f}mm)" if solve.error_mm is not None
                  else "solved")
    if solve.truth_mm is not None and solve.truth_yaw_deg is not None:
        viz.draw_pose(frame, field_map.field_to_map_px, solve.truth_mm[0],
                      solve.truth_mm[1], solve.truth_yaw_deg,
                      OVERLAY_TRUTH_COLOR, "truth", thickness=2)

    return imageio.resize_longest_side(frame, OVERLAY_MAX_SIDE)


def draw_matches(patch: np.ndarray, map_gray: np.ndarray,
                 patch_px: np.ndarray, map_px: np.ndarray,
                 mask: np.ndarray | None = None, limit: int = 0,
                 label: str | None = None) -> np.ndarray:
    """Every match drawn from the patch to the map, inliers picked out.

    `mask` marks the RANSAC inliers of a solved fit. Those are drawn solid and
    bright; everything else that the matcher proposed is drawn faint, thin, and
    with a smaller marker of its own, so the frame shows what was *rejected* as
    well as what was kept. That is the point of drawing the rejects: a fix
    carried by three strong matches and a fix carried by none look identical if
    only the inliers are shown, and the faint marks are what tell them apart.

    Size carries the distinction, not colour: the frame gets scaled down for a
    video and the inliers draw over the rejects, so hue alone would not survive.
    `limit=0` means no cap, i.e. draw every match.
    """
    n = len(patch_px)
    if mask is None:
        inlier_idx = np.zeros(0, int)
        reject_idx = np.arange(n)
    else:
        keep = mask.astype(bool)
        inlier_idx = np.flatnonzero(keep)
        reject_idx = np.flatnonzero(~keep)
    if limit > 0:
        inlier_idx = inlier_idx[:limit]
        reject_idx = reject_idx[:max(0, limit - len(inlier_idx))]

    left = cv.cvtColor(patch, cv.COLOR_GRAY2BGR)
    right = cv.cvtColor(map_gray, cv.COLOR_GRAY2BGR)
    h = max(left.shape[0], right.shape[0])
    left = cv.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv.BORDER_CONSTANT,
                             value=(40, 40, 40))
    right = cv.copyMakeBorder(right, 0, h - right.shape[0], 0, 0,
                              cv.BORDER_CONSTANT, value=(40, 40, 40))
    canvas = np.hstack([left, right])
    off = left.shape[1]
    # H.264 with yuv420p needs even dimensions, and the two panels side by side
    # can easily come out odd. Pad the right edge by one column here so the
    # frames are directly encodable and no ffmpeg scale filter is needed.
    if canvas.shape[1] % 2:
        canvas = cv.copyMakeBorder(canvas, 0, 0, 0, 1, cv.BORDER_CONSTANT,
                                   value=(40, 40, 40))
    if canvas.shape[0] % 2:
        canvas = cv.copyMakeBorder(canvas, 0, 1, 0, 0, cv.BORDER_CONSTANT,
                                   value=(40, 40, 40))
    rng = np.random.default_rng(0)

    # Rejects first, so the inliers draw over them where the two overlap. Every
    # reject still gets a marker on both panels, just a smaller one than an
    # inlier: the line alone is hard to place against a busy map, and the whole
    # point of drawing them is to see where the matcher went wrong.
    for i in reject_idx:
        a = tuple(np.rint(patch_px[i]).astype(int))
        b = tuple(np.rint(map_px[i] + [off, 0]).astype(int))
        cv.line(canvas, a, b, (78, 78, 118), MATCH_REJECT_LINE_THICKNESS,
                cv.LINE_AA)
        cv.circle(canvas, a, MATCH_REJECT_RADIUS, (90, 90, 150), -1, cv.LINE_AA)
        cv.circle(canvas, b, MATCH_REJECT_RADIUS, (90, 90, 150), -1, cv.LINE_AA)

    for i in inlier_idx:
        a = tuple(np.rint(patch_px[i]).astype(int))
        b = tuple(np.rint(map_px[i] + [off, 0]).astype(int))
        cv.line(canvas, a, b, tuple(int(v) for v in rng.integers(80, 255, 3)),
                MATCH_LINE_THICKNESS, cv.LINE_AA)
        # Hollow centre on each dot keeps the exact keypoint visible when the
        # dot is this large, instead of burying it under a filled disc.
        cv.circle(canvas, a, MATCH_DOT_RADIUS, (0, 255, 0), -1, cv.LINE_AA)
        cv.circle(canvas, a, MATCH_DOT_INNER_RADIUS, (0, 0, 0), -1, cv.LINE_AA)
        cv.circle(canvas, b, MATCH_DOT_RADIUS, (0, 0, 255), -1, cv.LINE_AA)
        cv.circle(canvas, b, MATCH_DOT_INNER_RADIUS, (0, 0, 0), -1, cv.LINE_AA)

    # A count is the only way to tell a frame that is mostly rejects from one
    # that is mostly inliers at a glance, and the ratio is the signal that says
    # whether a fit is trustworthy. The two swatches beside it say which marker
    # size is which, so the key is on the frame rather than in a commit message.
    stats = (f"{len(inlier_idx)} inlier  {len(reject_idx)} rejected"
             if mask is not None else f"{n} matches")
    cv.putText(canvas, stats, (10, 52), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               (220, 220, 220), 1, cv.LINE_AA)
    key_y = 74
    cv.circle(canvas, (16, key_y), MATCH_DOT_RADIUS, (0, 255, 0), -1, cv.LINE_AA)
    cv.putText(canvas, "inlier", (30, key_y + 6), cv.FONT_HERSHEY_SIMPLEX, 0.45,
               (200, 200, 200), 1, cv.LINE_AA)
    cv.circle(canvas, (100, key_y), MATCH_REJECT_RADIUS, (90, 90, 150), -1,
              cv.LINE_AA)
    cv.putText(canvas, "rejected", (112, key_y + 6), cv.FONT_HERSHEY_SIMPLEX,
               0.45, (200, 200, 200), 1, cv.LINE_AA)
    if label:
        cv.putText(canvas, label, (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2, cv.LINE_AA)
    return canvas


def draw_accuracy_inset(canvas: np.ndarray, solve: "Solve") -> np.ndarray:
    """A zoomed inset of the solved and true positions, with both headings.

    The separation drawn between the two arrows is the position error multiplied
    by `INSET_ERROR_GAIN`, clamped so the arrow cannot leave the panel. Drawn at
    true scale the error on a well-behaved set is sub-pixel and the panel says
    nothing; exaggerated, the direction and rough magnitude of the residual are
    readable. The printed error is the real one, and the scale bar and agreement
    ring are drawn at the same exaggerated scale as the arrows so the panel is
    internally consistent - it is the relationship to the ground, not the
    internal geometry, that the gain breaks.

    Both arrows are always drawn. With a near-zero error they overlap, so the
    truth arrow is drawn first and the solved one over it; the truth stays
    visible because the drawn length differs slightly and the tails separate.
    """
    if solve.robot_mm is None or solve.truth_mm is None:
        return canvas

    h, w = canvas.shape[:2]
    size = int(round(min(h, w) * INSET_SIZE_FRAC))
    size = max(INSET_MIN_PX, min(INSET_MAX_PX, size))
    # Bottom-left corner, measured from the left edge rather than mirrored from
    # the right, so the margin is the same as it was on the other side.
    x0 = INSET_MARGIN_PX
    y0 = h - size - INSET_MARGIN_PX
    if x0 + size > w or y0 < 0:
        return canvas

    panel = np.full((size, size, 3), INSET_BG_COLOR, np.uint8)
    cx = cy = size / 2.0
    # Millimetres -> inset pixels, at the exaggerated scale. Field y is up, the
    # image row grows down.
    scale = (size / 2.0) / INSET_RANGE_MM * INSET_ERROR_GAIN

    def to_inset(dx_mm: float, dy_mm: float) -> tuple[int, int]:
        return (int(round(cx + dx_mm * scale)),
                int(round(cy - dy_mm * scale)))

    ox, oy = solve.robot_mm
    # Offsets from the solved pose; the solved pose is pinned at the centre so
    # the panel does not move around within the frame.
    tdx = (solve.truth_mm[0] - ox) * INSET_ERROR_GAIN
    tdy = (solve.truth_mm[1] - oy) * INSET_ERROR_GAIN
    # Clamp so a large error bends the arrow back inside the panel instead of
    # running off it and getting clipped to nothing.
    reach = size / 2.0 - 30.0
    mag = float(np.hypot(tdx, tdy)) * scale
    if mag > reach and mag > 0:
        k = reach / mag
        tdx, tdy = tdx * k, tdy * k

    # Agreement ring, drawn at the exaggerated scale like everything else.
    r = int(round(ACCURACY_AGREE_MM * scale))
    if 0 < r < size:
        cv.circle(panel, (int(cx), int(cy)), r, (90, 90, 90), 1, cv.LINE_AA)

    t = to_inset(tdx, tdy)
    s = to_inset(0.0, 0.0)
    err = solve.error_mm

    # Truth first, so the solved arrow stays on top where they overlap.
    if solve.truth_yaw_deg is not None:
        a = np.radians(solve.truth_yaw_deg)
        tip = (int(round(t[0] + np.cos(a) * INSET_ARROW_PX)),
               int(round(t[1] - np.sin(a) * INSET_ARROW_PX)))
        cv.arrowedLine(panel, t, tip, OVERLAY_TRUTH_COLOR, 3, cv.LINE_AA,
                       tipLength=0.22)
        cv.drawMarker(panel, t, OVERLAY_TRUTH_COLOR, cv.MARKER_CROSS, 16, 2,
                      cv.LINE_AA)

    # The error line, visible whenever the two poses are further apart than the
    # markers are wide, so the separation is traceable rather than implied.
    if abs(t[0] - s[0]) + abs(t[1] - s[1]) > 6:
        cv.line(panel, s, t, (150, 150, 150), 1, cv.LINE_AA)

    if solve.heading_deg is not None:
        a = np.radians(solve.heading_deg)
        tip = (int(round(s[0] + np.cos(a) * INSET_ARROW_PX)),
               int(round(s[1] - np.sin(a) * INSET_ARROW_PX)))
        cv.arrowedLine(panel, s, tip, OVERLAY_COLOR, 3, cv.LINE_AA,
                       tipLength=0.22)
    cv.drawMarker(panel, s, OVERLAY_COLOR, cv.MARKER_CROSS, 16, 2, cv.LINE_AA)

    cv.putText(panel, "solved", (14, size - 40), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               OVERLAY_COLOR, 2, cv.LINE_AA)
    cv.putText(panel, "truth", (100, size - 40), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               OVERLAY_TRUTH_COLOR, 2, cv.LINE_AA)

    # Scale bar: the largest round number of *real* millimetres that still fits,
    # drawn at the exaggerated scale and labelled with the real value so the
    # panel's own scale is readable without knowing the gain.
    bar_mm = 100.0
    while bar_mm > 1.0 and bar_mm * scale > size - 110:
        bar_mm /= 2.0
    bar_px = int(round(bar_mm * scale))
    by = size - 14
    cv.line(panel, (14, by), (14 + bar_px, by), (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(panel, f"{bar_mm:g}mm", (14, by - 6), cv.FONT_HERSHEY_SIMPLEX,
               0.42, (255, 255, 255), 1, cv.LINE_AA)

    if err is not None:
        tone = (OVERLAY_TRUTH_COLOR if err <= ACCURACY_AGREE_MM
                else (0, 215, 255) if err <= ACCURACY_CLOSE_MM else OVERLAY_COLOR)
        text = f"error {err:.1f} mm"
        if solve.error_deg is not None:
            text += f"  {solve.error_deg:+.1f} deg"
        cv.putText(panel, text, (14, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, tone,
                   2, cv.LINE_AA)
        cv.putText(panel, f"drawn x{INSET_ERROR_GAIN:.0f}", (14, 46),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv.LINE_AA)

    canvas[y0:y0 + size, x0:x0 + size] = panel
    cv.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (200, 200, 200), 1,
                 cv.LINE_AA)
    return canvas


def draw_keypoints(image: np.ndarray, keypoints: list,
                   colour: tuple[int, int, int] = (0, 255, 0),
                   label: str | None = None) -> np.ndarray:
    """Every detected keypoint marked on its own image, as a standalone picture.

    Unlike the match view this shows the *detector's* output, before matching or
    RANSAC filtered anything: no lines, no inlier/reject distinction, just what
    BRISK found. That is the right picture for asking whether detection is the
    weak link - a frame with plenty of matches but sparse keypoints points at
    the detector, one with dense keypoints and few matches points at matching.

    Drawn as circles with a centre dot rather than crosses, because at this
    density crosses merge into a solid mass and the individual features stop
    being countable.

    The invalid region of a rectified patch (black, where the camera saw no
    ground) is shaded, and keypoints standing on that boundary are drawn in a
    separate colour, because they are an artefact of the warp's intensity cliff
    rather than features on the field. They are worth seeing - they are what the
    detector picked up - but they must not be mistaken for usable features. On
    the measured set none of them survive matching, so this only affects the
    picture; the pose is unaffected either way.

    No padding is needed at the image edges: BRISK discards any candidate whose
    sampling pattern would fall outside the image, so no keypoint is ever
    reported closer to a border than half its descriptor pattern - measured at
    20 px on the 500x750 map at 4 mm/px, with zero keypoints inside that band.
    """
    frame = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    # Downscale before drawing so the markers stay distinguishable: at 1 mm/px
    # the map is 2000x3000 and a readable marker would need to be ~20 px.
    frame = imageio.resize_longest_side(frame, FEATURE_MAX_SIDE)
    scale = frame.shape[1] / float(image.shape[1])

    # Which pixels held no ground, i.e. the warp's fill. Pixel value alone does
    # not identify it: the map legitimately contains black markings (2% of its
    # pixels are exactly 0), so `image == 0` would mark real content as an
    # artefact. What does identify it is that unwarped ground cannot reach the
    # frame edge - the warp's valid region is bounded by the camera's field of
    # view - so the fill is the zero-valued component touching the border.
    invalid = np.zeros(image.shape, bool)
    if (image == 0).any():
        n_lab, labels = cv.connectedComponents((image == 0).astype(np.uint8))
        border = np.zeros(n_lab, bool)
        border[labels[0, :]] = True
        border[labels[-1, :]] = True
        border[labels[:, 0]] = True
        border[labels[:, -1]] = True
        border[0] = False                    # label 0 is the non-zero background
        invalid = border[labels]

    edge_px = np.zeros(image.shape, bool)
    if invalid.any() and not invalid.all():
        # Widen the fill by the descriptor sampling radius the artefact spans,
        # so "on the edge" means the same thing here as in the measurements
        # (3 px at this grid). Dilate the *valid* side back and take the ring.
        edge_px = (cv.dilate(invalid.astype(np.uint8),
                             np.ones((7, 7), np.uint8)).astype(bool)
                   & ~invalid)

    if invalid.any():
        shade = frame.copy()
        shade[invalid] = (0, 0, 90)          # dark red, unmistakably not ground
        frame = cv.addWeighted(shade, 0.55, frame, 0.45, 0)

    on_edge = 0
    for kp in keypoints:
        px, py = int(round(kp.pt[0])), int(round(kp.pt[1]))
        c = (int(round(kp.pt[0] * scale)), int(round(kp.pt[1] * scale)))
        if not (0 <= c[0] < frame.shape[1] and 0 <= c[1] < frame.shape[0]):
            continue
        radius = int(np.clip(kp.size * scale / 2.0, 1.5, 9.0))
        # A keypoint is "on the boundary" if its own pixel or its immediate
        # neighbourhood straddles the valid/invalid split.
        if edge_px.any() and 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
            if edge_px[py, px] or invalid[py, px]:
                on_edge += 1
                cv.circle(frame, c, radius, FEATURE_EDGE_COLOR, 1, cv.LINE_AA)
                cv.circle(frame, c, 1, FEATURE_EDGE_COLOR, -1, cv.LINE_AA)
                continue
        cv.circle(frame, c, radius, colour, 1, cv.LINE_AA)
        cv.circle(frame, c, 1, colour, -1, cv.LINE_AA)

    text = f"{len(keypoints)} keypoints"
    if on_edge:
        # The split matters more than the total: it says how much of what the
        # detector found is usable.
        text += f"  ({on_edge} on warp edge)"
    if label:
        text = f"{label}  {text}"
    # Shrink the text to fit rather than letting it run off the right edge; the
    # capture names are long and the patch panel is only a few hundred px wide.
    font, thickness = 0.7, 2
    while font > 0.3:
        (tw, _), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font, thickness)
        if tw <= frame.shape[1] - 20:
            break
        font -= 0.05
        if font <= 0.5:
            thickness = 1
    cv.putText(frame, text, (10, 26), cv.FONT_HERSHEY_SIMPLEX, font,
               (255, 255, 255), thickness, cv.LINE_AA)
    return frame


# --- Main ------------------------------------------------------------------


def build_parser(parser) -> None:
    parser.add_argument("--map", default=None,
                        help="Field map to match against. Default: "
                             "input/FieldBW.png.")
    parser.add_argument("--no-prior", action="store_true",
                        help="Offer the whole map instead of the odometry window. "
                             "Kept only so the window's effect can be re-measured.")
    parser.add_argument("--write-steps", action="store_true",
                        help="Also save each capture's rectified patch and "
                             "match view into out/features/.")
    parser.add_argument("--frames", action="store_true",
                        help="Save one match_<id>.png per capture into out/frames/ "
                             "for building a video with ffmpeg.")
    parser.add_argument("--feature-images", action="store_true",
                        help="Save every detected keypoint on the patch and on "
                             "the map as separate images in out/features_images/ "
                             "(the detector's raw output, no matching applied).")
    parser.add_argument("--scale", type=float, default=MM_PER_PX,
                        help=f"Working grid in mm/px (default {MM_PER_PX}). "
                             "Both the map and the patch are resampled to it.")
    parser.add_argument("--detect-scale", type=float, default=DETECT_SCALE,
                        help="Extra decimation before keypoint detection only, "
                             f"1.0 = detect at the working grid (default "
                             f"{DETECT_SCALE}). Does not change the grid the fit "
                             "is solved on.")
    parser.add_argument("--no-scale-thresholds", action="store_true",
                        help="Keep MIN_INLIERS and friends at their 2 mm/px "
                             "values instead of rescaling them with --scale. "
                             "Use to separate 'the grid is too coarse' from "
                             "'the thresholds no longer fit the grid'.")
    parser.add_argument("--prior-noise", type=float, nargs=2,
                        metavar=("POS_MM", "YAW_DEG"), default=None,
                        help="Perturb the ground-truth prior by a uniform "
                             "+/-POS_MM and +/-YAW_DEG before the matcher sees "
                             "it, to test the window against a realistic "
                             "odometry error. Default is off (exact prior). "
                             "Seeded per capture, so runs are repeatable.")
    parser.add_argument("--sweep", type=float, nargs="+", metavar="MM_PER_PX",
                        help="Run the whole set once per scale and print a "
                             "solve-rate/speed/accuracy table, then exit.")


def sweep_scales(scales: list[float], captures: list[str], map_path: str,
                 args, out_dir: str) -> int:
    """Run the capture set once per scale and print the trade-off table.

    The point of exposing this is that "make it faster" and "keep the solves"
    pull against each other, and where the knee sits is a property of the data,
    not of the code. Sweeping reports solve rate next to time and error so the
    choice is visible rather than assumed - a scale that halves the runtime by
    dropping a third of the fixes is a worse pipeline, not a faster one.
    """
    probe = imageio.imread_gray(captures[0])

    print(f"scale sweep over {len(captures)} capture(s), "
          f"detect-scale {args.detect_scale:.2f}")
    print(f"{'mm/px':>6} {'ms/frame':>9} {'speedup':>8} {'solved':>8} {'%':>6} "
          f"{'err mean':>9} {'err max':>8} {'head mean':>10}")
    base_ms = None
    for scale in sorted(scales):
        configure_scale(scale, args.detect_scale,
                        scale_thresholds=not args.no_scale_thresholds)
        warper = GroundWarper(probe.shape[1], probe.shape[0], scale)
        field_map = load_map(map_path, scale)
        detector = build_detector()
        matcher = build_matcher()
        map_kps, map_desc = detect(field_map.gray, detector)
        map_px = keypoint_px(map_kps)

        errs, heads, elapsed, solved = [], [], 0.0, 0
        for path in captures:
            t0 = time.perf_counter()
            solve, _ = solve_capture(path, warper, field_map, detector, matcher,
                                     map_desc, map_px, reference_pose(path))
            elapsed += time.perf_counter() - t0
            if solve.robot_mm is not None:
                solved += 1
                if solve.error_mm is not None:
                    errs.append(solve.error_mm)
                    heads.append(abs(solve.error_deg))

        ms = elapsed / len(captures) * 1000.0
        if base_ms is None:
            base_ms = ms
        e = np.array(errs) if errs else np.array([np.nan])
        hd = np.array(heads) if heads else np.array([np.nan])
        print(f"{scale:6.2f} {ms:9.1f} {base_ms / ms:7.2f}x "
              f"{solved:4d}/{len(captures):<3d} "
              f"{100.0 * solved / len(captures):5.1f} "
              f"{e.mean():8.1f}mm {e.max():7.1f}mm {hd.mean():9.2f}deg")
    return 0


def run(captures: list[str], out_dir: str, args) -> int:
    map_path = args.map or paths.map_path("features")
    if not os.path.exists(map_path):
        print(f"[error] map not found: {map_path}", file=sys.stderr)
        return 2

    if args.sweep:
        return sweep_scales(args.sweep, captures, map_path, args, out_dir)

    probe = imageio.imread_gray(captures[0])
    configure_scale(args.scale, args.detect_scale,
                    scale_thresholds=not args.no_scale_thresholds)
    warper = GroundWarper(probe.shape[1], probe.shape[0], args.scale)
    field_map = load_map(map_path, args.scale)

    # The map never changes, so its features are computed once for every capture.
    detector = build_detector()
    matcher = build_matcher()
    t0 = time.perf_counter()
    map_kps, map_desc = detect(field_map.gray, detector)
    map_build_ms = (time.perf_counter() - t0) * 1000.0
    field_map.keypoints = map_kps
    field_map.descriptors = map_desc
    field_map.points_px = keypoint_px(map_kps)
    map_px = field_map.points_px

    if not args.quiet:
        print(f"map       : {os.path.basename(map_path)} "
              f"{field_map.gray.shape[1]}x{field_map.gray.shape[0]} px @ "
              f"{field_map.mm_per_px:.3f} mm/px "
              f"(native {field_map.native_mm_per_px:.3f})")
        print(f"patch     : rectified {warper.out_size[0]}x{warper.out_size[1]} "
              f"px @ {warper.mm_per_px:.3f} mm/px")
        print(f"detector  : BRISK threshold {BRISK_THRESHOLD}, "
              f"{len(map_kps)} map keypoints in {map_build_ms:.0f} ms")
        print(f"grid      : {args.scale:.2f} mm/px working, "
              f"detect at {args.detect_scale:.2f}x, "
              f"min inliers {ACTIVE['min_inliers']:.0f}, "
              f"ransac {ACTIVE['ransac_px']:.1f} px")
        print(f"matcher   : Hamming, ratio {RATIO_TEST}, cross-checked")
        if args.no_prior:
            print("prior     : none, whole map offered to the matcher")
        else:
            noise = args.prior_noise
            if noise is None:
                print(f"prior     : window +/-{PRIOR_POS_MM:.0f} mm, "
                      f"+/-{PRIOR_YAW_DEG:.0f} deg, exact (from filename)")
            else:
                print(f"prior     : window +/-{PRIOR_POS_MM:.0f} mm, "
                      f"+/-{PRIOR_YAW_DEG:.0f} deg, noisy "
                      f"+/-{noise[0]:.0f} mm / +/-{noise[1]:.0f} deg (uniform)")
        print()

    solves, views = [], {}
    for path in captures:
        if args.no_prior:
            prior = None
        else:
            truth = reference_pose(path)
            if args.prior_noise is None:
                prior = truth
            else:
                # Seeded on the capture name so each capture gets a fixed
                # offset: reproducible runs, and two configs compared on the
                # same noise rather than on two different draws. `hash()` is
                # randomised per process, so the name's bytes are hashed
                # instead - string hashing would give a different draw each run.
                seed = int.from_bytes(
                    hashlib.sha256(os.path.basename(path).encode()).digest()[:4],
                    "big")
                prior = noisy_prior(truth, args.prior_noise[0],
                                    args.prior_noise[1], seed=seed)
        solve, view = solve_capture(path, warper, field_map, detector, matcher,
                                    map_desc, map_px, prior,
                                    keep_matches=args.frames)
        solves.append(solve)
        if view.get("rectified") is not None:
            views[os.path.basename(path)[:-4]] = view

        if args.quiet:
            continue

        # How far the prior was moved before the matcher saw it. Worth printing
        # next to the error: a large prior offset with a small final error is
        # the window doing its job, whereas a large offset that fails to solve
        # means the window hid the true match.
        prior_note = ""
        if prior is not None and solve.truth_mm is not None:
            dx = prior[0] - solve.truth_mm[0]
            dy = prior[1] - solve.truth_mm[1]
            dyaw = pose.wrap_deg(prior[2] - solve.truth_yaw_deg)
            if abs(dx) + abs(dy) + abs(dyaw) > 1e-9:
                prior_note = (f"  prior=({dx:+6.1f},{dy:+6.1f})mm "
                              f"{dyaw:+5.1f}deg")

        if solve.robot_mm is None:
            print(f"{solve.capture}\n   NO FIX  ({solve.keypoints} kp, "
                  f"{solve.good_matches} matches, "
                  f"{solve.map_dropped} map kp outside window)")
        else:
            print(f"{solve.capture}\n"
                  f"   robot=({solve.robot_mm[0]:+7.1f},{solve.robot_mm[1]:+7.1f})"
                  f"mm  heading={solve.heading_deg:+7.2f}deg"
                  f"  [{solve.keypoints} kp, {solve.good_matches} match, "
                  f"{solve.inliers} inlier, scale {solve.fitted_scale:.3f}, "
                  f"{solve.ms:.0f} ms]")
            if solve.truth_mm is not None:
                print(f"      truth=({solve.truth_mm[0]:+7.1f},"
                      f"{solve.truth_mm[1]:+7.1f})mm  yaw="
                      f"{solve.truth_yaw_deg:+7.2f}deg  ->  "
                      f"error={solve.error_mm:6.1f}mm {solve.error_deg:+6.2f}deg"
                      f"{prior_note}")

    fixed = [s for s in solves if s.robot_mm is not None]
    print(f"\nSolved {len(fixed)}/{len(solves)} capture(s).")
    if fixed:
        times = np.array([s.ms for s in fixed])
        print(f"Timing : {times.mean():.0f} ms mean, "
              f"{np.median(times):.0f} ms median, {times.max():.0f} ms max")

    checked = [s for s in fixed if s.error_mm is not None]
    if checked:
        pos = np.array([s.error_mm for s in checked])
        head = np.array([abs(s.error_deg) for s in checked])
        print(f"Error  : {len(checked)} fix(es) vs filename -> position "
              f"mean {pos.mean():.1f} / max {pos.max():.1f} mm, "
              f"heading mean {head.mean():.2f} / max {head.max():.2f} deg")

    if not args.no_output:
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, "feature_solves.json")
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump([s.as_dict() for s in solves], fh, indent=2)
            fh.write("\n")
        print(f"Output : {out_json}")

    if args.write_steps and views:
        steps_dir = os.path.join(out_dir, "features")
        os.makedirs(steps_dir, exist_ok=True)
        for name, view in views.items():
            cv.imwrite(os.path.join(steps_dir, f"{name}_rectified.png"),
                       view["rectified"])
            if "matches" in view:
                cv.imwrite(os.path.join(steps_dir, f"{name}_matches.png"),
                           view["matches"])
            if "overlay" in view:
                cv.imwrite(os.path.join(steps_dir, f"{name}_pose.png"),
                           view["overlay"])
        print(f"Steps  : {steps_dir}/")

    if args.feature_images:
        feats_dir = os.path.join(out_dir, "features_images")
        os.makedirs(feats_dir, exist_ok=True)
        # The map is the same for every capture, so it is written once. Doing it
        # per capture would produce byte-identical copies of a 2000x3000 image.
        cv.imwrite(os.path.join(feats_dir, "map_keypoints.png"),
                   draw_keypoints(field_map.gray, field_map.keypoints,
                                  FEATURE_COLOR, "map"))
        written_feats = 0
        for path in captures:
            view = views.get(os.path.basename(path)[:-4])
            if view is None or "patch_keypoints" not in view:
                continue
            stem = capture_stem(path)
            cv.imwrite(os.path.join(feats_dir, f"patch_{stem}_keypoints.png"),
                       draw_keypoints(view["rectified"],
                                      view["patch_keypoints"], FEATURE_COLOR,
                                      os.path.basename(path)[:-4]))
            written_feats += 1
        print(f"Feats  : {feats_dir}/ (map + {written_feats} patch image(s))")

    if args.frames and views:
        frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        # A run writes only its own captures, so an earlier run's frames would
        # survive as leftover indices and the sequence would interleave two
        # different orderings. Start clean instead.
        for stale in glob.glob(os.path.join(frames_dir, "match_*.png")):
            os.remove(stale)
        written = 0
        for path, solve in zip(captures, solves):
            view = views.get(os.path.basename(path)[:-4])
            if view is None or "matches" not in view:
                continue
            # Number by the capture id, not by position, so match_9 follows
            # match_8 rather than sorting after match_98.
            stem = capture_stem(path)
            # The accuracy inset needs error_mm, which only exists after the
            # truth is parsed above, so it is added here rather than inside
            # `draw_matches`. Written on a copy: the stored view keeps the bare
            # match panel for --write-steps and any later reuse.
            frame = draw_accuracy_inset(view["matches"].copy(), solve)
            cv.imwrite(os.path.join(frames_dir, f"match_{stem}.png"), frame)
            written += 1
        # The hint has to name files the way the caller's shell will see them.
        # Both paths are therefore built relative to the current directory,
        # never one relative and one bare: a bare `out/match_video.mp4` resolves
        # against the caller's cwd, which is `PythonVision/` for `run.sh`, and
        # there is no `out/` there - only `<algo>/out/`. The video belongs
        # beside this algorithm's other outputs, so it is written into `out_dir`.
        video_path = os.path.join(out_dir, "match_video.mp4")
        print(f"Frames : {frames_dir}/ ({written} frame(s))")
        print("         ffmpeg -y -framerate 30 -pattern_type glob -i "
              f"'{os.path.relpath(frames_dir)}/match_*.png' "
              f"-c:v libx264 -pix_fmt yuv420p '{os.path.relpath(video_path)}'")

    return 0 if fixed else 1


# `capture_sort_key` and the id regex live in `common.pose`; named here so the
# call sites read the same as they did when they were module globals.
capture_sort_key = pose.capture_sort_key


def capture_stem(path: str) -> str:
    """A capture's id as `%04d` for numbering frames, or its name if it has none."""
    match = pose.CAPTURE_ID.match(os.path.basename(path))
    return f"{int(match.group(1)):04d}" if match else \
        os.path.splitext(os.path.basename(path))[0]


if __name__ == "__main__":
    # Run standalone with the same flags the harness would pass along.
    import argparse

    _parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _parser.add_argument("captures", nargs="*")
    _parser.add_argument("--no-output", action="store_true")
    _parser.add_argument("--quiet", action="store_true")
    build_parser(_parser)
    _args, _extra = _parser.parse_known_args(sys.argv[1:])
    _paths = ([paths.resolve_capture("features", n) for n in _args.captures]
              or sorted(paths.captures("features"), key=capture_sort_key))
    raise SystemExit(run(_paths, paths.output_dir("features"), _args))
