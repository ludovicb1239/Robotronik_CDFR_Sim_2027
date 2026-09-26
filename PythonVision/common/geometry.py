"""The fixed camera mounting, the field outline, and the ground-plane warp.

The camera is bolted to the robot, so its height, pitch and field of view are
constants and the pixel -> ground mapping is the same for every capture. Every
algorithm starts by rectifying a capture with that mapping, which is why it
lives here rather than in any one of them.

Two of the three algorithms also carry their own copy of parts of this, because
they were written to be readable end to end: `gradient` inlines the homography
and a tighter ground-bounds loop, and `features` reuses `aruco`'s `GroundWarper`
directly. Those stay where they are; this module is the shared statement of the
geometry, and the copies are checked against it.
"""

from __future__ import annotations

import math

import cv2 as cv
import numpy as np

# --- Field -----------------------------------------------------------------

# Field outline in millimetres, matching PosEstimator.FIELD_OUTLINE and the map.
FIELD_WIDTH_MM = 2000.0
FIELD_HEIGHT_MM = 3000.0

# Where each tag sits on the field, in millimetres. The four tags are also the
# field's fixed landmarks, so both the ArUco algorithm and the optical-flow
# initialisation in `gradient` need them.
TAG_FIELD_MM = {
    20: (-400.0, -900.0),
    21: (-400.0, 900.0),
    22: (400.0, -900.0),
    23: (400.0, 900.0),
}

# --- Camera -----------------------------------------------------------------

CAMERA_VFOV_DEG = 70.0          # vertical field of view
CAMERA_PITCH_DEG = 45.0         # fixed downward tilt
CAMERA_HEIGHT_MM = 233.2        # height above the ground plane

# Distance in millimetres from the camera's ground projection to the centre of
# the patch it sees, along the heading. The filename records the camera pose but
# a renderer places the patch centre, so this offset is required to reconcile
# the two. Measured as the mid-point of the valid content rows (~686 mm); 680
# validated against the overlays by eye.
CAMERA_PATCH_CENTRE_MM = 680.0

# The renderer's padding of the map, in millimetres and in pixel value. The
# camera footprint is wider than the field, so the map is padded so that a patch
# can be placed near an edge and still land entirely inside the image.
REFERENCE_PAD_MM = 1500.0
REFERENCE_PAD_VALUE = 128


def focal_px(height: int, vfov_deg: float = CAMERA_VFOV_DEG) -> float:
    """Focal length in pixels for a vertical field of view and image height."""
    return (height / 2.0) / math.tan(math.radians(vfov_deg / 2.0))


def camera_rotation(pitch_deg: float = CAMERA_PITCH_DEG,
                    heading_xz: tuple[float, float] = (0.0, -1.0)) -> np.ndarray:
    """Camera-to-world rotation, columns (right, up, forward).

    The camera is pitched down by `pitch_deg` from the heading given in the
    ground plane. `heading_xz` defaults to looking along -Z, which is the
    convention the map and the filename use.
    """
    p = math.radians(pitch_deg)
    hx, hz = heading_xz
    norm = math.hypot(hx, hz) or 1.0
    forward = np.array([hx / norm * math.cos(p), -math.sin(p), hz / norm * math.cos(p)])
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(right, forward), forward])


def capture_homography(width: int, height: int,
                       vfov_deg: float = CAMERA_VFOV_DEG,
                       pitch_deg: float = CAMERA_PITCH_DEG,
                       height_mm: float = CAMERA_HEIGHT_MM,
                       heading_xz: tuple[float, float] = (0.0, -1.0)) -> np.ndarray:
    """Pixel -> ground homography, giving `(X, Z)` in millimetres.

    A pixel `(u, v)`, with `v` measured downward, defines a camera ray whose
    direction is `rotation @ k_inv @ [u, v, 1]`. Intersecting that ray with the
    ground plane makes both ground coordinates ratios over the same denominator,
    so the whole mapping collapses into one 3x3 homography.

    The negated vertical term in `k_inv` keeps image rows, which grow downward,
    consistent with the camera's up axis; without it the warp is mirrored.
    """
    f = focal_px(height, vfov_deg)
    k_inv = np.array([[1.0 / f, 0.0, -width / 2.0 / f],
                      [0.0, -1.0 / f, height / 2.0 / f],
                      [0.0, 0.0, 1.0]])
    rotation = camera_rotation(pitch_deg, heading_xz)
    m = rotation @ k_inv

    # Ray/ground intersection: scale each ray until its vertical component
    # reaches Y = -height_mm, so the result is (X, Z, denominator).
    cam_y_m = height_mm / 1000.0
    h = np.vstack([-cam_y_m * m[0], -cam_y_m * m[2], m[1]])
    return np.diag([1000.0, 1000.0, 1.0]) @ h


def ground_bounds(homography: np.ndarray, width: int, height: int,
                  samples: int = 65) -> tuple[float, float, float, float]:
    """Ground area an image covers, as `(x0, z0, x1, z1)` in millimetres.

    Sampled over a grid rather than at the four corners: the bottom rows land
    behind the camera and are thrown far off to the sides, so corner-only bounds
    would be far too wide. Row 2 is the depth denominator, and a real ground hit
    needs it negative.
    """
    xs = np.linspace(0, width - 1, samples)
    ys = np.linspace(0, height - 1, samples)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    projected = np.hstack([grid, np.ones((len(grid), 1))]) @ homography.T

    depth = projected[:, 2]
    keep = depth < -1e-9
    gx = projected[keep, 0] / depth[keep]
    gz = projected[keep, 1] / depth[keep]
    return float(gx.min()), float(gz.min()), float(gx.max()), float(gz.max())


def patch_scale_transform(bounds: tuple[float, float, float, float],
                          mm_per_px: float) -> tuple[np.ndarray, tuple[int, int]]:
    """mm -> px scale/translate for a ground rectangle, and the output size.

    Composes with a capture homography to give the full pixel -> rectified-pixel
    warp, and returns the output `(width, height)` in pixels. A small pad keeps
    the sampling from clipping the footprint's edge.
    """
    x0, z0, x1, z1 = bounds
    pad = 0.02 * max(x1 - x0, z1 - z0)
    x0, z0, x1, z1 = x0 - pad, z0 - pad, x1 + pad, z1 + pad

    shape = (max(1, round((x1 - x0) / mm_per_px)),
             max(1, round((z1 - z0) / mm_per_px)))
    transform = np.array([[1.0 / mm_per_px, 0.0, -x0 / mm_per_px],
                          [0.0, 1.0 / mm_per_px, -z0 / mm_per_px],
                          [0.0, 0.0, 1.0]])
    return transform, shape


def corners_px(shape: tuple[int, ...], transform: np.ndarray) -> np.ndarray:
    """The four image corners of `shape`, through a 2x3 affine matrix."""
    h, w = shape[:2]
    corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.float32)
    pts = corners * np.array([w - 1, h - 1], np.float32)
    return (transform[:, :2] @ pts.T).T + transform[:, 2]


def valid_region_mask(rectified: np.ndarray) -> np.ndarray:
    """1 where a rectified pixel holds content, 0 in the empty trapezoid wedges.

    Found by flooding the zero-valued background inwards from the border, so
    genuinely black scene content enclosed by the patch is kept. A rectified
    patch is a trapezoid inside a rectangle, and painting the empty wedges over
    a map would read as black content rather than as "no data".
    """
    content = (rectified > 0).astype(np.uint8)
    h, w = content.shape[:2]
    if h < 3 or w < 3:
        return content

    scratch = np.zeros((h + 2, w + 2), np.uint8)
    outside = content.copy()
    empties = np.argwhere(outside == 0)
    on_edge = ((empties[:, 0] == 0) | (empties[:, 0] == h - 1)
               | (empties[:, 1] == 0) | (empties[:, 1] == w - 1))
    for y, x in empties[on_edge]:
        if outside[y, x] == 0:
            cv.floodFill(outside, scratch, (int(x), int(y)), 2)

    mask = np.ones((h, w), np.uint8)
    mask[outside == 2] = 0
    return mask
