"""Rectifying a capture flat onto the ground plane.

`GroundWarper` was the ArUco algorithm's, but it is not ArUco-specific: it warps
a capture onto the ground, reports the plane's bounds, and converts rectified
pixels back to the robot's own frame. `features` already imported it, and the
`gradient` algorithm needs the same warp (it re-derives the homography inline
because its rectification is inside an optimiser and is skipped entirely in
marker mode).

Ground positions are reported in the robot's frame: x forwards, y left, origin
on the ground directly under the camera. A lens pitched down by less than half
its vertical field of view never sees straight down, so nothing nearer than
`near_x_mm` can appear in a capture; the frame's top row is `far_x_mm` away.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .geometry import (
    CAMERA_HEIGHT_MM,
    CAMERA_PITCH_DEG,
    CAMERA_VFOV_DEG,
    capture_homography,
    ground_bounds,
)


def near_x_mm(height_mm: float = CAMERA_HEIGHT_MM,
              pitch_deg: float = CAMERA_PITCH_DEG,
              vfov_deg: float = CAMERA_VFOV_DEG) -> float:
    """Nearest ground distance the camera can see, along the heading.

    The bottom image row owns the steepest ray, so this is where the frame
    starts. Nothing nearer than it can appear, which is why a rectified patch is
    a strip rather than a disc around the robot.
    """
    return height_mm / np.tan(np.radians(pitch_deg - vfov_deg / 2.0))


def far_x_mm(height_mm: float = CAMERA_HEIGHT_MM,
             pitch_deg: float = CAMERA_PITCH_DEG,
             vfov_deg: float = CAMERA_VFOV_DEG) -> float:
    """Farthest ground distance the camera can see, along the heading."""
    return height_mm / np.tan(np.radians(pitch_deg + vfov_deg / 2.0))


class GroundWarper:
    """Warps a capture flat onto the ground, built once and reused.

    The mounting is fixed, so the homography, the ground bounds and the output
    size are the same for every capture of a given size and scale; building the
    object once per run keeps that work off the per-capture path.

    Ground positions are reported in the robot's frame: x forwards, y left.

        x = -raw_y       (forwards)
        y = -raw_x       (left)

    Both homography axes run opposite to the robot frame's, so both are negated.
    Verified against captures whose true pose is known: with these signs the tag
    offset comes out right to ~2 mm on every capture, whereas either sign on its
    own leaves one axis out by hundreds of millimetres.

    x needs no offset beyond the sign: `capture_homography`'s y comes out of the
    projection already measured from the ground point under the robot, because
    at the image centre the ray is the mounting pitch and height/tan(pitch) is
    exactly the distance to that point. Adding a constant here biases every
    position by it, which is what a stray `far_x_mm` term used to do - 41 mm of
    error on every fix.
    """

    def __init__(self, width: int, height: int, mm_per_px: float) -> None:
        self.mm_per_px = mm_per_px
        self.h = capture_homography(width, height)
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
        """The capture as a top-down patch, at this warper's mm per pixel."""
        return cv.warpPerspective(gray, self.m, self.out_size,
                                  flags=cv.INTER_LINEAR,
                                  borderMode=cv.BORDER_CONSTANT, borderValue=0)

    def robot_mm(self, px: np.ndarray) -> np.ndarray:
        """Rectified pixels -> robot-frame (x forwards, y left) in millimetres."""
        raw_x = px[:, 0] * self.mm_per_px + self.x0
        raw_y = px[:, 1] * self.mm_per_px + self.y0
        return np.stack([-raw_y, -raw_x], 1)
