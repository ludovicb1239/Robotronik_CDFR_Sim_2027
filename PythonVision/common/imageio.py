"""Reading and writing images, and the small conversions the scripts repeat."""

from __future__ import annotations

import os

import cv2 as cv
import numpy as np


def imread_gray(path: str) -> np.ndarray:
    """Read a single-channel 8-bit image, raising rather than returning None.

    A missing or unreadable capture is a real failure - there is nothing to
    localize against - so it raises instead of letting a `None` propagate into
    the pipeline and fail somewhere less obvious.
    """
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_longest_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest edge is `max_side`, or return it unchanged.

    `0` or less means no limit, which is the "save at native resolution" case.
    Used on every written image: the map is 2000x3000 and the rectified patch is
    thousands of pixels wide, so saving full size makes contact sheets and
    overlays unreadable.
    """
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return cv.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                     interpolation=cv.INTER_AREA)


def save_img(img: np.ndarray, path: str, max_side: int = 0) -> None:
    """Write an image, optionally downscaling its longest edge first."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv.imwrite(path, resize_longest_side(img, max_side))


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Normalise grayscale, mask, float or colour input into 8-bit BGR.

    Pipeline stages are saved with this so an intermediate view of a mask or a
    float accumulator still opens in an image viewer. A 0/1 mask is treated as a
    mask rather than as a picture, because normalising it by its range would
    turn a nearly-empty mask into a solid white image.
    """
    if img.ndim == 3:
        return img
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = (np.zeros_like(img, np.uint8) if hi - lo < 1e-9
               else ((img - lo) * (255.0 / (hi - lo))).astype(np.uint8))
    elif img.max() <= 1:            # a 0/1 mask, not a picture
        img = img * 255
    return cv.cvtColor(img, cv.COLOR_GRAY2BGR)


class StepWriter:
    """Writes each pipeline stage to disk, numbered in call order.

    Numbering with the call order rather than with a name is what makes the
    stage images read as a sequence: `01_warped_ground_plane.png` came before
    `02_...`, whatever the stages are called.
    """

    def __init__(self, output_dir: str | None, enabled: bool = True) -> None:
        self.output_dir = output_dir
        self.enabled = enabled and output_dir is not None
        self._step = 0

    def save(self, img: np.ndarray, stage: str) -> None:
        if not self.enabled:
            return
        self._step += 1
        os.makedirs(self.output_dir, exist_ok=True)
        cv.imwrite(os.path.join(self.output_dir, f"{self._step:02d}_{stage}.png"),
                   to_bgr(img))
