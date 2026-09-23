"""Rectify `capture_*.png` onto the ground plane, then render it onto `FieldBW.png`.

The camera is bolted to the robot, so its pose relative to the ground plane is
constant and the pixel -> ground warp is the same for every capture. The robot's
(x, y, yaw) comes from the filename, which OnboardCamScript writes when saving,
so nothing is estimated.

Usage:
    python opencv.py                 # every capture_*.png
    python opencv.py capture_x.png   # specific files
    python opencv.py --steps         # also write each pipeline stage
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass

import cv2 as cv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

REFERENCE_NAME = "FieldBW.png"
CAPTURE_GLOB = "capture_*.png"
OUTPUT_DIR = os.path.join(HERE, "out")

# Fixed camera mounting.
CAMERA_VFOV_DEG = 70.0
CAMERA_PITCH_DEG = 45.0
CAMERA_HEIGHT_MM = 233.2

# Distance in mm from the camera's ground projection to the patch centre, along
# the heading. The filename gives the camera position but the overlay places the
# patch centre, so this offset is required. Measured as the mid-point of the
# valid content rows (~686 mm); 680 validated against the overlays by eye.
CAMERA_PATCH_CENTRE_MM = 680.0

# Field outline in mm, matching PosEstimator.FIELD_OUTLINE.
FIELD_WIDTH_MM = 2000.0
FIELD_HEIGHT_MM = 3000.0

# The camera footprint is wider than the field, so the map is padded with
# mid-grey to give the patch somewhere to land.
REFERENCE_PAD_MM = 1500.0
REFERENCE_PAD_VALUE = 128

# 0 saves at native resolution.
PREVIEW_MAX_SIDE = 0

QUAD_COLOR = (0, 0, 255)
QUAD_THICKNESS = 3
LABEL_COLOR = (0, 0, 255)
LABEL_SCALE = 1.1
LABEL_THICKNESS = 2

CORNER_ORDER = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.float32)


def imread_gray(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_for_display(img: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> np.ndarray:
    """Downscale to `max_side` on the longest edge; 0 or less means no limit."""
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return cv.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                     interpolation=cv.INTER_AREA)


def save_img(img: np.ndarray, path: str) -> None:
    cv.imwrite(path, resize_for_display(img))


def corners_px(shape: tuple[int, ...], M: np.ndarray) -> np.ndarray:
    """The four image corners of `shape` mapped through a 2x3 affine matrix."""
    h, w = shape[:2]
    pts = CORNER_ORDER * np.array([w - 1, h - 1], np.float32)
    return (M[:, :2] @ pts.T).T + M[:, 2]


def draw_label(img: np.ndarray, quad_px: np.ndarray, label: str) -> np.ndarray:
    """Outline a quad and write a label above its top-left corner."""
    cv.polylines(img, [np.int32(quad_px).reshape(-1, 1, 2)], True, QUAD_COLOR,
                 QUAD_THICKNESS, cv.LINE_AA)
    h, w = img.shape[:2]
    tx = int(np.clip(quad_px[:, 0].min(), 0, max(w - 1, 0)))
    ty = int(np.clip(quad_px[:, 1].min(), 0, max(h - 1, 0)))
    org = (tx, max(ty - 10, 20))
    (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, LABEL_SCALE,
                                 LABEL_THICKNESS)
    cv.rectangle(img, (org[0] - 4, org[1] - th - 6), (org[0] + tw + 4, org[1] + 6),
                 (255, 255, 255), cv.FILLED)
    cv.putText(img, label, org, cv.FONT_HERSHEY_SIMPLEX, LABEL_SCALE,
               LABEL_COLOR, LABEL_THICKNESS, cv.LINE_AA)
    return img


def affine_from_centre(centre_px: float, centre_py: float, yaw_deg: float,
                       scale: float, patch_shape: tuple[int, int]) -> np.ndarray:
    """2x3 matrix placing a patch centred on its own centre at a pose."""
    ph, pw = patch_shape[:2]
    a = np.radians(yaw_deg)
    ca, sa = np.cos(a), np.sin(a)
    XY = scale * np.array([[ca, -sa], [sa, ca]])
    t = np.array([centre_px, centre_py], float) - XY @ np.array([pw / 2.0, ph / 2.0])
    return np.array([[XY[0, 0], XY[0, 1], t[0]], [XY[1, 0], XY[1, 1], t[1]]])


def valid_region_mask(rectified: np.ndarray) -> np.ndarray:
    """1 where a rectified pixel holds content, 0 in the empty trapezoid wedges.

    Found by flooding the zero-valued background inwards from the border, so
    genuinely black scene content enclosed by the patch is kept.
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


class StepWriter:
    """Writes each pipeline stage to disk, numbered in call order."""

    def __init__(self, output_dir: str | None, enabled: bool = True) -> None:
        self.output_dir = output_dir
        self.enabled = enabled and output_dir is not None
        self._step = 0

    def save(self, img: np.ndarray, stage: str) -> None:
        if not self.enabled:
            return
        self._step += 1
        os.makedirs(self.output_dir, exist_ok=True)
        vis = to_bgr(img)
        cv.imwrite(os.path.join(self.output_dir, f"{self._step:02d}_{stage}.png"), vis)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Normalise grayscale/float/mask input into 8-bit BGR for saving."""
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


@dataclass
class CameraModel:
    """Fixed camera mounting and the ground-plane mapping it implies."""

    width: int
    height: int
    vfov_deg: float = CAMERA_VFOV_DEG
    pitch_deg: float = CAMERA_PITCH_DEG
    height_mm: float = CAMERA_HEIGHT_MM
    heading_xz: tuple[float, float] = (0.0, -1.0)

    @property
    def focal_px(self) -> float:
        return (self.height / 2.0) / np.tan(np.radians(self.vfov_deg / 2.0))

    @property
    def rotation(self) -> np.ndarray:
        """Camera-to-world rotation, columns (right, up, forward)."""
        p = np.radians(self.pitch_deg)
        hx, hz = self.heading_xz
        norm = np.hypot(hx, hz) or 1.0
        forward = np.array([hx / norm * np.cos(p), -np.sin(p), hz / norm * np.cos(p)])
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        return np.column_stack([right, np.cross(right, forward), forward])

    def capture_to_ground(self) -> np.ndarray:
        """Pixel -> ground homography, returning (X, Z) in millimetres.

        A pixel (u, v) with v downward gives the camera direction
        d = M @ [u, v, 1]; the ray hits y = 0 at t = -C_y / d_y, so both ground
        coordinates and the denominator are linear in [u, v, 1] and collapse
        into a single homography. The negated vertical term is what keeps image
        rows (which grow downward) consistent with the camera's up axis.
        """
        f = self.focal_px
        cx, cy = self.width / 2.0, self.height / 2.0
        k_inv = np.array([[1.0 / f, 0.0, -cx / f],
                          [0.0, -1.0 / f, cy / f],
                          [0.0, 0.0, 1.0]])
        M = self.rotation @ k_inv
        cx_m, cy_m, cz_m = 0.0, self.height_mm / 1000.0, 0.0
        depth = M[1]
        H = np.vstack([cx_m * depth - cy_m * M[0],
                       cz_m * depth - cy_m * M[2],
                       depth])
        return np.diag([1000.0, 1000.0, 1.0]) @ H

    def footprint_mm(self, samples: int = 65) -> tuple[float, float, float, float]:
        """Ground area covered, as (x0, z0, x1, z1) in mm.

        Sampled across the whole image rather than at the corners only, so the
        bounds follow the true projective curvature of the footprint.
        """
        H = self.capture_to_ground()
        xs = np.linspace(0, self.width - 1, samples)
        ys = np.linspace(0, self.height - 1, samples)
        grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        projected = np.hstack([grid, np.ones((len(grid), 1))]) @ H.T

        # Row 2 is d_y; a ground hit needs t = -C_y / d_y > 0, so d_y < 0.
        depth = projected[:, 2]
        valid = depth < -1e-9
        if not valid.any():
            return (0.0, 0.0, 0.0, 0.0)
        gx = projected[valid, 0] / depth[valid]
        gz = projected[valid, 1] / depth[valid]
        return (float(gx.min()), float(gz.min()), float(gx.max()), float(gz.max()))


def rectify_capture(capture_gray, camera, scale=1.0, steps=None):
    """Warp a capture onto the ground plane.

    Returns the rectified patch, its mm-per-pixel, and the ground bounds it
    covers. Far ground has the most negative Z and lands at row 0, so the patch
    is a top-down view with the robot at the bottom.
    """
    H = camera.capture_to_ground()
    x0, z0, x1, z1 = camera.footprint_mm()

    pad = 0.02 * max(x1 - x0, z1 - z0)      # so sampling does not clip the edge
    x0, z0, x1, z1 = x0 - pad, z0 - pad, x1 + pad, z1 + pad

    out_w = max(1, int(round((x1 - x0) * scale)))
    out_h = max(1, int(round((z1 - z0) * scale)))
    T = np.array([[scale, 0.0, -x0 * scale],
                  [0.0, scale, -z0 * scale],
                  [0.0, 0.0, 1.0]])
    rectified = cv.warpPerspective(
        capture_gray, T @ H, (out_w, out_h),
        flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    if steps is not None:
        steps.save(rectified, "warped_ground_plane")
    return rectified, 1.0 / scale, (x0, z0, x1, z1)


@dataclass
class FieldReference:
    """The field map at 1 mm/px, padded, with the field origin at (pad_px, pad_px).

    Every mm <-> pixel conversion must go through the helpers, or the padding
    shifts each pose by a constant.
    """

    image: np.ndarray
    mm_per_px: float
    pad_px: int = 0

    @property
    def field_px(self) -> tuple[int, int]:
        return (self.image.shape[1] - 2 * self.pad_px,
                self.image.shape[0] - 2 * self.pad_px)

    def field_mm_to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return (self.pad_px + x_mm / self.mm_per_px,
                self.pad_px + y_mm / self.mm_per_px)


def load_field_reference(path: str, pad_mm: float = REFERENCE_PAD_MM,
                         pad_value: int = REFERENCE_PAD_VALUE) -> FieldReference:
    """Load the map and pad it so a wider-than-field patch has somewhere to land.

    Padding is not cosmetic: a patch can only be placed where it fits entirely,
    so against the bare field the footprint would have no valid position at all.
    """
    img = imread_gray(path)
    mm_per_px = FIELD_WIDTH_MM / float(img.shape[1])
    if abs(mm_per_px - FIELD_HEIGHT_MM / float(img.shape[0])) > 1e-3:
        print(f"[warn] {img.shape[1]}x{img.shape[0]} px for "
              f"{FIELD_WIDTH_MM:.0f}x{FIELD_HEIGHT_MM:.0f} mm; pixels are not square.",
              file=sys.stderr)

    pad_px = int(round(pad_mm / mm_per_px))
    if pad_px > 0:
        img = cv.copyMakeBorder(img, pad_px, pad_px, pad_px, pad_px,
                                cv.BORDER_CONSTANT, value=int(pad_value))
    return FieldReference(img, mm_per_px, pad_px)


def parse_pose_from_filename(path: str) -> tuple[float, float, float]:
    """Read the camera pose from a name like capture_<...>_x<mm>_y<mm>_yaw<deg>.

    Values are field millimetres from the field centre and refer to the camera at
    the centre of the image. Raises ValueError if any is missing, rather than
    defaulting to zero and silently placing the capture at the field centre.
    """
    values: dict[str, float] = {}
    for token in os.path.splitext(os.path.basename(path))[0].split("_"):
        for key in ("x", "y", "yaw", "pitch"):
            if token.startswith(key) and not token[len(key):len(key) + 1].isalpha():
                try:
                    values[key] = float(token[len(key):])
                except ValueError:
                    pass
                break

    missing = [k for k in ("x", "y", "yaw") if k not in values]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)}: missing {', '.join(missing)} in filename; "
            "expected '..._x<mm>_y<mm>_yaw<deg>_pitch<deg>.png'.")
    return values["x"], values["y"], values["yaw"]


def pose_to_centre_px(reference: FieldReference,
                      pose: tuple[float, float, float]) -> tuple[float, float, float]:
    """Field-centred camera pose -> patch centre in padded pixels, and render yaw.

    Reconciliation, both of which shift or mirror every overlay if wrong:

    * Origin: the pose is relative to the field centre, the renderer works from
      the map's top-left, so half the field is added.
    * Y: field +Y points up the image while rows increase downward, so Y is
      negated for the row and the render yaw is (-yaw + 90).

    The filename gives the camera; the renderer places the patch, which sits
    CAMERA_PATCH_CENTRE_MM further along the heading. Both components are added
    with the same sign - negating Y would move a yaw -90 capture ~1.4 m, about
    half the map.
    """
    x_mm, y_mm, yaw = pose
    a = np.radians(yaw)
    x_mm += np.cos(a) * CAMERA_PATCH_CENTRE_MM
    y_mm += np.sin(a) * CAMERA_PATCH_CENTRE_MM

    px, py = reference.field_mm_to_px(x_mm + FIELD_WIDTH_MM / 2.0,
                                      -y_mm + FIELD_HEIGHT_MM / 2.0)
    return px, py, -yaw + 90


def render_overlay(reference: FieldReference, reference_bgr: np.ndarray,
                   rectified: np.ndarray, rectified_mm_per_px: float,
                   centre_px: float, centre_py: float, yaw_deg: float,
                   label: str, blend: float = 0.5) -> np.ndarray:
    """Blend the patch onto the map at a pose, tinted red, with its outline drawn."""
    M = affine_from_centre(centre_px, centre_py, yaw_deg,
                           rectified_mm_per_px / reference.mm_per_px,
                           rectified.shape)
    out_shape = reference.image.shape[:2]
    placed = cv.warpAffine(rectified, M, out_shape[::-1], flags=cv.INTER_LINEAR,
                           borderMode=cv.BORDER_CONSTANT, borderValue=0)

    # Fade the trapezoid's empty wedges so they do not paint black over the map.
    coverage = cv.warpAffine(valid_region_mask(rectified), M, out_shape[::-1],
                             flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT,
                             borderValue=0).astype(np.float32)
    coverage = cv.GaussianBlur(coverage, (0, 0), 3.0)[..., None] * blend

    out = reference_bgr.astype(np.float32)
    patch = cv.cvtColor(placed, cv.COLOR_GRAY2BGR).astype(np.float32)
    patch[..., 0] *= 0.25                                   # B
    patch[..., 1] *= 0.25                                   # G
    patch[..., 2] = np.clip(patch[..., 2] * 1.3 + 40, 0, 255)  # R
    out = np.clip(out * (1.0 - coverage) + patch * coverage, 0, 255).astype(np.uint8)
    return draw_label(out, corners_px(rectified.shape, M), label)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rectify captures onto the ground plane and overlay them on "
                    "the field map.")
    parser.add_argument("captures", nargs="*",
                        help=f"Capture name(s). Default: every {CAPTURE_GLOB}.")
    parser.add_argument("--reference", default=os.path.join(HERE, REFERENCE_NAME))
    parser.add_argument("--pitch", type=float, default=CAMERA_PITCH_DEG)
    parser.add_argument("--vfov", type=float, default=CAMERA_VFOV_DEG)
    parser.add_argument("--height-mm", type=float, default=CAMERA_HEIGHT_MM)
    parser.add_argument("--no-show", action="store_true",
                        help="Accepted for run.sh compatibility; nothing is shown.")
    parser.add_argument("--no-output", action="store_true",
                        help="Do not write output files.")
    parser.add_argument("--steps", action="store_true",
                        help="Write each pipeline stage to out/steps/<capture>/.")
    return parser.parse_args(argv)


def resolve_captures(args: argparse.Namespace) -> list[str]:
    if not args.captures:
        return sorted(glob.glob(os.path.join(HERE, CAPTURE_GLOB)))
    paths = []
    for name in args.captures:
        path = name if os.path.isabs(name) else os.path.join(HERE, name)
        if os.path.exists(path):
            paths.append(path)
        else:
            print(f"[warn] not found: {path}", file=sys.stderr)
    return paths


def contact_sheet(overlays: list[tuple[str, np.ndarray]], tile_w: int = 620,
                  per_row: int = 3) -> np.ndarray:
    """Overlays tiled in a grid with a title bar, for judging a run at a glance."""
    tiles = []
    for name, img in overlays:
        tile = cv.resize(img, (tile_w, max(1, int(img.shape[0] * tile_w / img.shape[1]))),
                         interpolation=cv.INTER_AREA)
        bar = np.full((34, tile_w, 3), 255, np.uint8)
        cv.putText(bar, name, (6, 23), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                   cv.LINE_AA)
        tiles.append(np.vstack([bar, tile]))

    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    return np.vstack(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if not os.path.exists(args.reference):
        print(f"[error] reference image not found: {args.reference}", file=sys.stderr)
        return 2

    capture_paths = resolve_captures(args)
    if not capture_paths:
        print(f"[error] no captures matching {CAPTURE_GLOB}", file=sys.stderr)
        return 2

    reference = load_field_reference(args.reference)
    reference_bgr = cv.imread(args.reference, cv.IMREAD_COLOR)
    if reference.pad_px > 0:
        reference_bgr = cv.copyMakeBorder(
            reference_bgr, reference.pad_px, reference.pad_px, reference.pad_px,
            reference.pad_px, cv.BORDER_CONSTANT, value=(REFERENCE_PAD_VALUE,) * 3)

    fw, fh = reference.field_px
    print(f"Reference : {os.path.basename(args.reference)} ({fw}x{fh} px field + "
          f"{reference.pad_px} px pad, {reference.mm_per_px:.3f} mm/px)")
    print(f"Field     : {FIELD_WIDTH_MM:.0f} x {FIELD_HEIGHT_MM:.0f} mm")
    print(f"Captures  : {len(capture_paths)}\n")

    if not args.no_output:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    overlays: list[tuple[str, np.ndarray]] = []
    for path in capture_paths:
        name = os.path.basename(path)
        capture_gray = imread_gray(path)

        try:
            pose = parse_pose_from_filename(path)
        except ValueError as exc:
            print(f"{name}\n   [SKIP] {exc}")
            continue

        camera = CameraModel(capture_gray.shape[1], capture_gray.shape[0],
                             vfov_deg=args.vfov, pitch_deg=args.pitch,
                             height_mm=args.height_mm)
        steps = StepWriter(os.path.join(OUTPUT_DIR, "steps", name[:-4]),
                           enabled=args.steps and not args.no_output)

        rectified, mm_per_px, footprint = rectify_capture(
            capture_gray, camera, scale=1.0 / reference.mm_per_px, steps=steps)

        x_mm, y_mm, yaw_in = pose
        centre_px, centre_py, yaw = pose_to_centre_px(reference, pose)
        print(f"{name}\n   pose    : x={x_mm:.1f}mm y={y_mm:.1f}mm yaw={yaw_in:.1f}deg")
        print(f"   rectify : {rectified.shape[1]}x{rectified.shape[0]} px @ "
              f"{mm_per_px:.3f} mm/px, covers "
              f"{footprint[2] - footprint[0]:.0f} x {footprint[3] - footprint[1]:.0f} mm")

        label = f"{name}  x={x_mm:.0f} y={y_mm:.0f} yaw={yaw:.1f}"
        composed = render_overlay(reference, reference_bgr, rectified, mm_per_px,
                                  centre_px, centre_py, yaw, label)

        if not args.no_output:
            save_img(rectified, os.path.join(OUTPUT_DIR, f"{name}_rectified.png"))
            save_img(composed, os.path.join(OUTPUT_DIR, f"{name}_overlay.png"))
            overlays.append((name, composed))

    print(f"\nRendered {len(overlays)}/{len(capture_paths)} captures.")
    if overlays and not args.no_output:
        sheet = os.path.join(OUTPUT_DIR, "all_poses_overlay.png")
        save_img(contact_sheet(overlays), sheet)
        print(f"Pose overlays : {OUTPUT_DIR}/*_overlay.png")
        print(f"Contact sheet : {sheet}")

    return 0 if overlays else 1


if __name__ == "__main__":
    raise SystemExit(main())
