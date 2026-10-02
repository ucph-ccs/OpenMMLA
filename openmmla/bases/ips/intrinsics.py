"""The intrinsics a tag detector needs for the frames it is given.

A camera is calibrated at one frame size: its `Cameras` entry's `calibration_resolution`
([width, height]), which IPS Intrinsics writes from the checkerboard images. An entry
without it (one written before the key, or saved by a form that does not carry it) says nothing
of its size, and the principal point stands in: cx and cy lie near the centre of the calibration
images, so frames within 10% of (2cx, 2cy) are read with the intrinsics as they are, and others
are scaled from the common frame size nearest (2cx, 2cy) (1920x1080 for the Logitech C920, MacBook
Air and iPhone profiles). A frame of another size than the calibration's, such as the 960x540
recordings of 2024-12-10, needs fx, fy, cx and cy scaled by the width and height ratios: read with
the full-size intrinsics, every tag comes out twice as far away and off to one side, and no
position is in metres. The size compared is the frame's before `rotate` turns it; a fisheye
camera's frames are remapped with its own K first and are not scaled."""
from __future__ import annotations

import logging

# width and height ratios further apart than this mean the frame has another aspect than the
# calibration images (a cropped mode of the sensor): scaling cannot make such intrinsics right
ASPECT_TOLERANCE = 0.01
# how far (relative, per side) a frame size may lie from twice the principal point and still be the
# size the intrinsics were calibrated at; a principal point further off the frame's centre is warned of
PRINCIPAL_POINT_TOLERANCE = 0.10
# the frame sizes cameras commonly give, landscape (a portrait size is one of these turned)
COMMON_FRAME_SIZES = (
    (640, 480), (800, 600), (960, 720), (1024, 768), (1280, 960), (1600, 1200), (2048, 1536),
    (2592, 1944), (3264, 2448), (4032, 3024),
    (640, 360), (960, 540), (1280, 720), (1600, 900), (1920, 1080), (2560, 1440), (3840, 2160),
    (1280, 800), (1920, 1200),
)


def calibration_resolution(camera_config: dict | None) -> tuple[int, int] | None:
    """(width, height) the camera's `Cameras` entry says its intrinsics were calibrated at, None
    when it does not say (FrameIntrinsics then goes by the principal point)."""
    value = (camera_config or {}).get('calibration_resolution') if isinstance(camera_config, dict) else None
    try:
        width, height = (int(round(float(v))) for v in value)
        if width > 0 and height > 0:
            return width, height
    except (TypeError, ValueError):
        pass
    return None


def principal_point_size(params) -> tuple[float, float]:
    """(2cx, 2cy): about the size of the images `params` (fx, fy, cx, cy) were calibrated on."""
    return 2.0 * float(params[2]), 2.0 * float(params[3])


def size_offset(size, reference) -> float:
    """how far `size` lies from `reference`, both (width, height): the larger relative difference."""
    return max(abs(float(size[0]) - float(reference[0])) / float(size[0]),
               abs(float(size[1]) - float(reference[1])) / float(size[1]))


def inferred_calibration_size(params) -> tuple[tuple[int, int], bool]:
    """((width, height), found) the intrinsics `params` were calibrated at, as their principal
    point gives it: the common frame size nearest (2cx, 2cy) when one lies within
    PRINCIPAL_POINT_TOLERANCE (found), else (2cx, 2cy) rounded."""
    estimate = principal_point_size(params)
    candidates = [*COMMON_FRAME_SIZES, *((h, w) for w, h in COMMON_FRAME_SIZES)]
    nearest = min(candidates, key=lambda size: size_offset(size, estimate))
    if size_offset(nearest, estimate) <= PRINCIPAL_POINT_TOLERANCE:
        return nearest, True
    return (max(1, int(round(estimate[0]))), max(1, int(round(estimate[1])))), False


def scaled_params(params, calibration_size, frame_size) -> list[float]:
    """fx, fy, cx, cy of `params` (calibrated at `calibration_size`) for frames of `frame_size`,
    both (width, height): fx and cx scaled by the width ratio, fy and cy by the height ratio."""
    fx, fy, cx, cy = (float(v) for v in params)
    sx = float(frame_size[0]) / float(calibration_size[0])
    sy = float(frame_size[1]) / float(calibration_size[1])
    return [fx * sx, fy * sy, cx * sx, cy * sy]


class FrameIntrinsics:
    """the detector's camera_params for each frame size, said once in the log when they are
    scaled, and warned of when the frame's aspect differs from the calibration's or the principal
    point lies far from the frame's centre.

    `calibration_size` is the entry's `calibration_resolution`; None when it has none, and the
    principal point then gives the size (`calibration_size` holds the inferred one,
    `calibration_size_stated` says which)."""

    def __init__(self, params, calibration_size=None, camera: str = '', logger: logging.Logger | None = None):
        self.params = [float(v) for v in params]
        self.calibration_size_stated = calibration_size is not None
        if self.calibration_size_stated:
            self.calibration_size = (int(calibration_size[0]), int(calibration_size[1]))
            self._inferred_found = True
        else:
            self.calibration_size, self._inferred_found = inferred_calibration_size(self.params)
        self.camera = camera
        self.logger = logger
        self._by_size: dict[tuple[int, int], list[float]] = {}
        self._said_inferred = False

    def for_frame(self, width: int, height: int) -> list[float]:
        size = (int(width), int(height))
        params = self._by_size.get(size)
        if params is None:
            params = self._by_size[size] = self._resolve(size)
        return params

    def calibration_size_for(self, size: tuple[int, int]) -> tuple[int, int]:
        """the size the intrinsics are taken to be calibrated at for frames of `size`: the entry's;
        else the frame's own when it lies within PRINCIPAL_POINT_TOLERANCE of (2cx, 2cy), else the
        size inferred from the principal point."""
        if self.calibration_size_stated:
            return self.calibration_size
        if size_offset(size, principal_point_size(self.params)) <= PRINCIPAL_POINT_TOLERANCE:
            return size
        return self.calibration_size

    def _resolve(self, size: tuple[int, int]) -> list[float]:
        name = self.camera or '?'
        calibration_size = self.calibration_size_for(size)
        if not self.calibration_size_stated and not self._said_inferred and self.logger is not None:
            self._said_inferred = True
            cx, cy = self.params[2], self.params[3]
            estimate = f"{2 * cx:.0f}x{2 * cy:.0f}"
            message = (f"Camera {name}'s Cameras entry has no calibration_resolution: its principal point "
                       f"({cx:.1f}, {cy:.1f}) puts the calibration images at about {estimate}")
            if self._inferred_found:
                self.logger.info(f"{message}. Frames within {PRINCIPAL_POINT_TOLERANCE:.0%} of {estimate} are read "
                                 f"unscaled, others are scaled from {self.calibration_size[0]}x"
                                 f"{self.calibration_size[1]}.")
            else:
                self.logger.warning(f"{message}, which is no common frame size: frames of another size are "
                                    f"scaled from it. Set calibration_resolution in the entry to the size the "
                                    f"camera was calibrated at.")
        if size == calibration_size:
            params = list(self.params)
        else:
            params = scaled_params(self.params, calibration_size, size)
            sx, sy = size[0] / calibration_size[0], size[1] / calibration_size[1]
            if self.logger is not None:
                cal = f"{calibration_size[0]}x{calibration_size[1]}"
                self.logger.info(
                    f"Camera {name} was calibrated at {cal} and the frames are {size[0]}x{size[1]}: "
                    f"its intrinsics are scaled by {sx:.3f} x {sy:.3f} to fx {params[0]:.1f}, fy {params[1]:.1f}, "
                    f"cx {params[2]:.1f}, cy {params[3]:.1f}.")
                if abs(sx - sy) > ASPECT_TOLERANCE * max(sx, sy):
                    self.logger.warning(
                        f"The frames of camera {name} ({size[0]}x{size[1]}) have another aspect than "
                        f"its calibration ({cal}): the scaled intrinsics are only approximate, calibrate the camera "
                        f"at this size for metric positions.")
        if self.logger is not None and size_offset(size, principal_point_size(params)) > PRINCIPAL_POINT_TOLERANCE:
            self.logger.warning(
                f"Camera {name}'s principal point for its {size[0]}x{size[1]} frames, ({params[2]:.1f}, "
                f"{params[3]:.1f}), lies far from their centre ({size[0] / 2:.0f}, {size[1] / 2:.0f}): the "
                f"intrinsics were probably calibrated at another size than {calibration_size[0]}x"
                f"{calibration_size[1]}, and the positions are off. Check (or set) the entry's "
                f"calibration_resolution.")
        return params
