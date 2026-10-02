import math

import numpy as np

from .transform import quaternion_angle_degrees, quaternion_to_rotation, rotation_to_quaternion, slerp_quaternions


class PoseStabilizer:
    """Exponential smoothing of each tag's pose, for display only.

    The IPS base draws the smoothed pose in its window; what it publishes, and so what the
    synchronizer stores, is the raw pose of every detection. `smoothing` is the weight of the
    history (0.7: the new pose moves the drawn one 30 % of the way), 0 draws the raw pose.

    A tag starts again from its raw pose when it was not seen for more than `reset_seconds` (a tag
    seen again after minutes is drawn where it is, not pulled towards where it was), when time runs
    backwards, and when the new rotation lies more than `jump_degrees` from the smoothed one (a
    turn, or the flip between the two poses a small planar tag can be read in: drawn as it is,
    not swept through every angle in between). Rotations are smoothed as unit quaternions along
    the shorter arc (slerp, signs aligned), so a rotation near half a turn is averaged as itself;
    averaging Rodrigues vectors, as the base did, turns two readings of nearly the same half turn
    into a rotation near none.
    """

    def __init__(self, smoothing: float = 0.7, reset_seconds: float = 2.0, jump_degrees: float = 45.0):
        self.smoothing = float(smoothing)
        self.reset_seconds = float(reset_seconds)
        self.jump_degrees = float(jump_degrees)
        self.history = {}  # tag_id -> (quaternion, 3x1 translation, last seen)

    def reset(self, tag_id=None):
        """forget one tag's history, or every tag's."""
        if tag_id is None:
            self.history.clear()
        else:
            self.history.pop(tag_id, None)

    def update(self, tag_id, R, t, timestamp: float | None = None):
        """Update the pose of the tag with smoothing.

        Args:
            tag_id: AprilTag ID
            R: 3x3 rotation matrix
            t: 3x1 translation vector
            timestamp: when the frame was taken (seconds); None never resets on a gap

        Returns:
            stabilized_R: 3x3 smoothed rotation matrix
            stabilized_t: 3x1 smoothed translation vector
        """
        R = np.asarray(R, dtype=float).reshape(3, 3)
        t = np.asarray(t, dtype=float).reshape(3, 1)
        if self.smoothing <= 0 or np.linalg.det(R) < 0.5:
            # no smoothing, or not a rotation (nothing to smooth it with)
            self.history.pop(tag_id, None)
            return R, t
        q = rotation_to_quaternion(R)
        previous = self.history.get(tag_id)
        if previous is not None:
            prev_q, prev_t, last = previous
            gap = None if timestamp is None or last is None else float(timestamp) - last
            if (gap is not None and (gap > self.reset_seconds or gap < 0)) \
                    or quaternion_angle_degrees(prev_q, q) > self.jump_degrees:
                previous = None
        if previous is None:
            self.history[tag_id] = (q, t, None if timestamp is None else float(timestamp))
            return R, t

        alpha = 1.0 - self.smoothing
        smoothed_q = slerp_quaternions(prev_q, q, alpha)
        smoothed_t = self.smoothing * prev_t + alpha * t
        self.history[tag_id] = (smoothed_q, smoothed_t, None if timestamp is None else float(timestamp))
        return quaternion_to_rotation(smoothed_q), smoothed_t


def outward_normal_2d(R) -> np.ndarray:
    """the tag's outward normal (-column 2 of its rotation, z pointing into the tag) laid on the
    camera's x-z plane, as a unit 3-vector; zeros when it points straight along y."""
    normal = -np.asarray(R, dtype=float).reshape(3, 3)[:, 2]
    flat = np.array([normal[0], 0.0, normal[2]])
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > 1e-9 and math.isfinite(norm) else np.zeros(3)
