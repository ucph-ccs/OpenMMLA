import cv2
import numpy as np


class PoseStabilizer:
    def __init__(self, smoothing=0.7):
        self.smoothing = smoothing
        self.rotation_history = {}  # tag_id -> Rodrigues vector
        self.translation_history = {}

    def update(self, tag_id, R, t):
        """Update the pose of the tag with smoothing.

        Args:
            tag_id: AprilTag ID
            R: 3x3 numpy rotation matrix
            t: 3x1 numpy translation vector

        Returns:
            stabilized_R: 3x3 smoothed rotation matrix
            stabilized_t: 3x1 smoothed translation vector
        """
        rvec, _ = cv2.Rodrigues(R)
        t = np.asarray(t).reshape(3, 1)

        # if first time
        if tag_id not in self.rotation_history:
            self.rotation_history[tag_id] = rvec
            self.translation_history[tag_id] = t
            return R, t

        # smoothing
        prev_rvec = self.rotation_history[tag_id]
        prev_t = self.translation_history[tag_id]

        smoothed_rvec = self.smoothing * prev_rvec + (1 - self.smoothing) * rvec
        smoothed_t = self.smoothing * prev_t + (1 - self.smoothing) * t

        stabilized_R, _ = cv2.Rodrigues(smoothed_rvec)
        self.rotation_history[tag_id] = smoothed_rvec
        self.translation_history[tag_id] = smoothed_t

        return stabilized_R, smoothed_t


class NormalVectorStabilizer:
    def __init__(self, smoothing=0.7):
        self.smoothing = smoothing
        self.history = {}  # tag_id -> normal vector

    def update(self, tag_id, current_normal):
        current_normal = current_normal / np.linalg.norm(current_normal)

        if tag_id not in self.history:
            self.history[tag_id] = current_normal
            return current_normal

        prev = self.history[tag_id]
        smoothed = self.smoothing * prev + (1 - self.smoothing) * current_normal
        smoothed /= np.linalg.norm(smoothed)

        self.history[tag_id] = smoothed
        return smoothed


class TagRelationTracker:
    def __init__(self, min_consistent_frames=2):
        self.history = {}  # (a, b) -> [frame1, frame2, ...]
        self.frame_counter = 0
        self.min_consistent_frames = min_consistent_frames

    def step(self):
        self.frame_counter += 1

    def update(self, tag_id_a, tag_id_b, is_seeing):
        key = tuple(sorted([tag_id_a, tag_id_b]))
        if key not in self.history:
            self.history[key] = []

        if is_seeing:
            self.history[key].append(self.frame_counter)
            self.history[key] = self.history[key][-self.min_consistent_frames:]
        else:
            self.history[key] = []

    def is_confirmed(self, tag_id_a, tag_id_b):
        key = tuple(sorted([tag_id_a, tag_id_b]))
        return len(self.history.get(key, [])) >= self.min_consistent_frames
