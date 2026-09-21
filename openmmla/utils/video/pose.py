"""Whole-body pose estimation for the VFA server's /features: the persons in a frame and their
COCO-17 keypoints, from an Ultralytics YOLO pose model. The model is imported when the estimator
is made, so that a server without the vfa-server extra, and a test, can import this module."""
import os

import numpy as np

# the COCO-17 keypoint order every YOLO pose model answers in
COCO_KEYPOINTS = (
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
)


class PoseEstimator:
    """the persons of a frame with their keypoints, from a YOLO pose model.

    `model` is a weights file (yolo11n-pose.pt, yolov8m-pose.pt ...): a bare name is looked for
    under `weights_dir` and fetched there by Ultralytics when it is not there yet, so a server
    fetches it once, at start, and never during a request."""

    def __init__(self, model: str = 'yolo11n-pose.pt', weights_dir: str | None = None,
                 device: str | None = None, confidence: float = 0.25):
        from ultralytics import YOLO  # the vfa-server extra; not needed to import this module

        self.model_name = os.path.basename(str(model))
        path = str(model)
        if weights_dir and not os.path.isabs(path):
            # a bare name or a relative path lives under weights_dir, never under the process's
            # working directory
            path = os.path.join(weights_dir, path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.weights_path = path
        self.device = device
        self.confidence = float(confidence)
        self.model = YOLO(path)
        # how many keypoints the model answers per person, when it says (a pose model does)
        kpt_shape = getattr(getattr(self.model, 'model', None), 'kpt_shape', None)
        self.keypoints_count = int(kpt_shape[0]) if kpt_shape else None

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """the persons in `image_bgr` (an OpenCV image), each {bbox: [x1, y1, x2, y2], score,
        keypoints: [[x, y, confidence] x 17]} in pixels from the top-left corner."""
        kwargs = {'conf': self.confidence, 'verbose': False}
        if self.device:
            kwargs['device'] = self.device
        persons = []
        for result in self.model.predict(image_bgr, **kwargs):
            if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            xy = result.keypoints.xy.cpu().numpy()
            conf = result.keypoints.conf
            # a model without a visibility channel: a keypoint it placed counts as seen, one it
            # left at the origin as hidden
            conf = conf.cpu().numpy() if conf is not None else (np.abs(xy).sum(axis=-1) > 0).astype(float)
            for i in range(len(boxes)):
                keypoints = [[float(x), float(y), float(c)] for (x, y), c in zip(xy[i], conf[i])]
                persons.append({'bbox': [float(v) for v in boxes[i]], 'score': float(scores[i]),
                                'keypoints': keypoints})
        return persons
