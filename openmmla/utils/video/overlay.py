"""Drawing what the frame analyzer's features endpoint answered onto a frame: the person boxes
with tag and head yaw, the skeletons, the faces and a line from each face to where its gaze
lands. Used by the VFA base's live window and by the frame_features example."""
from __future__ import annotations

import cv2
import numpy as np

# the limbs of the COCO-17 skeleton, by keypoint index
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 1), (0, 2), (1, 3), (2, 4)]
COLORS = [(0, 200, 0), (0, 140, 255), (255, 0, 150), (255, 200, 0), (200, 0, 255), (0, 255, 255)]
KEYPOINT_CONFIDENCE = 0.3


def draw_features(image: np.ndarray, frame: dict, note: str | None = None) -> np.ndarray:
    """draw one frame's features onto `image` (BGR, any size: the coordinates are scaled from
    the frame's width and height to the image's) and return it. `note` is written in the
    corner (how old the features are, say)."""
    height, width = image.shape[:2]
    sx = width / float(frame.get('width') or width)
    sy = height / float(frame.get('height') or height)
    thickness = max(1, int(round(min(width, height) / 540)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55 * thickness

    def pt(x, y):
        return int(round(float(x) * sx)), int(round(float(y) * sy))

    for name, polygon in (frame.get('zones') or {}).items():
        points = np.array([pt(x, y) for x, y in polygon], dtype=np.int32)
        if len(points) >= 3:
            cv2.polylines(image, [points], True, (180, 180, 180), thickness)
            cv2.putText(image, str(name), tuple(points[0]), font, scale, (180, 180, 180), thickness)
    for index, person in enumerate(frame.get('persons', [])):
        color = COLORS[index % len(COLORS)]
        x1, y1 = pt(person['bbox'][0], person['bbox'][1])
        x2, y2 = pt(person['bbox'][2], person['bbox'][3])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, f"{person.get('person_id')}  yaw {person.get('head_yaw')}", (x1, max(14, y1 - 6)),
                    font, scale, color, thickness)
        keypoints = person.get('keypoints') or []
        for a, b in SKELETON:
            if a < len(keypoints) and b < len(keypoints) \
                    and keypoints[a][2] >= KEYPOINT_CONFIDENCE and keypoints[b][2] >= KEYPOINT_CONFIDENCE:
                cv2.line(image, pt(keypoints[a][0], keypoints[a][1]), pt(keypoints[b][0], keypoints[b][1]), color, thickness)
        for x, y, confidence in keypoints:
            if confidence >= KEYPOINT_CONFIDENCE:
                cv2.circle(image, pt(x, y), 2 * thickness, (0, 0, 255), -1)
        face = person.get('face_bbox')
        gaze = person.get('gaze') or {}
        if face:
            fx1, fy1 = pt(face[0], face[1])
            fx2, fy2 = pt(face[2], face[3])
            cv2.rectangle(image, (fx1, fy1), (fx2, fy2), color, max(1, thickness - 1))
            point = gaze.get('point')
            if point and gaze.get('inout') is not None:
                target = gaze.get('target') or {}
                label = str(target.get('category', ''))
                if target.get('person_id'):
                    label += f" {target['person_id']}"
                if target.get('zone'):
                    label += f" {target['zone']}"
                end = pt(point[0], point[1])
                cv2.line(image, ((fx1 + fx2) // 2, (fy1 + fy2) // 2), end, color, thickness)
                cv2.circle(image, end, 3 * thickness, color, -1)
                cv2.putText(image, f"{label} ({float(gaze['inout']):.2f})", (end[0] + 6, end[1] - 6), font, scale, color, thickness)
    if note:
        cv2.putText(image, note, (10, height - 12), font, scale, (255, 255, 255), thickness)
    return image
