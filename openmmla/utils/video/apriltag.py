"""
- Description: This module provides functions to detect AprilTags' position and ID in an image.
"""
import os

import cv2
import numpy as np

from .image import load_image


def detect_apriltags(image_input, tag_detector, normalize=True, render=True, show=True, save=False, save_path=None):
    """
    Detect the apriltags in an image and return the positions of the tags.

    Args:
        image_input: Either a string (file path) or bytes (image data)
        tag_detector: AprilTag detector object
        normalize: If True, return coordinates normalized to [0,1]. If False, return pixel coordinates
        render: Render the detected tags on the image or not
        show: Show the detected image or not
        save: Save the detected image or not
        save_path: Path to save the detected image

    Returns:
        dict: The positions of the detected tags. If normalize=True, positions are normalized to [0,1]
              where (0,0) is bottom-left and (1,1) is top-right.
              If normalize=False, positions are in pixel coordinates from bottom-left.
    """
    tag_pos = {}
    image = load_image(image_input)

    height, width, _ = image.shape
    print(f"Image resolution: {width}x{height} (Width x Height)")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tags = tag_detector.detect(gray)  # without pose estimation

    for tag in tags:
        center = np.mean(tag.corners, axis=0)
        corners = np.int32(tag.corners)

        if normalize:
            x = float(f'{center[0] / width:.4f}')
            y = float(f'{1.0 - (center[1] / height):.4f}')  # Flip Y coordinate
        else:
            x = int(center[0])
            y = int(height - center[1])  # Flip Y coordinate

        print(f"Person ID {tag.tag_id} center position: [{x}, {y}]")
        tag_pos[tag.tag_id] = [x, y]

        if render:
            tag_position = (int(corners[0][0]), int(corners[0][1]) - 10)  # Adjust position above the tag
            text = f"{tag.tag_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 5
            font_thickness = 14
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x, text_y = tag_position
            # Draw black border
            cv2.rectangle(image, (text_x - 20, text_y - text_size[1] - 20), (text_x + text_size[0] + 10, text_y + 30),
                          (0, 0, 0), -1)
            # Draw white background
            cv2.rectangle(image, (text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0], text_y + 20),
                          (255, 255, 255), -1)
            # Put black text
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    if show:
        cv2.imshow('AprilTag Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save:
        if isinstance(image_input, str):
            dirname, filename = os.path.split(image_input)
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(dirname, f"{name}_detected{ext}")
        elif save_path is None:
            raise ValueError("save_path must be provided when saving image data")

        cv2.imwrite(save_path, image)
        print(f"Detected image saved as {save_path}")

    return tag_pos