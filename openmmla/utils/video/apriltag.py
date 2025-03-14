"""This module contains utility functions to detect AprilTags in an image.

- detect_apriltags: Detect the apriltags in an image and return the ids, positions of the tags.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .image import load_image


def detect_apriltags(image_input, tag_detector, normalize=True, render=True, show=True, save=False, save_path=None):
    """Detect the apriltags in an image and return the ids, positions of the tags.

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

    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    for tag in tags:
        corners = np.int32(tag.corners)  # Convert corners to integer coordinates
        center = np.mean(corners, axis=0).astype(int)  # Compute center of the tag

        # Calculate bounding box width and height
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        box_width = max_x - min_x
        box_height = max_y - min_y
        min_dimension = min(box_width, box_height)

        # Normalize coordinates if required
        if normalize:
            x = float(f'{center[0] / width:.4f}')
            y = float(f'{1.0 - (center[1] / height):.4f}')  # Flip Y coordinate
        else:
            x = int(center[0])
            y = int(height - center[1])  # Flip Y coordinate

        print(f"Tag ID {tag.tag_id} center position: [{x}, {y}]")
        tag_pos[tag.tag_id] = [x, y]

        if render:
            draw.polygon([tuple(c) for c in corners], fill="black")  # Fill the bounding box in black
            font_size = max(int(min_dimension * 0.5), 20)  # Ensure a minimum font size
            font = ImageFont.load_default(size=font_size)

            text = str(tag.tag_id)
            text_size = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]
            text_x = center[0] - text_width // 2
            text_y = center[1] - text_height // 2
            draw.text((text_x, text_y), text, font=font, fill="white")

    if show:
        image_pil.show()

    if save:
        if isinstance(image_input, str):
            dirname, filename = os.path.split(image_input)
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(dirname, f"{name}_detected{ext}")
        elif save_path is None:
            raise ValueError("save_path must be provided when saving image data")

        image_pil.save(save_path)
        print(f"Detected image saved as {save_path}")

    return tag_pos
