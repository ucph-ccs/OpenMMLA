"""This module contains utility functions to detect AprilTags in an image.

- detect_apriltags: Detect the apriltags in an image and return the ids, positions of the tags.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, Tuple

from .image import load_image


def detect_apriltags(image_input, tag_detector, normalize=True, render=True, show=True, save=False, save_path=None) -> Tuple[Dict[int, list], Optional[Image.Image]]:
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
        Tuple containing:
        - dict: The positions of the detected tags. If normalize=True, positions are normalized to [0,1]
              where (0,0) is bottom-left and (1,1) is top-right.
              If normalize=False, positions are in pixel coordinates from bottom-left.
        - PIL.Image.Image or None: The rendered PIL image if render=True, otherwise None.
    """
    tag_pos = {}
    image = load_image(image_input)

    height, width, _ = image.shape
    print(f"Image resolution: {width}x{height} (Width x Height)")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tags = tag_detector.detect(gray)  # without pose estimation

    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Initialize rendered_image to a copy of the original image
    rendered_image = image_pil.copy()
    
    if render:
        draw = ImageDraw.Draw(rendered_image)
        
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

            draw.polygon([tuple(c) for c in corners], fill="black") # type: ignore
            font_size = max(int(min_dimension * 0.5), 30)
            font = ImageFont.load_default(size=font_size)

            text = str(tag.tag_id)
            text_size = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]
            text_x = center[0] - text_width // 2
            text_y = center[1] - text_height
            draw.text((text_x, text_y), text, font=font, fill="white")
    else:
        # Just process the tag positions without rendering
        for tag in tags:
            corners = tag.corners.astype(np.int32)
            center = np.mean(corners, axis=0).astype(int)
            
            if normalize:
                x = float(f'{center[0] / width:.4f}')
                y = float(f'{1.0 - (center[1] / height):.4f}')
            else:
                x = int(center[0])
                y = int(height - center[1])
                
            print(f"Tag ID {tag.tag_id} center position: [{x}, {y}]")
            tag_pos[tag.tag_id] = [x, y]

    if show and render:
        rendered_image.show()

    if save and render:
        if isinstance(image_input, str):
            dirname, filename = os.path.split(image_input)
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(dirname, f"{name}_detected{ext}")
        elif save_path is None:
            raise ValueError("save_path must be provided when saving image data")

        rendered_image.save(save_path)
        print(f"Detected image saved as {save_path}")

    return tag_pos, rendered_image
