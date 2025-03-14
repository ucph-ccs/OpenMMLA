"""This module contains utility functions for working with images, including load image, encode image to base 64 string.

- load_image: Load an image from either a file path or image data.
- encode_image_base64: Encode an image to base64 format.
"""

import base64

import cv2
import numpy as np


def load_image(image_input):
    """Load an image from either a file path or image data.

    Args:
        image_input: Either a string (file path) or bytes (image data)

    Returns:
        numpy.ndarray: The loaded image
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        raise ValueError("Invalid input type. Expected string (file path) or bytes (image data).")

    if image is None:
        raise ValueError("Failed to load image")

    return image


def encode_image_base64(image_input):
    """Encode an image to base64 format.

    Args:
        image_input: Either a string (file path) or bytes (image data)

    Returns:
        str: The image in base64 format
    """
    if isinstance(image_input, str):
        with open(image_input, 'rb') as file:
            image_data = file.read()
    elif isinstance(image_input, bytes):
        image_data = image_input
    else:
        raise ValueError("Invalid input type. Expected string (file path) or bytes (image data).")

    return "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')
