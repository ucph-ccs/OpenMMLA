"""This module contains utility functions to detect faces and estimate gaze using Gazelle."""
import math
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

from .image import load_image


def detect_gaze(
        image_input: Any,
        face_detector: Any,
        gazelle_model: torch.nn.Module,
        gazelle_transform: Any,
        device: str = 'cpu',
        normalize_bbox: bool = True,
        normalize_target: bool = True,
        render: bool = True,
        show: bool = False,
        inout_thresh: float = 0.5,
        render_heatmap: bool = False,
        save: bool = False,
        save_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Optional[Image.Image]]:
    """Detect faces, estimate gaze, and optionally render visualizations.

    Args:
        image_input: Image data (bytes) or file path (str).
        face_detector: Face detection function (e.g., RetinaFace.detect_faces).
        gazelle_model: Loaded Gazelle model.
        gazelle_transform: Preprocessing transform for Gazelle.
        device: Torch device ('cuda' or 'cpu').
        normalize_bbox: If True, return face bounding boxes normalized to [0,1] (top-left origin).
        normalize_target: If True, return gaze target coordinates normalized to [0,1] (bottom-left origin).
        render: If True, render bounding boxes, gaze lines, and scores onto the image.
        show: If True (and render=True), display the rendered image.
        inout_thresh: Confidence threshold for drawing the gaze line.
        render_heatmap: If True (and render=True), render individual heatmaps (can be slow).
        save: If True (and render=True), save the rendered image.
        save_path: Path to save the rendered image (required if save=True).

    Returns:
        A tuple containing:
        - list: A list of dictionaries, each containing info for one detected face:
            {
                'face_bbox': [xmin, ymin, xmax, ymax] (normalized TL or pixels TL),
                'gaze_target': [x, y] (normalized BL or pixels TL, or None),
                'inout_score': float (probability gaze is in frame, or None),
                'original_index': int (index in the original face_bboxes_pixels list)
            }
        - Image.Image | None: The rendered PIL image if render=True, otherwise None.
    """
    gaze_results: List[Dict[str, Any]] = []

    try:
        # 1. Load Image - assuming load_image returns numpy array (H, W, C) BGR by default
        np_image_bgr = load_image(image_input)
        if np_image_bgr is None or np_image_bgr.size == 0:
            raise ValueError("Failed to load image or image is empty.")

        # Convert to RGB for PIL and RetinaFace if needed
        np_image_rgb = cv2.cvtColor(np_image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(np_image_rgb)
        
        # Initialize rendered_image to a copy of the original image
        rendered_image = pil_image.copy().convert("RGBA")
        
        height, width, _ = np_image_rgb.shape

        if width == 0 or height == 0:
            raise ValueError("Loaded image has zero width or height.")

        # 2. Detect Faces (using RGB numpy array)
        faces_resp = face_detector(np_image_rgb)

        if not isinstance(faces_resp, dict) or not faces_resp:
            print("No faces detected or unexpected face detector response.")
            return gaze_results, rendered_image

        face_bboxes_pixels = [details['facial_area'] for key, details in faces_resp.items()]
        valid_face_indices: List[int] = []
        norm_face_bboxes_tl: List[List[float]] = []

        # Validate and normalize bboxes
        for i, bbox in enumerate(face_bboxes_pixels):
            safe_bbox = [
                max(0, int(bbox[0])), max(0, int(bbox[1])),
                min(width, int(bbox[2])), min(height, int(bbox[3]))
            ]
            if safe_bbox[0] < safe_bbox[2] and safe_bbox[1] < safe_bbox[3]:
                norm_face_bboxes_tl.append([
                    safe_bbox[0] / width, safe_bbox[1] / height,
                    safe_bbox[2] / width, safe_bbox[3] / height
                ])
                valid_face_indices.append(i)
            else:
                print(f"Warning: Skipping invalid face bbox {bbox} -> {safe_bbox}")

        if not norm_face_bboxes_tl:
            print("No valid face bboxes found after filtering.")
            return gaze_results, rendered_image

        # 3. Prepare Gazelle Input (using PIL image)
        img_tensor = gazelle_transform(pil_image).unsqueeze(0).to(device)
        gazelle_input = {
            "images": img_tensor,
            "bboxes": [norm_face_bboxes_tl]
        }

        # 4. Run Gazelle Inference
        with torch.no_grad():
            gazelle_output = gazelle_model(gazelle_input)

        # 5. Process Output
        heatmaps = gazelle_output['heatmap'][0]
        inout_scores_tensor = gazelle_output.get('inout')
        inout_scores = inout_scores_tensor[0] if inout_scores_tensor is not None else None

        for idx_norm, original_face_idx in enumerate(valid_face_indices):
            bbox_pixels = face_bboxes_pixels[original_face_idx]
            bbox_norm_tl = norm_face_bboxes_tl[idx_norm]
            score_tensor = inout_scores[idx_norm] if inout_scores is not None and idx_norm < len(inout_scores) else None
            inout_score = score_tensor.item() if score_tensor is not None else None
            heatmap = heatmaps[idx_norm] if idx_norm < len(heatmaps) else None

            gaze_target_coords = None
            if heatmap is not None and inout_score is not None and inout_score > 0.0:
                heatmap_np = heatmap.detach().cpu().numpy()
                if heatmap_np.size > 0:
                    max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
                    target_y_norm_hm = max_index[0] / heatmap_np.shape[0]
                    target_x_norm_hm = max_index[1] / heatmap_np.shape[1]

                    if normalize_target:
                        gaze_target_coords = [target_x_norm_hm, 1.0 - target_y_norm_hm]
                    else:
                        gaze_target_coords = [int(target_x_norm_hm * width), int(target_y_norm_hm * height)]
                else:
                    print(f"Warning: Empty heatmap for face index {original_face_idx}")

            if normalize_bbox:
                face_bbox_to_store = bbox_norm_tl
            else:
                face_bbox_to_store = [
                    max(0, int(bbox_pixels[0])), max(0, int(bbox_pixels[1])),
                    min(width, int(bbox_pixels[2])), min(height, int(bbox_pixels[3]))
                ]

            gaze_results.append({
                'face_bbox': face_bbox_to_store,
                'gaze_target': gaze_target_coords,
                'inout_score': inout_score,
                'original_index': original_face_idx
            })

        # 6. Render Visualization
        if render:
            draw = ImageDraw.Draw(rendered_image)
            colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
            font_size = max(int(min(width, height) * 0.025), 40)
            try:
                font = ImageFont.load_default(size=font_size)
            except IOError:
                print("Warning: Default font not found. Using basic font.")
                font = ImageFont.load_default()

            # Store face bounding box details first
            face_details = []
            for result in gaze_results:
                bbox_to_draw = result['face_bbox']
                original_index = result['original_index']
                color = colors[original_index % len(colors)]

                if normalize_bbox:
                    xmin, ymin, xmax, ymax = [
                        bbox_to_draw[0] * width, bbox_to_draw[1] * height,
                        bbox_to_draw[2] * width, bbox_to_draw[3] * height
                    ]
                else:
                    xmin, ymin, xmax, ymax = bbox_to_draw

                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                
                inout_score = result['inout_score']
                gaze_target = result['gaze_target']
                
                # Store details for drawing after heatmaps
                face_details.append({
                    'bbox': (xmin, ymin, xmax, ymax),
                    'color': color,
                    'inout_score': inout_score,
                    'gaze_target': gaze_target,
                    'original_index': original_index
                })
                
            # First render heatmaps if enabled
            if render_heatmap:
                for face_detail in face_details:
                    original_index = face_detail['original_index']
                    if original_index < len(heatmaps):
                        heatmap_tensor = heatmaps[original_index]
                        if heatmap_tensor is not None:
                            heatmap_np = heatmap_tensor.detach().cpu().numpy()
                            if heatmap_np.size > 0:
                                heatmap_img = Image.fromarray((heatmap_np * 255).astype(np.uint8)).resize(pil_image.size,
                                                                                                        Image.Resampling.BILINEAR)
                                cmap = cm.get_cmap('viridis')
                                heatmap_color_rgba = cmap(np.array(heatmap_img) / 255.)
                                heatmap_color_rgb = (heatmap_color_rgba[:, :, :3] * 255).astype(np.uint8)
                                heatmap_rgba = Image.fromarray(heatmap_color_rgb).convert("RGBA")
                                # Resize alpha to match the dimensions of heatmap_rgba
                                alpha = Image.fromarray((heatmap_np * 180).astype(np.uint8)).resize(pil_image.size,
                                                                                                   Image.Resampling.BILINEAR)
                                heatmap_rgba.putalpha(alpha)
                                rendered_image = Image.alpha_composite(rendered_image, heatmap_rgba)
            
            # Now draw bounding boxes and gaze lines (on top of heatmaps)
            draw = ImageDraw.Draw(rendered_image)  # Recreate the draw object
            for face_detail in face_details:
                xmin, ymin, xmax, ymax = face_detail['bbox']
                color = face_detail['color']
                inout_score = face_detail['inout_score']
                gaze_target = face_detail['gaze_target']
                
                # Draw face bounding box
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=max(int(min(width, height) * 0.005), 2))

                if inout_score is not None:
                    text = f"in: {inout_score:.2f}"
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Position text at the top left of the bounding box
                    text_x = xmin
                    text_y = ymin - text_height - 15
                    
                    # Then draw the text in the specified color
                    draw.text((text_x, text_y), text, fill=color, font=font)

                if inout_score is not None and inout_score >= inout_thresh and gaze_target is not None:
                    if normalize_target:
                        target_x_px = int(gaze_target[0] * width)
                        target_y_px = int((1.0 - gaze_target[1]) * height)
                    else:
                        target_x_px, target_y_px = int(gaze_target[0]), int(gaze_target[1])

                    bbox_center_x_px = (xmin + xmax) / 2
                    bbox_center_y_px = (ymin + ymax) / 2

                    radius = max(int(min(width, height) * 0.004), 2)
                    draw.ellipse([(target_x_px - radius, target_y_px - radius),
                                  (target_x_px + radius, target_y_px + radius)], fill=color)
                    draw.line([(bbox_center_x_px, bbox_center_y_px), (target_x_px, target_y_px)],
                              fill=color, width=max(int(min(width, height) * 0.003), 2))

            if show:
                rendered_image.show(title="Gaze Detection Results")

            if save:
                if not save_path:
                    if isinstance(image_input, str):
                        import os
                        dirname, filename = os.path.split(image_input)
                        name, ext = os.path.splitext(filename)
                        save_path = os.path.join(dirname, f"{name}_gaze_detected{ext}")
                    else:
                        raise ValueError("save_path must be provided when saving image from bytes")
                try:
                    # Convert RGBA to RGB before saving as JPEG
                    if rendered_image.mode == 'RGBA':
                        rendered_image = rendered_image.convert('RGB')
                    rendered_image.save(save_path)
                except Exception as e:
                    print(f"Error during image save: {e}")
                print(f"Gaze detection image saved to {save_path}")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_input}")
        return [], None
    except ImportError as e:
        print(
            f"Error: Missing dependency for gaze detection: {e}. Please install retina-face, torch, cv2, and matplotlib.")
        return [], None
    except Exception as e:
        print(f"Error during gaze detection: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return [], None

    return gaze_results, rendered_image
