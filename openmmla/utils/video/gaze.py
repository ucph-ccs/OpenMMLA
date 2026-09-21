"""This module contains utility functions to detect faces and estimate where each face looks: a
gaze backend (PaGE or Gaze-LLE, behind one interface) answers, per face, a heatmap
over the image and the probability that the gaze lands inside it."""
import os
from typing import Any

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .image import load_image


class GazeBackend:
    """what a gaze model answers for one image: a heatmap per face (a 2-D array over the image,
    any resolution, the hottest cell being the gaze target) and, when it can tell, the
    probability that the gaze lands inside the frame. Subclasses wrap one model each."""
    name = 'gaze'
    model_name = ''

    def predict(self, pil_image: Image.Image, norm_boxes: list[list[float]]) -> tuple[list[np.ndarray], list[float | None]]:
        """the heatmaps and in-frame probabilities of the faces at `norm_boxes` ([x1, y1, x2, y2]
        in [0, 1], top-left origin), one each, in the order given."""
        raise NotImplementedError


class GazelleBackend(GazeBackend):
    """Gaze-LLE (fkryan/gazelle on torch.hub): a frozen DINOv2 encoder with a small gaze decoder,
    64 x 64 heatmaps, in/out-of-frame from the _inout checkpoints."""
    name = 'gazelle'

    def __init__(self, model: torch.nn.Module, transform: Any, device: str = 'cpu', model_name: str = ''):
        self.model, self.transform, self.device, self.model_name = model, transform, device, model_name

    @classmethod
    def load(cls, model_name: str = 'gazelle_dinov2_vitl14_inout', device: str = 'cpu') -> 'GazelleBackend':
        model, transform = torch.hub.load('fkryan/gazelle', model_name, trust_repo=True)
        model.eval()
        model.to(device)
        return cls(model, transform, device, model_name)

    def predict(self, pil_image, norm_boxes):
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model({"images": img_tensor, "bboxes": [norm_boxes]})
        heatmaps = [h.detach().cpu().numpy() for h in output['heatmap'][0]]
        inout = output.get('inout')
        scores = [float(s.item()) for s in inout[0]] if inout is not None else [None] * len(heatmaps)
        return heatmaps, scores


def gaze_backend_name(backend: str | None, model_name: str | None = None) -> str:
    """the backend a config means: the one it names ('page' for PaGE, 'gazelle' for Gaze-LLE),
    else the one its checkpoint name belongs to (gazelle_* is Gaze-LLE), else PaGE."""
    key = str(backend or '').strip().lower()
    if key:
        return key
    if str(model_name or '').strip().lower().startswith('gazelle'):
        return 'gazelle'
    return 'page'


def load_gaze_backend(backend: str | None, model_name: str | None, device: str = 'cpu',
                      head_scale: float | None = None) -> GazeBackend:
    """the gaze backend named by the config (see gaze_backend_name); `model_name` picks its
    checkpoint, None its default; `head_scale` widens the face box into the head crop PaGE
    looks at (None: its default)."""
    key = gaze_backend_name(backend, model_name)
    if key == 'gazelle':
        return GazelleBackend.load(model_name or 'gazelle_dinov2_vitl14_inout', device)
    if key == 'page':
        from .gaze_page import DEFAULT_HEAD_SCALE, PageBackend
        return PageBackend.load(model_name, device, DEFAULT_HEAD_SCALE if head_scale is None else head_scale)
    raise ValueError(f"unknown gaze backend '{backend}': use gazelle or page")


def detect_gaze(
        image_input: Any,
        face_detector: Any,
        gazelle_model: torch.nn.Module | None = None,
        gazelle_transform: Any = None,
        device: str = 'cpu',
        normalize_bbox: bool = True,
        normalize_target: bool = True,
        render: bool = True,
        show: bool = False,
        inout_thresh: float = 0.5,
        render_heatmap: bool = False,
        save: bool = False,
        save_path: str | None = None,
        backend: GazeBackend | None = None,
        raise_errors: bool = False,
) -> tuple[list[dict[str, Any]], Image.Image | None]:
    """Detect faces, estimate gaze, and optionally render visualizations.

    Args:
        image_input: Image data (bytes) or file path (str).
        face_detector: Face detection function (e.g., RetinaFace.detect_faces).
        gazelle_model: Loaded Gazelle model (the older way to name the backend; `backend` is the newer).
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
        backend: the gaze model as a GazeBackend (Gaze-LLE, PaGE ...); given, the two gazelle
            arguments are not needed.
        raise_errors: True raises what the face detector or the gaze model raise instead of
            printing it and answering no faces (the overlay path tolerates a miss, the features
            endpoint reports it).

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
    gaze_results: list[dict[str, Any]] = []
    if backend is None:
        if gazelle_model is None or gazelle_transform is None:
            raise ValueError("detect_gaze needs a gaze backend, or a gazelle model and its transform")
        backend = GazelleBackend(gazelle_model, gazelle_transform, device)

    try:
        image = load_image(image_input)
        height, width, _ = image.shape
        print(f"Image resolution: {width}x{height} (Width x Height)")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        rendered_image = pil_image.copy().convert("RGBA")

        # 2. Detect Faces (using RGB numpy array)
        faces_resp = face_detector(rgb_image)

        if not isinstance(faces_resp, dict) or not faces_resp:
            print("No faces detected or unexpected face detector response.")
            return gaze_results, rendered_image

        face_bboxes_pixels = [details['facial_area'] for key, details in faces_resp.items()]
        valid_face_indices: list[int] = []
        norm_face_bboxes_tl: list[list[float]] = []

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

        # 3. and 4. the backend answers a heatmap and an in-frame probability per face
        heatmaps, inout_scores = backend.predict(pil_image, norm_face_bboxes_tl)

        # 5. Process Output
        for idx_norm, original_face_idx in enumerate(valid_face_indices):
            bbox_pixels = face_bboxes_pixels[original_face_idx]
            bbox_norm_tl = norm_face_bboxes_tl[idx_norm]
            inout_score = inout_scores[idx_norm] if idx_norm < len(inout_scores) else None
            inout_score = float(inout_score) if inout_score is not None else None
            heatmap = heatmaps[idx_norm] if idx_norm < len(heatmaps) else None

            gaze_target_coords = None
            if heatmap is not None and inout_score is not None and inout_score > 0.0:
                heatmap_np = np.asarray(heatmap)
                if heatmap_np.size > 0:
                    max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
                    # the centre of the hottest cell: its corner would put every gaze half a
                    # cell up and left (a cell is 1/64 of the image on each side)
                    target_y_norm_hm = (max_index[0] + 0.5) / heatmap_np.shape[0]
                    target_x_norm_hm = (max_index[1] + 0.5) / heatmap_np.shape[1]

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
            font = ImageFont.load_default(size=font_size)

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
                        heatmap_array = heatmaps[original_index]
                        if heatmap_array is not None:
                            heatmap_np = np.asarray(heatmap_array, dtype=np.float32)
                            if heatmap_np.size > 0:
                                heatmap_img = Image.fromarray((heatmap_np * 255).astype(np.uint8)).resize(
                                    pil_image.size,
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
                if save_path is None:
                    raise ValueError("save_path must be provided when saving image data")
                if rendered_image.mode == 'RGBA':
                    rendered_image = rendered_image.convert('RGB')
                rendered_image.save(save_path)
                print(f"Gaze detection image saved to {save_path}")

    except Exception as e:
        if raise_errors:
            raise
        print(f"Error during gaze detection: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return [], None

    return gaze_results, rendered_image
