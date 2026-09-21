"""PaGE (Practical Gaze Estimator, Ye et al. 2026, https://arxiv.org/abs/2607.04860) as a gaze
backend: the released checkpoints on HuggingFace (Octopus1/page-vits, page-vitsplus, page-vitb,
page-vithplus) carry their own model code and their DINOv3 backbone weights, so loading one needs
transformers (>= 4.56 for DINOv3; the checkpoints are made with 5.6.2) and no other download."""
import contextlib
import io
import warnings
from typing import Any

import numpy as np
import torch
from PIL import Image

from .gaze import GazeBackend

# the ViT-B distilled student: human-level on GazeFollow, VAT and ChildPlay at ~90M parameters;
# page-vits is the light one (a third of the FLOPs), page-vithplus the 840M teacher
DEFAULT_MODEL = 'Octopus1/page-vitb'
# the face detector answers a tight face box (eyes to chin) while PaGE learned from head boxes
# (hair to chin), and the crop is what its head branch sees: the box is widened about its
# centre by this factor before the crop, the answer keeps the face box
DEFAULT_HEAD_SCALE = 1.3
# PaGE's own loader reloads the DINOv3 weights under the checkpoint's key layout; when that
# step fails it warns and leaves the backbone random, which would answer plausible nonsense
RELOAD_SKIPPED = 'version-safe backbone reload skipped'


def _to_device(value: Any, device: str) -> Any:
    """the processor's nested answer (tensors in dicts and lists) moved to the device"""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


class PageBackend(GazeBackend):
    """PaGE: a scene branch and a head branch (both DINOv3) with cross attention between them,
    64 x 64 heatmaps and an in/out-of-frame probability per head."""
    name = 'page'

    def __init__(self, model: Any, processor: Any, device: str = 'cpu', model_name: str = DEFAULT_MODEL,
                 head_scale: float = DEFAULT_HEAD_SCALE):
        self.model, self.processor, self.device, self.model_name = model, processor, device, model_name
        self.head_scale = max(1.0, float(head_scale or 1.0))

    @classmethod
    def load(cls, model_name: str | None = None, device: str = 'cpu', head_scale: float = DEFAULT_HEAD_SCALE) -> 'PageBackend':
        """the checkpoint named (a HuggingFace repo id or a local directory), with its own code
        (trust_remote_code: modeling_page.py from Octopus1/PaGE) building the model"""
        from transformers import AutoImageProcessor, AutoModel
        from transformers.utils import logging as hf_logging

        name = model_name or DEFAULT_MODEL
        # transformers reports the DINOv3 keys as missing/unexpected before PaGE's own loader
        # remaps them; that report is noise, so it is muted while the model loads
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter('always')
                model = AutoModel.from_pretrained(name, trust_remote_code=True)
                processor = AutoImageProcessor.from_pretrained(name, trust_remote_code=True)
        finally:
            hf_logging.set_verbosity(verbosity)
        skipped = [str(w.message) for w in caught if RELOAD_SKIPPED in str(w.message)]
        if skipped:
            raise RuntimeError(f"PaGE {name} loaded without its backbone weights: {skipped[0]}")
        model.eval()
        model.to(device)
        return cls(model, processor, device, name, head_scale)

    def head_box(self, box: list[float]) -> tuple[float, float, float, float]:
        """the face box widened by head_scale about its centre, kept inside the frame"""
        x1, y1, x2, y2 = (float(v) for v in box)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w, half_h = (x2 - x1) / 2 * self.head_scale, (y2 - y1) / 2 * self.head_scale
        return (max(0.0, cx - half_w), max(0.0, cy - half_h), min(1.0, cx + half_w), min(1.0, cy + half_h))

    def predict(self, pil_image: Image.Image, norm_boxes: list[list[float]]) -> tuple[list[np.ndarray], list[float | None]]:
        if not norm_boxes:
            return [], []
        width, height = pil_image.size
        crops, boxes = [], []
        for box in norm_boxes:
            x1, y1, x2, y2 = self.head_box(box)
            left, top = int(round(x1 * width)), int(round(y1 * height))
            # the head crop is the model's second input; never an empty one
            right, bottom = max(left + 1, int(round(x2 * width))), max(top + 1, int(round(y2 * height)))
            crops.append(pil_image.crop((left, top, right, bottom)))
            boxes.append((x1, y1, x2, y2))
        inputs = _to_device(self.processor(pil_image, head_crops=crops, bboxes=[boxes]), self.device)
        with torch.inference_mode():
            output = self.model(inputs)
        heatmaps = [np.asarray(h.detach().float().cpu().numpy()) for h in output['heatmap'][0]]
        inout = output.get('inout')
        if inout is None:
            return heatmaps, [None] * len(heatmaps)
        scores = [float(s) for s in inout[0].detach().float().cpu().numpy().reshape(-1)]
        return heatmaps, scores
