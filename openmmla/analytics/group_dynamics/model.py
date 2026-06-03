"""optional TCN model for OpenMMLA-GD experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import GROUP_STATE_LABELS


@dataclass(frozen=True, slots=True)
class RobustTCNConfig:
    geometry_dim: int
    pose_video_dim: int = 0
    audio_text_dim: int = 0
    hidden_dim: int = 64
    num_classes: int = len(GROUP_STATE_LABELS)
    dropout: float = 0.2
    modality_dropout: float = 0.2


def build_robust_tcn(config: RobustTCNConfig):
    """build a small gated late-fusion TCN when torch is installed."""
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise RuntimeError("torch is required and must import cleanly to build the OpenMMLA-GD TCN model.") from exc

    class TemporalBlock(nn.Module):
        def __init__(self, channels: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def forward(self, inputs):
            outputs = self.net(inputs)
            outputs = outputs[..., : inputs.shape[-1]]
            return outputs + inputs

    class RobustGroupStateTCN(nn.Module):
        def __init__(self, cfg: RobustTCNConfig):
            super().__init__()
            self.cfg = cfg
            self.geometry_projection = nn.Linear(cfg.geometry_dim, cfg.hidden_dim)
            self.pose_video_projection = nn.Linear(cfg.pose_video_dim, cfg.hidden_dim) if cfg.pose_video_dim else None
            self.audio_text_projection = nn.Linear(cfg.audio_text_dim, cfg.hidden_dim) if cfg.audio_text_dim else None
            self.temporal = nn.Sequential(
                TemporalBlock(cfg.hidden_dim, cfg.dropout),
                TemporalBlock(cfg.hidden_dim, cfg.dropout),
            )
            self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_classes)

        def forward(self, geometry, pose_video=None, audio_text=None, modality_mask=None):
            fused = self.geometry_projection(geometry)
            weight = torch.ones((*geometry.shape[:2], 1), device=geometry.device, dtype=geometry.dtype)

            if self.pose_video_projection is not None and pose_video is not None:
                projected = self.pose_video_projection(pose_video)
                gate = _branch_gate(modality_mask, branch_index=1, like=weight)
                gate = _apply_modality_dropout(gate, self.cfg.modality_dropout, self.training)
                fused = fused + projected * gate
                weight = weight + gate

            if self.audio_text_projection is not None and audio_text is not None:
                projected = self.audio_text_projection(audio_text)
                gate = _branch_gate(modality_mask, branch_index=2, like=weight)
                gate = _apply_modality_dropout(gate, self.cfg.modality_dropout, self.training)
                fused = fused + projected * gate
                weight = weight + gate

            fused = fused / weight.clamp_min(1.0)
            encoded = self.temporal(fused.transpose(1, 2)).transpose(1, 2)
            return self.classifier(encoded)

    def _branch_gate(modality_mask, branch_index: int, like):
        if modality_mask is None:
            return torch.ones_like(like)
        if modality_mask.shape[-1] <= branch_index:
            return torch.ones_like(like)
        return modality_mask[..., branch_index : branch_index + 1].to(dtype=like.dtype, device=like.device)

    def _apply_modality_dropout(gate, probability: float, training: bool):
        if not training or probability <= 0:
            return gate
        keep = torch.rand_like(gate) > probability
        return gate * keep.to(dtype=gate.dtype)

    return RobustGroupStateTCN(config)
