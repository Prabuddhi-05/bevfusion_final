from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Debug shapes before fusion
        for i, x in enumerate(inputs):
            print(f"[ConvFuser] Input {i} shape: {x.shape}", flush=True)

        # Ensure spatial dimensions match
        h, w = inputs[1].shape[-2:]  # LiDAR feature size (usually reference)
        if inputs[0].shape[-2:] != (h, w):
            print(f"[ConvFuser] Resizing camera features from {inputs[0].shape[-2:]} to {(h, w)}", flush=True)
            inputs[0] = F.interpolate(inputs[0], size=(h, w), mode='bilinear', align_corners=False)

        return super().forward(torch.cat(inputs, dim=1))

