# Copyright 2026 nevlife
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Publish DINOv3 patch features, reduced to three PCA components, as an RGB image."""

import numpy as np
from sensor_msgs.msg import Image
import torch
from torch.nn import functional as F
from .common import CAMERA, INFER_SIZE, build_backbone, get_device, normalize, to_tensor


def load(context):
    return build_backbone(context.weights).to(get_device()).eval()


@torch.no_grad()
def infer(model, sample):
    _, rgb = sample[CAMERA]
    images = F.interpolate(to_tensor(rgb, next(model.parameters()).device), INFER_SIZE,
                           mode='bilinear', antialias=True)
    tokens = model.forward_features(normalize(images))[0, model.num_prefix_tokens:]
    grid = (INFER_SIZE[0] // 16, INFER_SIZE[1] // 16)
    centered = tokens.float() - tokens.float().mean(0)
    _, _, components = torch.pca_lowrank(centered, q=3, center=False)
    projected = centered @ components
    low, high = projected.amin(0), projected.amax(0)
    colors = ((projected - low) / (high - low + 1e-6) * 255).to(torch.uint8)
    colors = colors.view(*grid, 3).cpu().numpy()
    colors = np.repeat(np.repeat(colors, 16, axis=0), 16, axis=1)
    return Image(height=colors.shape[0], width=colors.shape[1], encoding='rgb8',
                 step=colors.shape[1] * 3, data=colors.tobytes())
