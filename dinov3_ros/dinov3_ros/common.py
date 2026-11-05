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

"""Backbone and image handling that DINOv3 training and inference share."""

import numpy as np
import timm
import torch

CAMERA = '/camera/image'
BACKBONE = 'vit_small_patch16_dinov3.lvd1689m'
INFER_SIZE = (256, 448)
CROP = 224
STEPS = 20
BATCH = 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_backbone(weights):
    model = timm.create_model(BACKBONE, pretrained=True)
    if weights:
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    return model


def to_tensor(rgb, device):
    pixels = torch.from_numpy(np.array(rgb)).to(device)
    return pixels.permute(2, 0, 1).unsqueeze(0).float() / 255


def normalize(images):
    return (images - MEAN.to(images.device)) / STD.to(images.device)
