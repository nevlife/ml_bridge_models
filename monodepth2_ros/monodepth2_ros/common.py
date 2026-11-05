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

"""
Networks and image handling that monodepth2 training and inference share.

The sources are not vendored: MONODEPTH2_DIR names a checkout, and context.weights a
folder with encoder.pth, depth.pth, pose_encoder.pth, and pose.pth (default
MONODEPTH2_DIR/models/mono_640x192).
"""

import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.environ['MONODEPTH2_DIR'])
import layers  # noqa: E402, I100
import networks  # noqa: E402, I100

CAMERA = '/camera/image'
INFO = '/camera/camera_info'
SIZE = (288, 512)
STEPS = 20
BATCH = 4
DEPTH_RANGE = (0.1, 100.0)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_folder(context):
    return Path(context.weights or Path(os.environ['MONODEPTH2_DIR'], 'models', 'mono_640x192'))


def build_nets(context, device):
    folder = weights_folder(context)
    nets = {'encoder': networks.ResnetEncoder(18, False),
            'pose_encoder': networks.ResnetEncoder(18, False, 2)}
    nets['depth'] = networks.DepthDecoder(nets['encoder'].num_ch_enc, range(4))
    nets['pose'] = networks.PoseDecoder(nets['pose_encoder'].num_ch_enc, 1, 2)
    for name, net in nets.items():
        state = torch.load(folder / f'{name}.pth', map_location='cpu', weights_only=True)
        net.load_state_dict({k: v for k, v in state.items() if k in net.state_dict()})
        net.to(device)
    return nets


def to_batch(rgbs, device):
    pixels = torch.from_numpy(np.stack([np.array(rgb) for rgb in rgbs])).to(device)
    images = pixels.permute(0, 3, 1, 2).float() / 255
    return F.interpolate(images, SIZE, mode='bilinear', antialias=True)


def intrinsics(cameras, shapes, device):
    """4x4 K and its inverse for the resized images, from 3x3 camera matrices."""
    ks = []
    for camera, (height, width) in zip(cameras, shapes):
        k = np.eye(4, dtype=np.float32)
        k[:3, :3] = camera
        k[0] *= SIZE[1] / width
        k[1] *= SIZE[0] / height
        ks.append(k)
    k = torch.from_numpy(np.stack(ks)).to(device)
    return k, torch.linalg.inv(k)


def predict_depth(nets, images):
    disp = nets['depth'](nets['encoder'](images))[('disp', 0)]
    return layers.disp_to_depth(disp, *DEPTH_RANGE)[1], disp
