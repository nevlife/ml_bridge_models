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

"""Publish monodepth2 depth as 32FC1, in relative scale."""

from sensor_msgs.msg import Image
import torch
from .common import CAMERA, predict_depth, get_device, to_batch, build_nets


def load(context):
    nets = build_nets(context, get_device())
    for net in nets.values():
        net.eval()
    return nets


@torch.no_grad()
def infer(nets, sample):
    _, rgb = sample[CAMERA]
    depth, _ = predict_depth(nets, to_batch([rgb], next(nets['encoder'].parameters()).device))
    depth = depth[0, 0].float().cpu().numpy()
    return Image(height=depth.shape[0], width=depth.shape[1], encoding='32FC1',
                 step=depth.shape[1] * 4, data=depth.tobytes())
