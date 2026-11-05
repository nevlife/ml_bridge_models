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
Model and preprocessing that OmniVLA-edge training and inference share.

The sources are not vendored: OMNIVLA_DIR names a checkout of
https://github.com/NHirose/OmniVLA and context.weights the omnivla-edge.pth of
https://huggingface.co/NHirose/omnivla-edge. Only the pose goal is wired: it hides
the language, image, and satellite tokens, so CLIP is neither needed nor loaded.
The model's frame is x forward, y left, in units of WAYPOINT_SPACING_M.
"""

import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from .poses import limit_range

sys.path.insert(0, str(Path(os.environ['OMNIVLA_DIR'], 'inference')))
from model_omnivla_edge import OmniVLA_edge  # noqa: E402, I100

CAMERA = '/camera/image'
GOAL = '/goal_pose'
ODOM = '/odom'
WAYPOINT_SPACING_M = 0.1
GOAL_RADIUS_M = 30.0
CONTEXT = 5
HORIZON = 8
SIZE = (96, 96)
LARGE_SIZE = (224, 224)
POSE_ONLY = 4
TEXT_FEATURES = 512
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
STEPS = 20
BATCH = 8
FRAME = 'base_link'


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(context, device):
    model = OmniVLA_edge(context_size=CONTEXT, len_traj_pred=HORIZON, learn_angle=True,
                         obs_encoder='efficientnet-b0', obs_encoding_size=1024,
                         late_fusion=False, mha_num_attention_heads=4,
                         mha_num_attention_layers=4, mha_ff_dim_factor=4)
    if not context.weights:
        raise ValueError('omnivla_edge needs weights:= the path of omnivla-edge.pth')
    model.load_state_dict(torch.load(context.weights, map_location='cpu', weights_only=True))
    return model.to(device)


def to_pixels(rgb, size):
    """Resize like the upstream PIL pipeline and normalize to CHW."""
    array = np.array(Image.fromarray(np.asarray(rgb)).resize(size), dtype=np.float32)
    return (torch.from_numpy(array).permute(2, 0, 1) / 255 - MEAN) / STD


def observation(frames):
    """Stack the context, oldest first, and the current frame at the large size."""
    context = torch.cat([to_pixels(frame, SIZE) for frame in frames])
    return context, to_pixels(frames[-1], LARGE_SIZE)


def goal_vector(forward, left, yaw):
    forward, left = limit_range(forward, left, GOAL_RADIUS_M)
    return np.array([forward / WAYPOINT_SPACING_M, left / WAYPOINT_SPACING_M,
                     math.cos(yaw), math.sin(yaw)], dtype=np.float32)


def run_model(model, context_images, large, goals):
    """Run the pose-only branch: the other goals are masked, their inputs blank."""
    count = len(goals)
    device = goals.device
    black = ((torch.zeros(3, *SIZE) - MEAN) / STD).to(device).repeat(count, 1, 1, 1)
    current = context_images[:, -3:]
    actions, _, _ = model(context_images, goals, torch.cat([black, black, current], 1), current,
                          torch.full((count,), POSE_ONLY, device=device),
                          torch.zeros(count, TEXT_FEATURES, device=device), large)
    return actions
