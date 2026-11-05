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
OmniVLA-edge (Hirose et al.) local path from recent images and a pose goal.

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

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .poses import limit_range, planar, relative, yaw_of

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


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _model(context, device):
    model = OmniVLA_edge(context_size=CONTEXT, len_traj_pred=HORIZON, learn_angle=True,
                         obs_encoder='efficientnet-b0', obs_encoding_size=1024,
                         late_fusion=False, mha_num_attention_heads=4,
                         mha_num_attention_layers=4, mha_ff_dim_factor=4)
    if not context.weights:
        raise ValueError('omnivla_edge needs weights:= the path of omnivla-edge.pth')
    model.load_state_dict(torch.load(context.weights, map_location='cpu', weights_only=True))
    return model.to(device)


def _pixels(rgb, size):
    """Resize like the upstream PIL pipeline and normalize to CHW."""
    array = np.array(Image.fromarray(np.asarray(rgb)).resize(size), dtype=np.float32)
    return (torch.from_numpy(array).permute(2, 0, 1) / 255 - MEAN) / STD


def _observation(frames):
    """Stack the context, oldest first, and the current frame at the large size."""
    context = torch.cat([_pixels(frame, SIZE) for frame in frames])
    return context, _pixels(frames[-1], LARGE_SIZE)


def _goal_vector(forward, left, yaw):
    forward, left = limit_range(forward, left, GOAL_RADIUS_M)
    return np.array([forward / WAYPOINT_SPACING_M, left / WAYPOINT_SPACING_M,
                     math.cos(yaw), math.sin(yaw)], dtype=np.float32)


def _forward(model, context_images, large, goals):
    """Run the pose-only branch: the other goals are masked, their inputs blank."""
    count = len(goals)
    device = goals.device
    black = ((torch.zeros(3, *SIZE) - MEAN) / STD).to(device).repeat(count, 1, 1, 1)
    current = context_images[:, -3:]
    actions, _, _ = model(context_images, goals, torch.cat([black, black, current], 1), current,
                          torch.full((count,), POSE_ONLY, device=device),
                          torch.zeros(count, TEXT_FEATURES, device=device), large)
    return actions


def load(context):
    return _model(context, _device()).eval()


@torch.no_grad()
def infer(model, sample):
    goal = sample[GOAL]
    frames = sample[CAMERA + '@past']
    if goal is None or len(frames) < CONTEXT or not all(frames):
        return None
    device = next(model.parameters()).device
    context, large = _observation([f[1] for f in frames] + [sample[CAMERA][1]])
    pose = goal[1].pose
    vector = _goal_vector(pose.position.x, pose.position.y, yaw_of(pose.orientation))
    actions = _forward(model, context[None].to(device), large[None].to(device),
                       torch.from_numpy(vector)[None].to(device))[0].cpu().numpy()
    path = PathMessage()
    path.header.frame_id = FRAME
    for forward, left, cosine, sine in actions:
        waypoint = PoseStamped()
        waypoint.header.frame_id = FRAME
        waypoint.pose.position.x = float(forward) * WAYPOINT_SPACING_M
        waypoint.pose.position.y = float(left) * WAYPOINT_SPACING_M
        half = math.atan2(sine, cosine) / 2
        waypoint.pose.orientation.z, waypoint.pose.orientation.w = math.sin(half), math.cos(half)
        path.poses.append(waypoint)
    return path


def _collate(samples):
    """
    Keep samples with a full context and full future; label them from the odometry.

    Only arrays leave a worker: decoded bag messages do not pickle.
    """
    batch = []
    for s in samples:
        frames, future = s[CAMERA + '@past'], s[ODOM + '@future']
        if len(frames) < CONTEXT or not all(frames) or not all(future):
            continue
        context, large = _observation([f[1] for f in frames] + [s[CAMERA][1]])
        origin = planar(s[ODOM][1].pose.pose)
        steps = [relative(origin, planar(f[1].pose.pose)) for f in future]
        actions = np.array([[x / WAYPOINT_SPACING_M, y / WAYPOINT_SPACING_M,
                             math.cos(yaw), math.sin(yaw)] for x, y, yaw in steps],
                           dtype=np.float32)
        batch.append((context, large, _goal_vector(*steps[-1]), actions))
    return batch


def train(dataset, context):
    if dataset is None:
        raise ValueError('omnivla_edge trains on bags with images and odometry: pass bags')
    device = _device()
    model = _model(context, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2,
                                         collate_fn=_collate, drop_last=True)
    step = 0
    while step < STEPS:
        for batch in loader:
            if not batch:
                continue
            images, large, goals, actions = zip(*batch)
            predicted = _forward(model, torch.stack(images).to(device),
                                 torch.stack(large).to(device),
                                 torch.from_numpy(np.stack(goals)).to(device))
            loss = F.mse_loss(predicted, torch.from_numpy(np.stack(actions)).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            context.logger.info(f'step {step}/{STEPS} loss {loss.item():.3f} '
                                f'(batch {len(batch)})')
            if step == STEPS:
                break
    out = Path(context.out_dir or '.', 'omnivla_edge')
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'omnivla-edge.pth')
    context.logger.info(f'weights: {out}')
