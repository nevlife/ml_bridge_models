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

"""Publish an OmniVLA-edge path from recent images and a pose goal."""

import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
import torch

from .poses import yaw_of
from .common import (
    CAMERA,
    CONTEXT,
    FRAME,
    GOAL,
    WAYPOINT_SPACING_M,
    get_device,
    run_model,
    goal_vector,
    build_model,
    observation,
)


def load(context):
    return build_model(context, get_device()).eval()


@torch.no_grad()
def infer(model, sample):
    goal = sample[GOAL]
    frames = sample[CAMERA + '@past']
    if goal is None or len(frames) < CONTEXT or not all(frames):
        return None
    device = next(model.parameters()).device
    context, large = observation([f[1] for f in frames] + [sample[CAMERA][1]])
    pose = goal[1].pose
    vector = goal_vector(pose.position.x, pose.position.y, yaw_of(pose.orientation))
    actions = run_model(model, context[None].to(device), large[None].to(device),
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
