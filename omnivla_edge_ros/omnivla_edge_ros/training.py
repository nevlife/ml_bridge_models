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

"""Train OmniVLA-edge without ROS: the future odometry of a bag is the label."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .poses import planar, relative
from .common import (
    BATCH,
    CAMERA,
    CONTEXT,
    ODOM,
    STEPS,
    WAYPOINT_SPACING_M,
    get_device,
    run_model,
    goal_vector,
    build_model,
    observation,
)


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
        context, large = observation([f[1] for f in frames] + [s[CAMERA][1]])
        origin = planar(s[ODOM][1].pose.pose)
        steps = [relative(origin, planar(f[1].pose.pose)) for f in future]
        actions = np.array([[x / WAYPOINT_SPACING_M, y / WAYPOINT_SPACING_M,
                             math.cos(yaw), math.sin(yaw)] for x, y, yaw in steps],
                           dtype=np.float32)
        batch.append((context, large, goal_vector(*steps[-1]), actions))
    return batch


def train(dataset, context):
    if dataset is None:
        raise ValueError('omnivla_edge trains on bags with images and odometry: pass bags')
    device = get_device()
    model = build_model(context, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2,
                                         collate_fn=_collate, drop_last=True)
    step = 0
    while step < STEPS:
        for batch in loader:
            if not batch:
                continue
            images, large, goals, actions = zip(*batch)
            predicted = run_model(model, torch.stack(images).to(device),
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
