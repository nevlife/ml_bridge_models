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
Train monodepth2 without ROS: a few steps of its self-supervised loss on bag images.

The previous and next frames are warped into the current one by the predicted depth
and poses, with SSIM and L1, auto-masking, and edge-aware smoothness.
"""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from .common import (
    layers,
    BATCH,
    CAMERA,
    INFO,
    SIZE,
    STEPS,
    predict_depth,
    get_device,
    to_batch,
    intrinsics,
    build_nets,
    )


def _collate(samples):
    """
    Keep samples with a previous and next frame and the camera info.

    Only arrays leave a worker: decoded bag messages do not pickle.
    """
    usable = [s for s in samples if s[CAMERA + '@past'][0] and s[CAMERA + '@future'][0]
              and s[INFO]]
    return [(np.array(s[CAMERA + '@past'][0][1]), np.array(s[CAMERA][1]),
             np.array(s[CAMERA + '@future'][0][1]),
             np.asarray(s[INFO][1].k, dtype=np.float32).reshape(3, 3)) for s in usable]


def _photometric(ssim, predicted, target):
    l1 = (target - predicted).abs().mean(1, keepdim=True)
    return 0.85 * ssim(predicted, target).mean(1, keepdim=True) + 0.15 * l1


def train(dataset, context):
    if dataset is None:
        raise ValueError('monodepth2 trains on bag images: pass bags')
    device = get_device()
    nets = build_nets(context, device)
    for net in nets.values():
        net.train()
    params = [p for net in nets.values() for p in net.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-5)
    ssim = layers.SSIM().to(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2,
                                         collate_fn=_collate, drop_last=True)
    step = 0
    while step < STEPS:
        for batch in loader:
            if not batch:
                continue
            prev, cur, nxt, cameras = zip(*batch)
            frames = {-1: to_batch(prev, device), 0: to_batch(cur, device),
                      1: to_batch(nxt, device)}
            k, inv_k = intrinsics(cameras, [rgb.shape[:2] for rgb in cur], device)
            backproject = layers.BackprojectDepth(len(batch), *SIZE).to(device)
            project = layers.Project3D(len(batch), *SIZE).to(device)
            depth, disp = predict_depth(nets, frames[0])
            losses = []
            for other in (-1, 1):
                pair = [frames[other], frames[0]] if other < 0 else [frames[0], frames[other]]
                axisangle, translation = nets['pose']([nets['pose_encoder'](torch.cat(pair, 1))])
                pose = layers.transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=other < 0)
                grid = project(backproject(depth, inv_k), k, pose)
                warped = F.grid_sample(frames[other], grid, padding_mode='border',
                                       align_corners=True)
                losses.append(_photometric(ssim, warped, frames[0]))
            identity = [_photometric(ssim, frames[o], frames[0]) for o in (-1, 1)]
            identity = [i + torch.randn_like(i) * 1e-5 for i in identity]
            reprojection = torch.cat(losses + identity, 1).min(1)[0].mean()
            normalized = disp / (disp.mean((2, 3), keepdim=True) + 1e-7)
            loss = reprojection + 1e-3 * layers.get_smooth_loss(normalized, frames[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            context.logger.info(f'step {step}/{STEPS} loss {loss.item():.4f} '
                                f'(batch {len(batch)})')
            if step == STEPS:
                break
    out = Path(context.out_dir or '.', 'monodepth2')
    out.mkdir(parents=True, exist_ok=True)
    for name, net in nets.items():
        torch.save(net.state_dict(), out / f'{name}.pth')
    context.logger.info(f'weights: {out}')
