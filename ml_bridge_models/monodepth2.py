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
monodepth2 (Niantic) depth and pose from one camera.

The monodepth2 sources are not vendored: MONODEPTH2_DIR names a checkout, and
context.weights a folder with encoder.pth, depth.pth, pose_encoder.pth, and
pose.pth (default MONODEPTH2_DIR/models/mono_640x192). train() runs a few steps
of its self-supervised loss on the bag: the previous and next frames warped into
the current one by the predicted depth and poses, SSIM and L1 with auto-masking,
and edge-aware smoothness. infer() publishes the depth as 32FC1 (relative scale).
"""

import os
from pathlib import Path
import sys

import numpy as np
from sensor_msgs.msg import Image
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


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _weights(context):
    return Path(context.weights or Path(os.environ['MONODEPTH2_DIR'], 'models', 'mono_640x192'))


def _nets(context, device):
    folder = _weights(context)
    nets = {'encoder': networks.ResnetEncoder(18, False),
            'pose_encoder': networks.ResnetEncoder(18, False, 2)}
    nets['depth'] = networks.DepthDecoder(nets['encoder'].num_ch_enc, range(4))
    nets['pose'] = networks.PoseDecoder(nets['pose_encoder'].num_ch_enc, 1, 2)
    for name, net in nets.items():
        state = torch.load(folder / f'{name}.pth', map_location='cpu', weights_only=True)
        net.load_state_dict({k: v for k, v in state.items() if k in net.state_dict()})
        net.to(device)
    return nets


def _images(rgbs, device):
    pixels = torch.from_numpy(np.stack([np.array(rgb) for rgb in rgbs])).to(device)
    images = pixels.permute(0, 3, 1, 2).float() / 255
    return F.interpolate(images, SIZE, mode='bilinear', antialias=True)


def _intrinsics(cameras, shapes, device):
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


def _depth(nets, images):
    disp = nets['depth'](nets['encoder'](images))[('disp', 0)]
    return layers.disp_to_depth(disp, *DEPTH_RANGE)[1], disp


def load(context):
    nets = _nets(context, _device())
    for net in nets.values():
        net.eval()
    return nets


@torch.no_grad()
def infer(nets, sample):
    _, rgb = sample[CAMERA]
    depth, _ = _depth(nets, _images([rgb], next(nets['encoder'].parameters()).device))
    depth = depth[0, 0].float().cpu().numpy()
    return Image(height=depth.shape[0], width=depth.shape[1], encoding='32FC1',
                 step=depth.shape[1] * 4, data=depth.tobytes())


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
    device = _device()
    nets = _nets(context, device)
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
            frames = {-1: _images(prev, device), 0: _images(cur, device),
                      1: _images(nxt, device)}
            k, inv_k = _intrinsics(cameras, [rgb.shape[:2] for rgb in cur], device)
            backproject = layers.BackprojectDepth(len(batch), *SIZE).to(device)
            project = layers.Project3D(len(batch), *SIZE).to(device)
            depth, disp = _depth(nets, frames[0])
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
