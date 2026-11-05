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
Train DINOv3 without ROS: a few steps of DINO-style self-distillation on bag images.

Two augmented crops, a student and an EMA teacher, and a cross-entropy between their
centered, sharpened outputs.
"""

import copy

import numpy as np
import torch
from torch.nn import functional as F
from .common import BATCH, CAMERA, CROP, STEPS, build_backbone, get_device, normalize


class _Head(torch.nn.Module):

    def __init__(self, dim, out=4096, hidden=1024):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, out))

    def forward(self, x):
        return self.mlp(x)


def _views(images):
    """Two random resized crops with flips and brightness jitter, on the GPU."""
    batch, _, height, width = images.shape
    views = []
    for _ in range(2):
        crops = []
        for image in images:
            scale = float(torch.empty(1).uniform_(0.4, 1.0))
            h, w = int(height * scale), int(min(width, height * scale * width / height))
            y = int(torch.randint(0, height - h + 1, (1,)))
            x = int(torch.randint(0, width - w + 1, (1,)))
            crop = F.interpolate(image[None, :, y:y + h, x:x + w], (CROP, CROP),
                                 mode='bilinear', antialias=True)
            if torch.rand(1) < 0.5:
                crop = crop.flip(-1)
            crops.append(crop * float(torch.empty(1).uniform_(0.7, 1.3)))
        views.append(normalize(torch.cat(crops).clamp(0, 1)))
    return views


def _collate(samples):
    return torch.stack([torch.from_numpy(np.array(s[CAMERA][1])) for s in samples])


def train(dataset, context):
    if dataset is None:
        raise ValueError('DINOv3 trains on bag images: pass bags')
    device = get_device()
    student = build_backbone(context.weights).to(device).train()
    student_head = _Head(student.num_features).to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    teacher_head = copy.deepcopy(student_head).requires_grad_(False)
    center = torch.zeros(1, 4096, device=device)
    params = list(student.parameters()) + list(student_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.04)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2,
                                         collate_fn=_collate, drop_last=True)
    step = 0
    while step < STEPS:
        for pixels in loader:
            images = pixels.to(device).permute(0, 3, 1, 2).float() / 255
            views = _views(images)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                with torch.no_grad():
                    targets = [teacher_head(teacher(v)).float() for v in views]
                outputs = [student_head(student(v)).float() for v in views]
            loss = sum(-(F.softmax((t - center) / 0.04, -1) *
                         F.log_softmax(s / 0.1, -1)).sum(-1).mean()
                       for t, s in [(targets[0], outputs[1]), (targets[1], outputs[0])]) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for t, s in zip(list(teacher.parameters()) + list(teacher_head.parameters()),
                                params):
                    t.mul_(0.996).add_(s.detach(), alpha=0.004)
                center.mul_(0.9).add_(torch.cat(targets).mean(0, keepdim=True), alpha=0.1)
            step += 1
            context.logger.info(f'step {step}/{STEPS} loss {loss.item():.4f}')
            if step == STEPS:
                break
    path = f'{context.out_dir or "."}/dinov3_teacher.pt'
    torch.save(teacher.state_dict(), path)
    context.logger.info(f'weights: {path}')
