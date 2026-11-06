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
Train DINOv3 without ROS: its own train.py, unmodified, on bag images.

DINOV3_DIR names a checkout of https://github.com/facebookresearch/dinov3. train.py reads
images through make_dataset, the only part replaced here. Inference runs the timm model, which
has its own layout, so the last teacher is converted with timm's filter for these weights.
"""

import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import timm
from timm.models.eva import checkpoint_filter_fn
import torch

from .common import BACKBONE, BATCH, CAMERA, STEPS

CONFIG = 'dinov3/configs/train/vitl_im1k_lin834.yaml'
OPTIONS = ('student.arch=vit_small', 'student.n_storage_tokens=4', 'student.drop_path_rate=0.1',
           'student.mask_k_bias=true', 'train.num_workers=0', 'train.compile=false',
           'optim.epochs=1', 'optim.warmup_epochs=0', 'teacher.warmup_teacher_temp_epochs=0',
           'checkpointing.period=1000000')


class _Images(torch.utils.data.Dataset):

    def __init__(self, dataset, transform, target_transform):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = Image.fromarray(np.array(self.dataset[index][CAMERA][1]))
        return self.transform(image), self.target_transform(None)


def _save(run, out):
    """Write the EMA teacher backbone in timm's layout."""
    path = run / 'eval' / f'training_{STEPS - 1}' / 'teacher_checkpoint.pth'
    state = torch.load(path, map_location='cpu', weights_only=True)['teacher']
    teacher = {k.removeprefix('backbone.'): v for k, v in state.items()
               if k.startswith('backbone.')}
    model = timm.create_model(BACKBONE, pretrained=False)
    model.load_state_dict(checkpoint_filter_fn(teacher, model))
    torch.save(model.state_dict(), out)


def train(dataset, context):
    if dataset is None:
        raise ValueError('dinov3 trains on bag images: pass bags')
    code = os.environ['DINOV3_DIR']
    sys.path.insert(0, code)
    from dinov3.train import train as original

    original.make_dataset = lambda dataset_str, transform, target_transform: _Images(
        dataset, transform, target_transform)
    run = Path(context.out_dir or '.', 'dinov3')
    run.mkdir(parents=True, exist_ok=True)
    sys.argv = ['train.py', '--config-file', str(Path(code, CONFIG)), '--output-dir', str(run),
                '--no-resume', f'train.batch_size_per_gpu={BATCH}',
                f'train.OFFICIAL_EPOCH_LENGTH={STEPS}', f'train.output_dir={run}',
                f'evaluation.eval_period_iterations={STEPS}', *OPTIONS]
    original.main()
    _save(run, run / 'backbone.pt')
    context.logger.info(f'weights: {run / "backbone.pt"}')
