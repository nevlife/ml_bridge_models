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
Train monodepth2 without ROS: its own Trainer, unmodified, on bag images.

The Trainer reads files through a dataset class and a split list. MonoDataset stays; only
those two are replaced, so the loss, the networks, and the saved folder are the original's.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import PIL.Image
from torchvision.transforms import functional as vision

from .common import BATCH, CAMERA, INFO, SIZE, STEPS, weights_folder

# Pillow removed the name that MonoDataset reads.
PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import datasets as monodepth_datasets  # noqa: E402, I100
from datasets.mono_dataset import MonoDataset  # noqa: E402, I100
from options import MonodepthOptions  # noqa: E402, I100
import trainer as monodepth_trainer  # noqa: E402, I100

PAST, CURRENT, FUTURE = range(3)


def _jitter(parameters):
    """Return what ColorJitter.get_params returned in the torchvision that MonoDataset wants."""
    order, brightness, contrast, saturation, hue = parameters
    steps = [(vision.adjust_brightness, brightness), (vision.adjust_contrast, contrast),
             (vision.adjust_saturation, saturation), (vision.adjust_hue, hue)]

    def apply(image):
        for index in order:
            adjust, value = steps[int(index)]
            image = adjust(image, value)
        return image

    return apply


class _OldLoader:
    """Give the iterators of a DataLoader the next() method that Trainer.val calls."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        return next(self.iterator)

    def next(self):  # noqa: A003
        return next(self.iterator)


def _usable(dataset):
    """Return the indices with a previous and next frame and the camera info."""
    usable = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample[CAMERA + '@past'][0] and sample[CAMERA + '@future'][0] and sample[INFO]:
            usable.append(index)
    return usable


def _dataset_class(dataset, usable):
    first = dataset[usable[0]]
    height, width = first[CAMERA][1].shape[:2]
    k = np.asarray(first[INFO][1].k, dtype=np.float32).reshape(3, 3)

    class BagDataset(MonoDataset):
        """Frame numbers are 3 * sample + position, so frame_idxs -1 and 1 find their frames."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.K = np.array([[k[0, 0] / width, 0, k[0, 2] / width, 0],
                               [0, k[1, 1] / height, k[1, 2] / height, 0],
                               [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        def check_depth(self):
            return False

        def preprocess(self, inputs, color_aug):
            if isinstance(color_aug, tuple):
                color_aug = _jitter(color_aug)
            super().preprocess(inputs, color_aug)

        def get_color(self, folder, frame_index, side, do_flip):
            sample = dataset[usable[frame_index // 3]]
            position = frame_index % 3
            if position == CURRENT:
                rgb = sample[CAMERA][1]
            else:
                rgb = sample[CAMERA + ('@past' if position == PAST else '@future')][0][1]
            image = Image.fromarray(np.array(rgb))
            return image.transpose(Image.FLIP_LEFT_RIGHT) if do_flip else image

    return BagDataset


def train(dataset, context):
    if dataset is None:
        raise ValueError('monodepth2 trains on bag images: pass bags')
    usable = _usable(dataset)[:STEPS * BATCH]
    lines = [f'bag {3 * i + CURRENT} l' for i in range(len(usable))]
    monodepth_datasets.KITTIRAWDataset = _dataset_class(dataset, usable)
    monodepth_trainer.readlines = lambda path: lines
    options = MonodepthOptions().parser.parse_args([
        '--model_name', 'monodepth2', '--log_dir', context.out_dir or '.',
        '--height', str(SIZE[0]), '--width', str(SIZE[1]), '--batch_size', str(BATCH),
        '--num_epochs', '1', '--num_workers', '0', '--weights_init', 'scratch',
        '--load_weights_folder', str(weights_folder(context))])
    trainer = monodepth_trainer.Trainer(options)
    trainer.val_loader = _OldLoader(trainer.val_loader)
    trainer.val_iter = iter(trainer.val_loader)
    trainer.train()
    context.logger.info('weights: ' + str(
        Path(context.out_dir or '.', 'monodepth2', 'models', 'weights_0')))
