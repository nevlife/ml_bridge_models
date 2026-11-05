#!/usr/bin/env python3
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
VAD inference worker; run inside the VAD container, which has no ROS.

Frames arrive on stdin and paths leave on stdout as length-prefixed pickles.
Calibration and the pipeline come from the first frame of an annotation file; each
request replaces the images and ego state. EOF on stdin ends the worker.
"""

import argparse
import importlib
import os
import pickle
import struct
import sys

# Importing the libraries already prints to stdout: keep it for the protocol first.
OUTPUT = os.fdopen(os.dup(1), 'wb')
os.dup2(2, 1)

from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel, collate  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HEADER = struct.Struct('<Q')
SCENE = 'live'


def read_message(stream):
    header = stream.read(HEADER.size)
    if len(header) < HEADER.size:
        return None
    return pickle.loads(stream.read(HEADER.unpack(header)[0]))


def write_message(stream, value):
    data = pickle.dumps(value, protocol=4)
    stream.write(HEADER.pack(len(data)) + data)
    stream.flush()


def can_bus(request):
    """Lay out the ego state as in NuScenes CAN bus records, angles last."""
    w, x, y, z = request['rotation']
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) % (2 * np.pi)
    return np.concatenate([request['translation'], request['rotation'],
                           request['acceleration'], request['rotation_rate'],
                           request['velocity'], [yaw, np.degrees(yaw)]])


def prepare(dataset, request, index):
    """Run the test pipeline with the request in place of files and annotations."""
    results = dataset.get_data_info(0)
    results.update(scene_token=SCENE, frame_idx=index, sample_idx=str(index),
                   timestamp=request['stamp_s'], can_bus=can_bus(request),
                   ego_fut_cmd=np.array(request['command'], dtype=np.float32),
                   ego_his_trajs=np.zeros_like(results['ego_his_trajs']),
                   ego_lcf_feat=np.zeros_like(results['ego_lcf_feat']))
    dataset.pre_pipeline(results)
    images = [frame[..., ::-1].astype(np.float32) for frame in request['images']]
    stacked = images[0].shape + (len(images),)
    results.update(filename=results['img_filename'], img=images, img_shape=stacked,
                   ori_shape=stacked, pad_shape=stacked, scale_factor=1.0,
                   img_norm_cfg={'mean': np.zeros(3, dtype=np.float32),
                                 'std': np.ones(3, dtype=np.float32), 'to_rgb': False})
    for transform in dataset.pipeline.transforms[1:]:
        results = transform(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('annotations')
    args = parser.parse_args()
    source = sys.stdin.buffer

    cfg = Config.fromfile(args.config)
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.test.update(ann_file=args.annotations, test_mode=True)
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.pop('map_ann_file', None)
    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(0), device_ids=[0]).eval()
    write_message(OUTPUT, 'ready')

    index = 0
    while (request := read_message(source)) is not None:
        collated = collate([prepare(dataset, request, index)], samples_per_gpu=1)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **collated)
        plans = result[0]['pts_bbox']['ego_fut_preds']
        chosen = plans[int(np.argmax(request['command']))].cpu().double().cumsum(0)
        write_message(OUTPUT, chosen.numpy())
        index += 1


if __name__ == '__main__':
    main()
