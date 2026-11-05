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

"""Train VAD without ROS: the original tools/train.py in its container."""

import os
from pathlib import Path
import subprocess
from .common import CONFIG, IMAGE, docker


def train(dataset, context):
    if dataset is not None:
        raise ValueError('vad trains on its annotated scenes in VAD_DATA: pass no bags')
    if not context.weights:
        raise ValueError('vad needs weights:= the checkpoint to start from')
    out = Path(context.out_dir or '.', 'vad').resolve()
    out.mkdir(parents=True, exist_ok=True)
    weights = Path(context.weights).resolve()
    resnet = Path(os.environ['VAD_RESNET']).resolve()
    arguments = docker((out, '/out', 'rw'), (weights.parent, '/weights', 'ro'),
                        (resnet, '/resnet.pth', 'ro')) + [
        '--entrypoint', 'python', IMAGE, '/tools/vad_train.py', CONFIG, '--work-dir', '/out',
        '--no-validate', '--cfg-options', 'data.train.data_root=/data/',
        'data.train.ann_file=/data/vad_train.pkl', f'load_from=/weights/{weights.name}',
        'model.pretrained.img=/resnet.pth', 'runner.max_epochs=1', 'total_epochs=1',
        'data.workers_per_gpu=2', 'log_config.interval=5']
    subprocess.run(arguments, check=True)
    context.logger.info(f'weights: {out}/latest.pth')
