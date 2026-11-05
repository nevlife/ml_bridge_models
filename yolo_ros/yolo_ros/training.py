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
Train YOLO (ultralytics) without ROS.

Recorded bags have no box labels, so train() fine-tunes on the coco8 sample set that
ultralytics downloads.
"""

from ultralytics import YOLO
from .common import DEFAULT_WEIGHTS


def train(dataset, context):
    if dataset is not None:
        context.logger.info(f'{len(dataset)} bag samples have no labels; training on coco8')
    model = YOLO(context.weights or DEFAULT_WEIGHTS)
    result = model.train(data='coco8.yaml', epochs=1, imgsz=320, batch=4, workers=0,
                         project=context.out_dir or 'runs', name='yolo', exist_ok=True,
                         plots=False)
    context.logger.info(f'weights: {result.save_dir}/weights/best.pt')
