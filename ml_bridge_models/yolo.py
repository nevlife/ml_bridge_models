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
YOLO detection (ultralytics) on one camera.

Recorded bags have no box labels, so train() fine-tunes on the coco8 sample set
that ultralytics downloads; inference runs on the camera topic.
"""

from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

CAMERA = '/camera/image'
DEFAULT_WEIGHTS = 'yolo11n.pt'


def load(context):
    return YOLO(context.weights or DEFAULT_WEIGHTS)


def train(dataset, context):
    if dataset is not None:
        context.logger.info(f'{len(dataset)} bag samples have no labels; training on coco8')
    model = YOLO(context.weights or DEFAULT_WEIGHTS)
    result = model.train(data='coco8.yaml', epochs=1, imgsz=320, batch=4, workers=0,
                         project=context.out_dir or 'runs', name='yolo', exist_ok=True,
                         plots=False)
    context.logger.info(f'weights: {result.save_dir}/weights/best.pt')


def infer(model, sample):
    _, rgb = sample[CAMERA]
    # ultralytics reads NumPy images as BGR.
    result = model.predict(rgb[..., ::-1], verbose=False)[0]
    message = Detection2DArray()
    for (x0, y0, x1, y1), cls, score in zip(result.boxes.xyxy.tolist(),
                                             result.boxes.cls.tolist(),
                                             result.boxes.conf.tolist()):
        detection = Detection2D()
        detection.bbox.center.position.x = (x0 + x1) / 2
        detection.bbox.center.position.y = (y0 + y1) / 2
        detection.bbox.size_x = x1 - x0
        detection.bbox.size_y = y1 - y0
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = result.names[int(cls)]
        hypothesis.hypothesis.score = score
        detection.results.append(hypothesis)
        message.detections.append(detection)
    return message
