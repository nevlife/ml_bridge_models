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

"""YOLO detection (ultralytics) on one camera."""

import numpy as np
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from .common import CAMERA, DEFAULT_WEIGHTS


def load(context):
    return YOLO(context.weights or DEFAULT_WEIGHTS)


def infer(model, sample):
    _, rgb = sample[CAMERA]
    # ultralytics reads NumPy images as BGR and keeps the last one, so it gets its own copy.
    result = model.predict(np.array(rgb[..., ::-1]), verbose=False)[0]
    message = Detection2DArray()
    boxes = result.boxes
    for (x0, y0, x1, y1), cls, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(),
                                            boxes.conf.tolist()):
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
