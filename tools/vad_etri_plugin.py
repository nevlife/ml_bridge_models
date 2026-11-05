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
Give the unmodified VAD dataset its map from lanes.pkl; loaded before the VAD tools.

The dataset builds VectorizedLocalMap from nuScenes map files, which converted ETRI
scenes do not have. LaneMap has the same interface and labels every lane a divider.
"""

import os
from pathlib import Path
import pickle

import numpy as np
from nuscenes.eval.common.utils import Quaternion
from projects.mmdet3d_plugin.datasets import nuscenes_vad_dataset
from shapely.geometry import box, LineString

DIVIDER = 0
# The sources predate NumPy 1.24, which removed these aliases.
for alias in (bool, int, float, object):
    if alias.__name__ not in vars(np):
        setattr(np, alias.__name__, alias)


class LaneMap:
    """Vectors of the lanes around a pose, in its lidar frame."""

    def __init__(self, dataroot, patch_size, map_classes, fixed_ptsnum_per_line=-1,
                 padding_value=-10000, **_):
        with open(Path(os.environ.get('VAD_LANES', Path(dataroot, 'lanes.pkl'))), 'rb') as stream:
            self.lanes = pickle.load(stream)
        self.patch_size = patch_size
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

    def gen_vectorized_samples(self, location, translation, rotation):
        height, width = self.patch_size
        patch = box(-width / 2, -height / 2, width / 2, height / 2)
        matrix = Quaternion(rotation).rotation_matrix
        lines = []
        for lane in self.lanes[location]:
            local = (np.asarray(lane)[:, :3] - np.asarray(translation)) @ matrix
            cropped = LineString(local[:, :2]).intersection(patch)
            if cropped.geom_type == 'MultiLineString':
                cropped = max(cropped.geoms, key=lambda g: g.length)
            if cropped.geom_type == 'LineString' and cropped.length > 0:
                lines.append(cropped)
        vectors = nuscenes_vad_dataset.LiDARInstanceLines(
            lines, sample_dist=1, num_samples=250, padding=False, fixed_num=self.fixed_num,
            padding_value=self.padding_value, patch_size=self.patch_size)
        return {'gt_vecs_pts_loc': vectors, 'gt_vecs_label': [DIVIDER] * len(lines)}


nuscenes_vad_dataset.VectorizedLocalMap = LaneMap
