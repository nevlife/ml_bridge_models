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
VAD (Jiang et al.) planning from six cameras, in its own container.

VAD needs a Python and mmcv that cannot share a process with this ROS, so train()
and infer() run in the container VAD_IMAGE. VAD_DIR names a checkout of
https://github.com/hustvl/VAD, used unmodified, VAD_DATA the output of
vad_etri_to_nuscenes.py, and VAD_RESNET the ResNet-50 file of its configs.
context.weights is the VAD checkpoint: train() starts from it, infer() runs it.
container/vad_worker.py reads frames from a pipe; calibration comes from the annotations.
"""

import os
from pathlib import Path
import struct

CAMERAS = ('camera_front', 'camera_front_right', 'camera_front_left', 'camera_rear_wide',
           'camera_rear_left', 'camera_rear_right')
TOPICS = tuple(f'/{name}/image' for name in CAMERAS)
ODOM = '/odom'
IMU = '/imu'
GOAL = '/goal_pose'
FRAME = 'base_link'
TURN_M = 2.0
RIGHT, LEFT, STRAIGHT = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
HEADER = struct.Struct('<Q')
ROOT = '/workspace/VAD'
CONTAINER = Path(__file__).resolve().parent / 'container'
IMAGE = os.environ.get('VAD_IMAGE', 'etri-vad:cu128')
CONFIG = os.environ.get('VAD_CONFIG', 'projects/configs/VAD/VAD_tiny_stage_2.py')


def docker(*mounts):
    """Return the docker command with the sources, data, and tools mounted read-only."""
    code = os.environ['VAD_DIR']
    data = os.environ['VAD_DATA']
    command = ['docker', 'run', '--rm', '--gpus', 'all', '--ipc=host',
               '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'HOME=/tmp',
               '-e', f'PYTHONPATH={ROOT}', '-w', ROOT,
               '-v', f'{code}:{ROOT}:ro', '-v', f'{data}:/data:ro',
               '-v', f'{CONTAINER}:/tools:ro']
    for host, container, mode in mounts:
        command += ['-v', f'{host}:{container}:{mode}']
    return command


def command_for(goal):
    """Return VAD's driving command for a goal (forward, left) in the robot frame."""
    if goal is None:
        return STRAIGHT
    left = goal[1]
    if left >= TURN_M:
        return LEFT
    return RIGHT if left <= -TURN_M else STRAIGHT
