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
and infer() run in the container VAD_IMAGE. VAD_DIR names the VAD sources with the
config VAD_CONFIG, and VAD_DATA the output of tools/vad_prepare.py beside
train/<scene>/ images. Infer needs no ROS in the container: tools/vad_worker.py
reads frames from a pipe. Calibration comes from the annotation file.
"""

import os
from pathlib import Path
import pickle
import struct
import subprocess

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage

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
TOOLS = Path(__file__).resolve().parent.parent / 'tools'
IMAGE = os.environ.get('VAD_IMAGE', 'etri-vad:cu128')
CONFIG = os.environ.get('VAD_CONFIG', 'projects/configs/VAD/VAD_etri_tiny.py')


def _docker(*mounts):
    """Return the docker command with the sources, data, and tools mounted read-only."""
    code = os.environ['VAD_DIR']
    data = os.environ['VAD_DATA']
    command = ['docker', 'run', '--rm', '--gpus', 'all', '--ipc=host',
               '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'HOME=/tmp',
               '-e', f'PYTHONPATH={ROOT}', '-w', ROOT,
               '-v', f'{code}:{ROOT}:ro', '-v', f'{data}:{ROOT}/data/etri:ro',
               '-v', f'{TOOLS}:/tools:ro']
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


class _Worker:
    """A container that answers one frame at a time over its stdin and stdout."""

    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint).resolve()
        # The image's entry point prints a banner to stdout, which is the protocol here.
        arguments = _docker((checkpoint.parent, '/weights', 'ro')) + [
            '-i', '--entrypoint', 'python', IMAGE, '/tools/vad_worker.py', CONFIG,
            f'/weights/{checkpoint.name}', 'data/etri/vad_replay.pkl']
        self.process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        if self._read() != 'ready':
            raise RuntimeError('the VAD worker did not start')

    def _read(self):
        header = self.process.stdout.read(HEADER.size)
        if len(header) < HEADER.size:
            raise RuntimeError(f'the VAD worker ended with code {self.process.poll()}')
        return pickle.loads(self.process.stdout.read(HEADER.unpack(header)[0]))

    def ask(self, request):
        data = pickle.dumps(request, protocol=4)
        self.process.stdin.write(HEADER.pack(len(data)) + data)
        self.process.stdin.flush()
        return self._read()


def load(context):
    if not context.weights:
        raise ValueError('vad needs weights:= the path of a checkpoint')
    return _Worker(context.weights)


def infer(worker, sample):
    odom = sample[ODOM][1]
    imu = sample[IMU][1]
    goal = sample[GOAL]
    pose = odom.pose.pose
    plan = worker.ask({
        'images': [sample[topic][1] for topic in TOPICS],
        'stamp_s': sample['@time'] * 1e-9,
        'translation': [pose.position.x, pose.position.y, pose.position.z],
        'rotation': [pose.orientation.w, pose.orientation.x, pose.orientation.y,
                     pose.orientation.z],
        'acceleration': [imu.linear_acceleration.x, imu.linear_acceleration.y,
                         imu.linear_acceleration.z],
        'rotation_rate': [imu.angular_velocity.x, imu.angular_velocity.y,
                          imu.angular_velocity.z],
        'velocity': [odom.twist.twist.linear.x, odom.twist.twist.linear.y,
                     odom.twist.twist.linear.z],
        'command': command_for(None if goal is None else (goal[1].pose.position.x,
                                                          goal[1].pose.position.y)),
    })
    path = PathMessage()
    path.header.frame_id = FRAME
    for forward, left in plan:
        waypoint = PoseStamped()
        waypoint.header.frame_id = FRAME
        waypoint.pose.position.x, waypoint.pose.position.y = float(forward), float(left)
        waypoint.pose.orientation.w = 1.0
        path.poses.append(waypoint)
    return path


def train(dataset, context):
    if dataset is not None:
        raise ValueError('vad trains on its annotated scenes in VAD_DATA: pass no bags')
    out = Path(context.out_dir or '.', 'vad').resolve()
    out.mkdir(parents=True, exist_ok=True)
    arguments = _docker((out, '/out', 'rw')) + [
        IMAGE, 'python', 'tools/train.py', CONFIG, '--work-dir', '/out', '--no-validate',
        '--cfg-options', 'data.train.ann_file=data/etri/vad_train.pkl',
        'runner.max_epochs=1', 'total_epochs=1', 'data.workers_per_gpu=2',
        'log_config.interval=5']
    subprocess.run(arguments, check=True)
    context.logger.info(f'weights: {out}/latest.pth')
