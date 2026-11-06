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

"""Run VAD on camera topics through a worker in its container."""

from pathlib import Path
import pickle
import subprocess

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
import numpy as np
from .common import (CONFIG, FRAME, GOAL, HEADER, IMAGE, IMU, INFOS, ODOM, TOPICS, command_for,
                     docker)


class _Worker:
    """A container that answers one frame at a time over its stdin and stdout."""

    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint).resolve()
        # The image's entry point prints a banner to stdout, which is the protocol here.
        arguments = docker((checkpoint.parent, '/weights', 'ro')) + [
            '-i', '--entrypoint', 'python', IMAGE, '/tools/vad_worker.py', CONFIG,
            f'/weights/{checkpoint.name}', '/data/vad_replay.pkl']
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
        'intrinsics': [None if sample[topic] is None else np.reshape(sample[topic][1].k, (3, 3))
                       for topic in INFOS],
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
