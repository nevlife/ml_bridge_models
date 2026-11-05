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

"""Make a VAD test bag from replay.json of vad_etri_to_nuscenes.py: cameras, ego, goals."""

import argparse
import json
from pathlib import Path

import cv2
from geometry_msgs.msg import PoseStamped
from vad_ros.common import CAMERAS
from nav_msgs.msg import Odometry
import numpy as np
from rclpy.serialization import serialize_message
import rosbag2_py
from sensor_msgs.msg import Image, Imu

DATA_PREFIX = '/data/'


def _stamped(message, stamp_ns, frame):
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp_ns, 10**9)
    message.header.frame_id = frame
    return message


def _image(path, stamp_ns, frame):
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    message = _stamped(Image(), stamp_ns, frame)
    message.height, message.width = rgb.shape[:2]
    message.encoding = 'rgb8'
    message.step = rgb.shape[1] * 3
    message.data = rgb.tobytes()
    return message


def _odometry(row, stamp_ns):
    message = _stamped(Odometry(), stamp_ns, 'map')
    message.child_frame_id = 'base_link'
    pose = message.pose.pose
    pose.position.x, pose.position.y, pose.position.z = row['translation']
    (pose.orientation.w, pose.orientation.x, pose.orientation.y,
     pose.orientation.z) = row['rotation']
    bus = row['can_bus']
    twist = message.twist.twist
    twist.linear.x, twist.linear.y, twist.linear.z = bus[13:16]
    twist.angular.x, twist.angular.y, twist.angular.z = bus[10:13]
    return message


def _imu(row, stamp_ns):
    message = _stamped(Imu(), stamp_ns, 'base_link')
    bus = row['can_bus']
    acceleration, rate = message.linear_acceleration, message.angular_velocity
    acceleration.x, acceleration.y, acceleration.z = bus[7:10]
    rate.x, rate.y, rate.z = bus[10:13]
    return message


def _goal(row, stamp_ns):
    if not any(row['future_mask']):
        return None
    forward, left = np.sum(row['future'], axis=0)
    message = _stamped(PoseStamped(), stamp_ns, 'base_link')
    message.pose.position.x, message.pose.position.y = float(forward), float(left)
    message.pose.orientation.w = 1.0
    return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='folder of replay.json and images/')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = Path(args.data)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=args.output, storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    topics = {f'/{name}/image': 'sensor_msgs/msg/Image' for name in CAMERAS}
    topics.update({'/odom': 'nav_msgs/msg/Odometry', '/imu': 'sensor_msgs/msg/Imu',
                   '/goal_pose': 'geometry_msgs/msg/PoseStamped'})
    for index, (name, kind) in enumerate(topics.items()):
        writer.create_topic(rosbag2_py.TopicMetadata(id=index, name=name, type=kind,
                                                     serialization_format='cdr'))
    rows = json.loads((data / 'replay.json').read_text())
    for row in rows:
        stamp_ns = int(row['stamp_ms'] * 1e6)
        for name in CAMERAS:
            path = data / row['images'][name].removeprefix(DATA_PREFIX)
            writer.write(f'/{name}/image', serialize_message(_image(path, stamp_ns, name)),
                         stamp_ns)
        writer.write('/odom', serialize_message(_odometry(row, stamp_ns)), stamp_ns)
        writer.write('/imu', serialize_message(_imu(row, stamp_ns)), stamp_ns)
        goal = _goal(row, stamp_ns)
        if goal is not None:
            writer.write('/goal_pose', serialize_message(goal), stamp_ns)
    print(f'{len(rows)} frames in {args.output}')


if __name__ == '__main__':
    main()
