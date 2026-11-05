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
Make a navigation test bag: camera, odometry in a forward-x frame, and goals.

The goal of each image is where the odometry shows the robot HORIZON_S later, in the
robot frame at that image, so a model's path can be compared with the path driven.
"""

import argparse
import math

from geometry_msgs.msg import PoseStamped
from omnivla_edge_ros.poses import planar, relative
from nav_msgs.msg import Odometry
import numpy as np
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py
from sensor_msgs.msg import Image

HORIZON_S = 2.664


def _reader(uri):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    return reader


def _stamp_ns(message):
    return message.header.stamp.sec * 10**9 + message.header.stamp.nanosec


def _forward_x(odometry, offset):
    """Turn the odometry's child frame by offset about z, onto a forward-x frame."""
    x, y, yaw = planar(odometry.pose.pose)
    yaw += offset
    message = Odometry()
    message.header = odometry.header
    message.header.frame_id = 'odom'
    message.child_frame_id = 'base_link'
    message.pose.pose.position.x, message.pose.pose.position.y = x, y
    message.pose.pose.orientation.z = math.sin(yaw / 2)
    message.pose.pose.orientation.w = math.cos(yaw / 2)
    return message


def _pose_at(times, poses, time_ns):
    index = int(np.searchsorted(times, time_ns))
    if index == 0 or index == len(times):
        return None
    start, end = poses[index - 1], poses[index]
    ratio = (time_ns - times[index - 1]) / (times[index] - times[index - 1])
    turn = math.remainder(end[2] - start[2], math.tau)
    return tuple(a + ratio * (b - a) for a, b in zip(start[:2], end[:2])) + (
        start[2] + ratio * turn,)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-bag', required=True)
    parser.add_argument('--camera-topic', required=True)
    parser.add_argument('--odom-bag', required=True)
    parser.add_argument('--odom-topic', default='/Odometry')
    parser.add_argument('--forward-offset-deg', type=float, default=0.0,
                        help='yaw from the odometry child frame to forward x')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    odometry = []
    reader = _reader(args.odom_bag)
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == args.odom_topic:
            odometry.append(_forward_x(deserialize_message(data, Odometry),
                                       math.radians(args.forward_offset_deg)))
    times = np.array([_stamp_ns(m) for m in odometry])
    poses = [planar(m.pose.pose) for m in odometry]

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=args.output, storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    for index, (name, kind) in enumerate(((args.camera_topic, 'sensor_msgs/msg/Image'),
                                          ('/odom', 'nav_msgs/msg/Odometry'),
                                          ('/goal_pose', 'geometry_msgs/msg/PoseStamped'))):
        writer.create_topic(rosbag2_py.TopicMetadata(id=index, name=name, type=kind,
                                                     serialization_format='cdr'))
    rows = [(time, '/odom', serialize_message(m)) for time, m in zip(times, odometry)]
    reader = _reader(args.camera_bag)
    while reader.has_next():
        topic, data, log_ns = reader.read_next()
        if topic != args.camera_topic:
            continue
        rows.append((log_ns, topic, data))
        stamp = _stamp_ns(deserialize_message(data, Image))
        origin = _pose_at(times, poses, stamp)
        target = _pose_at(times, poses, stamp + int(HORIZON_S * 1e9))
        if origin is None or target is None:
            continue
        forward, left, yaw = relative(origin, target)
        goal = PoseStamped()
        goal.header.frame_id = 'base_link'
        goal.header.stamp.sec, goal.header.stamp.nanosec = divmod(stamp, 10**9)
        goal.pose.position.x, goal.pose.position.y = forward, left
        goal.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.orientation.w = math.cos(yaw / 2)
        rows.append((log_ns, '/goal_pose', serialize_message(goal)))
    rows.sort(key=lambda row: row[0])
    for log_ns, topic, data in rows:
        writer.write(topic, data, int(log_ns))
    print(f'{len(rows)} messages in {args.output}')


if __name__ == '__main__':
    main()
