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

"""Planar poses for navigation models: x forward, y left, in metres and radians."""

import math


def yaw_of(q):
    """Return the yaw of a quaternion message."""
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def planar(pose):
    """Return (x, y, yaw) of a Pose message."""
    return pose.position.x, pose.position.y, yaw_of(pose.orientation)


def relative(origin, target):
    """Return target as (forward, left, yaw) in the robot frame at origin."""
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    c, s = math.cos(origin[2]), math.sin(origin[2])
    return c * dx + s * dy, -s * dx + c * dy, target[2] - origin[2]


def limit_range(forward, left, radius):
    """Pull a point onto the circle of this radius when it lies beyond it."""
    distance = math.hypot(forward, left)
    if distance <= radius:
        return forward, left
    return forward * radius / distance, left * radius / distance
