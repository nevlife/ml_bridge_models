import math
from types import SimpleNamespace

from ml_bridge_models.poses import limit_range, relative, yaw_of
import pytest


def test_relative_puts_a_point_ahead_on_the_forward_axis():
    robot = (10.0, 5.0, math.pi / 2)
    forward, left, yaw = relative(robot, (10.0, 7.0, math.pi / 2))
    assert forward == pytest.approx(2.0)
    assert left == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)


def test_relative_keeps_left_positive():
    forward, left, _ = relative((0.0, 0.0, 0.0), (1.0, 3.0, 0.0))
    assert (forward, left) == pytest.approx((1.0, 3.0))


def test_yaw_of_a_quarter_turn():
    half = math.pi / 4
    q = SimpleNamespace(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))
    assert yaw_of(q) == pytest.approx(math.pi / 2)


def test_limit_range_only_shortens_far_points():
    assert limit_range(3.0, 4.0, 10.0) == (3.0, 4.0)
    assert limit_range(30.0, 40.0, 10.0) == pytest.approx((6.0, 8.0))
