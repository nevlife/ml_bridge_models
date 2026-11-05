from vad_ros.common import command_for, LEFT, RIGHT, STRAIGHT


def test_no_goal_is_straight():
    assert command_for(None) == STRAIGHT


def test_goal_side_selects_the_turn():
    assert command_for((10.0, 3.0)) == LEFT
    assert command_for((10.0, -3.0)) == RIGHT
    assert command_for((10.0, 1.0)) == STRAIGHT
