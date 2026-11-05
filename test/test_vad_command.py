import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'vad', Path(__file__).parents[1] / 'ml_bridge_models' / 'vad.py')
vad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vad)


def test_no_goal_is_straight():
    assert vad.command_for(None) == vad.STRAIGHT


def test_goal_side_selects_the_turn():
    assert vad.command_for((10.0, 3.0)) == vad.LEFT
    assert vad.command_for((10.0, -3.0)) == vad.RIGHT
    assert vad.command_for((10.0, 1.0)) == vad.STRAIGHT
