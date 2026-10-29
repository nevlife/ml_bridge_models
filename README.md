# ml_bridge_models

Models for [ml_bridge](https://github.com/nevlife/ml_bridge): one module per model
(`load`, `train`, `infer`) and its spec in `specs/`. It is a pip package for the
model's virtual environment, not a ROS package (`COLCON_IGNORE`).

| Model | Module | Spec | Training data | Inference input |
| --- | --- | --- | --- | --- |
| YOLO (ultralytics) | `ml_bridge_models.yolo` | `specs/yolo.yaml` | coco8 (bags have no box labels) | one camera |

## Environment

Create the virtual environment from the ROS Python so that rclpy and the message
packages stay importable, keep NumPy at the version the ROS binaries were built
with, and build `ml_bridge` with an interpreter line that follows the active
environment.

```bash
/usr/bin/python3 -m venv --system-site-packages venvs/yolo
echo "numpy==1.26.4" > venvs/yolo/constraints.txt
venvs/yolo/bin/pip install -c venvs/yolo/constraints.txt ultralytics
venvs/yolo/bin/pip install --no-deps -e ml_bridge_models
colcon build --symlink-install --packages-select ml_bridge --python-shebang "/usr/bin/env python3"
```

## Run

```bash
source venvs/yolo/bin/activate
ros2 run ml_bridge run --ros-args -p spec:=ml_bridge_models/specs/yolo.yaml \
  -p model:=ml_bridge_models.yolo -p mode:=train -p out_dir:=runs
ros2 run ml_bridge run --ros-args -p spec:=ml_bridge_models/specs/yolo.yaml \
  -p model:=ml_bridge_models.yolo -p mode:=infer -p weights:=runs/yolo/weights/best.pt \
  -r /camera/image:=/my_camera/image_raw
```

ultralytics is AGPL-3.0; this repository only imports it.

## License

MIT. See [LICENSE](LICENSE).
