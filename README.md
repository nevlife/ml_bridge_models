# ml_bridge_models

Models for [ml_bridge](https://github.com/nevlife/ml_bridge): one module per model
(`load`, `train`, `infer`) and its spec in `specs/`. It is a pip package for the
model's virtual environment, not a ROS package (`COLCON_IGNORE`).

| Model | Module | Spec | Training data | Inference input |
| --- | --- | --- | --- | --- |
| YOLO (ultralytics) | `ml_bridge_models.yolo` | `specs/yolo.yaml` | coco8 (bags have no box labels) | one camera |
| DINOv3 ViT-S/16 (timm) | `ml_bridge_models.dinov3` | `specs/dinov3.yaml` | bag images, DINO-style self-distillation (no labels) | one camera; publishes PCA of patch features as an image |
| monodepth2 (Niantic) | `ml_bridge_models.monodepth2` | `specs/monodepth2.yaml` | bag images with the previous and next frame and camera_info, its self-supervised reprojection loss | one camera; publishes depth as 32FC1 (relative scale) |
| OmniVLA-edge (Hirose et al.) | `ml_bridge_models.omnivla_edge` | `specs/omnivla_edge.yaml`, `specs/omnivla_edge_train.yaml` | bag images and odometry; the future odometry is the label | one camera and a goal pose; publishes a path |
| VAD tiny (Jiang et al.) | `ml_bridge_models.vad` | `specs/vad.yaml` | annotated ETRI scenes in its container | six cameras, odometry, IMU, and a goal that picks the command; publishes a path |

monodepth2 is not vendored: set `MONODEPTH2_DIR` to a checkout of
https://github.com/nianticlabs/monodepth2 (non-commercial license) and pass
`weights:=` a folder with `encoder.pth`, `depth.pth`, `pose_encoder.pth`, and
`pose.pth` (default `$MONODEPTH2_DIR/models/mono_640x192`).

OmniVLA-edge is not vendored either: set `OMNIVLA_DIR` to a checkout of
https://github.com/NHirose/OmniVLA and pass `weights:=` the `omnivla-edge.pth` of
https://huggingface.co/NHirose/omnivla-edge. Only the pose goal is wired, so CLIP is
not needed. Install the extra `omnivla_edge`. `/goal_pose` is in the robot frame, x
forward and y left. The model comes from robots near 0.5 m/s: on a faster robot its
path stays short and only its direction is comparable with the path driven.
`tools/make_nav_bag.py` joins a camera bag and an odometry bag and writes the goals.

VAD needs an older Python and mmcv, so it runs in a container (`VAD_IMAGE`, default
`etri-vad:cu128`, which has no ROS). Set `VAD_DIR` to the sources and `VAD_DATA` to the
output of `tools/vad_prepare.py` (run in the container) beside `train/<scene>/`.
`tools/vad_worker.py` answers frames over a pipe, and calibration comes from the first
frame of the annotation file. `tools/make_vad_bag.py` makes a replay bag. Its 8.8 MB
images need a transport that keeps them: best effort over Cyclone UDP delivered no
sample, rmw_zenoh with shared memory delivered them.

DataLoader workers send their batches through pickle, and decoded bag messages do
not pickle: a `collate_fn` that returns a message makes the loader wait forever.
Return arrays, as the modules here do.

## Environment

Create the virtual environment from the ROS Python so that rclpy and the message
packages stay importable, keep NumPy at the version the ROS binaries were built
with, and build `ml_bridge` with an interpreter line that follows the active
environment.

```bash
/usr/bin/python3 -m venv --system-site-packages venvs/yolo
echo "numpy==1.26.4" > venvs/yolo/constraints.txt
venvs/yolo/bin/pip install -c venvs/yolo/constraints.txt ultralytics timm==1.0.30
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
ros2 run ml_bridge run --ros-args -p spec:=ml_bridge_models/specs/dinov3.yaml \
  -p model:=ml_bridge_models.dinov3 -p mode:=train -p bags:="['/data/run1']" \
  -p out_dir:=runs -r /camera/image:=/my_camera/image_raw
```

Topic remapping applies to the bag topics in train mode as well.

ultralytics is AGPL-3.0; this repository only imports it.

## License

MIT. See [LICENSE](LICENSE).
