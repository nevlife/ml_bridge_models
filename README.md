# ml_bridge_models

Model packages for [ml_bridge](https://github.com/nevlife/ml_bridge). Each model is one
ROS 2 package: inference runs as a ROS node, and training runs without ROS. Model
sources are used as they are.

| Package | Model | Sources | Training data | Inference input | Output |
| --- | --- | --- | --- | --- | --- |
| `yolo_ros` | YOLO | `ultralytics` | coco8 | camera | `~/detections` |
| `dinov3_ros` | DINOv3 ViT-S/16 | `timm` | bag images | camera | `~/features` |
| `monodepth2_ros` | monodepth2 | `MONODEPTH2_DIR` | bag images and camera_info | camera | `~/depth` |
| `omnivla_edge_ros` | OmniVLA-edge | `OMNIVLA_DIR` | bag images and odometry | camera and goal | `~/path` |
| `vad_ros` | VAD-Tiny | `VAD_DIR`, in a container | converted ETRI scenes | six cameras, odometry, IMU, goal | `~/path` |

## Package layout

```text
<model>_ros/
  config/spec.yaml       topics and alignment, shared by training and inference
  launch/infer.launch.py
  <model>_ros/common.py     preprocessing that both sides share, no ROS
  <model>_ros/training.py   train(dataset, context), no ROS
  <model>_ros/inference.py  load(context), infer(state, sample), ROS messages
  requirements.txt       Python packages of the model
```

## Environment

Create the inference environment from the ROS Python so that rclpy stays importable,
and keep NumPy at the version the ROS binaries were built with.

```bash
/usr/bin/python3 -m venv --system-site-packages venvs/models
echo "numpy==1.26.4" > venvs/models/constraints.txt
venvs/models/bin/pip install -c venvs/models/constraints.txt \
  -r ml_bridge_models/yolo_ros/requirements.txt
source venvs/models/bin/activate
colcon build --symlink-install --python-shebang "/usr/bin/env python3" \
  --packages-up-to yolo_ros
```

Training needs no ROS: put `rosbag2_dataset`, `ml_bridge`, and the model package on
`PYTHONPATH` in any environment that has the model's requirements.

## Train

```bash
python -m ml_bridge.training --spec dinov3_ros/config/spec.yaml \
  --model dinov3_ros.training --bags /data/run1 \
  --remap /camera/image:=/my_camera/image_raw --out-dir runs
```

`--remap` maps spec topics to bag topics. `omnivla_edge_ros` trains with
`config/train_spec.yaml`, whose future odometry is the label.

## Infer

```bash
ros2 launch dinov3_ros infer.launch.py camera:=/my_camera/image_raw \
  weights:=runs/dinov3_teacher.pt
```

Each launch file names its topic arguments.

## Models

monodepth2 is not vendored: set `MONODEPTH2_DIR` to a checkout of
https://github.com/nianticlabs/monodepth2 (non-commercial license). `weights` is a
folder with `encoder.pth`, `depth.pth`, `pose_encoder.pth`, and `pose.pth`.

OmniVLA-edge is not vendored either: set `OMNIVLA_DIR` to a checkout of
https://github.com/NHirose/OmniVLA and pass the `omnivla-edge.pth` of
https://huggingface.co/NHirose/omnivla-edge. Only the pose goal is wired, so CLIP is
not needed. The goal is in the robot frame, x forward and y left.
`make_nav_bag.py` joins a camera bag and an odometry bag and writes the goals.

VAD needs an older Python and mmcv, so training and inference run in a container
(`VAD_IMAGE`, default `etri-vad:cu128`, which has no ROS). Set `VAD_DIR` to an unmodified
checkout of https://github.com/hustvl/VAD, `VAD_DATA` to the output of
`vad_etri_to_nuscenes.py` (run in the container), and `VAD_RESNET` to the ResNet-50 file
of its configs. Pass the VAD-Tiny checkpoint of its README as weights. The conversion
changes conventions, not content: the lidar frame turns from forward +x to forward +y,
names map to nuScenes classes, and images are undistorted and cropped. `make_vad_bag.py`
makes a replay bag. Replay it on rmw_zenoh with shared memory.

DataLoader workers send their batches through pickle, and decoded bag messages do not
pickle: a `collate_fn` that returns a message makes the loader wait forever. Return
arrays, as the training modules here do.

## License

MIT. Each model has its own license.
