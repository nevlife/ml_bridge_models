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
Convert ETRI scenes to the nuScenes conventions of the VAD sources; run in the container.

The frame convention differs, not the content: ETRI's lidar frame looks forward along
+x and nuScenes' along +y, so every lidar-frame value turns by TURN about z. Names map
to nuScenes classes, images are undistorted and cropped, keyframes are 2 Hz. Lanes go
to lanes.pkl for vad_etri_plugin.py. Writes vad_train.pkl, vad_replay.pkl, replay.json.
"""

import argparse
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
from pyquaternion import Quaternion

TURN = np.pi / 2
ROTATION = np.array([[0.0, -1.0], [1.0, 0.0]])
NAMES = {'Car': 'car', 'Pedestrian': 'pedestrian', 'Cyclist': 'bicycle'}
CROP_SIZE = (1920, 1080)
KEEP_TOP = ('camera_front_left', 'camera_front_right', 'camera_rear_left',
            'camera_rear_right', 'camera_rear_wide')
KEYFRAME_EVERY = 5
TURN_M = 2.0


def turn_xy(values):
    """Turn the last axis, pairs of x and y, by TURN."""
    values = np.asarray(values, dtype=np.float64)
    return (values.reshape(-1, 2) @ ROTATION.T).reshape(values.shape)


def command(future_deltas):
    """Follow the VAD converter: right is +x at the end of the path, in its frame."""
    lateral = np.cumsum(future_deltas, axis=0)[-1, 0]
    if lateral >= TURN_M:
        return np.array([1, 0, 0], dtype=np.float32)
    if lateral <= -TURN_M:
        return np.array([0, 1, 0], dtype=np.float32)
    return np.array([0, 0, 1], dtype=np.float32)


def convert_image(name, cam, source, target):
    raw = cv2.imread(str(source))
    height, width = raw.shape[:2]
    raw_k = np.asarray(cam['cam_intrinsic_raw'], dtype=np.float64)
    new_k = np.asarray(cam['cam_intrinsic'], dtype=np.float64)
    distortion = np.asarray(cam['distortion'], dtype=np.float64)
    if cam['is_fisheye']:
        maps = cv2.fisheye.initUndistortRectifyMap(raw_k, distortion[:4], np.eye(3), new_k,
                                                   (width, height), cv2.CV_32FC1)
    else:
        maps = cv2.initUndistortRectifyMap(raw_k, distortion, None, new_k, (width, height),
                                           cv2.CV_32FC1)
    flat = cv2.remap(raw, maps[0], maps[1], cv2.INTER_LINEAR)
    left = (width - CROP_SIZE[0]) // 2
    top = 0 if name in KEEP_TOP else height - CROP_SIZE[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), flat[top:top + CROP_SIZE[1], left:left + CROP_SIZE[0]],
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    shifted = new_k.copy()
    shifted[0, 2] -= left
    shifted[1, 2] -= top
    return shifted


def convert_info(info, images, out):
    new = dict(info)
    boxes = np.array(info['gt_boxes'], dtype=np.float64)
    boxes[:, :2] = turn_xy(boxes[:, :2])
    boxes[:, 6] -= TURN
    new['gt_boxes'] = boxes
    new['gt_names'] = np.array([NAMES.get(n, n) for n in info['gt_names']])
    new['gt_velocity'] = turn_xy(info['gt_velocity'])
    new['gt_agent_fut_trajs'] = turn_xy(info['gt_agent_fut_trajs']).astype(np.float32)
    goals = np.array(info['gt_agent_fut_goal'], dtype=np.float32)
    moving = goals < 9
    goals[moving] = (goals[moving] + 2) % 8
    new['gt_agent_fut_goal'] = goals
    lcf = np.array(info['gt_agent_lcf_feat'], dtype=np.float32)
    lcf[:, 0:2] = turn_xy(lcf[:, 0:2])
    lcf[:, 2] += TURN
    lcf[:, 3:5] = turn_xy(lcf[:, 3:5])
    new['gt_agent_lcf_feat'] = lcf
    new['gt_ego_his_trajs'] = turn_xy(info['gt_ego_his_trajs']).astype(np.float32)
    new['gt_ego_fut_trajs'] = turn_xy(info['gt_ego_fut_trajs']).astype(np.float32)
    new['gt_ego_fut_cmd'] = command(new['gt_ego_fut_trajs'])
    ego = np.array(info['gt_ego_lcf_feat'], dtype=np.float32)
    ego[:2] = turn_xy(ego[:2])
    new['gt_ego_lcf_feat'] = ego
    new['lidar2ego_rotation'] = list(Quaternion(axis=[0, 0, 1], angle=-TURN).q)
    new['sweeps'] = []
    turn = np.eye(3)
    turn[:2, :2] = ROTATION
    cams = {}
    for name, cam in info['cams'].items():
        camera = dict(cam)
        relative = Path(cam['data_path']).relative_to('data/etri')
        target = out / 'images' / relative
        camera['cam_intrinsic'] = convert_image(name, cam, images / relative, target)
        camera['data_path'] = str(Path('/data/images') / relative)
        camera['sensor2lidar_rotation'] = turn @ cam['sensor2lidar_rotation']
        camera['sensor2lidar_translation'] = turn @ cam['sensor2lidar_translation']
        cams[name] = camera
    new['cams'] = cams
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--images', required=True, help='folder of train/<scene>/')
    parser.add_argument('--train-scene', required=True)
    parser.add_argument('--replay-scene', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    with open(args.annotations, 'rb') as stream:
        data = pickle.load(stream)
    out = Path(args.out)
    metadata = {k: v for k, v in data['metadata'].items() if k != 'map_lanes'}

    def scene(token):
        frames = [i for i in data['infos'] if i['scene_token'] == token][::KEYFRAME_EVERY]
        return [convert_info(i, Path(args.images), out) for i in frames]

    train, replay = scene(args.train_scene), scene(args.replay_scene)
    for name, infos in (('vad_train.pkl', train), ('vad_replay.pkl', replay)):
        with open(out / name, 'wb') as stream:
            pickle.dump({'infos': infos, 'metadata': metadata}, stream)
    lanes = {info['map_location']: data['metadata']['map_lanes'].get(info['scene_token'], [])
             for info in train + replay}
    with open(out / 'lanes.pkl', 'wb') as stream:
        pickle.dump(lanes, stream)
    source = {i['frame_idx']: i for i in data['infos']
              if i['scene_token'] == args.replay_scene}
    rows = [{'stamp_ms': i['timestamp'],
             'images': {name: cam['data_path'] for name, cam in i['cams'].items()},
             'translation': list(i['ego2global_translation']),
             'rotation': list(i['ego2global_rotation']),
             'can_bus': [float(v) for v in i['can_bus']],
             'future': source[i['frame_idx']]['gt_ego_fut_trajs'].tolist(),
             'future_mask': i['gt_ego_fut_masks'].tolist()} for i in replay]
    (out / 'replay.json').write_text(json.dumps(rows))
    print(f'{len(train)} train frames, {len(replay)} replay frames')


if __name__ == '__main__':
    main()
