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
Prepare ETRI scenes for ml_bridge_models.vad; run inside the VAD container.

Writes to OUT a training annotation file with TRAIN_FRAMES frames of one scene, a
replay annotation file of another scene, and replay.json for make_vad_bag.py.
"""

import argparse
import json
from pathlib import Path
import pickle

TRAIN_FRAMES = range(100, 140)
REPLAY_EVERY = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--train-scene', required=True)
    parser.add_argument('--replay-scene', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    with open(args.annotations, 'rb') as stream:
        data = pickle.load(stream)
    out = Path(args.out)

    def write(name, infos, scene):
        lanes = {scene: data['metadata']['map_lanes'].get(scene, [])}
        metadata = {**data['metadata'], 'map_lanes': lanes}
        with open(out / name, 'wb') as stream:
            pickle.dump({'infos': infos, 'metadata': metadata}, stream)

    frames = [i for i in data['infos'] if i['scene_token'] == args.train_scene]
    write('vad_train.pkl', [i for i in frames if i['frame_idx'] in TRAIN_FRAMES],
          args.train_scene)
    replay = [i for i in data['infos'] if i['scene_token'] == args.replay_scene]
    write('vad_replay.pkl', replay, args.replay_scene)
    rows = [{'stamp_ms': i['timestamp'],
             'images': {name: cam['data_path'] for name, cam in i['cams'].items()},
             'translation': list(i['ego2global_translation']),
             'rotation': list(i['ego2global_rotation']),
             'can_bus': [float(v) for v in i['can_bus']],
             'future': i['gt_ego_fut_trajs'].tolist(),
             'future_mask': i['gt_ego_fut_masks'].tolist()}
            for i in replay[::REPLAY_EVERY]]
    (out / 'replay.json').write_text(json.dumps(rows))
    print(f'{len(frames)} train scene frames, {len(rows)} replay frames')


if __name__ == '__main__':
    main()
