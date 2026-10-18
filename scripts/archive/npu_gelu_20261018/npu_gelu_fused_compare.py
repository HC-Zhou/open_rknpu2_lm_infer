#!/usr/bin/env python3
"""Paired complete-ViT comparison of two fused layouts/collectors."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np
from npu_gelu_fused_vit import difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before-bin', 'after-bin', 'model', 'cache', 'reference-dir', 'red', 'blue', 'red512', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {}
    for image in ('red', 'blue', 'red512'):
        runs = dict(before=[], after=[])
        features = {}
        for index, kind in enumerate(('before', 'after', 'after', 'before')):
            out = args.output/f'{image}-{index}-{kind}'
            print(f'Running {image} {kind} {index}', flush=True)
            command = [str(getattr(args, kind+'_bin')), '--model', str(args.model), '--image', str(getattr(args, image)),
                       '--vision-dtype', 'fp16', '--tune-cache', str(args.cache), '--output', str(out)]
            with (args.output/f'{image}-{index}-{kind}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               pass_fds=(int(os.environ['LLM_INFER_LOCK_FD']),))
            report = json.loads((out/'profile.json').read_text())
            runs[kind].extend(report['samples'])
            f = np.fromfile(out/'features.f32', dtype='<f4')
            if kind in features:
                assert np.array_equal(f.view(np.uint32), features[kind].view(np.uint32))
            features[kind] = f
        equal = np.array_equal(features['before'].view(np.uint32), features['after'].view(np.uint32))
        row = dict(repeatable=True, features_bitwise_equal=bool(equal))
        gold = np.fromfile(args.reference_dir/f'c7-fp32-{image}/features.f32', dtype='<f4')
        for kind, samples in runs.items():
            row[kind] = dict(median_ms=statistics.median(s['total_ms'] for s in samples),
                             samples_ms=[s['total_ms'] for s in samples],
                             stages_ms={key: statistics.median(s['stages_ms'][key] for s in samples)
                                        for key in samples[0]['stages_ms']},
                             features_vs_fp32=difference(gold, features[kind]))
        row['speedup'] = row['before']['median_ms']/row['after']['median_ms']
        result[image] = row
        (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({image: row}), flush=True)
        assert equal, 'layout/collector changed feature bits'


if __name__ == '__main__':
    main()
