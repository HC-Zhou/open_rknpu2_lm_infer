#!/usr/bin/env python3
"""ABBA compare normal vision_profile against a frozen fused implementation."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import numpy as np


def difference(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    assert x.shape == y.shape and np.isfinite(y).all()
    return dict(relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),
                max_abs=float(np.abs(x-y).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before-bin', 'after-bin', 'model', 'cache', 'reference-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--image', type=Path, action='append', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {}
    for image in args.image:
        runs = dict(before=[], after=[])
        features = {}
        for index, kind in enumerate(('before', 'after', 'after', 'before')):
            out = args.output/f'{image.stem}-{index}-{kind}'
            command = [str(getattr(args, kind+'_bin')), '--model', str(args.model),
                       '--image', str(image), '--vision-dtype', 'fp16',
                       '--tune-cache', str(args.cache), '--output', str(out)]
            print('Running', image.name, kind, index, flush=True)
            fd = os.environ.get('LLM_INFER_LOCK_FD')
            with (args.output/f'{image.stem}-{index}-{kind}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               pass_fds=(int(fd),) if fd else ())
            report = json.loads((out/'profile.json').read_text())
            runs[kind].extend(report['samples'])
            f = np.fromfile(out/'features.f32', dtype='<f4')
            if kind in features:
                assert np.array_equal(f.view(np.uint32), features[kind].view(np.uint32))
            features[kind] = f
        equal = np.array_equal(features['before'].view(np.uint32), features['after'].view(np.uint32))
        row = dict(repeatable=True, features_bitwise_equal=bool(equal))
        gold = np.fromfile(args.reference_dir/f'c7-fp32-{image.stem}/features.f32', dtype='<f4')
        for kind, samples in runs.items():
            row[kind] = dict(median_ms=statistics.median(s['total_ms'] for s in samples),
                             samples_ms=[s['total_ms'] for s in samples],
                             stages_ms={key: statistics.median(s['stages_ms'][key] for s in samples)
                                        for key in samples[0]['stages_ms']},
                             features_vs_fp32=difference(gold, features[kind]))
        row['speedup'] = row['before']['median_ms']/row['after']['median_ms']
        result[image.stem] = row
        (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({image.stem: row}), flush=True)
        assert equal, 'formal integration changed feature bits'


if __name__ == '__main__':
    main()
