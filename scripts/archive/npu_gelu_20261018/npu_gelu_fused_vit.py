#!/usr/bin/env python3
"""Compare CPU GELU, separate NPU GELU and fused UP in complete ViT calls."""
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
                cosine=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),
                max_abs=float(np.abs(x-y).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cpu-bin', 'separate-bin', 'fused-bin', 'model', 'cache', 'reference-dir', 'red', 'blue', 'red512', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binaries = dict(cpu=args.cpu_bin, separate=args.separate_bin, fused=args.fused_bin)
    result = {}
    for image in ('red', 'blue', 'red512'):
        runs = {kind: [] for kind in binaries}
        features = {}
        for index, kind in enumerate(('cpu', 'separate', 'fused', 'fused', 'separate', 'cpu')):
            out = args.output/f'{image}-{index}-{kind}'
            print(f'Running {image} {kind} {index}', flush=True)
            command = [str(binaries[kind]), '--model', str(args.model), '--image', str(getattr(args, image)),
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
        gold = np.fromfile(args.reference_dir/f'c7-fp32-{image}/features.f32', dtype='<f4')
        row = dict(repeatable=True)
        for kind, samples in runs.items():
            row[kind] = dict(median_ms=statistics.median(s['total_ms'] for s in samples),
                             samples_ms=[s['total_ms'] for s in samples],
                             stages_ms={key: statistics.median(s['stages_ms'][key] for s in samples)
                                        for key in samples[0]['stages_ms']},
                             features_vs_fp32=difference(gold, features[kind]))
        row['fused_vs_cpu'] = difference(features['cpu'], features['fused'])
        row['fused_vs_separate'] = difference(features['separate'], features['fused'])
        row['fused_speedup_vs_cpu'] = row['cpu']['median_ms']/row['fused']['median_ms']
        row['fused_speedup_vs_separate'] = row['separate']['median_ms']/row['fused']['median_ms']
        result[image] = row
        (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({image: row}), flush=True)


if __name__ == '__main__':
    main()
