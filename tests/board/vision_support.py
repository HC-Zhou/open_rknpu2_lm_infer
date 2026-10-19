"""Shared arguments for explicit RK3588 multimodal end-to-end tests."""
import argparse
import os
from pathlib import Path


def arguments(description, terminal=False, photos_required=True):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--photos', required=photos_required, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', type=Path, default=Path('.cache'))
    parser.add_argument('--spec-type', choices=('none', 'draft-mtp'), default='none')
    parser.add_argument('--spec-draft-n-max', type=int, default=3)
    parser.add_argument('--spec-draft-p-min', type=float, default=0.)
    parser.add_argument('--spec-draft-sampling', choices=('greedy', 'probabilistic'), default='greedy')
    if terminal:
        parser.add_argument('--python-deps', type=Path)
    args = parser.parse_args()
    for name in ('binary', 'model', 'photos', 'output'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def command(args):
    return [str(args.binary), '--model', str(args.model), '--ctx-size', '2048',
            '--ubatch-size', '128', '--threads', '4', '--cpu-mask', '0xf0',
            '--npu-mask', '7', '--tune-cache', str(args.cache),
            '--spec-type', args.spec_type, '--spec-draft-n-max', str(args.spec_draft_n_max),
            '--spec-draft-p-min', str(args.spec_draft_p_min),
            '--spec-draft-sampling', args.spec_draft_sampling]


def environment(log_level='info'):
    env = dict(os.environ, RUST_LOG=log_level, NO_COLOR='1', TERM='xterm-256color')
    # The test runner retains the global lock. Production children must start
    # without inheriting it or receiving the inherited-lock bypass variables.
    for key in ('OPEN_RKNPU2_LOCK_FD', 'LLM_INFER_LOCK_FD', 'OPEN_RKNPU_API_KEY'):
        env.pop(key, None)
    return env
