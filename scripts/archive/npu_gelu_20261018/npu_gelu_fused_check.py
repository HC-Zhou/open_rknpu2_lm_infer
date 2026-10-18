#!/usr/bin/env python3
"""Validate fused FP32-carrier output, panel layout, tails and input updates."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('LLM_INFER_LOCK_FD'), 'run under scripts/run_board.py'
    lib = ct.CDLL(str(args.bridge.resolve()))
    lib.npu_gelu_fused_create.argtypes = [ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
    lib.npu_gelu_fused_create.restype = ct.c_void_p
    lib.npu_gelu_fused_run.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]
    lib.npu_gelu_fused_run.restype = ct.c_void_p
    lib.npu_gelu_fused_guard.argtypes = [ct.c_void_p]
    lib.npu_gelu_fused_free.argtypes = [ct.c_void_p]
    lib.npu_gelu_fused_error.restype = ct.c_char_p
    report = []
    for m, n, k in [(16, 32, 32), (128, 128, 32), (256, 4096, 1024), (1024, 4096, 1024)]:
        weight = np.zeros((k, n), dtype=np.float16)
        weight[np.arange(n)%k, np.arange(n)] = 1
        bias = ((np.arange(n)%5-2)/8).astype(np.float32)
        x = (((np.arange(m*k)%97)-48)/16).astype(np.float16).reshape(m, k)
        anchors = [-65504, -100, -32, -16.125, -8, -4, -2, -1, -.1, -.001, 0., .001, .1, 1, 2, 4, 8, 16.125, 32, 100, 65504]
        x[0, :len(anchors)] = anchors
        p = lib.npu_gelu_fused_create(n, k, weight.ctypes.data, bias.ctypes.data)
        assert p, lib.npu_gelu_fused_error()
        try:
            for flip in (False, True):
                current = -x if flip else x
                stages = np.zeros(5, dtype=np.float64)
                address = lib.npu_gelu_fused_run(p, m, current.ctypes.data, stages.ctypes.data)
                assert address, lib.npu_gelu_fused_error()
                raw = np.ctypeslib.as_array(ct.cast(address, ct.POINTER(ct.c_float)), shape=(m*n,)).copy()
                got = (raw.reshape(m//8, n//4, 8, 4).transpose(0, 2, 1, 3).reshape(m, n)/8192).astype(np.float16)
                a = current[:, np.arange(n)%k].astype(np.float64)+bias
                ref = .5*a*(1+np.tanh(.7978845608028654*(a+.044715*a*a*a)))
                error = got.astype(np.float64)-ref
                guard = bool(lib.npu_gelu_fused_guard(p))
                row = dict(M=m, N=n, K=k, flipped=flip, guard_ok=guard,
                           max_abs=float(np.abs(error).max()),
                           relative_l2=float(np.linalg.norm(error)/np.linalg.norm(ref)),
                           central_max_abs=float(np.abs(error[np.abs(a)<=4]).max()), stages_us=stages.tolist())
                report.append(row)
                print(json.dumps(row), flush=True)
                assert guard and np.isfinite(got).all()
                assert np.all(np.abs(error) <= .002+.0005*np.abs(ref)), 'fused numerical/layout mismatch'
        finally:
            lib.npu_gelu_fused_free(p)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
