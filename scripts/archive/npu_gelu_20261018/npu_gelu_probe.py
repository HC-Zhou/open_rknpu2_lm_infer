#!/usr/bin/env python3
"""Minimal FP16 NPU GELU LUT experiment, reusing the validated SiLU recipe.

Uncached buffers and one core: correctness only, not a production latency result.
Every call reloads the LUT, including the before/after SiLU canaries.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def gelu(x):
    x = np.asarray(x, dtype=np.float64)
    return .5*x*(1+np.tanh(.7978845608028654*(x+.044715*x*x*x)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--count', default=256, type=int)
    parser.add_argument('--tails', action='store_true')
    parser.add_argument('--encoding', choices=('biased-half', 'raw-q13', 'raw-half'), default='biased-half')
    args = parser.parse_args()
    if args.count % 8 or not 32 <= args.count <= 524288 or (args.count > 4096 and args.count % 4096):
        parser.error('32..4096 by 8, or up to 524288 by 4096 for width128/channel tiling')
    if not os.environ.get('LLM_INFER_LOCK_FD'):
        parser.error('run under scripts/run_board.py')
    sys.path.insert(0, str(args.reference_root.resolve()))
    from activation import silu_regs
    from probe import Device, d, patch
    args.output.mkdir(parents=True, exist_ok=True)
    device = Device()
    report = {'count': args.count, 'tails': args.tails, 'encoding': args.encoding,
              'scope': 'single-core uncached correctness probe'}

    def run(x, kind):
        am, ar = device.alloc(x.nbytes)
        cm, cr = device.alloc(x.nbytes+4096)
        # Bound the LUT input before its fixed index gain. Outside [-4,4],
        # GELU is zero/identity to within 7.1e-5; restore those tails below.
        staged = np.clip(x, -4, 4).astype(np.float16) if kind == 'gelu' else x
        am[:x.nbytes] = staged.tobytes()
        cm[:] = b'\xa5'*len(cm)
        regs = silu_regs(min(len(x), 4096), ar.dma_addr, cr.dma_addr, 'float16', load=kind == 'silu')
        if len(x) > 4096:
            # Same physical [channels/8,width,8] view as FloatReduce::add_regs.
            # A unary map preserves flat host order under this reinterpretation.
            width, channels = 128, len(x)//128
            for reg, value in {0x4024:width*16, 0x4030:width-1, 0x403c:((channels-1)<<16)|(channels-1),
                               0x4058:channels-1, 0x405c:width-1, 0x40c0:width*16}.items():
                patch(regs, d.T_DPU, reg, value)
            for reg, value in {0x500c:width-1, 0x5014:channels-1, 0x5040:width*16}.items():
                patch(regs, d.T_RDMA, reg, value)
        if kind == 'gelu':
            words = []
            for bank in (0, 1):
                words.append(d.e(d.T_DPU, 0x4100, (2|bank)<<16))
                for i in range(513):
                    value = float(gelu((i-512 if bank == 0 else i)/128))
                    if args.encoding == 'raw-q13':
                        code = int(np.clip(np.rint(value*8192), -32768, 32767))
                    elif args.encoding == 'raw-half':
                        code = int(np.float16(value+.25).view(np.uint16))
                    else:
                        code = int(np.clip(np.rint((value+.25)*4096), -32768, 32767))
                        assert code > 0
                    words.append(d.e(d.T_DPU, 0x4104, code))
            patch(regs, d.T_DPU, 0x4068, int(np.float16(4096).view(np.uint16))<<16)
            patch(regs, d.T_DPU, 0x4088, 12<<12)
            if args.encoding.startswith('raw-'):
                patch(regs, d.T_DPU, 0x4010, 0x28000002)
                patch(regs, d.T_DPU, 0x4084, 1)
                patch(regs, d.T_DPU, 0x4088, 0)
            regs = words+regs
        device.prepare([(regs, 0x18)])
        timing = device.run()
        if kind == 'gelu' and args.encoding == 'raw-q13':
            out = (np.frombuffer(cm, dtype=np.int16, count=len(x)).astype(np.float32)/8192).astype(np.float16)
        else:
            out = np.frombuffer(cm, dtype=np.float16, count=len(x)).copy()
        assert cm[x.nbytes:x.nbytes+4096] == b'\xa5'*4096, 'output guard changed'
        if kind == 'gelu':
            if args.encoding != 'raw-q13':
                out = (out.astype(np.float32)-np.float32(.25)).astype(np.float16)
            out[x > 4] = x[x > 4]
            out[x < -4] = np.float16(-0.)
            out[(out == 0) & np.signbit(x)] = np.float16(-0.)
        return out, timing

    try:
        anchors = np.array([-8, -2, -1, -.5, .5, 1, 2, 8], dtype=np.float16)
        ref = anchors.astype(np.float64)/(1+np.exp(-anchors.astype(np.float64)))
        before, _ = run(anchors, 'silu')
        assert np.all(np.isfinite(before)) and np.all(np.abs(before-ref) <= .003+.001*np.abs(ref)), (anchors, before, ref)
        x = np.linspace(-8, 8, args.count).astype(np.float16)
        values = [-16, -8, -4, -2, -1, -.1, -.001, 0, .001, .1, 1, 2, 4, 8, 16]
        if args.tails:
            values += [-65504, -100, -32, -16.125, 16.125, 32, 100, 65504]
        x[:len(values)] = values
        output, timing = run(x, 'gelu')
        expected = gelu(x)
        delta = np.abs(output.astype(np.float64)-expected)
        report.update(input=x[:32].astype(float).tolist(), output=output[:32].astype(float).tolist(),
                      expected=expected[:32].tolist(), max_abs=float(delta.max()),
                      relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(expected)),
                      central_relative_l2=float(np.linalg.norm(delta[np.abs(x) <= 4])/np.linalg.norm(expected[np.abs(x) <= 4])),
                      nonfinite=int(np.count_nonzero(~np.isfinite(output))), timing=timing,
                      guard_ok=True)
        after, _ = run(anchors, 'silu')
        report['canary_restored'] = bool(np.array_equal(before.view(np.uint16), after.view(np.uint16)))
        report['passed'] = bool(report['canary_restored'] and np.all(np.isfinite(output))
                                and np.all(delta <= .003+.001*np.abs(expected)))
        np.savez(args.output/'tensors.npz', input=x, output=output, reference=expected)
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        return 0 if report['passed'] else 1
    finally:
        device.close()


if __name__ == '__main__':
    raise SystemExit(main())
