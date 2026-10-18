#!/usr/bin/env python3
"""Bounded direct-FP16 LUT output and output-converter offset probe."""
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
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encoding', choices=('q13', 'biased-q12-offset', 'half-bits', 'f32-proc'), required=True)
    args = parser.parse_args()
    if not os.environ.get('LLM_INFER_LOCK_FD'):
        parser.error('run under scripts/run_board.py')
    sys.path.insert(0, str(args.reference_root.resolve()))
    from activation import silu_regs
    from probe import Device, d, patch
    args.output.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-16, 16, 4096).astype(np.float16)
    anchors = [-65504, -100, -32, -16.125, -8, -4, -2, -1, -.1, -.001, -0., 0.,
               .001, .1, 1, 2, 4, 8, 16.125, 32, 100, 65504]
    x[:len(anchors)] = anchors
    dev = Device()
    report = {'encoding': args.encoding, 'scope': 'uncached one-core direct output correctness'}
    try:
        am, ar = dev.alloc(x.nbytes)
        cm, cr = dev.alloc(x.nbytes+4096)
        am[:x.nbytes] = x.tobytes()
        cm[:] = b'\xa5'*len(cm)
        regs = silu_regs(len(x), ar.dma_addr, cr.dma_addr, 'float16', load=False)
        patch(regs, d.T_DPU, 0x4068, int(np.float16(4096).view(np.uint16))<<16)
        patch(regs, d.T_DPU, 0x4088, 13<<12)
        # Slope in LUT-code units / index units. The index gain is 4096.
        patch(regs, d.T_DPU, 0x4128, 16384<<16)
        patch(regs, d.T_DPU, 0x412c, 13<<5)
        if args.encoding == 'biased-q12-offset':
            patch(regs, d.T_DPU, 0x4080, (-1024)&0xffffffff)
            patch(regs, d.T_DPU, 0x4088, 12<<12)
            patch(regs, d.T_DPU, 0x4128, 8192<<16)
        if args.encoding == 'half-bits':
            patch(regs, d.T_DPU, 0x4010, 0x28000002)
            patch(regs, d.T_DPU, 0x4084, 1)
            patch(regs, d.T_DPU, 0x4088, 0)
        if args.encoding == 'f32-proc':
            patch(regs, d.T_DPU, 0x4010, (2<<29)|(2<<26)|5)
            patch(regs, d.T_RDMA, 0x5044, 0x178a9)
        words = []
        for bank in (0, 1):
            words.append(d.e(d.T_DPU, 0x4100, (2|bank)<<16))
            for i in range(513):
                t = (i-512 if bank == 0 else i)/128
                y = float(gelu(t))
                if args.encoding == 'biased-q12-offset':
                    code = int(np.rint((y+.25)*4096))
                elif args.encoding == 'half-bits':
                    code = int(np.float16(y).view(np.uint16))
                else:
                    code = int(np.rint(y*8192))
                words.append(d.e(d.T_DPU, 0x4104, int(np.clip(code, -32768, 65535))))
        dev.prepare([(words+regs, 0x18)])
        report['timing'] = dev.run()
        got = np.frombuffer(cm, dtype=np.float16, count=len(x)).copy()
        ref = gelu(x)
        error = got.astype(np.float64)-ref
        central = np.abs(x)<=4
        report.update(input=x[:len(anchors)].astype(float).tolist(),
                      output=got[:len(anchors)].astype(float).tolist(),
                      reference=ref[:len(anchors)].tolist(),
                      max_abs=float(np.abs(error).max()),
                      central_max_abs=float(np.abs(error[central]).max()),
                      central_relative_l2=float(np.linalg.norm(error[central])/np.linalg.norm(ref[central])),
                      nonfinite=int(np.count_nonzero(~np.isfinite(got))),
                      guard_ok=cm[x.nbytes:x.nbytes+4096] == b'\xa5'*4096)
        np.savez(args.output/'tensors.npz', input=x, output=got, reference=ref)
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        assert report['guard_ok'], 'output guard changed'
    finally:
        dev.close()


if __name__ == '__main__':
    main()
