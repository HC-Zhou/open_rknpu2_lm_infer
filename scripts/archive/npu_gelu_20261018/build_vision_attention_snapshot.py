#!/usr/bin/env python3
"""Instrument Attention state/timings in a copied fused-GELU ViT crate."""
import argparse
from pathlib import Path
import shutil
import subprocess

from build_npu_gelu_fused_snapshot import replace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    subprocess.run(['python3',str(root/'tests/board/build_npu_gelu_fused_snapshot.py'),
                    '--output',str(args.output),'--name','vision_attention_probe',
                    '--bridge','logs/npu-gelu-bridge/libnpu_gelu_fused_fixed.so'],check=True)
    out=args.output.resolve()
    shutil.copy2(root/'tests/board/vision_attention_snapshot.rs',out/'src/npu/attention_probe.rs')
    replace(out/'src/npu.rs','mod attention;','mod attention;\nmod attention_probe;')
    path=out/'src/npu/attention.rs'
    source=path.read_text()
    boundary=source.index('    /// Stateless noncausal attention')
    half,rest=source[:boundary],source[boundary:]
    marker='            check(\n                unsafe {\n                    rk_npu_attention_f16_run_half('
    assert half.count(marker)==1
    half=half.replace(marker,'            let mut timing=rk_npu_attention_f16_timings::default();\n'+marker,1)
    assert half.count('                        ptr::null_mut(),')==1
    half=half.replace('                        ptr::null_mut(),','                        &mut timing,',1)
    marker='            for h in 0..heads {\n                for r in 0..m {\n                    let t ='
    assert half.count(marker)==1
    half=half.replace(marker,'            super::attention_probe::record(self.cache.raw.cast(), w.workspace.raw.cast(), &timing, rows, offset);\n'+marker,1)
    path.write_text(half+rest)
    print('Instrumented Attention state:',out)


if __name__=='__main__':main()
