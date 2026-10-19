#!/usr/bin/env python3
"""Small dense-PyTorch oracle for the fixed C7 MTP coordinate transform."""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tests/fixtures/mtp/adapter.json'))
    args = parser.parse_args()
    residues = {i*i % 19 for i in range(1, 19)}
    h20 = torch.ones(20, 20, dtype=torch.float64)
    for i in range(1, 20):
        for j in range(1, 20):
            h20[i,j] = 1 if (i-j) % 19 in residues else -1
    h128 = torch.ones(1, 1, dtype=torch.float64)
    while len(h128) < 128:
        h128 = torch.cat([torch.cat([h128,h128],1),torch.cat([h128,-h128],1)],0)
    signs = 2*torch.randint(0,2,(2560,),generator=torch.Generator().manual_seed(42))-1
    rotation = signs[:,None]*torch.kron(h20,h128)/(2560**0.5)
    torch.testing.assert_close(rotation@rotation.T,torch.eye(2560,dtype=torch.float64))
    x = (torch.arange(2560)*17%107-53).double()/32
    indices = [0,1,7,63,127,128,255,256,511,1023,1279,1280,2047,2559]
    report = dict(torch=torch.__version__,rotation='signed Hadamard 20x128',seed=42,
        input='(i*17%107-53)/32',indices=indices,forward=(x@rotation)[indices].tolist(),
        inverse=(x@rotation.T)[indices].tolist())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
