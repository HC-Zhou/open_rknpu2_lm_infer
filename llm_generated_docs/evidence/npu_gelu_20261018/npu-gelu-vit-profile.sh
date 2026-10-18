#!/usr/bin/env bash
set -euo pipefail
root=logs/npu-gelu-vit
mkdir -p "$root"
model=/home/firefly/mnt/3090x4/llm/w4a8_quant/transfer/qwen35/models/C7
for image in red blue red512; do
    path=/home/firefly/mnt/3090x4/tmp/qwen35-vision-20261018/$image.png
    if [ "$image" = red512 ]; then path=/home/firefly/mnt/3090x4/tmp/floating-mm-util-20261018/red512.png; fi
    index=0
    for kind in cpu npu npu cpu; do
        if [ "$kind" = cpu ]; then bin=logs/vision-committed-baseline-20261018/vision_profile
        else bin=target/float-autotune/release/examples/vision_profile_npu; fi
        taskset -c 4-7 "$bin" --model "$model" --image "$path" --vision-dtype fp16 \
            --tune-cache .cache/mm-tune-20261018 --output "$root/$image-$index-$kind" \
            > "$root/$image-$index-$kind.log" 2>&1
        index=$((index+1))
    done
done
python3 - <<'PY'
from pathlib import Path
import json,statistics
import numpy as np
root=Path('logs/npu-gelu-vit')
gold=Path('/home/firefly/mnt/3090x4/tmp/vision-fp16-storage-20261018/fp32-gold/reference')
report={}
def difference(ref,value):
    x,y=ref.astype(np.float64),value.astype(np.float64)
    assert np.isfinite(y).all() and x.shape==y.shape
    return dict(relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),
                cosine=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),max_abs=float(np.abs(x-y).max()))
for image in ('red','blue','red512'):
    paths=[root/f'{image}-{i}-{kind}' for i,kind in enumerate(('cpu','npu','npu','cpu'))]
    features=[np.fromfile(p/'features.f32',dtype='<f4') for p in paths]
    assert np.array_equal(features[0].view(np.uint32),features[3].view(np.uint32))
    assert np.array_equal(features[1].view(np.uint32),features[2].view(np.uint32))
    row=dict(features_vs_cpu=difference(features[0],features[1]),repeatable=True)
    for kind,indices in [('cpu',(0,3)),('npu',(1,2))]:
        samples=[sample for i in indices for sample in json.loads((paths[i]/'profile.json').read_text())['samples']]
        row[kind]={'median_ms':statistics.median(s['total_ms'] for s in samples),
                   'samples_ms':[s['total_ms'] for s in samples],
                   'stages_ms':{key:statistics.median(s['stages_ms'][key] for s in samples) for key in samples[0]['stages_ms']}}
    row['speedup']=row['cpu']['median_ms']/row['npu']['median_ms']
    path=gold/f'c7-fp32-{image}/features.f32'
    if path.exists():
        reference=np.fromfile(path,dtype='<f4')
        row['cpu_vs_fp32']=difference(reference,features[0]);row['npu_vs_fp32']=difference(reference,features[1])
    report[image]=row
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert all(row['features_vs_cpu']['relative_l2']<.02 for row in report.values()),'ViT drift exceeds research gate'
PY
