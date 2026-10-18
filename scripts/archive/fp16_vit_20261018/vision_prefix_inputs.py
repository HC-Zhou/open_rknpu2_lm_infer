#!/usr/bin/env python3
"""Prepare prefix-replacement controls and paired old/new FP32 suffix cuts."""
import json
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from vision_stage_oracle import BASE,ROOT,load,save,map_tensors,model


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    net,_=model()
    interventions=[]
    summary={}
    board='/home/firefly/mnt/3090x4/tmp/vision-fp16-storage-20261018'
    def board_path(path):
        return str(path).replace(str(BASE),board)
    with torch.inference_mode(),torch.autocast('cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
        for image in ('red','blue'):
            old_dir=BASE/f'precision/{image}-original-fp32/trace'
            actual=ROOT/f'{image}-actual/trace'
            args=map_tensors(torch.load(ROOT/f'gold-{image}/block-args.pt',weights_only=True),lambda x:x.cuda())
            inp=load(actual/'p0-l23-vision_block_input.f32')
            last=net.blocks[23](inp,**args)
            path=ROOT/f'{image}-last-fp32.f32'
            save(path,last)
            def add(name,at,path):
                interventions.append(dict(name=image+'-'+name,at=at,path=board_path(path) if path else '',
                    image=f'/home/firefly/mnt/3090x4/tmp/qwen35-vision-20261018/{image}.png'))
            add('baseline',-2,None)
            add('noop-prefix0',0,actual/'p0-l0-vision_hidden.f32')
            add('patch-old',-1,old_dir/'p0-l-1-vision_patch_position.f32')
            add('patch-gold',-1,ROOT/f'gold-{image}/input.f32')
            add('prefix1-old',0,old_dir/'p0-l0-vision_hidden.f32')
            add('prefix2-old',1,old_dir/'p0-l1-vision_hidden.f32')
            add('prefix1-gold',0,ROOT/f'gold-{image}/p0-l0-vision_hidden.f32')
            add('last-fp32',23,path)
            add('baseline-after',-2,None)
            old_features=load(BASE/f'rotary/profile-{image}-1-no-scan/features.f32',2560)
            new_features=load(BASE/f'rotary/profile-{image}-2-fp16/features.f32',2560)
            total=(new_features.double()-old_features.double()).reshape(-1)
            total2=total.square().sum();norm=old_features.double().norm()
            def suffix(x,index):
                for j in range(index+1,24):x=net.blocks[j](x,**args)
                return net.merger(x)
            previous=torch.zeros_like(total)
            cuts=[]
            names=[(-1,'vision_patch_position')]+[(i,'vision_hidden') for i in range(24)]
            for i,name in names:
                filename=f'p0-l{i}-{name}.f32'
                delta=(suffix(load(actual/filename),i).double()-suffix(load(old_dir/filename),i).double()).reshape(-1)
                change=delta-previous
                cuts.append(dict(layer=i,stage='patch_position' if i<0 else 'block',
                    projection_on_new_old_difference=(torch.dot(change,total)/total2).item(),
                    relative_effect_l2=(change.norm()/norm).item()))
                previous=delta
            change=total-previous
            cuts.append(dict(stage='merger',projection_on_new_old_difference=(torch.dot(change,total)/total2).item(),
                             relative_effect_l2=(change.norm()/norm).item()))
            assert abs(sum(z['projection_on_new_old_difference'] for z in cuts)-1)<1e-8
            summary[image]=cuts
            print(image,[(z.get('layer',z['stage']),round(z['projection_on_new_old_difference']*100,2)) for z in cuts],flush=True)
    (ROOT/'interventions.json').write_text(json.dumps(interventions,indent=2)+'\n')
    (ROOT/'paired-prefix-summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
