#!/usr/bin/env python3
"""Local-error and error-propagation probes for ViT, using FP32 CUDA oracles."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Also supports staging these three scripts together in the artifact directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'qwen35'))
import vision_reference as upstream
from vision_precision_reference import visual_state

BASE = Path('/data007/zt/tmp/vision-fp16-storage-20261018')
ROOT = BASE/'precision-stages'
GOLD = BASE/'fp32-gold/reference'
MODEL = Path('/data007/zt/llm/w4a8_quant/transfer/qwen35/models/C7')


def load(path, width=1024):
    return torch.from_numpy(np.fromfile(path, dtype='<f4').reshape(-1, width)).cuda()


def save(path, value):
    upstream.dump(path, value)


def map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(map_tensors(x, fn) for x in value)
    if isinstance(value, dict):
        return {k:map_tensors(v, fn) for k,v in value.items()}
    return value


def model():
    definitions = upstream.definitions(BASE.parent/'qwen35-vision-20261018/upstream')
    config = json.loads((MODEL/'config.json').read_text())['vision_config']
    config.update(num_attention_heads=config['num_heads'],
                  rope_parameters={'rope_type':'axial','rope_theta':10000}, _attn_implementation='sdpa')
    result = definitions['Qwen3_5VisionModel'](SimpleNamespace(**config)).eval().float().cuda()
    result.load_state_dict(visual_state(MODEL), strict=True)
    return result, definitions


def prepare(net):
    manifest = json.loads((GOLD/'manifest.json').read_text())
    for image in ('red', 'blue'):
        directory = ROOT/f'gold-{image}'
        directory.mkdir(exist_ok=True)
        def capture(module, args, kwargs):
            save(directory/'input.f32', args[0])
            torch.save(map_tensors(kwargs, lambda x:x.detach().cpu()), directory/'block-args.pt')
        handle = net.blocks[0].register_forward_pre_hook(capture, with_kwargs=True)
        checks = []
        for i, block in enumerate(net.blocks):
            def check(module, args, output, i=i):
                name = f'p0-l{i}-vision_hidden.f32'
                expected = load(GOLD/f'c7-fp32-{image}'/name)
                assert torch.equal(output, expected), (image, i)
                shutil.copy2(GOLD/f'c7-fp32-{image}'/name, directory/name)
            checks.append(block.register_forward_hook(check))
        pixels = load(GOLD/f'{image}-patches.f32', 1536)
        grid = torch.tensor([manifest['runs'][f'c7-fp32-{image}']['grid']], device='cuda')
        result = net(pixels, grid).pooler_output
        assert torch.equal(result, load(GOLD/f'c7-fp32-{image}/features.f32', 2560))
        handle.remove()
        for handle in checks:
            handle.remove()
        print('prepared', image, flush=True)


def analyze(net, definitions):
    def compare(ref, value):
        x, y = ref.reshape(-1).double(), value.reshape(-1).double()
        assert x.numel() == y.numel() and torch.isfinite(y).all()
        delta = y-x
        return dict(relative_l2=(delta.norm()/x.norm().clamp_min(1e-30)).item(),
                    error_l2=delta.norm().item(), reference_l2=x.norm().item(),
                    max_abs=delta.abs().max().item())
    def rounded(x):
        return x.half().float()
    def attention(q, k, v, scale):
        def head(x):
            return x.reshape(-1,16,64).transpose(0,1).unsqueeze(0)
        out = F.scaled_dot_product_attention(head(q),head(k),head(v),scale=scale)
        return out.transpose(1,2).reshape(-1,1024)
    def qkv_parts(values, kwargs):
        q,k,v = values.reshape(-1,3,16,64).permute(1,0,2,3).unbind(0)
        q,k = definitions['apply_rotary_pos_emb_vision'](q,k,*kwargs['position_embeddings'])
        return q.reshape(-1,1024),k.reshape(-1,1024),v.reshape(-1,1024)
    def block_parts(block, x, kwargs):
        n1=block.norm1(x); qkv=block.attn.qkv(n1)
        q,k,v=qkv_parts(qkv,kwargs)
        a=attention(q,k,v,block.attn.scaling)
        p=block.attn.proj(a); mid=x+p; n2=block.norm2(mid)
        up=block.mlp.linear_fc1(n2); gelu=block.mlp.act_fn(up)
        down=block.mlp.linear_fc2(gelu); out=mid+down
        return dict(norm1=n1,qkv=qkv,q_rope=q,k_rope=k,attention=a,proj=p,
                    residual1=mid,norm2=n2,up=up,gelu=gelu,down=down,hidden=out)
    def finish_mlp(block, mid, norm=None, up=None, gelu=None, down=None):
        if down is None:
            if gelu is None:
                if up is None:
                    up=block.mlp.linear_fc1(block.norm2(mid) if norm is None else norm)
                gelu=block.mlp.act_fn(up)
            down=block.mlp.linear_fc2(gelu)
        return mid+down
    def finish_attn(block, x, kwargs, norm=None, qkv=None, q=None,k=None,v=None,a=None,p=None):
        if p is None:
            if a is None:
                if q is None:
                    if qkv is None:qkv=block.attn.qkv(block.norm1(x) if norm is None else norm)
                    q,k,v=qkv_parts(qkv,kwargs)
                a=attention(q,k,v,block.attn.scaling)
            p=block.attn.proj(a)
        return finish_mlp(block,x+p)
    def attention_emulation(q,k,v,scale):
        q=q.reshape(-1,16,64).transpose(0,1)
        k=k.reshape(-1,16,64).transpose(0,1)
        v=v.reshape(-1,16,64).transpose(0,1)
        qscale=rounded(q*float(np.float16(scale)))
        score=qscale@k.transpose(-1,-2)
        scores16=rounded(score)
        p32=torch.softmax(score,dim=-1)
        p_score16=torch.softmax(scores16,dim=-1)
        p16=rounded(p_score16)
        def rows(x):return x.transpose(0,1).reshape(-1,1024)
        outputs=dict(q_scale=rows(p32@v),score_round=rows(p_score16@v),probability_round=rows(p16@v))
        # Both tested images fit in the runtime's one 512-key PV tile. A final
        # FP32 division uses the separately FP16-rounded P@1 denominator.
        numerator=rounded(p16@v)
        denominator=rounded(p16.sum(-1,keepdim=True))
        outputs['pv_round_normalize']=rounded(rows(numerator/denominator))
        stats=dict(score_min=score.min().item(),score_max=score.max().item(),
                   score_round=compare(score,scores16),
                   probability_error_from_score=compare(p32,p_score16),
                   probability_round=compare(p_score16,p16),
                   changed_argmax_fraction=(score.argmax(-1)!=scores16.argmax(-1)).float().mean().item(),
                   max_value_abs=v.abs().max().item())
        return outputs,stats
    profiles=json.loads((BASE/'storage-load-only.json').read_text())['profiles']
    report=dict(scope='ViT only; same-input local oracles and ordered FP32 suffix replacement',images={})
    for image in ('red','blue'):
        kwargs=map_tensors(torch.load(ROOT/f'gold-{image}/block-args.pt',weights_only=True),lambda x:x.cuda())
        rows=load(ROOT/f'gold-{image}/input.f32').shape[0]
        gold_features=load(GOLD/f'c7-fp32-{image}/features.f32',2560)
        current_features=load(ROOT/f'{image}-actual/features.f32',2560)
        assert torch.equal(current_features,load(BASE/f'rotary/profile-{image}-2-fp16/features.f32',2560))
        total=(current_features.double()-gold_features.double()).reshape(-1)
        total2=total.square().sum()
        norm_gold=gold_features.double().norm()
        def effect(before,after):
            delta=(after.double()-before.double()).reshape(-1)
            return dict(relative_effect_l2=(delta.norm()/norm_gold).item(),
                        projection_on_total_error=(torch.dot(delta,total)/total2).item(),
                        cut_error=compare(gold_features,after))
        def suffix(hidden,index):
            for j in range(index+1,24):hidden=net.blocks[j](hidden,**kwargs)
            return net.merger(hidden)
        result=dict(operators={},blocks={},prefix_replacement=[],detailed_replacement={},
                    features=compare(gold_features,current_features))
        actual=[]
        for mode in ('actual','gold-input'):
            block_rows=[]; operator_rows=[]
            for i,block in enumerate(net.blocks):
                trace=ROOT/f'{image}-{mode}/trace'
                names=['block_input','norm1','qkv','qkv_unrounded','q_rope','k_rope','attention',
                       'proj','proj_unrounded','residual1','norm2','up','up_unrounded','gelu','down','down_unrounded','hidden']
                t={name:load(trace/f'p0-l{i}-vision_{name}.f32',3072 if name.startswith('qkv') else 4096 if name.startswith('up') or name=='gelu' else 1024) for name in names}
                gold_input=load(ROOT/f'gold-{image}'/('input.f32' if i==0 else f'p0-l{i-1}-vision_hidden.f32'))
                gold_out=load(ROOT/f'gold-{image}/p0-l{i}-vision_hidden.f32')
                if mode=='gold-input':assert torch.equal(t['block_input'],rounded(gold_input))
                elif i:assert torch.equal(t['block_input'],actual[-1]['hidden'])
                replay=block_parts(block,t['block_input'],kwargs)
                assert torch.equal(replay['hidden'],block(t['block_input'],**kwargs)),(image,mode,i,'manual FP32 block')
                mid=t['residual1']; mlp_replay=finish_mlp(block,mid)
                block_rows.append(dict(layer=i,input_error=compare(gold_input,t['block_input']),
                    output_error=compare(gold_out,t['hidden']),
                    inherited_input_through_fp32_block=compare(gold_out,replay['hidden']),
                    local_block_error=compare(replay['hidden'],t['hidden']),
                    attention_group_local=compare(replay['residual1'],mid),
                    mlp_group_local=compare(mlp_replay,t['hidden']),
                    fp32_input_error_gain=((replay['hidden']-gold_out).double().norm()/
                        (t['block_input']-gold_input).double().norm().clamp_min(1e-30)).item()))
                ops={}
                for label,linear,inp in [('qkv',block.attn.qkv,t['norm1']),('proj',block.attn.proj,t['attention']),
                                         ('up',block.mlp.linear_fc1,t['norm2']),('down',block.mlp.linear_fc2,t['gelu'])]:
                    weight=rounded(linear.weight)
                    exact=F.linear(inp,weight,linear.bias)
                    selection=[p for p in profiles if (p['m'],p['k'],p['n'])==(rows,inp.shape[1],linear.weight.shape[0])]
                    assert len(selection)==1
                    kt=selection[0]['k_tile']
                    parts=[rounded(F.linear(inp[:,a:a+kt],weight[:,a:a+kt])) for a in range(0,inp.shape[1],kt)] if kt else [rounded(F.linear(inp,weight))]
                    while len(parts)>1:parts=[rounded(parts[a]+parts[a+1]) if a+1<len(parts) else parts[a] for a in range(0,len(parts),2)]
                    emulation=parts[0]+linear.bias
                    assert torch.equal(rounded(t[label+'_unrounded']),t[label])
                    ops[label]=dict(k_tile=kt,npu_core_vs_fp32=compare(exact,t[label+'_unrounded']),
                        extra_bias_output_round=compare(t[label+'_unrounded'],t[label]),
                        total=compare(exact,t[label]),boundary_emulation_vs_npu=compare(emulation,t[label+'_unrounded']))
                for label,inp,norm in [('norm1',t['block_input'],block.norm1),('norm2',mid,block.norm2)]:
                    exact=norm(inp)
                    ops[label]=dict(total=compare(exact,t[label]),vs_rounded_fp32=compare(rounded(exact),t[label]))
                q,k,v=t['qkv'].reshape(rows,3,1024).unbind(1)
                qr,kr=definitions['apply_rotary_pos_emb_vision'](q.reshape(rows,16,64),k.reshape(rows,16,64),*kwargs['position_embeddings'])
                ops['rope']=dict(q=compare(qr,t['q_rope']),k=compare(kr,t['k_rope']))
                exact=attention(t['q_rope'],t['k_rope'],v,block.attn.scaling)
                approximations,stats=attention_emulation(t['q_rope'],t['k_rope'],v,block.attn.scaling)
                ops['attention']=dict(total=compare(exact,t['attention']),emulation_vs_npu=compare(approximations['pv_round_normalize'],t['attention']),**stats)
                ops['gelu']=dict(total=compare(block.mlp.act_fn(t['up']),t['gelu']))
                ops['residual1']=dict(total=compare(t['block_input']+t['proj'],mid))
                ops['residual2']=dict(total=compare(mid+t['down'],t['hidden']))
                operator_rows.append(dict(layer=i,operators=ops))
                if mode=='actual':actual.append(t)
            result['blocks'][mode]=block_rows; result['operators'][mode]=operator_rows
            print(image,mode,'local block oracles complete',flush=True)
        # First attribute complete blocks: each cut substitutes the recorded
        # NPU prefix, then executes the remaining graph in FP32 on the GPU.
        previous=gold_features; boundary=[gold_features]
        patch=load(ROOT/f'{image}-actual/trace/p0-l-1-vision_patch_position.f32')
        point=suffix(patch,-1)
        result['prefix_replacement'].append(dict(stage='patch_position',**effect(previous,point)))
        previous=point; boundary=[point]
        for i,t in enumerate(actual):
            point=suffix(t['hidden'],i)
            result['prefix_replacement'].append(dict(layer=i,stage='block',**effect(previous,point)))
            previous=point;boundary.append(point)
        trace=ROOT/f'{image}-actual/trace'
        merger_norm=load(trace/'p0-l-1-vision_merger_norm.f32').reshape(-1,4096)
        merger_up=load(trace/'p0-l-1-vision_merger_up.f32',4096)
        merger_gelu=load(trace/'p0-l-1-vision_merger_gelu.f32',4096)
        for stage,point in [('merger_norm',net.merger.linear_fc2(net.merger.act_fn(net.merger.linear_fc1(merger_norm)))),
                            ('merger_up',net.merger.linear_fc2(net.merger.act_fn(merger_up))),
                            ('merger_gelu',net.merger.linear_fc2(merger_gelu)),('merger_down',current_features)]:
            result['prefix_replacement'].append(dict(stage=stage,**effect(previous,point)));previous=point
        assert abs(sum(x['projection_on_total_error'] for x in result['prefix_replacement'])-1)<1e-8
        ranked=sorted([x for x in result['prefix_replacement'] if x['stage']=='block'],key=lambda x:abs(x['projection_on_total_error']),reverse=True)
        chosen=sorted({10,23,*[x['layer'] for x in ranked[:3]]})
        result['detailed_layers']=chosen
        print(image,'suffix block cuts complete; detailed layers',chosen,flush=True)
        for i in chosen:
            t=actual[i];block=net.blocks[i];x=t['block_input'];mid=t['residual1']
            v=t['qkv'].reshape(rows,3,1024)[:,2]
            approximations,stats=attention_emulation(t['q_rope'],t['k_rope'],v,block.attn.scaling)
            cuts=[('norm1',lambda:finish_attn(block,x,kwargs,norm=t['norm1'])),
                  ('qkv_core',lambda:finish_attn(block,x,kwargs,qkv=t['qkv_unrounded'])),
                  ('qkv_bias_output_round',lambda:finish_attn(block,x,kwargs,qkv=t['qkv'])),
                  ('rope',lambda:finish_attn(block,x,kwargs,q=t['q_rope'],k=t['k_rope'],v=v))]
            for name,value in approximations.items():
                cuts.append(('attention_'+name,lambda value=value:finish_attn(block,x,kwargs,a=value)))
            cuts += [('attention_emulation_residual',lambda:finish_attn(block,x,kwargs,a=t['attention'])),
                ('proj_core',lambda:finish_attn(block,x,kwargs,p=t['proj_unrounded'])),
                ('proj_bias_output_round',lambda:finish_attn(block,x,kwargs,p=t['proj'])),
                ('residual1_round',lambda:finish_mlp(block,mid)),
                ('norm2',lambda:finish_mlp(block,mid,norm=t['norm2'])),
                ('up_core',lambda:finish_mlp(block,mid,up=t['up_unrounded'])),
                ('up_bias_output_round',lambda:finish_mlp(block,mid,up=t['up'])),
                ('gelu',lambda:finish_mlp(block,mid,gelu=t['gelu'])),
                ('down_core',lambda:finish_mlp(block,mid,down=t['down_unrounded'])),
                ('down_bias_output_round',lambda:finish_mlp(block,mid,down=t['down'])),
                ('residual2_round',lambda:t['hidden'])]
            detail=[];previous=boundary[i]
            for stage,finish in cuts:
                point=suffix(finish(),i)
                detail.append(dict(stage=stage,**effect(previous,point)));previous=point
            assert torch.equal(previous,boundary[i+1])
            macro=next(z for z in result['prefix_replacement'] if z.get('layer')==i)
            assert abs(sum(z['projection_on_total_error'] for z in detail)-macro['projection_on_total_error'])<1e-8
            result['detailed_replacement'][str(i)]=detail
            print(image,'detailed layer',i,'complete',flush=True)
        report['images'][image]=result
        print(image,'largest block projections',[(x['layer'],round(x['projection_on_total_error'],4),round(100*x['relative_effect_l2'],4)) for x in ranked[:8]],flush=True)
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('analysis complete',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','analyze'))
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net, definitions = model()
    with torch.inference_mode(), torch.autocast('cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
        (prepare(net) if args.mode == 'prepare' else analyze(net, definitions))


if __name__ == '__main__':
    main()
