"""Snapshot and instrument two MM units; relink with the existing native objects."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--native-build', type=Path, default=Path('build/fp-mm-tune'))
p.add_argument('--source-snapshot', type=Path,
               help='directory containing the archived revision6 *.cpp.original files')
p.add_argument('--ablation', choices=('none', 'narrow', 'pack', 'both'), default='none')
args = p.parse_args()
root = Path('native/rk_npu_matmul').resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
source = out/'source'
source.mkdir()
for name in ('fp16_stage_trace.h', 'fp16_stage_trace.cpp'):
    shutil.copy2(Path(__file__).resolve().parent/name, source/name)

def replace(s, old, new):
    assert s.count(old) == 1, (old, s.count(old))
    return s.replace(old, new)

hashes = {}
for name in ('rk_npu_mm.cpp', 'rk_npu_mm_tune.cpp'):
    original = args.source_snapshot/(name+'.original') if args.source_snapshot else root/'src'/name
    s = original.read_text()
    hashes[name] = hashlib.sha256(s.encode()).hexdigest()
    (source/(name+'.original')).write_text(s)
    if name.endswith('_tune.cpp'):
        s = replace(s, 'bool narrow(const float* src, uint16_t* dst, size_t count, rk_npu_mm_dtype type) {\n    size_t j = 0;',
                    'bool narrow(const float* src, uint16_t* dst, size_t count, rk_npu_mm_dtype type) {\n    size_t j = 0;\n    { fp16_trace::Scope measured(fp16_trace::narrow);')
        s = replace(s, '    for (j = 0; j < count; ++j) if (!finite(dst[j], type)) return false;',
                    '    }\n    fp16_trace::Scope measured(fp16_trace::narrow_validate);\n    for (j = 0; j < count; ++j) if (!finite(dst[j], type)) return false;')
        s = replace(s, '             const float* bias, rk_npu_mm_dtype type) {',
                    '             const float* bias, rk_npu_mm_dtype type) {\n    { fp16_trace::Scope measured(fp16_trace::collect);')
        s = replace(s, '    for (size_t j = 0; j < count; ++j) if (!std::isfinite(dst[j])) return false;',
                    '    }\n    fp16_trace::Scope measured(fp16_trace::collect_validate);\n    for (size_t j = 0; j < count; ++j) if (!std::isfinite(dst[j])) return false;')
        if args.ablation in ('narrow', 'both'):
            signature = 'bool narrow(const float* src, uint16_t* dst, size_t count, rk_npu_mm_dtype type) {'
            fast = '''
    if (type == RK_NPU_MM_F16) {
        fp16_trace::Scope measured(fp16_trace::narrow);
        int invalid=0;
#pragma omp parallel for schedule(static) reduction(|:invalid) if(count >= (1u<<18))
        for (int64_t block=0;block<int64_t((count+7)/8);++block) {
            const size_t j=size_t(block)*8;
            if (j+8<=count) {
                const auto lo=vcvt_f16_f32(vld1q_f32(src+j));
                const auto hi=vcvt_f16_f32(vld1q_f32(src+j+4));
                const auto raw=vreinterpretq_u16_f16(vcombine_f16(lo,hi));
                vst1q_u16(dst+j,raw);
                const auto mask=vdupq_n_u16(0x7c00);
                invalid|=vmaxvq_u16(vceqq_u16(vandq_u16(raw,mask),mask))!=0;
            } else for(size_t q=j;q<count;++q) {
                dst[q]=bits::float_to_half(src[q]); invalid|=!finite(dst[q],type);
            }
        }
        return !invalid;
    }
'''
            s = replace(s, signature, signature+fast)
        for stage, expr in (
            ('pack', 'rk_npu_mm_pack_a(w->plan.get(),a,w->a.size()*2,&w->device_a)'),
            ('sync_in', 'rk_npu_mem_sync(w->ctx,&w->device_a,RK_NPU_SYNC_TO_DEVICE)'),
            ('gemm', 'rk_npu_mm_run_device_packed_b(w->work.get(),&w->device_a,weight,&w->partials)'),
            ('reduce', 'w->reduce->run()'),
            ('sync_out', 'rk_npu_mem_sync(w->ctx,&w->reduce->output(),RK_NPU_SYNC_FROM_DEVICE)'),
            ('unpack', 'rk_npu_mm_unpack_c(w->collect_plan.get(),&w->reduce->output(),w->c.data(),w->c.size()*2)')):
            s = replace(s, expr, f'FP_STAGE({stage}, {expr})')
    else:
        if args.ablation in ('pack', 'both'):
            anchor = '    // Preserve the existing vectorized batch packers for compact unsplit IO.'
            fast = '''    // Diagnostic ablation: contiguous split-A row copies, same device layout.
    if (!weight && !d.trans_a && i.split_count>1 && i.options.a_layout==RK_NPU_MM_LAYOUT_NORMAL) {
        const int ak=align_up(i.part_k,64/sizeof(T));
        const uint64_t per=p->one.input_bytes/sizeof(T);
#pragma omp parallel for schedule(static) if(total >= (1u<<17))
        for(int64_t job=0;job<int64_t(p->items)*d.M;++job) {
            const int item=job/d.M, row=job%d.M;
            const int batch=item/i.split_count, k0=(item%i.split_count)*i.part_k;
            const int valid=std::min(i.part_k,d.K-k0);
            auto* target=dst+uint64_t(item)*per+uint64_t(row)*ak;
            const auto* input=src+uint64_t(batch)*d.batch_stride_a+uint64_t(row)*d.lda+k0;
            std::memcpy(target,input,size_t(valid)*sizeof(T));
            if(valid<ak)std::memset(target+valid,0,size_t(ak-valid)*sizeof(T));
        }
        return RK_NPU_OK;
    }
'''
            s = replace(s, anchor, fast+anchor)
        for stage, expr in (
            ('pack', 'rk_npu_mm_pack_a(&w->plan, a, ab, &w->a)'),
            ('sync_in', 'rk_npu_mem_sync(w->domain->ctx, &w->a, RK_NPU_SYNC_TO_DEVICE)'),
            ('gemm', 'submit(*w, &w->a, weight, &w->c)'),
            ('sync_out', 'rk_npu_mem_sync(w->domain->ctx, &w->c, RK_NPU_SYNC_FROM_DEVICE)'),
            ('unpack', 'rk_npu_mm_unpack_c(&w->plan, &w->c, c, cb)')):
            expected = 2 if stage in ('sync_in', 'sync_out') else 1
            assert s.count(expr) == expected, (expr, s.count(expr))
            s = s.replace(expr, f'FP_STAGE({stage}, {expr})')
        s = replace(s, '    int invalid = 0;\n#pragma omp parallel for',
                    '    int invalid = 0;\n    { fp16_trace::Scope measured(fp16_trace::fused_input);\n#pragma omp parallel for')
        s = replace(s, '    if (invalid) return RK_NPU_ERR_PARAM;\n    int rc =',
                    '    }\n    if (invalid) return RK_NPU_ERR_PARAM;\n    int rc =')
        s = replace(s, '    invalid = 0;\n#pragma omp parallel for',
                    '    { fp16_trace::Scope measured(fp16_trace::fused_output);\n    invalid = 0;\n#pragma omp parallel for')
        s = replace(s, '    return invalid ? RK_NPU_ERR_PARAM : RK_NPU_OK;',
                    '    }\n    return invalid ? RK_NPU_ERR_PARAM : RK_NPU_OK;')
        for stage, expr in (
            ('gemm', 'submit(*w, &w->a, const_cast<rk_npu_mem*>(&b->mem), &w->c)'),):
            s = replace(s, expr, f'FP_STAGE({stage}, {expr})')
    (source/name).write_text('#include "fp16_stage_trace.h"\n'+s)

objects = out/'objects'
objects.mkdir()
replaced = {'rk_npu_mm.cpp.o', 'rk_npu_mm_tune.cpp.o'}
base_objects = args.native_build/'CMakeFiles/rk_npu_matmul.dir/src'
for path in sorted(base_objects.glob('*.o')):
    if path.name not in replaced:
        shutil.copy2(path, objects/path.name)
flags = ['-O2', '-fPIC', '-march=armv8.2-a+fp16', '-fopenmp', '-std=gnu++17',
         '-DRK_NPU_ENABLE_TRACE=0', '-Drk_npu_matmul_EXPORTS',
         '-I'+str(root/'include'), '-I'+str(root/'src'), '-I'+str(source)]
for path in sorted(source.glob('*.cpp')):
    subprocess.run(['g++', *flags, '-c', str(path), '-o', str(objects/(path.name+'.o'))], check=True)
subprocess.run(['g++', '-shared', '-Wl,-soname,librk_npu_matmul.so', '-o', str(out/'librk_npu_matmul.so'),
                *map(str, sorted(objects.glob('*.o'))), '-fopenmp', '-ldl'], check=True)
(out/'source-hashes.json').write_text(json.dumps({'sources': hashes, 'ablation': args.ablation}, indent=2)+'\n')
print('diagnostic library:', out/'librk_npu_matmul.so')
