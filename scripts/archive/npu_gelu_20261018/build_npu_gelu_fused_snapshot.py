#!/usr/bin/env python3
"""Create an isolated ViT crate with fused GEMM/bias/GELU only for FFN UP."""
import argparse
import os
from pathlib import Path
import shutil


def replace(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, (path, old)
    path.write_text(text.replace(old, new, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', default='vision_profile_fused')
    parser.add_argument('--bridge', default='logs/npu-gelu-bridge/libnpu_gelu_fused.so')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root/'src', out/'src')
    for name in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'rust-toolchain.toml'):
        shutil.copy2(root/name, out/name)
    for name in ('native', 'tests', 'llm_generated_docs'):
        (out/name).symlink_to(os.path.relpath(root/name, out), target_is_directory=True)
    replace(out/'Cargo.toml', 'name = "vision_profile"', f'name = "{args.name}"')
    with (out/'Cargo.toml').open('a') as f:
        f.write('\n[workspace]\n')
    shutil.copy2(root/'tests/board/npu_gelu_fused_snapshot.rs', out/'src/npu/fused_probe.rs')
    if args.bridge != 'logs/npu-gelu-bridge/libnpu_gelu_fused.so':
        replace(out/'src/npu/fused_probe.rs', 'logs/npu-gelu-bridge/libnpu_gelu_fused.so', args.bridge)
    replace(out/'src/npu.rs', 'mod floating;', 'mod floating;\nmod fused_probe;')
    p = out/'src/npu/floating.rs'
    replace(p, 'pub struct FloatLinear {', 'pub struct FloatLinear {\n    fused: Option<super::fused_probe::Fused>,')
    replace(p, '        Ok(FloatLinear {', '''        let fused = if self.host_type == RK_NPU_MM_HOST_STORAGE && k == 1024 && n == 4096 {
            Some(super::fused_probe::Fused::new(n, k, &half, bias)?)
        } else { None };
        Ok(FloatLinear {
            fused,''')
    replace(p, 'impl FloatLinear {', '''impl FloatLinear {
    pub fn run_half_gelu(&self, pool: &rayon::ThreadPool, input: &[u16], output: &mut [u16], rows: usize) -> Result<()> {
        if let Some(fused) = &self.fused { return fused.apply(pool, input, output, rows); }
        self.run_half(input, output, rows)?;
        crate::ops::vision_half::gelu(pool, output, true);
        Ok(())
    }''')
    p = out/'src/model/qwen35/vision/storage.rs'
    signature = '    fn linear(layer: &FloatLinear, input: &[Self], output: &mut [Self], rows: usize) -> Result<()>;'
    replace(p, signature, signature+'''
    fn linear_gelu(layer: &FloatLinear, pool: &ThreadPool, input: &[Self], output: &mut [Self], rows: usize) -> Result<()> {
        Self::linear(layer, input, output, rows)?;
        Self::gelu(pool, output, true);
        Ok(())
    }''')
    replace(p, 'impl Element for u16 {', '''impl Element for u16 {
    fn linear_gelu(layer: &FloatLinear, pool: &ThreadPool, input: &[Self], output: &mut [Self], rows: usize) -> Result<()> {
        layer.run_half_gelu(pool, input, output, rows)
    }''')
    p = out/'src/model/qwen35/vision/encoder.rs'
    replace(p, '''            T::linear(&block.fc1, &normalized, &mut ff, rows)?;
            stages.mark("mlp_up_linear");
            T::gelu(pool, &mut ff, true);
            stages.mark("gelu");''', '''            T::linear_gelu(&block.fc1, pool, &normalized, &mut ff, rows)?;
            stages.mark("mlp_up_gelu");''')
    print('Created fused GELU snapshot:', out)


if __name__ == '__main__':
    main()
