#!/usr/bin/env python3
"""Copy the Rust crate and override FP16 FFN GELU only in that experiment copy."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root/'src', out/'src')
    for name in ('Cargo.lock', 'build.rs', 'rust-toolchain.toml'):
        shutil.copy2(root/name, out/name)
    manifest = (root/'Cargo.toml').read_text()
    assert manifest.count('name = "vision_profile"') == 1
    manifest = manifest.replace('name = "vision_profile"', 'name = "vision_profile_npu"')
    (out/'Cargo.toml').write_text(manifest+'\n[workspace]\n')
    # Relative links work under both host and RK3588 shared-folder mappings.
    import os
    for name in ('native', 'tests', 'llm_generated_docs'):
        (out/name).symlink_to(os.path.relpath(root/name, out), target_is_directory=True)
    path = out/'src/ops/vision_half.rs'
    source = path.read_text()
    source = source.replace('mod convert;', 'mod convert;\nmod npu_probe;', 1)
    signature = 'pub fn gelu(pool: &ThreadPool, values: &mut [u16], approximate: bool) {'
    assert source.count(signature) == 1
    source = source.replace(signature, signature+'\n    if approximate { npu_probe::apply(values); return; }', 1)
    path.write_text(source)
    shutil.copy2(root/'tests/board/npu_gelu_snapshot.rs', out/'src/ops/vision_half/npu_probe.rs')
    print('Created isolated NPU GELU crate:', out)


if __name__ == '__main__':
    main()
