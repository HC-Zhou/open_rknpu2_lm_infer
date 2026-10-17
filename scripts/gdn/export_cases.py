#!/usr/bin/env python3
"""Package the measured ggml GDN input/output pairs for independent kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="final-t4 benchmark directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress-source", type=Path, help="optional actual-ggml nonuniform-gate recordings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for tokens in [1, 2, 4, 7, 8, 32, 128, 512]:
        source = args.source / f"input-{tokens}.f32"
        raw = np.fromfile(source, dtype="<f4")
        position = 0

        def take(shape):
            nonlocal position
            count = int(np.prod(shape))
            result = raw[position:position + count].reshape(shape).copy()
            position += count
            return result

        q_full = take((tokens, 32, 128))
        k_full = take((tokens, 32, 128))
        q, k = q_full[:, ::2].copy(), k_full[:, ::2].copy()
        assert np.array_equal(q_full, np.repeat(q, 2, axis=1))
        assert np.array_equal(k_full, np.repeat(k, 2, axis=1))
        v = take((tokens, 32, 128))
        g = take((tokens, 32))
        beta = take((tokens, 32))
        initial = take((32, 128, 128))
        assert position == len(raw)
        expected_path = args.source / f"ggml-{tokens}.f32"
        expected = np.fromfile(expected_path, dtype="<f4")
        split = tokens * 32 * 128
        output = expected[:split].reshape(tokens, 32, 128).copy()
        final = expected[split:].reshape(32, 128, 128).copy()
        path = args.output / f"tokens-{tokens:04d}.npz"
        np.savez_compressed(path, q=q, k=k, v=v, g=g, beta=beta,
                            state_initial=initial, output=output, state_final=final)
        cases.append(dict(tokens=tokens, kind="timed-baseline", file=path.name, bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            original_expanded_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            ggml_output_and_state_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest()))
    if args.stress_source:
        for tokens in [1, 7, 128]:
            inputs = args.stress_source / f"stress-input-{tokens}.npz"
            data = dict(np.load(inputs))
            expected_path = args.stress_source / f"expected-{tokens}.f32"
            expected = np.fromfile(expected_path, dtype="<f4")
            split = tokens * 32 * 128
            path = args.output / f"stress-{tokens:04d}.npz"
            np.savez_compressed(path, **data, output=expected[:split].reshape(tokens, 32, 128),
                                state_final=expected[split:].reshape(32, 128, 128))
            cases.append(dict(tokens=tokens, kind="nonuniform-gate-correctness", file=path.name,
                bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                compact_inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
                ggml_output_and_state_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest()))
    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root / "native/cpu_kernels", args.output / "reference_cpu", dirs_exist_ok=True)
    for source, name in [(root / "LICENSE", "LICENSE"),
                         (root / "llm_generated_docs/THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")]:
        shutil.copyfile(source, args.output / "reference_cpu" / name)
    runner = args.output / "reference_runner"
    runner.mkdir(exist_ok=True)
    reference = root / "scripts/gdn/cpu_reference"
    shutil.copyfile(reference / "main.cpp", runner / "main.cpp")
    cmake = (reference / "CMakeLists.txt").read_text().replace(
        "${CMAKE_CURRENT_LIST_DIR}/../../../native/cpu_kernels", "${CMAKE_CURRENT_LIST_DIR}/../reference_cpu")
    (runner / "CMakeLists.txt").write_text(cmake)
    shutil.copyfile(root / "scripts/gdn/check_case.py", args.output / "check_case.py")
    manifest = dict(dtype="little-endian float32", key_heads=16, value_heads=32,
        key_dim=128, value_dim=128, head_mapping="key_head = value_head // 2",
        state_layout="[value_head, value_channel, key_channel]",
        query_key_normalized=True, query_scale_applied=False,
        output_scale="1/sqrt(key_dim)", gate="g is log_decay; apply exp(g) once",
        beta="already sigmoid; do not sigmoid again", value="post-convolution and SiLU",
        scope="recurrence only, before output gated RMSNorm and output projection",
        reference="actual ggml CPU gated_delta_net, de3ebb93bd8828496faefb44ed45059ec7ed3ea5",
        nonzero_initial_state=True, independent_windows=True,
        investigation_max_abs="1e-5 + 1e-4*max_abs(reference)",
        investigation_relative_l2=1e-4, cases=cases,
        standalone_sources={str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [args.output / "reference_cpu", runner] for p in sorted(folder.iterdir()) if p.is_file()},
        checker_sha256=hashlib.sha256((args.output / "check_case.py").read_bytes()).hexdigest())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(directory=str(args.output), cases=len(cases),
                         compressed_bytes=sum(row["bytes"] for row in cases)), indent=2))


if __name__ == "__main__":
    main()
