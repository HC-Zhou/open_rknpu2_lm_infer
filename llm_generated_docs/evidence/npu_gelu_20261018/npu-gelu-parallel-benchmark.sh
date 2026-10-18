#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > logs/npu-gelu-parallel-benchmark.exit' EXIT
export CARGO_TARGET_DIR=target/float-autotune
export CARGO_BUILD_JOBS=4
cargo build --release --locked --example npu_gelu_bench
bin="$CARGO_TARGET_DIR/release/examples/npu_gelu_bench"
bridge=logs/npu-gelu-bridge/libnpu_gelu_parallel.so
mkdir -p logs/npu-gelu-bench-parallel
"$bin" --bridge "$bridge" --finite-domain --mask 7 --output logs/npu-gelu-bench-parallel/finite.json > logs/npu-gelu-bench-parallel/finite.log
for count in 1048576 1572864 4194304; do
    "$bin" --bridge "$bridge" --count "$count" --mask 7 --output "logs/npu-gelu-bench-parallel/synthetic-$count.json" > "logs/npu-gelu-bench-parallel/synthetic-$count.log"
done
traces=/home/firefly/mnt/3090x4/tmp/vision-fp16-storage-20261018/precision-stages
for image in red blue; do
    for layer in 0 23; do
        "$bin" --bridge "$bridge" --input-f32 "$traces/$image-actual/trace/p0-l$layer-vision_up.f32" \
            --mask 7 --output "logs/npu-gelu-bench-parallel/$image-layer$layer.json" > "logs/npu-gelu-bench-parallel/$image-layer$layer.log"
    done
done
