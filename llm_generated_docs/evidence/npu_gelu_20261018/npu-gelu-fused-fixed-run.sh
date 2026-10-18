#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > logs/npu-gelu-fused-fixed.exit' EXIT
g++ -O3 -DNDEBUG -Wall -Wextra -Werror -fPIC -shared -std=gnu++17 -march=armv8.2-a+fp16 -mtune=cortex-a76 -Inative/rk_npu_matmul/include -Inative/rk_npu_matmul/src tests/board/npu_gelu_fused_bridge.cpp build/fp-mm-tune/librk_npu_matmul.a -Wl,--exclude-libs,ALL -Wl,-Bsymbolic -fopenmp -ldl -lpthread -o logs/npu-gelu-bridge/libnpu_gelu_fused_fixed.so
cargo fmt --manifest-path logs/npu-gelu-fused-fixed-snapshot/Cargo.toml
CARGO_TARGET_DIR=target/float-autotune CARGO_BUILD_JOBS=4 cargo clippy --manifest-path logs/npu-gelu-fused-fixed-snapshot/Cargo.toml --lib --release --locked -- -D warnings
CARGO_TARGET_DIR=target/float-autotune CARGO_BUILD_JOBS=4 cargo build --manifest-path logs/npu-gelu-fused-fixed-snapshot/Cargo.toml --release --locked --example vision_profile_fused_fixed
python3 tests/board/npu_gelu_fused_compare.py \
 --before-bin logs/npu-gelu-fused-original/vision_profile_fused \
 --after-bin target/float-autotune/release/examples/vision_profile_fused_fixed \
 --model /home/firefly/mnt/3090x4/llm/w4a8_quant/transfer/qwen35/models/C7 \
 --cache .cache/mm-tune-20261018 \
 --reference-dir /home/firefly/mnt/3090x4/tmp/vision-fp16-storage-20261018/fp32-gold/reference \
 --red /home/firefly/mnt/3090x4/tmp/qwen35-vision-20261018/red.png \
 --blue /home/firefly/mnt/3090x4/tmp/qwen35-vision-20261018/blue.png \
 --red512 /home/firefly/mnt/3090x4/tmp/floating-mm-util-20261018/red512.png \
 --output logs/npu-gelu-fused-fixed
