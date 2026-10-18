#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > logs/gelu-validation.exit' EXIT
export CARGO_TARGET_DIR=target/float-autotune CARGO_BUILD_JOBS=4
cargo clippy --release --locked --all-targets -- -D warnings
cargo test --release --locked -- --test-threads=4
cargo build --release --locked --bin open_rknpu2_lm_infer --example vision_profile
ctest --test-dir build/gelu-production --output-on-failure
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python3 scripts/run_board.py --wait-lock --report logs/gelu-final-native-clocks.json -- taskset -c 4-7 build/gelu-production/test_mm_gelu --board
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python3 scripts/run_board.py --wait-lock --report logs/gelu-rust-clocks.json -- taskset -c 4-7 cargo test --release --locked --lib fused_gelu_collects_tails_and_reuses_resident_weights -- --ignored --nocapture
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python3 scripts/run_board.py --wait-lock --report logs/gelu-formal-vit-clocks.json -- taskset -c 4-7 python3 tests/board/vision_fused_regression.py \
 --before-bin target/float-autotune/release/examples/vision_profile_fused_fixed \
 --after-bin target/float-autotune/release/examples/vision_profile \
 --model /home/firefly/mnt/3090x4/llm/w4a8_quant/transfer/qwen35/models/C7 \
 --cache .cache/mm-tune-20261018 \
 --reference-dir /home/firefly/mnt/3090x4/tmp/vision-fp16-storage-20261018/fp32-gold/reference \
 --image /home/firefly/mnt/3090x4/tmp/qwen35-vision-20261018/red.png \
 --image /home/firefly/mnt/3090x4/tmp/qwen35-vision-20261018/blue.png \
 --image /home/firefly/mnt/3090x4/tmp/floating-mm-util-20261018/red512.png \
 --output logs/gelu-formal-vit
