#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > logs/npu-gelu-fused-vit.exit' EXIT
python3 tests/board/npu_gelu_fused_vit.py \
 --cpu-bin logs/vision-committed-baseline-20261018/vision_profile \
 --separate-bin target/float-autotune/release/examples/vision_profile_npu \
 --fused-bin target/float-autotune/release/examples/vision_profile_fused \
 --model /home/firefly/mnt/3090x4/llm/w4a8_quant/transfer/qwen35/models/C7 \
 --cache .cache/mm-tune-20261018 \
 --reference-dir /home/firefly/mnt/3090x4/tmp/vision-fp16-storage-20261018/fp32-gold/reference \
 --red /home/firefly/mnt/3090x4/tmp/qwen35-vision-20261018/red.png \
 --blue /home/firefly/mnt/3090x4/tmp/qwen35-vision-20261018/blue.png \
 --red512 /home/firefly/mnt/3090x4/tmp/floating-mm-util-20261018/red512.png \
 --output logs/npu-gelu-fused-vit
