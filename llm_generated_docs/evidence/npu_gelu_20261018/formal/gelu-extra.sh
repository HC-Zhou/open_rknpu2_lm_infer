#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > logs/gelu-extra.exit' EXIT
model=/home/firefly/mnt/3090x4/llm/w4a8_quant/transfer/qwen35/models/C7
bin=target/float-autotune/release/examples/vision_profile
"$bin" --model "$model" --image /home/firefly/mnt/3090x4/tmp/qwen35-vision-20261018/red.png --vision-dtype fp32 --tune-cache .cache/mm-tune-20261018 --output logs/gelu-extra/fp32 --repeats 2
"$bin" --model "$model" --image logs/gelu-extra/red_odd.png --tune-cache .cache/mm-tune-20261018 --output logs/gelu-extra/odd --repeats 2
