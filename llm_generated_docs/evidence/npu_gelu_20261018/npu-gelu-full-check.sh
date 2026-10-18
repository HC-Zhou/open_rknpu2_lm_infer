#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; printf "%s\n" "$status" > logs/npu-gelu-full-check.exit' EXIT
bash logs/npu-gelu-parallel-benchmark.sh
bash logs/npu-gelu-vit-profile.sh
