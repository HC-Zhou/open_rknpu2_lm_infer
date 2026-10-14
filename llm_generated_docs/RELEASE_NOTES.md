# v0.2.0

- 各命令统一使用 llama.cpp 风格的参数：`-c/--ctx-size`、`-ub/--ubatch-size`、`-t/--threads`、`-n/--n-predict`、`-p/--prompt`、`-f/--file`、`--temp`、`-s/--seed`、`-cnv/--conversation` 和 `-sys/--system-prompt`。
- bench 使用 llama-bench 风格的 `-p/--n-prompt`、`-n/--n-gen`、`-d/--n-depth` 和 `-r/--repetitions`，支持逗号分隔的测试长度。
- `bench -p 0 -n 0 -pg 512,128` 在同一 KV 上连续执行 prefill512 和 decode128，分别报告两阶段及合计。
- `-c 0` 在 bench 中按测试需求自动设置容量，其他命令按模型支持的上下文上限设置容量，最多32768。
- 文档和板端比较脚本同步使用新参数名。原 `--context`、`--chunk`、`--max-tokens`、`--prompt-file`、`--chat-template` 及 bench 的 `--prompt`、`--generate`、`--depth` 已移除。

ARM64 二进制适用于 Debian 11 / glibc 2.31 或更新的系统，依赖 libstdc++6、libgomp1 和 RK3588 NPU DRM 驱动；模型独立提供。附件包含二进制和固定依赖的源码包。
