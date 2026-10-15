# v0.3.0

- 普通推理与评测只查询已有策略，缺失时使用合法默认配置；首次新长度不再触发在线性能搜索或调优缓存写入。
- 新增显式 `tune` 命令，支持选择batches和输出JSON，显示形状、缓存命中、候选进度、最佳耗时与累计时间。
- W4认证profile和缓存身份在权重创建时保留；读取策略不再reload代表权重或重复扫描原码，workspace淘汰独立于小量策略驻留。原有调优缓存格式保持兼容。
- 新增在线search/reload/实际fallback计数。三款模型缓存与回退路径完整logits检查一致；短prefill和已完成稳态数据见SHORT_PREFILL_TUNING_RESULTS。
- README中英文均补充完整性能准备与验证步骤。二进制附件提供固定频率脚本，源码/文档附件只收录版本控制内的文件。
- 本版本应用ubatch范围仍为1–128、默认64；更大ubatch后续独立验证。thinking与聊天模板语义保持原状。

# v0.2.0

- 各命令统一使用 llama.cpp 风格的参数：`-c/--ctx-size`、`-ub/--ubatch-size`、`-t/--threads`、`-n/--n-predict`、`-p/--prompt`、`-f/--file`、`--temp`、`-s/--seed`、`-cnv/--conversation` 和 `-sys/--system-prompt`。
- bench 使用 llama-bench 风格的 `-p/--n-prompt`、`-n/--n-gen`、`-d/--n-depth` 和 `-r/--repetitions`，支持逗号分隔的测试长度。
- `bench -p 0 -n 0 -pg 512,128` 在同一 KV 上连续执行 prefill512 和 decode128，分别报告两阶段及合计。
- `-c 0` 在 bench 中按测试需求自动设置容量，其他命令按模型支持的上下文上限设置容量，最多32768。
- 文档和板端比较脚本同步使用新参数名。原 `--context`、`--chunk`、`--max-tokens`、`--prompt-file`、`--chat-template` 及 bench 的 `--prompt`、`--generate`、`--depth` 已移除。

ARM64 二进制适用于 Debian 11 / glibc 2.31 或更新的系统，依赖 libstdc++6、libgomp1 和 RK3588 NPU DRM 驱动；模型独立提供。附件包含二进制和固定依赖的源码包。
