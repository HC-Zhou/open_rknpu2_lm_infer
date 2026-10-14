# v0.1.0

RK3588 原生 Rust LLM 运行时，支持 Qwen3、Granite 和 MiniCPM5 的 mixed W4A8/W8A8 checkpoint。

- CLI 流式生成、终端聊天和 OpenAI 兼容服务。
- 原生 NPU Linear、FP16 MHA/GQA/MQA、KV 前缀复用。
- WikiText-2 前缀 PPL、prefill 和带历史的 decode 测速。
- OpenMP 等待默认值和持久化调优缓存。
- 固定上游算子库 Git submodule、ARM64 CI 和 tag release。

ARM64 二进制适用于 Debian 11 / glibc 2.31 或更新的系统，依赖 libstdc++6、libgomp1 和 RK3588 NPU DRM 驱动；模型独立提供。附件包含二进制和固定依赖的源码包。
