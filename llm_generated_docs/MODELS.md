# Model checkpoints

支持稠密 Qwen3、Llama 和 Granite 的 compressed-tensors checkpoint。普通 BF16 或 GGUF 文件需要先转换为下述量化格式。

模型目录包含 `config.json`、`tokenizer.json`、`tokenizer_config.json`，以及单文件 `model.safetensors` 或 `model.safetensors.index.json` 指向的多个 shard。聊天模板优先读取 `chat_template.jinja`，其次读取 tokenizer_config 中的 Jinja 字符串。

| 项目 | 支持范围 |
|---|---|
| `model_type` | `qwen3`、`llama`、`granite` |
| 结构 | 稠密、SiLU、标准 RoPE，无 Linear bias |
| 原生 Attention | MHA/GQA/MQA；D 为32的正整数倍，Hq/Hkv 为正整数，Hkv≤32，context≤32768；算子库内部处理 query 分段 |
| 量化声明 | compressed-tensors，status=compressed |
| 权重 | 整数4/8 bit，静态、对称、channel scale |
| 激活 | 整数8 bit，动态、对称、token scale |
| W4 存储 | pack-quantized I32 weight_packed + I32/I64 weight_shape |
| W8 存储 | int-quantized I8 weight，或 pack-quantized |
| scale | 正且有限的 `[out_features,1]`，F32/F16/BF16 |
| zero point | 缺省或全零整数 |
| embedding / norms | BF16/F16/F32 |
| lm_head | BF16/F16/F32，可 tied；或 W4/W8，须 untied |

W4/W8 使用 compressed-tensors 的低位优先、unsigned-offset packing。加载后转换为算子库的 signed `B[K,N]` codes，保留 checkpoint 的 per-channel scales。

混合位宽按各模块声明加载。仅相邻同位宽的 QKV、gate/up 合并；混合位宽分别执行并在 Rust 中拼回输出。

CPU 执行 RMSNorm、RoPE、残差及部分 SwiGLU。RoPE 明确采用 FP32 乘法和固定 FMA 顺序，避免编译器乘加融合策略改变激活量化输入。Linear 使用 full-K `maxabs/127`、round-away、clamp `[-127,127]` 的动态 A8 ABI。原生 GQA 使用 FP16 KV 和 stable softmax。

| 模型 | 层数 | Hidden / FFN | Hq / Hkv / D | 架构差异 |
|---|---:|---|---|---|
| Qwen3-4B | 36 | 2560 / 9728 | 32 / 8 / 128 | Q/K RMSNorm，Attention scale=1/√128 |
| Granite-4.2-3B-W4A8-Mixed-QQQ-SQ | 40 | 2560 / 8192 | 40 / 8 / 64 | 无 Q/K Norm，Attention scale=1/64；保留 Granite 四项缩放 |
| MiniCPM5-2B-W4A8-Mixed-QQQ-Rot | 42 | 2048 / 6144 | 16 / 2 / 128 | Llama 架构，无 Q/K Norm，Attention scale=1/√128 |

Granite 的 SmoothQuant 与 MiniCPM5 的残差/value-head 旋转均已离线融合到 checkpoint，不增加在线算子。Qwen3 的 `C7-Rotate-QQQ` 同样使用离线融合的 signed Hadamard 20×128 残差旋转和128维 value-head 旋转；保留 Q/K Norm，36层共246个 W4 Linear 与7个 W8 Linear（含 lm_head）。聊天模板和 EOS 从各自 checkpoint 加载；MiniCPM5 的两个 EOS 均生效。`--stats` 的各 domain 中包含实际 Attention 形状、scale 和 core mask。

不同量化与 Attention 数值 ABI 的 logits 应分别验证；同输入数值比较使用 `eval`。设置 `OPEN_RKNPU2_TRACE_DIR` 可导出逐层 FP32 中间张量，文件名为 `pPOSITION-lLAYER-STAGE.f32`；该诊断模式将 gate/up 和 SwiGLU 分开执行。
