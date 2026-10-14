# RK3588 validation

验证平台为 RK3588 Linux，Rust 1.95.0、GCC 10。模型为 `Qwen3-4B-W4A8-Mixed-LMHeadW8`，36 层，vocab151936，原生 FP16 GQA，W8 NPU lm_head。

## 数值与功能

| 检查 | 结果 |
|---|---|
| Release 自动测试 | 30 passed，0 failed；3 项板端测试单独运行 |
| Clippy | `--all-targets --release --locked -- -D warnings` 通过 |
| OpenMP 启动 | libgomp 解析确认默认 `PASSIVE` / `0` 生效，部分或全部显式配置均保留 |
| NPU 算子测试 | W4/W8 padding、fused SwiGLU、MHA/GQA/MQA、KV append/grow/truncate/重写/reset 通过 |
| 全模型 logits | 22-token prompt 加3步 teacher-forced continuation，607744 个 FP32 值与 C++ native 实现完全相同 |
| 逐层数值 | 1302 个张量、50893440 个元素与 C++ native 实现逐值相同 |
| OpenAI SDK | 14 项集成检查通过 |
| TUI | 板端 PTY 验证中文输入、流式生成、reset、退出和终端恢复 |
| 独立源码构建 | 板端全新编译、模型 logits 与官方 SDK 验收通过 |

OpenAI SDK 使用官方 Python 客户端，覆盖聊天、SSE usage、prefix cache、文本补全、function tools、流式工具调用、工具历史、reasoning、JSON mode 的拒绝、context 超限、断流恢复、有界队列回压和队列恢复。

请求参数检查覆盖 developer 消息、文本 parts、负 seed、top_p=0 和 max_completion_tokens。输入与数值证据见 [logits-inputs.json](evidence/logits-inputs.json)、[native-logits.json](evidence/native-logits.json) 和 [api-sdk.json](evidence/api-sdk.json)。

原生 KV truncate 测试同时检查保留前缀、替换后缀、清空后重新 append，以及容量和 payload 保持不变。依赖检查确认可执行程序仅链接系统 C/C++、OpenMP 等库。

## 通用算子接口

默认构建使用 [rknpu2_matmul_open 固定提交](https://github.com/happyme531/rknpu2_matmul_open/commit/1ef44786a0ddc726c1225f06b7fdf3fe0978d7c2) 的 Git submodule。头维度按32对齐，query heads 是 KV heads 的正整数倍，1..32 KV heads。模型和 Rust 后端按这些公共参数加载；1..128 逻辑 query 行的分段、紧凑输出布局和实际用核选择由算子库处理。

RK3588 原生测试覆盖8种几何：MHA D32；GQA R3/D96、R5/D160、R4/D256、R4/D128、R5/D64、R8/D128；MQA R16/D192。每种覆盖1/3/31/32/64/128行、causal/no-mask/Boolean mask 和 KV 增长、truncate、替换后缀、reset。与 CPU reference 的最大 relative L2 为0.000415515。相同库提交下，Qwen3 的607744个回归 logits 逐值一致。

W8 共享权重分块由 `rk_npu_matmul_k_partition_query` 查询库的在线调优策略，覆盖大 K 波和尾波；KV 内存预算也通过库的大小查询计算。默认构建的 Clippy、30项自动测试、3项 NPU 测试和 C header 编译均通过。数值与依赖证据见 [generic-operators.json](evidence/generic-operators.json)。

## Granite 与 MiniCPM5

两种 checkpoint 的标准 RoPE、稠密 MLP 和 W4/W8 格式由同一 Rust 前向实现加载。Granite 使用 D64/R5 GQA、Attention scale=1/64；MiniCPM5 使用 D128/R8 GQA。两个模型均省略 Q/K Norm；MiniCPM5 的离线旋转不增加在线算子。

按 `.paper` 的 WikiText-2 前缀口径验证：32 个连续、独立的512-token窗口，context512、chunk64，每窗重置 KV；对零起始 target 位置257–511评分，共8160个目标。读取原始 `wiki.test.raw`，不添加 BOS/EOS 或 chat 模板。两个 tokenizer 的前16384个原生 ID 均与评测输入逐项相同。

| 模型 | 同权重 FP32 Attention / INT32 Linear reference | Rust / NPU FP16 GQA | PPL 变化 |
|---|---:|---:|---:|
| Granite-4.2-3B-W4A8-Mixed-QQQ-SQ | 16.267613 | 16.051859 | −1.33% |
| MiniCPM5-2B-W4A8-Mixed-QQQ-Rot | 13.772371 | 13.807292 | +0.25% |

NLL 使用 FP32 shifted exp、FP64词表求和和 log，与 `.paper` 的 scorer 算术一致。表中 PPL 由保存的逐 token NLL 独立重算；reference 为相同 checkpoint 的既有完整评测前32窗。逐窗配对值见 [model-adaptation-ppl.json](evidence/model-adaptation-ppl.json)。

短提示、数学提示和跨 chunk 提示的原生 chat token IDs 均与 Transformers 相同。完整模型能执行 prefill、teacher-forced decode 和 greedy 生成；Qwen3 回归的607744个 logits 与 C++ native reference 逐值相同。

两个新模型的 HTTP 普通聊天、原生预开启 thinking 的 SSE reasoning 拆分和 stream usage 均通过板端检查，见 [model-adaptation-api.json](evidence/model-adaptation-api.json)。

在板上复现前缀评分：

```sh
target/release/open_rknpu2_lm_infer --model /path/to/model --context 512 --chunk 64 \
  --no-tune ppl --tokens-file wiki32.json --report ppl32.json
```

`wiki32.json` 为各模型 tokenizer 对原始语料得到的前32个512-token窗口，二维数组，不添加特殊 token。每次独立调用由 `scripts/run_board.py` 持有共享板锁并固定、恢复频率。

## 三款模型的前缀 PPL

Qwen3、Granite 和 MiniCPM5 的32窗 PPL 使用上述相同评分口径，每个模型8160个目标，原生逐 token NLL 均独立重算。Qwen3 的前16384个原生 token IDs 与已保存的参考输入完全相同。

| 模型 | 同权重 FP32 Attention 参考 PPL | RK3588 PPL | PPL 变化 |
|---|---:|---:|---:|
| Qwen3-4B / C7-Rotate-QQQ | 12.5165 | 12.5317 | +0.12% |
| Granite-4.2-3B | 16.2676 | 16.0519 | -1.33% |
| MiniCPM5-2B | 13.7724 | 13.8073 | +0.25% |

参考为相同量化 checkpoint 的 FP32 Attention / INT32 Linear 实现。Qwen3 的 PPL checkpoint 为 `selection/models/C7-Rotate-QQQ`，残差和 value-head 旋转均已离线融合，无在线变换；原生 A8 执行使用 round-away。逐窗配对数据见 [model-ppl.json](evidence/model-ppl.json)。

固定频率的六项吞吐结果见 [PERFORMANCE.md](PERFORMANCE.md)。Granite/MiniCPM5 各64-token prefill 加两步 teacher-forced decode 的完整 logits，在调优与默认策略之间逐值相同，见 [tuned-model-logits.json](evidence/tuned-model-logits.json)。

原厂 RKLLM 1.3.1 的32窗 NLL 各8160个 target ID 已与三种原生输入逐项核对，并重新汇总其 PPL。相同频率下的性能与 PPL 发布配置对照见 [SDK_COMPARISON.md](SDK_COMPARISON.md)，原始样本和逐目标核验结果见 [sdk-comparison.json](evidence/sdk-comparison.json)。

## 复现

在板上执行：

```sh
cargo test --release --locked
cargo clippy --all-targets --release --locked -- -D warnings
cargo test --release --locked --lib npu::tests -- --ignored --test-threads=1
```

使用相同模型与 token IDs 分别导出 C++ reference 和本程序的 logits：

```sh
target/release/open_rknpu2_lm_infer eval --model /path/to/model \
  --context 4096 --no-tune --tokens-file prompt.json \
  --continuation-file continuation.json --dump-logits rust.f32
python3 scripts/compare_logits.py native.f32 rust.f32 \
  --vocab 151936 --max-nrmse 0 --report logits-comparison.json
```

API 集成检查需要先启动服务，并在客户端环境安装 `openai`：

```sh
python3 scripts/check_api.py --base-url http://127.0.0.1:8080/v1 \
  --api-key local --report api-sdk.json
```

## 限制

Qwen3 的逐值 reference 是同一量化与原生 Attention ABI 的 C++ 实现。Granite/MiniCPM5 的 FP16 Attention 会改变完整 logits；32窗 PPL 检查对应其实际质量影响，不表示与 FP32 Attention 逐值一致或完整语料、下游任务质量。

真实大模型验收覆盖上述三个 mixed checkpoint。其他符合格式的 checkpoint 仍需分别进行板端数值与资源验证。Granite/MiniCPM5 的 XML 工具调用格式尚未转换为 OpenAI `tool_calls`，详见 [API.md](API.md)。
