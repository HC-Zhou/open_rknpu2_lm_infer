# Qwen3.5 C7 文本主干上板验证

日期：2026-10-17，Asia/Shanghai（UTC+08:00）。本报告覆盖真实混合 W4/W8 checkpoint 的普通文本推理与端到端 PPL。

## Checkpoint 与实现

- 服务器目录：`/data007/zt/llm/w4a8_quant/transfer/qwen35/models/C7`。本机共享挂载为 `/home/zt/mnt/3090x4`，备用1挂载为 `/home/firefly/mnt/3090x4`。
- C7 为 fixed offline residual rotation + QQQ，在线变换数为 0。249 个 Linear 中 197 个 W4、52 个 W8；GDN A/B 小投影、3 个 down projection 和独立 lm_head 为 W8，embedding 为 BF16。
- 32 层文本主干：24 个 GDN、8 个 full attention。GDN 使用 CPU FP32 canonical state 与 NEON；Linear 使用 NPU INT32 完整累加；Attention 使用原生 FP16 GQA，带 Q/K norm、partial M-RoPE 和 sigmoid output gate。
- 首轮验证时 C7 索引不含 `mtp.*`；之后 checkpoint 已补入15个 BF16 MTP tensor，清单声明保持 source 原值、target coordinate alignment 待完成。原始 MTP 与 C7 residual rotation 的接口尚未对齐，不能直接拼接。当前仅实现文本 AR，不启用 MTP 和视觉。
- 首次加载发现 W8 `N=32/64` 的分片数少于请求的三个 NPU 核。已将核数限制为可用列 tile 数；独立测试覆盖 `K=2560,N=32/64,M=1/2/3/5/7`，对比确切整数 GEMM 与缩放。

## PPL 协议与结果

固定使用交接项目 `data/wiki_full.npy` 的**前 32 个窗口，保持原顺序**，不按结果挑选。每窗 512 token，独立重置 GDN/卷积/KV；消费位置 0–510，评分 target 位置 257–511，每窗 255 项，共 8,160 项。无 chat template、无额外 BOS/EOS。NLL 采用 FP32 shifted exp、FP64 sum/log。

FP32 参考直接取同一个 C7 已完成的 `results/C7/ppl.token_nll.npy`，与 `ppl.json` 的逐窗 NLL 一致性已校验；没有使用 BF16 原始模型或不同量化产物。参考为 GPU 整数 Linear + FP32 模型，A8 使用原 Torch rounding；原生运行时使用 round-away A8、CPU FP32 GDN 和 FP16 Attention。这些算术差异保留在报告中。

| 配置 | PPL | 相对同窗口 FP32 参考 | 结果 |
|---|---:|---:|---|
| C7 FP32 参考 | 8.601783301 | — | 8,160 targets |
| 备用1，ubatch=32 | 8.594176999 | −0.088427% | PASS |
| 备用1，默认 ubatch=128 | 8.594176999 | −0.088427% | PASS |

两种 ubatch 的 8,160 项 NLL **逐项完全一致**。原生输出头/主干 `tune_calls=0`、`weight_reload_calls=0`。配对结果见 [ppl32-comparison.json](evidence/qwen35_mtp/c7/ppl32-comparison.json) 和 [ppl128-comparison.json](evidence/qwen35_mtp/c7/ppl128-comparison.json)。

调查线保持计划中的绝对相对 PPL 变化 1%。用户确认本轮以端到端 PPL 验证整模型精度，浮点小漂移不阻塞文本主干。结果仅覆盖上述 32 窗，**不等同于**交接项目 580 窗、147,900 项的完整 WikiText 结果（后者参考 PPL 为 10.4760893）。

原严格 logits 检查、不同外部分块和批量/逐行评分的逐值诊断仍有差异，原阈值和失败记录保留；取消后 reset/recompute 的 logits 逐值一致。不能把 PPL 通过解释为 logits、greedy 全轨迹或任意 GDN 回滚已经等价。

默认 Rust 测试 51 passed、0 failed、7 ignored；随后显式执行其中 4 个 NPU 用例，全部通过，合计 55 项通过。余下 3 个既有 ubatch/model sweep 未执行。全 target release Clippy `-D warnings`、format 检查和 2 个 Python runner 测试通过。CLI 使用原生 chat template、greedy、默认 ubatch=128，对“请计算 12+7，只输出结果。”输出 `19`，`finish_reason=stop`，3 个 completion token（含停止 token）；验证了 tokenizer 的 end-of-turn 会正常停止。

## 后续CPU GDN与批量Attention版本

最新版本保留FP32 state，使用独立C++ GDN，并移除prefill的逐query Attention路径。32窗PPL为8.578578429（同权重参考8.601783301，−0.269768%），在1%调查线内。批量Attention前后8160项NLL逐值相等。最新 release 默认测试为54 passed、0 failed、7 ignored，额外显式执行 C7 Attention 几何 oracle 通过；format、全 target release Clippy通过。此结果单列保存于 [qwen35_batch_attn_20261017/](evidence/qwen35_batch_attn_20261017/)，不覆盖前述首轮二进制与PPL记录。完整性能与阶段诊断见 [QWEN35_VS_QWEN3_PERFORMANCE.md](QWEN35_VS_QWEN3_PERFORMANCE.md)。

随后完成公共模块重组；重构后的两模型 logits 与各2窗评分对照保持逐值一致，详见 [模块组织与重构验证](RUNTIME_MODULE_LAYOUT.md)。本报告完整32窗数字仍对应重组前的 `90770d...e8876` 二进制。

## 复现

先在有 NumPy 且能访问交接目录的主机导出固定窗口与配对参考：

```sh
python scripts/qwen35/prepare_ppl.py \
  --quant-project /path/to/w4a8_quant/transfer/qwen35 \
  --windows 32 --output /path/to/c7-ppl32
```

备用1执行（`MODEL` 和 `EVAL` 指向其可访问的目录）：

```sh
python3 scripts/run_board.py --wait-lock --cpu-khz 2256000 --skip-dmc \
  --report /tmp/c7-ppl-clocks.json -- \
  target/release/open_rknpu2_lm_infer --model "$MODEL" \
  --ctx-size 512 --ubatch-size 128 --no-tune ppl \
  --tokens-file "$EVAL/tokens.json" --score-start 257 --report /tmp/c7-ppl.json

python3 scripts/qwen35/compare_ppl.py \
  --native /tmp/c7-ppl.json --reference "$EVAL/reference.json" \
  --tokens "$EVAL/tokens.json" --output /tmp/c7-ppl-comparison.json
```

报告逐窗保存，可在完全相同输入和参数下继续未完成的 PPL。`prepare_ppl.py --windows 580` 可导出完整输入；本轮没有把未跑的完整集计入通过。

## 板端环境与边界

备用1为 `firefly-rk3588-node01`，Linux 5.10.198，32 GB RAM。使用共享提交锁；CPU mask `0xf0`，大核固定 2256000 kHz，NPU 固定 1 GHz。无 DMC devfreq，DDR 未受控。本报告只记录正确性；后续同板重测见 [QWEN35_VS_QWEN3_PERFORMANCE.md](QWEN35_VS_QWEN3_PERFORMANCE.md)。普通运行 `tune_search=false`，不做请求内搜索或重载权重。

板载 UTC 时钟为 2024 年，未修改系统时钟；报告日期来自开发主机。时钟上下限/governor 的恢复以各次 runner 的 `before/restored` 和 `restore_errors` 为准。

PPL 两轮、回归、CLI 及严格 logits 诊断共 5 次 runner 记录均核对完毕：`restore_errors=[]`，受控频率上下限和 governor 与运行前一致；无遗留测试进程。严格 logits 诊断的退出码 1 是保留的数值门槛失败，不是板端恢复失败。

证据目录：[evidence/qwen35_mtp/c7/](evidence/qwen35_mtp/c7/)，入口为 [validation.json](evidence/qwen35_mtp/c7/validation.json)。权重、完整 logits 和中间 activation trace 保留在共享临时目录，不复制进源码仓库。小型报告包含逐 token NLL、输入/参考哈希、测试日志、时钟记录、checkpoint 清单和 8 个分片的 SHA256。源码与板端二进制身份见 [source-manifest.json](evidence/qwen35_mtp/c7/source-manifest.json)。
