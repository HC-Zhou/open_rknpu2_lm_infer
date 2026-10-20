# Qwen3.5 DFlash

DFlash 为 Qwen3.5 提供并行草稿生成，复用目标模型的 embedding 和 lm_head。当前已验证的组合为未旋转的 Qwen3.5-4B C4 与 `z-lab/Qwen3.5-4B-DFlash`。草稿保留发布权重，以 FP16 GEMM、FP32 CPU 归一化和 RoPE 执行；原生算子库无需修改。旋转 C7 需要额外的草稿坐标适配，当前不支持。

## 使用

```sh
BIN=./bin/open_rknpu2_lm_infer
MODEL=models/Qwen3.5-4B-C4
DFLASH=models/Qwen3.5-4B-DFlash

"$BIN" -m "$MODEL" --spec-type draft-dflash --spec-draft-model "$DFLASH" \
  --spec-draft-n-max 7 run -cnv --thinking --temp 0 -n 128 --stats \
  -p 'Explain the difference between TCP and UDP, with examples.'

"$BIN" -m "$MODEL" --spec-type draft-dflash -md "$DFLASH" \
  --spec-draft-n-max 3 chat --thinking

"$BIN" -m "$MODEL" --spec-type draft-dflash --model-draft "$DFLASH" \
  --spec-draft-n-max 7 serve --port 8080
```

`--spec-draft-model`、`--model-draft`、`-md` 等价。`--spec-draft-n-max` 计算新提案数，不包含待验证的 anchor；3/7 分别对应 block 4/8。默认提案上限为 3，设为 0 可禁用推测解码。参数作用于整个 `run`、`chat`、`serve` 实例，API 的请求格式保持不变。

`--spec-draft-p-min` 与 MTP 使用相同的 top-10 置信度规则：低于阈值时丢弃当前提案并截断后缀。DFlash 的并行草稿已经计算完成，这个阈值缩短的是提交给目标模型的验证块。默认值 0 不截断。`--spec-draft-sampling greedy` 采用目标采样后匹配，`probabilistic` 采用拒绝采样；目标温度、重复惩罚和停止条件继续由公共生成引擎处理。

当前验证接口支持最多 7 个提案，非零上限必须小于 `--ubatch-size`。该草稿的滑窗路径目前要求 `--ctx-size` 不超过其 4096-token 窗口。MTP 的参数和使用方式不变。

## 调优与统计

```sh
"$BIN" -m "$MODEL" --spec-type draft-dflash -md "$DFLASH" \
  --spec-draft-n-max 7 tune --batches 1,4,8,128
```

调优准备目标模型的小 batch、草稿的噪声块，以及上下文投影。推理只读取策略缓存，不启动调优搜索。沿用 `--tune-cache`、CPU/NPU mask、线程数及物理 batch 参数。

`run --stats` 和 Rust `GenerationResult.timings` 包含 `spec_proposed`、`spec_accepted`、`spec_acceptance`、`spec_accept_length`、`spec_rounds`、`spec_draft_ms`、`spec_verify_ms`、`spec_repair_ms`。接受长度为每次验证实际推进的 decode token 数，包含 verifier bonus，不包含 prefill 产生的首个 token。草稿上下文的首次构建计入 prefill，接受前缀的上下文更新计入 repair。旧 `mtp_*` 计时字段保留为兼容别名。

Rust 调用使用统一配置：

```rust,ignore
options.speculation = SpeculativeConfig {
    kind: SpeculativeType::DraftDflash,
    model: Some(draft_directory),
    max_draft: 7,
    ..Default::default()
};
let engine = Engine::start(options)?;
```

选层特征随目标事务提交；拒绝后缀、请求重置和取消均同步处理草稿缓存。两种草稿共用采样、EOS、stop、流式输出和前缀复用路径。

## 接受长度与请求类型

同一条百科分析请求、greedy、128-token 输出上限的对照：

| 目标与执行路径 | thinking | block 4 | block 8 |
|---|---|---:|---:|
| 原始 BF16，官方 draft 参考代码 | 关闭 | 2.048 | 2.228 |
| C4，GPU 整数参考 | 关闭 | 2.153 | 2.268 |
| C4，板端实验路径 | 关闭 | 2.117 | 2.309 |
| 原始 BF16，官方 draft 参考代码 | 开启 | 3.256 | 5.292 |
| C4，GPU 整数参考 | 开启 | 3.629 | 6.048 |

这是单请求对照。开启 thinking 的前 128 个输出属于思考阶段。关闭 thinking 后接受长度低，在原始 BF16 加官方代码中同样出现，不能据此认定板端适配存在数值错误。

[官方模型卡](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash#benchmark-results) 使用 BF16、thinking、数学/代码/对话任务及最长 4096-token 输出；报告的 block 4/8 平均接受长度分别为 3.234–3.551 / 4.704–5.684。其统计为整轮 completion tokens / verify 次数，与上表的 decode 口径相差一个 prefill 首 token。

## 验证工具与来源

2026-10-20 板端 release 验证通过：Clippy 全 target 无警告，68 个库测试与 6 个输入测试通过；生产 CLI/API 共 19 项功能检查通过，包含 block 4/8、AR 对照、流式、stop、前缀复用、取消后恢复及 MTP1 回归。测试使用保存的 512-token 真实请求，普通功能请求生成上限为 16 token，取消场景上限为 128 token。

`tune` 命令验证通过，本次复用 596 次缓存；`tune_cache_misses`、`tune_calls`、`strategy_fallback_calls`、`weight_reload_calls` 均为 0。板端功能检查和调优命令的固定频率设置均已恢复，无恢复错误。

`tests/tools/speculative_api_smoke.py` 用保存的真实请求检查主命令、block 4/8、非流式/流式响应、stop、前缀复用、取消后恢复，以及 MTP1 回归。独立草稿算式参考位于 `tests/tools/dflash/reference.py`，输入夹具由 `dflash_bench` 工具导出。

草稿权重：`z-lab/Qwen3.5-4B-DFlash`，revision `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`，Apache-2.0。参考代码：[z-lab/dflash](https://github.com/z-lab/dflash)，revision `07ebd93db9f472af339b644bb70221ad8428328a`；MIT 许可与原作者署名保存在 `src/model/dflash/LICENSE`。
