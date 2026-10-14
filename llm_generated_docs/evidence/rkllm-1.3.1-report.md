# RKLLM W8A8 三模型测试

| 模型 | prefix PPL | pp512 | pp1024 | pp4096 | decode128@512 | decode128@1024 | decode128@3800 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B W8A8 | 12.8709 | 105.037 ± 0.031 | 94.619 ± 0.021 | 50.133 ± 0.084 | 5.772 ± 0.002 | 5.125 ± 0.004 | 2.310 ± 0.005 |
| MiniCPM5-2B W8A8 | 14.1293 | 211.052 ± 0.484 | 188.643 ± 0.082 | 98.874 ± 0.009 | 9.431 ± 0.006 | 8.985 ± 0.003 | 7.117 ± 0.003 |
| Granite-4.2-3B W8A8 | 15.7448 | 120.459 ± 0.107 | 106.277 ± 0.019 | 54.404 ± 0.004 | 6.806 ± 0.003 | 6.370 ± 0.006 | 3.932 ± 0.001 |

吞吐单位 tokens/s，均值 ± 样本标准差；每档预热 1 次、正式测量 3 次。三模型性能均为 2026-10-14 板端实测，使用 SDK phase timer、greedy 生成、每次调用前清空 KV。
CPU4–7，大核 2400 MHz，NPU 1000 MHz，DDR 2112 MHz；Runtime 1.3.1 / driver 0.9.8。正式样本的 SDK token 计数及频率恢复记录均通过核验。

Qwen3 PPL 复用 2026-10-06 同口径结果，原始 NLL 在本次重新汇总核对；当前性能输入 token 文件与该次 PPL 的记录一致。PPL 均为 WikiText-2 前 32 个独立 c512 窗口，评分目标 257–511，共 8160 tokens，FP64 logsumexp。

## Limitations

三个模型分别使用自身 tokenizer 和 token 文件。PPL 为前 32 窗口的 prefix PPL。长上下文 decode128 使用 3800 个输入 tokens；pp4096 单独测 prefill。decode128 为 SDK 的 128 步生成阶段，共回传 129 tokens。

performance_manifest.json 记录 Qwen3 的完整性能配置；performance/ 保存原始日志、JSON、资源和时钟。quality_source.json 指向历史 PPL 原始证据。summary.json 保存 Qwen3 结果，comparison.json 包含三模型结果及原始吞吐样本。

## 复现

在 .paper/ 目录执行，重测选择新的 output-dir。

```bash
remoterun '.venv-board-baselines/bin/python experiments/run_baseline_matrix.py --manifest experiments/results/rkllm_qwen3_matched_20261014/performance_manifest.json --output-dir experiments/results/rkllm_qwen3_matched_rerun'
```
