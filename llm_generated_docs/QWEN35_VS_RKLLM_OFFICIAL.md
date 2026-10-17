# Qwen3.5-4B 与 RKLLM 官方公开成绩对照

日期：2026-10-17，Asia/Shanghai。两边均为128-token输入、64-token输出。

官方参考为 [RK3588 benchmark](https://github.com/airockchip/rknn-llm/blob/f7390530443bf84f0394255a449d7cbe81e69d1c/benchmark.md#rk3588)，文件最后更新于2026-09-29的 `f739053`（release v1.3.1）。本轮只重测自有 runtime，RKLLM 数字取该公开表。

| 指标 | 官方 RKLLM | 自有 runtime | 自有/官方 | 吞吐提升 |
|---|---:|---:|---:|---:|
| Prefill近似比较，tok/s | 61.869 | 129.052 ± 1.512 | 2.086× | +108.59% |
| Decode，tok/s | 6.22 | 8.948 ± 0.007 | 1.439× | +43.87% |

官方给出 TTFT=2068.88ms、decode=6.22tok/s；prefill一栏由 `128000/2068.88` 折算，包含首个输出的工作。自有prefill执行完整主干但不含final norm/lm_head，逐样本 `head_ms=0`；因此prefill倍数是近似对照，不能解释为严格匹配的TTFT提升。

自有测量使用重构后 Release 二进制 `fbd441...642693`，备用1，CPU4 / 0xf0 / 2256MHz、NPU7 / 1GHz，DDR未受控；context4224、ubatch128、Attention tile512。固定seed42输入，同一进程每档预热1次、正式3次，prefill后保持状态连续decode64；不计加载、prepare、分词、采样与统计读取。正式测量的tune/reload/strategy-fallback计数均为0，head-free prefill连续性检查通过。

官方是W8A8，自有C7是197个W4与52个W8 Linear（含W8输出头）的混合方案。输入长度已对齐；量化、权重、输入内容和板端环境未完全匹配，结果描述两个系统方案相对公开成绩的表现，不单独归因于runtime算法。官方仓库的 [RK3588定频脚本](https://github.com/airockchip/rknn-llm/blob/f7390530443bf84f0394255a449d7cbe81e69d1c/scripts/fix_freq_rk3588.sh) 将大核设为2352MHz、NPU1GHz、DDR2112MHz；本次未据此作频率归一化。

首轮误用 `--no-tune`，禁用了策略缓存，context也为512，故该组只保留为 `preliminary-default-strategy*`，不进入上述表格，也不作为缓存收益的受控A/B。两次runner均正常结束，受控频率上下限和governor恢复。

证据入口：[comparison.json](evidence/qwen35_vs_rkllm_official_20261017/comparison.json)。目录含正式3个原始样本、阶段计时、命令/时钟恢复记录及官方数字的固定revision引用；没有改动模型实现。
