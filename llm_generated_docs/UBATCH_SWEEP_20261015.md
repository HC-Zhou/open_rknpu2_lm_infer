# ubatch128/192/256 整模型验证

2026-10-15，Qwen3-4B混合W4A8/W8A8，context8192，4线程、CPU0xf0、NPU7，Attention tile512；固定CPU2.4GHz/NPU1GHz/DDR2.112GHz。

## 应用改动

应用支持ubatch1–256。Linear不超过128时沿用2的幂档位，129–192使用192，193–256使用256；因此192行不再补到256。Attention每次公开调用封顶128，使用现有offset循环拆分，更底层Qwen Attention仍按最多32个逻辑token处理。算子库无需本轮改动。

按用户最新要求，默认ubatch从64改为128。默认tune原本跟随decode和主块，因此现在默认准备1/128；显式选择192或256时跟随为1/192或1/256。其他尾块可用--batches指定，已有有效缓存复用。README中英文与USAGE同步，既有chunk64历史性能复现命令显式指定64，历史数据不改写。

## 完整调用吞吐

每档每种输入预热1次、正式3次，计时包含全部模型层、原生KV/Attention和输出头；排除模型加载、离线策略搜索与prepare。先128、后192、最后256，是短程顺序小测，不是ABBA，不把几个百分点内差异解释成稳定优劣。

768是三档的公共倍数，分别使用6/4/3个主块；1024覆盖192的64-token尾块，128/256均整除。每个模型实例先显式离线补齐策略，随后检查正式计时与teacher-forced continuation期间调优、reload、实际fallback和profile数量都不增加。实验中的tune_search仍为true，但实际搜索增量严格为0；普通应用仍只读缓存。

| ubatch | pp768 tok/s | pp1024 tok/s | 对128的变化（768 / 1024） |
|---|---:|---:|---:|
| 128 | 168.50 | 157.58 | +0.00% / +0.00% |
| 192 | 164.20 | 156.93 | -2.55% / -0.41% |
| 256 | 164.48 | 158.03 | -2.39% / +0.28% |

本次192/256没有测出超过128的明确吞吐收益，因此选择默认128。该结论针对当前模型、缓存、资源和输入长度，不外推所有模型与长上下文。保留192/256供显式选择，停止扩展其他尺寸或模型矩阵。

## 正确性与检查

每个输入在最后一次完整prefill后再执行固定8-token continuation，比较9个位置的全部151936维FP32 logits。192和256相对128，两种输入共4组比较全部逐位一致，最大绝对误差0；128本身也保留结果和完整原始dump。8步continuation用于数值检查，不作为正式decode吞吐结论。

W4/W8 Linear CPU oracle及融合SwiGLU扩展到256，覆盖129、191、192、193、255、256等边界；Attention129/192/193/256行分别与不超过128行的基线分段执行比较，逐位一致。fmt、clippy(all-targets、release、-D warnings)、常规Rust测试与release构建通过。默认值修改不影响本次显式指定128/192/256的性能样本，无须重跑长测。

首次Attention测试因2线程配4个CPU的掩码配置错误，在NPU提交前失败；同进程随后仍完整执行了模型sweep，因此初次包装器退出101。修正测试掩码后，Attention用例独立补跑通过。初次包装器与最终Attention包装器的频率恢复均无错误。原始状态保留在证据中，不把首次退出码改写为成功。

## 证据与复现

[原始样本、固定输入IDs、计数增量、完整logits比较与时钟](evidence/ubatch-expanded-20261015.json)包含所有正式样本、构建SHA和资源配置。RSS来自同一模型实例离线调优后的进程，含调优临时分配器保留的内存，不能直接作为只读缓存服务的内存开销。

通常直接使用默认配置：

```sh
python3 scripts/run_board.py --report tune-clocks.json -- \
  target/release/open_rknpu2_lm_infer --model /path/to/model \
  --ctx-size 8192 --tune-cache /path/to/cache tune --report tune.json
```

显式复现192档的pp768/1024，先以相同模型、context和cache执行 `--ubatch-size 192 tune --batches 1,64,192`，再执行 `--ubatch-size 192 bench -p 768,1024 -n 0 -r 3`。256档改为256，调优1/256；128档可直接默认tune。
