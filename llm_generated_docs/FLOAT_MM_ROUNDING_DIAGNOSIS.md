# 为什么 FP16/FP32 归约的端到端偏移接近

> 历史实验记录：当前实现已收敛为默认启用 FP16 K tiling + NPU FP16 reduce，可用 `RK_NPU_MM_K_TILING=off` 做完整 K 精度对照。其余实验路径及专用工具已从当前代码删除，本文保留测量与分析，不再表示可选运行配置。实验源码与匹配二进制保存在共享 artifact。

2026-10-18。复用冻结的 revision5 二进制和已有只读 profile，红图 M256 分别跑完整 K、FP16 partial/CPU FP32 sum、FP32 partial/CPU FP32 sum 三次纯 ViT trace。未重跑语言模型，未更改算子实现。每种 trace 的最终 features 均与此前保存数组逐位一致。该实验用于定位误差轨迹，不用于计时。

## 已确认的精度边界

FP32 路径是 `FP16 A/B → GEMM FP32 partial → FP32 sum → FP16 output → FP32 widening/bias`，下一次 Linear 的输入还会转换为 FP16。`float_collect_fp32` 的目标仍为 uint16，包含 `vcvt_f16_f32`；这保持了现有逻辑 FP16 算子的输出语义。当前实验不是所有 Linear 的输入、输出和中间运算均保留 FP32。

因此“FP32 partial”不能等同于“全链 FP32”。但共同的 FP16 出口本身也不能证明它是主导误差来源；要做因果归因，需要锁定 Ktile/layout/host M/task M，再单独改变 partial/reduce/output 精度。

## 逐层 trace

下表为相对原完整 K 路径的 relative L2。行号为0起始 block index，最后一行在 merger 后。

| 位置 | FP16 partial + CPU sum | FP32 partial + CPU sum |
|---|---:|---:|
| patch + position | 0 | 0 |
| block 0 | 0.02611% | 0.00062% |
| block 3 | 0.14752% | 0.07011% |
| block 7 | 0.16088% | 0.09152% |
| block 15 | 0.42615% | 0.13110% |
| block 23 | 1.56243% | 0.79896% |
| 最终 features | 0.73289% | 0.41699% |

FP32 在第一个 block 的偏移约小42倍，到最终 features 约小1.76倍。误差沿后续计算逐渐出现并增长，第23个 block 的相对偏移增幅较大；merger 后相对偏移又减小。仅凭这些相对范数，不能把最后一层的增幅等同于某个单算子的绝对误差增幅，也不能单独归因于 Attention、归约或最终 FP16 舍入。

这两条实际调优路径的 Ktile、host/task M 等配置并未完全锁定；表格证明的是已有路径的误差轨迹，不是精确隔离累加位宽的单变量试验。

## 对“精度未通过”的解释修正

此前 logits 的 relative L2/KL 门限是相对原完整 K FP16 路径的**一致性门限**，不是相对完整高精度图文模型的准确率评估。完整 K 自己也有数值误差。以红图为例，相对 PyTorch FP32 参考：

| 位置 | 完整 K | FP16 partial/CPU sum | FP32 partial/CPU sum |
|---|---:|---:|---:|
| block 23 | 1.50974% | 1.14815% | 1.06641% |
| 最终 features | 0.59280% | 0.82625% | 0.57238% |

FP32 对部分位置更接近参考，但不能由此断言整个任务更准确。reference 实际运行于 **CUDA FP32，关闭 TF32，BF16 checkpoint 权重扩为 FP32**；此前部分报告把它称为 CPU FP32 reference，已更正标注。独立 dense CPU oracle 的称谓仍正确。

对已有 logits 逐行去掉均值后，relative L2 仍约13.9%～20.0%，所以此前较大的 logits 差异不能仅由整体常数平移解释。也不能直接把该百分比解释成准确率下降；已测10行 top1 均不变。

相近的“到基线距离”也不代表不同方案输出接近：FP16 CPU 与 FP32 CPU 的红图 features 彼此 relative L2 为0.7506%，蓝图为0.6101%。偏移方向也发生了变化。

## 下一项有区分度的对照

固定同一份真实 A/B、Ktile、layout、host M/task M，分别比较 FP16 partial、FP32 partial + 最终 FP16、FP32 partial + HOST_F32 直接输出，并同时记录舍入前后的结果。随后再看同一套视觉 features 通过同一个 decoder 的变化。当前数据尚未完成这一单变量因果对照，不应把“端到端 logits 改善有限”解释为 FP32 归约没有作用。

原始层 trace 留在共享 artifact 的 `floating-mm-variants-20261018/rounding-trace/`。结构化数据为 [rounding-diagnosis.json](evidence/float_reduce_variants_20261018/rounding-diagnosis.json)。当时的 `summarize_float_rounding.py` 不在当前工作树中；已保留的实验工具及其版本依赖见[实验归档](FP16_VIT_EXPERIMENT_ARCHIVE.md)。trace runner 已结束，频率恢复记录另存于证据目录。
