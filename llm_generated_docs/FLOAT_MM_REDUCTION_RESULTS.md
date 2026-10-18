# K 分块方案性能与精度总表

> 历史实验记录：当前实现已收敛为默认启用 FP16 K tiling + NPU FP16 reduce，可用 `RK_NPU_MM_K_TILING=off` 做完整 K 精度对照。其余实验路径及专用工具已从当前代码删除，本文保留测量与分析，不再表示可选运行配置。实验源码与匹配二进制保存在共享 artifact。

2026-10-18，RK3588。NPU 1GHz，DDR 2112MHz，大核 2.4GHz；CPU mask0xf0/4线程，NPU mask7。BF16 未参与本轮计算。

ViT 为已加载权重、清空 features cache 后的完整暖态 Encoder::encode，每路径6个调用取中位数；不含图片解码/resize、模型加载和语言模型。精度偏移相对原完整 K FP16 基线；features 取红/蓝图最大相对 L2，logits 取红/蓝/双图三个用例最大相对 L2，KL 取10行中的最大值。

| 方案 | ViT 256×256 | ViT 384×256 | features L2 最大 | logits L2 最大 | 最大 KL | top1 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 完整 K（默认） | 677.5ms | 1034.9ms | 0.000% | 0.00% | 0.000000 | 0/10 |
| FP16 partial + NPU FP16 tree¹ | 624.8ms | 966.6ms | 0.706% | 17.83% | 0.123529 | 0/10 |
| FP16 fused GEMM + EW | 626.3ms | 966.9ms | 0.694% | 19.17% | 0.100819 | 0/10 |
| FP16 partial + CPU FP32 sum | 667.0ms | 1023.0ms | 0.733% | 16.99% | 0.100565 | 0/10 |
| FP32 partial + NPU FP32 tree | 646.4ms | 965.9ms | 0.377% | 15.08% | 0.120159 | 0/10 |
| FP32 partial + CPU FP32 sum | 628.1ms | 967.8ms | 0.417% | 14.52% | 0.144045 | 0/10 |

¹ 独立 FP16 NPU tree 沿用上一轮 revision3 的定频配对结果，其余为本轮测量。不同批次间的小差异不判定为收益。融合蓝图保留了944–1118ms的原始样本；中位数相近不构成统计优越性结论。

FP32 NPU 数据采用 revision4 的搜索结果，revision5 复用这些 profile 和常驻权重时已核对红图 features 逐位一致。融合和两种 CPU 路径采用 revision5 新增的均衡 task 搜索，其他 Linear 的已选参数也可能变化。因此这是各方案实际调参后的对比，不是仅更换归约器的因果 A/B；FP32 CPU 在红图上更快不能单独归因于 CPU 累加。固定几何的独立/融合对照见分析文档。

完整 K features/logits 均逐位一致。其余已测 K 分块方案均未通过既定模型门限（logits relative L2<2%、逐行 KL<0.01、top1不变）；这不是任务准确率下降的百分比。FP32 fused feedback 原语存在半数通道错误，未纳入模型或有效性能测量。

FP32 NPU 路径在512×512上的完整 ViT 为2878.9ms，完整 K 为3007.3ms；其余新增方案未测该分辨率。按3 TFLOPS估算，已测最佳完整 ViT MFU仍约8.8%。

FP32 partial 的红/蓝图 features 相对 PyTorch FP32 reference（CUDA，禁用 TF32） L2为0.511%/0.374%，比完整 K 的约0.593%/0.428%更小，但 logits 偏移依然较大。不能用 primitive 通过或 features 更接近参考替代完整模型核对。

本表对应当时实验版本的默认完整 K；当前版本默认启用 FP16 K tiling + NPU FP16 reduce。设置 `RK_NPU_MM_K_TILING=off` 可做精度对照。详见 [自动调优说明](FLOAT_MM_AUTOTUNE.md)、[融合变慢的对照分析](FLOAT_MM_REDUCTION_COMPARISON.md) 和 [原始证据](evidence/float_reduce_variants_20261018/README.md)。

门限含义及 FP16/FP32 逐层误差的新核对见 [舍入诊断](FLOAT_MM_ROUNDING_DIAGNOSIS.md)：上述通过/未通过指与原路径的一致性，不能据此判定任务准确率或把完整 K 当作高精度金标准。
