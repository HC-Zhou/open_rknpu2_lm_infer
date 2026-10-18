# FP16 matmul 的 K 分块与归约比较

> 历史实验记录：当前实现已收敛为默认启用 FP16 K tiling + NPU FP16 reduce，可用 `RK_NPU_MM_K_TILING=off` 做完整 K 精度对照。其余实验路径及专用工具已从当前代码删除，本文保留测量与分析，不再表示可选运行配置。实验源码与匹配二进制保存在共享 artifact。

2026-10-18，RK3588，NPU 1GHz / DDR 2112MHz / 大核 2.4GHz，CPU 4线程、mask `0xf0`，NPU mask `7`。BF16 不参与本轮计算。

## 完整算子调用

以下均含 FP32 输入转换、pack、同步、submit/wait、输出收集和 FP32 bias，不含常驻权重准备。每个方案分别搜索参数，再做三组 ABBA，每条路径6个调用取中位数。不同批次的共同对照是原 revision3 独立 FP16 NPU tree；每组原始对照值保存在 [microbench.json](evidence/float_reduce_variants_20261018/microbench.json)，不能将跨批次的小差异解释为收益。

这是加入 revision5 均衡 task 搜索前的算子扫描：host M 固定，Ktile 为512/1024/2048，task cap 为0/64/128，另扫 Ntile 和支持的 A/C 布局。后面的均衡 task 对照及 [完整 ViT/精度总表](FLOAT_MM_REDUCTION_RESULTS.md) 使用更完整的搜索结果。因此不能把本表的 CPU/NPU 差距直接套到新版完整 ViT。

| 路径 | M256 K4096 N1024 | M1024 K4096 N1024 | M64 K4096 N4096 |
|---|---:|---:|---:|
| 完整 K，revision2 | 5.957ms | 22.597ms | 4.933ms |
| FP16 partial + 独立 NPU FP16 tree | 4.001ms | 15.052ms | 2.345ms |
| FP16 partial + CPU FP32 sum | 5.631ms | 21.937ms | 3.681ms |
| FP16 fused GEMM + EW | 4.258ms | 15.101ms | 2.525ms |
| FP32 partial + 独立 NPU FP32 tree | 4.503ms | 17.878ms | 2.638ms |
| FP32 partial + CPU FP32 sum | 4.797ms | 21.391ms | 3.064ms |

前三行完整 K / 独立 FP16 tree 的正式配对样本见 [revision3 证据](evidence/fp16_k_tile_autotune_20261018/README.md)。其余四行来自各自与独立 FP16 tree 的同进程配对。融合 FP16 的配对加速比分别为0.951×、0.995×、0.932×；当前没有优于独立归约。

## 为什么融合没有更快

当前融合仍将每个 K 分块的中间输出写回设备内存，下一块通过 RDMA 读入 EW。它省掉了独立 Add task 和对应提交，但并未将整条 K 链的累加器一直保留在片上。

为使输出能直接作为下一次 EW 操作数，融合使用专用 half-surface C：每个输出占用的缓冲空间是紧凑 FP16 的1.5倍。当前正确实现限制 A 为 normal；独立路径可以选择 panel A/C。缓冲空间增大不能直接等同于实际总线流量增加1.5倍，尚未测量物理流量。

融合要求一个 core 完成同一输出区域的整条 K 链。M256/N1024、task M128、Ntile0 时只有两个独立输出区域，实际只使用两核；独立 partial 可以将不同 K 的任务分配给三核。减小 task M 或 Ntile 会增加区域数，也会增加 task、权重重复读取等开销，因此仅“让三核都有任务”并不保证更快。上述三形状的参数扫描已覆盖这些取舍。

进一步固定 M256/K4096/N1024、Ktile1024、A normal、C 为各路径要求的布局，做一组进程级 ABBA，每进程预热2次、正式12次。下表为各路径24个正式完整调用的中位数；这些是两个进程内的重复调用，不视作24个独立实验：

| Ntile / task M | 允许核数 | 独立 FP16 tree | 融合 FP16 | 融合加速比 |
|---|---:|---:|---:|---:|
| 0 / 128 | 3 | 3.786ms | 4.078ms | 0.928× |
| 0 / 128 | 1 | 5.464ms | 5.317ms | 1.027× |
| 352 / 128 | 3 | 4.273ms | 4.211ms | 1.015× |
| 352 / 128 | 1 | 6.692ms | 6.477ms | 1.033× |
| 0 / 64 | 3 | 4.187ms | 4.371ms | 0.958× |
| 0 / 64 | 1 | 6.078ms | 5.825ms | 1.044× |

这里融合确实将每次调用的3次 submit 降为1次。Ntile0/task M128 的三核配置中，ioctl 记录确认融合只提交 mask3（两核），独立 GEMM 提交 mask7，之后两层 Add 分别使用 mask3/mask1。限制为单核后，融合在三组分块参数下均快2.7%～4.4%，说明省去独立 Add/提交本身有收益。Ntile352 使融合也能用三核，相同分块下略快，但4.211ms仍慢于独立路径不切 N 的3.786ms。

因此并行度和任务划分是已用对照确认的因素。C 布局及读写的独立贡献仍未单独测出，不能把全部差距归到 layout。driver hardware time 记录也不等同于 MAC 忙周期。原始24样本及 ioctl 汇总见 [variants.json](evidence/float_reduce_variants_20261018/variants.json) 的 `fusion_ablation`。当时的 `fp16_reduce_ablation.cpp` 与 `summarize_float_variants.py` 不在当前工作树中；已保留的实验工具及其版本依赖见[实验归档](FP16_VIT_EXPERIMENT_ARCHIVE.md)。

补充相同 Ntile0/Ktile1024/A normal 的均衡 task M 扫描（另一次进程级 ABBA，每路径24次调用）：

| task M cap | 独立 FP16 tree | 融合 FP16 |
|---|---:|---:|
| CBUF 默认160 | 3.843ms | 4.396ms |
| 128 | 3.769ms | 4.074ms |
| 96 | 3.832ms | 3.985ms |
| 88 | 3.829ms | 3.912ms |
| 86 | 3.838ms | 3.936ms |

cap86形成86+86+84三个区域，cap88形成88+88+80，均能使用三核。融合仍未超过独立归约，但相对 cap128 快约4%。原先按 task 数量去重会把160+96和128+128混为一类，漏掉实际不同的负载分布。revision5 已去掉这一去重方式，并加入按核数均分 M 的 task cap；相同 cap 数值仍去重。证据见 [balanced-ablation.json](evidence/float_reduce_variants_20261018/balanced-ablation.json)。

## 精度和默认取舍

原 revision3 的 FP16 partial + FP16 tree 在红/蓝/双图的10行 logits 中 top1 均相同，但 logits relative L2 为12.1%～17.8%，部分 KL 超过0.01，模型门限失败。完整 ViT 的收益约6%～8%，不能据此默认启用。默认仍为完整 K，保留 revision2 的逐位一致结果。

FP32 writer 探针使用不能由 FP16 精确表示的值，512/512 输出保留额外精度，dense CPU oracle 最大绝对误差 `7.45e-9`。FP32 独立 NPU Add tree 的5组测试覆盖2/3/4/16个 partial，均与 CPU FP32 两两相加逐位一致。252组库内 K 分块测试覆盖两种 partial 精度、CPU/NPU 归约和 FP16 融合，并包含不整齐 M/K/N、N tile、mask1/7、storage/FP32 host IO。

FP32 fused feedback 的两个已测寄存器配方有半数通道错误；没有纳入合法候选，也没有作为正确实现计时。失败日志与成功的 FP32 writer/tree 日志一并保留在 [证据目录](evidence/float_reduce_variants_20261018/)。

revision4 默认关闭 K tiling 后，重新核对红/蓝图 features 及红/蓝/双图10行 logits，均与 revision2 基线逐位一致。FP32 partial + NPU tree 完成 M256/M384 的14个 profile 和同一组模型用例：

| 指标 | 红图 | 蓝图 | 双图 |
|---|---:|---:|---:|
| features 相对完整 K 基线 L2 | 0.3773% | 0.3440% | — |
| features 相对 PyTorch FP32 reference（CUDA，禁用 TF32） L2 | 0.5106% | 0.3743% | — |
| logits 相对完整 K 基线 L2 | 13.461% | 13.794% | 15.079% |
| 最大逐行 KL | 0.120159 | 0.071230 | 0.105206 |
| top1 相同的行数 | 3/3 | 3/3 | 4/4 |

features 满足既定 cosine>0.9999、relative L2<2% 门限，并且比完整 K 的 PyTorch FP32 reference（CUDA，禁用 TF32） 误差更小；logits 不满足 relative L2<2%、逐行 KL<0.01 的门限。不能把额外 FP32 精度或 top1 相同解释成模型精度已通过，也不能仅凭这三组颜色问题判断广泛任务质量。FP32 partial 仍为可选实验路径。

当前用法以 [自动调优说明](FLOAT_MM_AUTOTUNE.md) 为准；旧算法选择变量和实验工具已移除。
