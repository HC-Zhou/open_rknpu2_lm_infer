# FP16 ViT 利用率与大 K 瓶颈

**后续实现更新（2026-10-18）**：删除运行时数值扫描，ViT改为FP16激活存储，21个HOST_STORAGE profile重新调优，RoPE改为预计算表＋NEON。256²/384×256/512²完整encode分别为490.609/748.705/2238.948ms，3T口径MFU为11.23%/11.26%/11.00%。256²当前最大两项是GELU213.797ms与Attention103.415ms；down56.011ms、up43.252ms。性能和精度限制见[最新FP16存储报告](VISION_FP16_STORAGE.md)。下文保留各次历史实验，不能将其“当前”措辞或FP32 host开销当作最新实现。

2026-10-18，RK3588 主板，NPU 1GHz、DDR 2112MHz、大核 2.4GHz；CPU 4线程/mask `0xf0`、NPU mask `7`。本页诊断对象是完整 K 的 FP16 revision2，K 分块后的结果见 [自动调优说明](FLOAT_MM_AUTOTUNE.md)。原始记录与统计见 [证据目录](evidence/floating_mm_util_20261018/README.md)。

## 实测结论

高 M 可以跑到接近峰值，但 K4096 的完整 K 内核受 CBUF 与任务几何限制，增大 M 没有消除瓶颈。`K1024/N4096` 在 M256 时已经达到 2.623 TFLOPS；M8192 为 2.668 TFLOPS。`K4096/N1024` 从 M256 增至 M4096，仍只有约 0.40 TFLOPS。

完整 ViT 的约 8% MFU 还包含显著 CPU 与数据整理开销。256×256 输入时，CPU GELU 约占 32%，MLP down 约占 22%，Attention 约占 18%。即使解决大 K matmul，也仍需处理 GELU 和 Attention 才能显著提高整体利用率。

## FP32激活阶段默认 K tiling 路径的瓶颈（历史）

当前默认保留 FP16 partial + NPU FP16 Add tree。以下引用此前同一保留路径的定频配对计时；revision6 清理后已核对策略及红/蓝图 features 一致，本次没有重新测一套阶段计时。原始配对记录见 [measurement.json](evidence/fp16_k_tile_autotune_20261018/measurement.json)。

| 阶段，24层累计；256×256 | 完整 K | K tiling + NPU reduce | 分块后占总耗时 |
|---|---:|---:|---:|
| CPU GELU | 217.1ms | 214.7ms | 34.4% |
| Attention 全流程 | 126.2ms | 128.1ms | 20.5% |
| MLP down，K4096 | 145.2ms | 97.5ms | 15.6% |
| MLP up | 50.7ms | 50.0ms | 8.0% |
| QKV split + RoPE | 44.2ms | 44.6ms | 7.1% |
| QKV Linear | 40.5ms | 40.2ms | 6.4% |
| 整次 encode | 677.5ms | 624.8ms | 100% |

MLP down 用时减少32.8%，约1.49倍加速；两层 merger Linear 合计8.48→4.20ms。两者贡献了几乎全部净收益，整次 encode 少52.7ms（时延下降7.8%，吞吐约1.084倍）。3T口径 MFU 从8.14%提高到8.82%。K tiling 的收益来自改变小 K plan 的 CBUF 分配和 task 几何，不减少原 GEMM 的乘加数；额外增加 partial 存储、NPU Add 和控制开销。

GELU 当前是 CPU Rayon 并行逐元素 `tanh` 近似路径，没有显式 NEON 向量实现。Attention 阶段包含 Q/K/V 布局转换、FP32→FP16、cache load/sync、QK/PV 和 CPU stable softmax，现有阶段计时尚未把它们逐项拆开，因此不能把全部128ms归因于其中某一个子算子。

512×512 时当前路径约2799.1ms，其中 Attention 872.0ms（31.2%）、GELU 851.5ms（30.4%）、MLP down 362.1ms（12.9%）。分辨率增大后 Attention 成为与 GELU 相当的主要项。

K tiling 没有解决 CPU 激活、Attention BMM/softmax、RoPE 和 host/device 张量流转。普通 Linear 的完整调用仍含 FP32→FP16、pack/sync、结果收集及 widening/bias；K 分块当前走分阶段 host IO，完整 K 则可使用已验证的转换与 pack/collection 融合。设备内核改善不能全部兑现为完整调用收益。配对试验中256×256每次 encode 的平均驱动硬件区间约196.5→108.5ms，但该统计含不同冷/热调用，不能直接与暖态总时延中位数相减，作为精确 CPU 耗时归因。

下一步应优先优化 GELU 的 CPU 向量化或评估设备执行；对 Attention 先拆 QK/softmax/PV/转换与同步，再决定改哪一部分；同时评估减少 Linear host IO 往返。仅继续调大 K matmul 已难以显著提升整体 MFU。

## 当前搜索空间重新调优 up/down

2026-10-18，补做 revision6、M256 的独立空缓存搜索及原配置/新配置同库同频配对复测。此前 K tiling 的 down 有过1798条候选计时，但后续 ViT 及 revision6 验证加载了该缓存，没有重新搜索新增的均衡 task M 候选。

| 完整 FP32 host 调用 | 原配置，本轮复测 | 新调优配置 | 时延下降 |
|---|---:|---:|---:|
| up，K1024/N4096 | 1.9458ms | 1.9002ms | 2.35% |
| down，K4096/N1024 | 4.0402ms | 3.9429ms | 2.41% |

两个新配置均仅将 task M cap 从CBUF默认改为128，测试输出与原配置逐位一致。up 仍为完整 K + panel8 A/C + fused host IO；down 仍为 K1024 四个 partial + normal A/panel8 C + NPU FP16 Add tree + staged host IO。down/up 为2.075倍，新增候选未消除差距。两者 FLOPs 相同，但不同执行路径的开销尚未逐项计时，不能将剩余差距全部归因于归约或转换。

每 shape 有4个独立进程、每进程8组 ABBA。这里只测这两个 M256 matmul，未替换默认缓存，未重测完整 ViT。原始记录、统计方法与恢复验证见 [复查报告](evidence/fp16_up_down_retune_20261018/README.md)。

## 常驻设备数据的 matmul

使用相同 raw MM plan，比对常驻 A/B/C 的 device 调用和含 FP32 输入转换、pack/sync、输出收集的完整 host 调用。每条路径预热后做3组 ABBA、6次调用，表中取中位数。FMA 按2 FLOPs；下表是 device 调用的墙钟有效算力，包含 submit/wait。

| K×N | M256 | M1024 | M4096 | M8192 |
|---|---:|---:|---:|---:|
| 1024×1024 | 1.924 T | 2.008 T | 2.037 T | — |
| 1024×4096 | 2.623 T | 2.656 T | 2.662 T | 2.668 T |
| 4096×1024 | 0.402 T | 0.403 T | 0.404 T | — |
| 4096×4096 | 0.470 T | 0.470 T | 0.472 T | — |

M256、K1024、N4096 的 device 调用约0.819ms，完整 host 调用约1.888ms，有效算力从2.623T变为1.137T。M1024 对应约3.234ms与7.776ms。因此计算内核接近峰值与完整调用的利用率是两种口径。

四种形状采用按三核均分、32对齐的 N tile。K1024 使用 panel8 A/C、task M cap128；K4096 使用 normal A/panel8 C、CBUF 默认 task M。该探针隔离完整 K 路径的限制，不代替自动调优选中策略的完整调用成绩。

## K 与 CBUF

`native/rk_npu_matmul/src/rk_npu_matmul_f16.cpp` 的 FP16 task 行数受12个 CBUF bank限制，每 bank 32768 bytes。权重 bank 数为 `max(2, ceil(K*2*32/32768))`；normal/panel A 由剩余 bank 容纳输入行，native A 另使用既有预算。对应上述探针：

| K | 权重 banks | normal/panel A 可用 banks | CBUF 最大 task 行数 | 探针实际 task 行数 |
|---|---:|---:|---:|---:|
| 1024 | 2 | 10 | 160 | cap128 |
| 4096 | 8 | 4 | 16 | 16 |

M1024、三路 N 分块时，K1024 为24个 task，K4096 为192个 task。增大逻辑 M 继续重复这些很薄的 task。K tiling 将每个 partial 的 K 缩小，使输入可采用更大的 task M；它的净收益取决于 partial 输出、归约及同步开销，必须按完整调用计时。

## 完整24层 ViT

模型权重加载后，先做一次未加分段计时的冷 encode，再预热一次，随后测3次。每次清空 image feature cache，保留常驻权重和 workspace。每次输出与首轮 features 逐位一致，feature cache hits 为0。512×512 使用另外生成的纯红图；两张较小图沿用视觉交接的固定输入。

| 输入宽×高 | patch M | 冷 encode | 热 encode 中位数 | FLOPs | 热 encode MFU，3T峰值 |
|---|---:|---:|---:|---:|---:|
| 256×256 | 256 | 704.570ms | 678.013ms | 165.356G | 8.129% |
| 384×256 | 384 | 1070.918ms | 1029.845ms | 252.866G | 8.185% |
| 512×512 | 1024 | 3083.318ms | 3016.216ms | 738.734G | 8.164% |

encode 包含 patch embedding、24层 Transformer、merger、CPU 处理、同步和输出收集，不含图像解码/resize、模型权重加载和语言模型。计算量公式见自动调优说明。

各阶段取3次 encode 的中位数，单位 ms；各列中位数之和不要求恰好等于总时延中位数。

| 阶段，24层累计 | M256 | M384 | M1024 |
|---|---:|---:|---:|
| CPU GELU | 215.569 | 324.660 | 857.259 |
| MLP down Linear，K4096 | 146.107 | 205.530 | 548.916 |
| Attention | 124.817 | 218.674 | 888.290 |
| MLP up Linear | 51.313 | 75.526 | 186.363 |
| QKV split + RoPE | 43.641 | 67.198 | 177.286 |
| QKV Linear | 39.684 | 61.164 | 152.447 |
| Attention output Linear | 17.535 | 24.968 | 64.282 |
| norm1 + norm2 | 16.975 | 26.580 | 64.813 |

## 计时与复现

`scripts/archive/fp16_vit_20261018/fp16_device_probe.cpp` 保留常驻设备数据对照；`tests/tools/vision_profile.rs` 提供完整 encode 和分阶段计时；归档中的 `npu_ioctl_profile.cpp` 使用受测库的私有 UAPI 记录 submit、sync 和驱动返回的 `hw_elapse_time`。历史汇总脚本为同目录的 `summarize_float_util.py`，使用边界见[实验归档](FP16_VIT_EXPERIMENT_ARCHIVE.md)。

三档 ViT 的每次 encode 平均驱动硬件区间分别为196.712、293.357、950.087ms。所有提交 mask 均为7，设备错误计数为0；未观察到只用单核的现象。驱动区间包含任务执行中的访存、流水与停顿，不是 MAC busy cycles；用模型 FLOPs 除以该区间只能得到该口径的有效算力，不能当作硬件指令级 MFU。

完整探针在共享板锁及固定频率下执行。runner 返回0、恢复错误为空，并核对 CPU/NPU/DDR 的 min/max/governor 恢复。原始数组和冻结的可执行文件留在 `/home/zt/mnt/3090x4/tmp/floating-mm-util-20261018/`，仓库证据收录日志与结构化结果。

后续 K 分块的独立 FP16/FP32 归约、融合 EW、任务并行度对照及完整模型精度见 [归约路径比较](FLOAT_MM_REDUCTION_COMPARISON.md)。该实验独立于本页完整 K 基线；融合路径在部分 task 几何下实际只用两核，不能沿用本页“提交均为 mask7”的结论。
