# NPU FFN GEMM/bias/GELU 融合：实现与验证

2026-10-18，RK3588 主板。历史 CPU 对照基线为外层 `93d552e`、算子库 `5b4e39c`。正式推理现默认使用融合 FFN，旧 Rust CPU tanh GELU 路径已删除；merger 使用模型规定的 erf GELU，仍由 Rust 计算。

## 结果

独立 NPU GELU 的完整调用在记录的 ViT UP 激活上快约10–12倍。替换24层 FFN 的近似 GELU 后，完整 ViT 快1.53–1.66倍，最终特征相对 CPU GELU 基线的 relative L2 偏移0.75%–1.91%。相对真正 PyTorch FP32 基线的误差也有增加。

同一任务中的 `GEMM → BS bias → BN 索引缩放 → GELU LUT` 已验证并接入隔离 ViT：逐通道 bias 通过 BRDMA 读取，不需要展开为 M×N。融合版本使用 FP32 DPU 处理和 FP32 carrier 输出，Rust 收集后写 FP16；这与直接 FP16 writer 的硬件输出路径不同。

最终融合版本在三张回归图上比独立 NPU GELU 再省约6.9%–7.5%完整 ViT 时间，相对 Rust CPU GELU 为1.67–1.81倍。相对 FP32 gold 的最终特征 relative L2 为1.81%、1.06%、2.55%；并非所有尺寸都比独立方案更差或更好。

## 独立 NPU GELU 的完整 ViT

同模型、同调优缓存、CPU4/A76、NPU mask7；CPU2.4GHz、NPU1GHz、DDR2.112GHz。每个分辨率按 CPU/NPU/NPU/CPU 顺序运行四个独立进程，每个进程预热一次、计时三次，以下取每条路径六个样本的中位数。计时是完整 encoder，权重驻留，每次清空特征缓存；不是含文本解码的请求时延。输入为前期红/蓝图实验素材。

| 输入 | CPU GELU ViT / ms | NPU GELU ViT / ms | 加速比 | NPU 对 CPU 特征偏移 | CPU 对 FP32 gold | NPU 对 FP32 gold |
|---|---:|---:|---:|---:|---:|---:|
| 红图256×256 | 513.236 | 311.785 | 1.646× | 1.0544% | 1.2706% | 1.7548% |
| 蓝图384×256 | 776.254 | 466.360 | 1.664× | 0.7500% | 0.8710% | 0.9262% |
| 红图512×512 | 2273.443 | 1482.247 | 1.534× | 1.9073% | 2.4234% | 2.8466% |

表中三个精度列均为 relative L2。CPU/NPU 各自的重复运行输出逐 bit 相同。红图256²的24层 GELU 累计由230.911ms降到22.429ms；512²由893.364ms降到82.675ms。Merger 的 erf GELU 保留 CPU，其他视觉阶段相同。

逐算子误差不能代替最终特征误差：在记录的 UP 激活上，局部 relative L2 只有0.00077%–0.03260%，经过整个 ViT 后仍会累积。这里的2%仅是探针的运行检查阈值，不能视作视觉任务精度通过标准。

## 独立 GELU 的完整调用

直接比较已提交的 Rust `vision_half::gelu` 与 NPU 探针，CPU均使用4个线程。NPU计时包含输入夹取/拷贝、缓存同步、每核 LUT 装载、提交等待、输出还原；创建工作区的时间另记。三组 ABBA，每条路径六个样本。

| UP 激活 | 元素数 | Rust CPU / ms | NPU / ms | 加速比 | 对 CPU relative L2 |
|---|---:|---:|---:|---:|---:|
| 红图 layer0 | 1,048,576 | 9.515 | 0.811 | 11.73× | 0.03260% |
| 红图 layer23 | 1,048,576 | 8.801 | 0.802 | 10.97× | 0.00086% |
| 蓝图 layer0 | 1,572,864 | 14.240 | 1.354 | 10.52× | 0.02289% |
| 蓝图 layer23 | 1,572,864 | 13.114 | 1.303 | 10.07× | 0.00077% |

最大绝对差均为0.001953125。输入反转后的输出与原输出反转逐 bit 相同；输出 guard、有限值与重复性检查通过。所有63,488种有限 FP16 位模式也通过；该全集的整体 relative L2 被大数值尾部主导，因此不用于论证真实激活精度。

## LUT 与输出

LUT 使用两组513项表，覆盖负/正两个 `[-4,0]`、`[0,4]` 区间，步长1/128，表值为 `round(gelu_pytorch_tanh(x) × 8192)`。BN 将索引乘4096。DPU 使用 FP16 输入、INT16 输出，直接写 Q13 编码；CPU再还原为 FP16。正尾 x>4 使用 x，负尾 x<−4 使用带符号零，并保留负零符号。

单个大任务采用已验证的 FloatReduce 物理视图 `[channels/8, width, 8]`，width128、channels至4096，即524,288元素。逐点操作不需要真实转置。每次提交中每个核的第一个任务完整装载 LUT；不依赖跨算子或 runtime PM 后的 LUT 留存。

当前 FP16 LUT writer 在零码附近存在转换异常；本方案用 INT16 writer 绕开。独立路径的夹取、尾部还原和 Q13 解码仍由 CPU 完成，因此不能把独立 NPU 时间当成纯设备流水线或融合时间。

## GEMM 与 bias 融合

归档的 `scripts/archive/npu_gelu_20261018/npu_gelu_fused_probe.cpp` 使用算子库的浮点 GEMM 寄存器生成器，在 GEMM 的 DPU 写回前启用 LUT。每个形状执行 plain/fused/plain，检查融合以及后续普通 GEMM 的状态恢复。

| M/N/K | 无 bias | 标量 bias | 逐通道 bias |
|---|---|---|---|
| 16/32/32 | 通过 | 通过 | 通过 |
| 32/64/32 | 通过 | 通过 | 通过 |
| 128/128/32 | 通过 | 通过 | 通过 |

探针使用可精确计算的稀疏投影，输入范围[-3,3]，逐通道 bias 为不同的 FP32 值，融合输出与数学 GELU 的最大绝对差0.0000571552；输出 guard 和前后普通 GEMM 检查通过。这是连通性与广播验证，不是普通稠密权重 GEMM 的精度或融合性能结果。

逐通道路径：BS `0x4040=0x20150` 开启外部 ALU Add；BRDMA `0x501c=2` 读取 ALU 系数，`0x5020` 指向紧凑 FP32 bias[N]；RDMA `0x5044=0x17850`，关闭主数据读取；PC enable使用0x1d。GEMM主数据直接来自 CORE，BS 加 bias 后经 BN 的4096索引缩放，EW LUT计算，最终 INT16 writer输出。

若某个 GEMM 做 K 分块，非线性必须放在最后一次归约之后：`gelu(sum(partials)+bias)`；不能对每个 partial 分别激活。本次融合只覆盖 K1024/N4096 的 ViT UP，采用完整 K；down 继续使用已有缓存及 K tiling。正式融合接口固定完整 K；其 panel8/N tile 几何独立于普通 GEMM 调优缓存，不冒用普通 GEMM 的计时成绩。

## 融合完整 ViT 对照

下表 CPU/独立列来自三路交错测量，最终融合列来自后续配对测量，均为每条路径六个正式样本的中位数。两轮使用相同频率、资源掩码、checkpoint、图片、调优缓存；不把跨窗口的全部差异归因于单一几何参数。精度只比较 ViT features，与文本量化无关。

| 输入 | CPU GELU / ms | CPU 对 FP32 L2 | 独立 NPU / ms | 独立对 FP32 L2 | 融合 / ms | 融合对 FP32 L2 |
|---|---:|---:|---:|---:|---:|---:|
| 红图256×256 | 507.947 | 1.2706% | 302.074 | 1.7548% | 281.237 | 1.8129% |
| 蓝图384×256 | 770.960 | 0.8710% | 460.186 | 0.9262% | 425.823 | 1.0615% |
| 红图512×512 | 2271.497 | 2.4234% | 1463.143 | 2.8466% | 1358.693 | 2.5484% |

独立方案与 CPU 输出和前轮实验逐 bit 相同。最终融合与其几何/收集对照组也在三个尺寸上逐 bit 相同；每条路径各自的重复输出均一致。融合相对 CPU 路径的 L2 距离为0.9941%、0.9185%、2.3747%；这与各自相对 FP32 gold 的距离是不同指标。

融合对照组采用大 M 时 N tile128及逐行收集，最终版本采用 N tile1376及按panel收集；两项一起比较，不作单因素归因：

| 输入 | 对照融合 / ms | 最终融合 / ms | 对照/最终 | features逐bit相同 |
|---|---:|---:|---:|---|
| 256×256 | 280.480 | 281.237 | 0.997× | 是 |
| 384×256 | 425.773 | 425.823 | 1.000× | 是 |
| 512×512 | 1417.142 | 1358.693 | 1.043× | 是 |

小尺寸的收集顺序变化未见稳定端到端收益；大尺寸不能把 profile 的 `task_m_tile` 当成 N tile。最终512²的阶段中位数：Attention721.549ms、down207.106ms、融合UP+GELU148.589ms。Attention已占约53%的完整时间，后续只改GELU的空间明显减小。

## FLOPs 与有效 FU

按用户指定的3 TFLOPS峰值，FMA计2 FLOPs。以模型需要的 Linear 与非因果 QK/PV 计算量作为分子；padding、K分块额外加法、分母辅助列和其他重复硬件工作不计入模型 FLOPs。GELU、norm、softmax、RoPE等标量 FLOPs未计入，但其耗时仍包含在完整时间或所属阶段时间中。

配置为24层、hidden1024、intermediate4096、16个64维头、merger输出2560，patch输入1536。M为patch数，每张图的主计算量为：

```text
F(M) = 24 × [24 × M × 1024² + 4 × M² × 1024]
     + 2 × M × 1536 × 1024
     + 2 × (M/4) × 4096² + 2 × (M/4) × 4096 × 2560
     = 620756992 × M + 98304 × M² FLOPs
MFU  = F(M) / (完整encode秒数 × 3e12)
```

| 输入 | 模型 GFLOPs | CPU GELU MFU | 独立 NPU MFU | 最终融合 MFU |
|---|---:|---:|---:|---:|
| 256×256 | 165.356 | 10.85% | 18.25% | 19.60% |
| 384×256 | 252.866 | 10.93% | 18.32% | 19.79% |
| 512×512 | 738.734 | 10.84% | 16.83% | 18.12% |

下面是最终融合版本的阶段中位数；每格为「完整阶段毫秒 / 有效FU」。block阶段均累计24层。UP+GELU只以UP GEMM的模型FLOPs作分子；Merger计两次FC计算量，时间包含其norm、分配和GELU。各分项分别取中位数，和总时间存在很小差异。

| 阶段 | 256×256 | 384×256 | 512×512 |
|---|---:|---:|---:|
| Patch投影 | 0.98 / 27.3% | 1.61 / 25.0% | 3.82 / 28.1% |
| QKV | 33.64 / 38.3% | 48.43 / 39.9% | 128.41 / 40.1% |
| Attention整体 | 102.25 / 2.1% | 173.20 / 2.8% | 721.55 / 4.8% |
| Attention输出投影 | 15.35 / 28.0% | 19.29 / 33.4% | 52.62 / 32.6% |
| UP+GELU | 42.98 / 40.0% | 63.57 / 40.5% | 148.59 / 46.2% |
| Down | 57.55 / 29.9% | 79.61 / 32.4% | 207.11 / 33.2% |
| Merger | 4.20 / 27.7% | 5.81 / 30.0% | 14.11 / 33.0% |
| 其余CPU/杂项 | 24.24 / — | 34.21 / — | 83.09 / — |

这不是纯NPU执行窗口的HFU，也不是驱动load：阶段时间包含打包、布局、缓存同步、提交等待、CPU处理及输出收集。Attention用有效QK+PV FLOPs作分子，耗时还包含softmax/LUT及相关处理，现有阶段记录不能分别给出QK核、PV核和softmax的利用率。CPU杂项不使用NPU的3T峰值计算FU。

Attention占完整时间36.4%/40.7%/53.1%，有效FU只有2.1%/2.8%/4.8%，是当前整体MFU的主要拖累；这仍不能单凭阶段时间判断CPU处理、同步或NPU矩阵核各占多少。计算数据和分项FLOPs见 [utilization.json](evidence/npu_gelu_20261018/utilization.json)。

## 融合实现

算子库 `src/rk_npu_mm_gelu.cpp` 复用已有的 `emit_f16_batch_tasks` 生成已支持的 panel8 GEMM 任务，再启用 BS bias 和 LUT。A 为 FP16，bias 为紧凑 FP32[N]。DPU 的处理类型设为 FP32，避免4096倍索引缩放将后层大激活截断；FP32 writer 保留 LUT 的整数码数值，零码也可正确输出。正端使用斜率外推，负端表首项固定为零，使范围外负值输出零；无需 CPU 夹取或尾部修补。

输出 BO 是 panel8/N4 的 FP32 carrier，承载约8192倍的 GELU 值；它并不等于高精度浮点 GELU，也不能据此推断CORE的MAC累加位宽。Rust 以8行 panel 为单位读取该 BO，缩放并直接写入正常 FP16 激活，不另建 FP32 Vec。同步、布局收集和 FP16 转换都计入完整 encoder 时间。每核每次提交的首个任务重新装载 LUT。

融合固定 A/C panel8，N tile 默认按允许核心数均分并向32取整；C7 的 N4096/mask7 得到1376，硬件 M 任务由原 CBUF 规则生成。公开配置可显式指定 N tile；当前未做融合专用自动调优。普通 Linear profile 的 `task_m_tile` 和缓存成绩不代表融合路径。

融合与独立方案按 `CPU / 独立 / 融合 / 融合 / 独立 / CPU` 测量；另对融合任务几何与收集顺序做 `before / after / after / before` 配对，每条路径各六个正式样本。保存每次进程的 features，并检查同一路径重复输出以及布局调整前后的逐 bit 一致性。完整耗时包括输出 BO 同步与 Rust 收集，不包含权重上传。

独立的稀疏权重检查覆盖 M16/128/256/1024、N32/128/4096、K32/1024，并加入±65504、零点、输入取反及输出 guard。中央区间对数学 GELU 的最大绝对差约0.000934；大值使用 FP16 舍入误差界检查。完整 ViT 再以真实稠密权重检验重复性及最终特征。

融合同时去掉了原 UP 输出和加 bias 后的 FP16 写回，不能把最终特征变化全部归因于 LUT。精度参考仍是同 C7 权重、同 patches、禁用 TF32/autocast 的 [PyTorch FP32 gold](VISION_FP16_PRECISION.md)。

直接 FP16 LUT writer 的零码异常仍存在；独立 DPU 的 FP32→FP16 任务还存在与消费者之间的 surface 布局不匹配，因此当前性能实现使用 Rust 收集。输出格式诊断已放入实验归档，不作为可用推理分支。

原型只覆盖 M 为8的倍数且不超过1024；正式接口对 M 向8、N/K 向32补零，Rust 收集器跳过 padding。局部和最终特征误差不等于视觉任务准确率；按本次选定取舍，正式默认启用融合，不保留 CPU FFN 开关。

正式 `rk_npu_mm_gelu_run_weights` 复用 `rk_npu_mm_tuned_weights` 中已有的完整 K 常驻权重，不重新上传或保留原始权重副本。Rust FloatBackend 按 (K,N) 共享工作区，仅保留当前 M；24层 UP 共享输入、输出、bias 和命令 BO，运行时重绑定权重地址和 bias。借用锁覆盖 run 和 Rust 收集，防止视图被下一层覆盖。FP32 存储模式也使用同一融合算子，输出直接收集为 FP32；该模式不能复现已删除的 CPU FFN 路径。

## 复现与证据

历史源码、快照生成器、动态库桥接、旧 CPU 对照和失败格式探针统一在 `scripts/archive/npu_gelu_20261018/`，依赖冻结基线，不再提供 Cargo example。以下为当时的历史命令（需在隔离的冻结 checkout 恢复归档前路径）；当前正式回归使用算子库 `test_mm_gelu --board`、Rust `fused_gelu_collects_tails_and_reuses_resident_weights` 和 `tests/board/vision_fused_regression.py`。NPU 寄存器实现使用 C++，CPU 布局收集与 FP16 转换使用 Rust。

```sh
g++ -O3 -DNDEBUG -fPIC -shared -std=gnu++17 -march=armv8.2-a+fp16 -mtune=cortex-a76 \
  -Inative/rk_npu_matmul/include -Inative/rk_npu_matmul/src \
  tests/board/npu_gelu_bridge.cpp build/fp-mm-tune/librk_npu_matmul.a \
  -Wl,--exclude-libs,ALL -Wl,-Bsymbolic -fopenmp -ldl -lpthread \
  -o logs/npu-gelu-bridge/libnpu_gelu_parallel.so
cargo build --release --locked --example npu_gelu_bench
python3 scripts/run_board.py --wait-lock --report logs/gelu-clocks.json -- \
  taskset -c 4-7 target/release/examples/npu_gelu_bench \
  --bridge logs/npu-gelu-bridge/libnpu_gelu_parallel.so --output logs/gelu.json
python3 tests/board/build_npu_gelu_snapshot.py --output logs/npu-gelu-snapshot-new
g++ -O3 -Wall -Wextra -Werror -std=gnu++17 -march=armv8.2-a+fp16 -mtune=cortex-a76 \
  -Inative/rk_npu_matmul/include -Inative/rk_npu_matmul/src \
  scripts/archive/npu_gelu_20261018/npu_gelu_fused_probe.cpp build/fp-mm-tune/librk_npu_matmul.a \
  -fopenmp -ldl -lpthread -o logs/npu-gelu-bridge/fused_bias_probe
python3 scripts/run_board.py --wait-lock --report logs/gelu-bias-clocks.json -- \
  logs/npu-gelu-bridge/fused_bias_probe 2

g++ -O3 -DNDEBUG -Wall -Wextra -Werror -fPIC -shared -std=gnu++17 \
  -march=armv8.2-a+fp16 -mtune=cortex-a76 \
  -Inative/rk_npu_matmul/include -Inative/rk_npu_matmul/src \
  tests/board/npu_gelu_fused_bridge.cpp build/fp-mm-tune/librk_npu_matmul.a \
  -Wl,--exclude-libs,ALL -Wl,-Bsymbolic -fopenmp -ldl -lpthread \
  -o logs/npu-gelu-bridge/libnpu_gelu_fused.so
python3 tests/board/build_npu_gelu_fused_snapshot.py --output logs/gelu-fused-new
CARGO_TARGET_DIR=target/float-autotune cargo build \
  --manifest-path logs/gelu-fused-new/Cargo.toml --release --locked --example vision_profile_fused
python3 scripts/run_board.py --wait-lock --report logs/gelu-fused-clocks.json -- \
  taskset -c 4-7 target/float-autotune/release/examples/vision_profile_fused \
  --model MODEL --image IMAGE --vision-dtype fp16 --tune-cache CACHE --output logs/gelu-fused-result
```

隔离快照只覆盖 FFN 近似 GELU；独立/融合的编译产物使用专用名称，不覆盖正式 `vision_profile`。三路完整 ViT 和融合配对工具分别为 `npu_gelu_fused_vit.py`、`npu_gelu_fused_compare.py`，原始报告、逐样本时间及运行脚本保存在 [本轮证据目录](evidence/npu_gelu_20261018/)。性能对照、有效融合路径的正确性检查、release Clippy 均通过；频率上下限/governor恢复，`restore_errors=[]`。C++桥接通过 `-Wall -Wextra -Werror` 编译；输出格式诊断的数值不匹配按归档中的限制处理。


## 正式接入的验证

算子库签名提交 `a9a2e06`，外层使用正式 FloatLinear ABI。release Clippy 全targets `-D warnings`、fmt、70项默认Rust测试和20项native CPU CTest通过；默认跳过的10项硬件测试中，本轮显式执行新增融合测试通过。正式native板端检查覆盖M3/129/260、N35/160/4096、K37/96/1024、mask1/2/4/3/7、权重/bias重绑定、±65504、零点和物理padding。Rust检查两种输出存储、N尾部、M尾部以及共享工作区切换层后的输出复现。全部使用FP16权重，没有运行BF16 CPU矩阵计算。

完整ViT与冻结融合原型按before/after/after/before配对；每个进程预热一次、正式三次，每条路径每尺寸六个样本。时间不包含图片预处理、模型加载或文本解码。精度以同一PyTorch FP32 gold为参照；下表不是与旧CPU GELU重新配对的测量。

| 输入 | 冻结融合原型 / ms | 正式共享工作区 / ms | 正式对FP32 gold L2 | 与原型逐bit相同 |
|---|---:|---:|---:|---|
| 256×256 | 279.682 | 283.253 | 1.8129% | 是 |
| 384×256 | 428.865 | 427.977 | 1.0615% | 是 |
| 512×512 | 1372.570 | 1384.314 | 2.5484% | 是 |

三档正式结果精度与原型完全一致。完整时间差为+1.28%、−0.21%、+0.86%，本轮没有观察到正式接入带来的额外端到端加速；共享权重/工作区的目的是避免原型按层重复分配和上传，不将这组小幅时间差归因于单一因素。

手动`--vision-dtype fp32`在256²上通过有限值及重复输出检查，相对FP32 gold的L2为0.9253%；这仍是FP16 NPU GEMM＋融合FFN，不能当作全精度计算。默认参数另外覆盖352×224图片、14×22网格（308个patch，M不是8的倍数），完整encode重复输出逐bit相同；该尺寸只验证运行与重复性，没有新的FP32 gold。所有runner返回0，CPU/NPU/DDR频率上下限及governor恢复，`restore_errors=[]`。正式样本、源码/二进制/输出hash及验证记录见[formal证据](evidence/npu_gelu_20261018/formal/)。

## Attention 路由检查

只读快照在256²、384×256、512²上记录到96/144/384个外层调用（含两次 encode）；全部 `cache_lut=1`、`can_lut=1`、`lut=1`、`mask_mode=NO_MASK`、`flags=STABLE_SOFTMAX`。最大 V 绝对值8.4141，低于该实现的63.90484回退阈值，未触发 CPU exp 回退。无 mask 会省去 mask BO 与读取，但 generic query 内部仍无条件按32行切块，即外层128行调用拆成4次。该项仅诊断，未混入融合 GELU 提交；不能从静态分块直接推断改成128行后的收益。探针 stderr 同步输出显著干扰耗时，因此只用来验证路由，不用它的 encoder 时间做性能结论。
