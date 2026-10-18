# FP16 / BF16 静态矩阵乘自动调优

2026-10-18。公共接口见 `native/rk_npu_matmul/include/rk_npu_mm_tune.h`，实现复用 `rk_npu_mm` 执行器与 W8/W4 的测量、排名和原子缓存工具。

当前仅保留禁用 K tiling 和 FP16 partial + NPU FP16 reduce 两种配置，默认启用后者。已移除的 CPU 归约、FP32 partial/归约和 GEMM+EW 融合只保留历史实验记录。六种旧路径的完整 ViT 耗时与精度偏移见 [性能与精度总表](FLOAT_MM_REDUCTION_RESULTS.md)，融合的 task 几何对照见 [归约路径比较](FLOAT_MM_REDUCTION_COMPARISON.md)。

## 支持范围

普通静态权重 MM：紧凑行主序 A[M,K]、B[K,N]、C[M,N]，原生 FP16 或 BF16 运算。HOST_STORAGE 接受相应精度的 uint16 位模式，可选 FP32 bias，结果在加bias后舍入回目标精度；HOST_F32 包含 FP32 输入转换、FP32 输出和可选 FP32 bias。BF16 使用原生 BF16 MAC 与存储，不经过 FP16。

FP16 搜索9种完整 A/D 配方：normal/normal、normal/native、native/normal、native/native、panel8/native、panel16/native、normal/panel8、panel8/panel8、panel16/panel16。panel 指每8/16行一个局部通道块，FP16 原子为8个元素；输入 pack、输出 collection、CNA/DPU 寄存器和 task 地址偏移同步匹配该布局。

host 物理 M 搜索实际逻辑 M 及16/32/64/96/128/192/256/384/512/768/1024中小于 M 且不小于 `max(16,M/8)` 的值；N tile 搜索0/128/256/512/1024/2048及按允许核数平分的32对齐值。另细搜硬件 task M 上限0/8/16/32/64/96/128/192/256，以及按允许核数的一至四倍分组均分物理 M 得到的候选，0表示按 CBUF 推导；这与 host 物理 M 分块是两个参数。panel task 保持8/16行对齐，并受对应 CBUF 和 stride 范围约束。当前 panel8 A 的 K≤1024、panel16 A 的 K≤2048；normal A + panel8 D 可用于 CBUF 能容纳完整 panel 的更大 K。

搜索先用完整调用扫描布局/M/N/K，每种布局与 K partition 的优胜配置再用用户指定的 warmup/loops/repeats 细测 task M 及两种 host IO：分阶段转换、转换与 device pack/输出收集+bias 融合。融合 IO 当前用于完整 K 路径。最终 fastest/stable 来自细测。只去重相同 cap 值，不再因 task 数量相同而删除候选：160+96与128+128同为两个 task，但负载平衡不同。M128 native/native、Ntile0、默认 task M、分阶段 IO 始终参与正式细测作为基线。尾部补零并返回紧凑逻辑输出；NPU mask、CPU affinity 和线程数固定。

BF16 保留原有 normal/native A/D、N tile 与 M32/64/128/256/512 搜索和缓存 revision1。按用户要求，本轮不运行 BF16 CPU 数值计算。FP16 使用 revision7；当前库只读取当前格式，不再保留实验 revision2–6 的兼容分支。清理前已将本机验证通过、数学路径兼容的旧策略一次性转存为当前格式。其他机器的旧缓存按 miss 处理，可用显式调优重建；普通推理只读缓存，不迁移、不搜索、不写入。

每次排名测量所选 host 类型的完整逻辑调用：HOST_F32 包含 FP32→目标精度及 FP32 widening；HOST_STORAGE 直接读写目标精度。两者均包含尾块填充、pack A、cache sync、submit/wait、unpack和可选bias。workspace 创建和权重打包上传单独计时。搜索返回最快及低抖动两个策略；普通调用不分配缓冲、不读环境、不搜索策略。按用户要求，已删除正常 MM 执行中的逐元素 Inf/NaN 扫描；显式调优器仍在计时之外用独立 oracle 检验输出。

## 两种配置与精度对照

| `RK_NPU_MM_K_TILING` | 行为 |
|---|---|
| 不设置、`fp16`、`on` 或 `1` | 默认启用 K tiling，分块结果为 FP16，在 NPU 上用 FP16 Add tree 归约 |
| `off` 或 `0` | 禁用 K tiling，使用完整 K 数值路径，适合精度/一致性对照 |

只保留这个环境开关。归约算法固定为 NPU FP16，没有其他算法选择入口；不增加模型选项或 CLI 参数。环境只在准备策略/权重时读取，修改后重新启动推理进程。两种配置使用独立的 cache identity。普通推理只读 profile，miss 时仍使用合法的完整 K 基线，不会在线调优。

默认启用表示允许搜索和使用 K 分块，并非强制所有 shape 分块：仅当 K>1024 时搜索 Ktile256/512/1024/2048，以及向上32对齐的 K/2、K/3；去重后保留小于 K 且不超过32个 partial 的项。调优仍可选择更快的完整 K 策略。partial 数与允许核数独立，每个 partial 可继续按 M/N 在允许核上调度。

K 分块的调优、常驻权重、partial 调度和 NPU reduce 均在算子库内部。packed partial 留在设备内，CPU 只同步和收集最终结果。每块 FP16 输出和逐层 FP16 Add 都可能产生舍入差异，因此默认加速路径不保证与完整 K 逐位一致。需要检查精度时，保持相同模型、输入和其他运行选项，手动关闭 K tiling：

```sh
RK_NPU_MM_K_TILING=off <原推理命令>
# 也可用 RK_NPU_MM_K_TILING=0
```

前面的图文实验 top1 全部不变，但 features/logits 相对原路径存在偏移。关闭 K tiling 用于测量这一偏移；完整 K FP16 本身不是全 FP32 精度金标准。

收敛后的回归、两模式缓存及红/蓝图输出复核见 [两种配置验证记录](evidence/fp16_two_modes_20261018/README.md)。

当前接口聚焦 batch=1 的静态 MM；BMM、外部 device buffer 与动态权重仍通过原有 raw MM API 执行。

## 缓存与正确性

缓存区分逻辑 M/N/K、运算精度、host IO/bias、CPU/NPU mask、K tiling policy、warmup/loops/repeats/timeout、驱动与调度 revision；选中策略还保存物理 M、硬件 task M、布局、N/K tile、融合 IO、核配置和完整调用代价。改变布局或 M 后新建匹配的 workspace。raw MM 会拒绝不兼容的 dtype/K/N/split partition/context/domain 权重。

`rk_npu_mm_tuned_weights_create` 由算子库创建常驻权重：加载完整 K 基线，以及同一 K/N/resource family 先前调优选中的 K partitions。显式调优保存 shape profile 时，同时保存与 M 无关的小型 partition 标记；正常推理只读这些记录。常驻对象不保留原始 host B，运行时不重扫或重打包权重。不同 M/layout/N tile 复用对应 partition；模型已加载后才出现的新 partition 使用合法基线，重载权重后方可使用新选择。每个已选 partition 占一份相应 packed B 内存。

浮点权重不影响溢出分区或代码路径，因此 profile 按 shape/resources 共享，不扫描权重建立身份。调优输入为目标精度舍入后的确定性合成值。独立 CPU oracle 通过8个按行变化的系数与非周期 B 生成完整输出；每次计时调用后检查所有输出。FP16 容差为 `0.003 + 0.005*abs(ref)`，BF16 为 `0.015 + 0.02*abs(ref)`。板端测试另使用独立随机输入与逐项 dense CPU 点积。submit 或设备错误立即终止搜索。

`rk_npu_mm_autotune_cache_load` 只读缓存，无设备工作与写入；缺失、过期或无效条目返回 `RK_NPU_ERR_CACHE_MISS`。`rk_npu_mm_autotune_cached` 显式允许搜索并保存。调用方负责应用 strategy 的 CPU affinity/threads；调优函数退出时恢复调用线程及参与的 OpenMP workers 的 affinity、线程数和 dynamic 配置。

## Rust 与 ViT

普通使用不必运行独立测试工具。正式 CLI 入口为：

```sh
open_rknpu2_lm_infer -m MODEL tune --vision
open_rknpu2_lm_infer -m MODEL tune --vision --image photo.jpg --image other.png --report vision-tune.json
```

不带图片时准备 M256/384/1024；带图片时按模型预处理后的 patch 数去重，自动生成七类投影及 merger 的 M/4。维度来自模型配置，不硬编码 C7 的 hidden/intermediate/out 大小；也不加载 checkpoint 张量或文本权重。`--vision-dtype` 决定 HOST_STORAGE/FP16 或 HOST_F32 对照缓存。已有条目直接复用，只有显式 `tune` 搜索 miss；`--batches` 仍用于文本调优，与 `--vision` 互斥。详细候选日志仅在 debug 日志级别启用。

`FloatBackend::new` 默认 FP16运算/FP32 host IO，`FloatBackend::bf16` 使用原生 BF16运算/FP32 host IO。`FloatBackend::fp16_storage` 直接读写 FP16 激活并在加bias后写回FP16；当前 ViT 使用这一入口。`FloatBackend::prepare(k,n,rows)` 和 `FloatLinear::prepare(rows)` 遵循现有 `ModelOptions`：

- `tune=false`：固定 M128 基线。
- `tune=true, tune_search=false`：只读 profile；miss 使用基线。
- `tune=true, tune_search=true`：显式准备时搜索 miss 并保存。

`FloatLinear::run` 使用 FP32 host IO；`FloatLinear::run_half` 使用 FP16 host IO。两者使用准备好的策略，通过一个 `rk_npu_mm_tuned_run_weights` 调用完成计算。Rust 不管理 K 循环、partial 或权重分区。视觉 Linear 自动读取 `<tune-cache>/floating/` 下的相应 FP16 storage profile，无需改变模型权重。host 类型参与缓存身份，因此旧 FP32 host profile 不能视为新 FP16 storage 的调优结果。workspace 按 K/N 共享，只保留每个 K/N 最近的逻辑 M，限制非方图形状变化导致的常驻内存增长。Qwen3.5 量化 decoder 与 Attention 调优保持各自入口。

ViT默认`--vision-dtype fp16`；手动`--vision-dtype fp32`恢复FP32中间张量和CPU计算，使用`FloatBackend::new`/HOST_F32的独立profile，NPU运算仍是FP16。参数适用于模型CLI、服务启动以及vision probe/profile；Rust使用`ModelOptions::vision_dtype`。没有对应环境变量。切换张量类型不改变K tiling算法，`RK_NPU_MM_K_TILING`继续独立控制是否允许FP16分块＋FP16 NPU归约。

ViT 的 hidden、normalized、QKV/Q/K/V、Attention输出、MLP中间结果和merger中间结果均使用FP16存储。CPU LayerNorm 的均值/方差使用FP32；Attention保留原来的FP32 CPU max、分块累加及归一化，通过新增的`rk_npu_attention_f16_run_half`直接写FP16结果。GELU、残差和位置/RoPE也写FP16。图片patches只在入口转换一次，最终features为兼容语言模型接口扩为FP32一次；正常执行不生成层间整块FP32激活。

21个HOST_STORAGE profile已完成显式搜索和只读复核。256²/384×256/512²完整暖态encode为490.609/748.705/2238.948ms，较同轮原FP32激活实现降低20.04%–22.56%时延；features和图文logits存在偏移，详见[FP16存储性能与精度报告](VISION_FP16_STORAGE.md)。下文revision2等表格保留历史结果，不代表当前FP16激活实现。

在 RK3588 上构建并调优，不需要加载 checkpoint：

```sh
cargo build --release --locked --example float_tune
python3 scripts/run_board.py --wait-lock --report results/float-tune-clocks.json -- \
  target/release/examples/float_tune --vision --rows 256,384,1024 --fp16-storage \
  --tune-cache .cache --report results/float-profiles.json
```

`--vision` 包含交接文档中的七种 C7 投影；merger 的逻辑 M 自动除以4。单个分辨率有7类 Linear shape，24个 Transformer block 共享其中4类 profile；21个调优 case 指 M256/384/1024 三档合计，并非每层分别调优。非方图使用实际 `grid_h*grid_w`，不会按图像 token 数误建 profile。当前 ViT 使用`--fp16-storage`调优的条目。普通 FP32 host IO matmul 使用 `--k K --n N --rows M`；加入 `--bf16` 选择 BF16，或加入`--fp16-storage`选择FP16输入输出；`--load-only` 验证读取现有条目、并报告 miss 使用的基线。之后视觉命令使用相同 `--tune-cache`、CPU/NPU mask 即可。常规文本 `tune` 命令保持文本形状准备语义。

默认准备启用 K tiling 的 profile：

```sh
target/release/examples/float_tune --vision --rows 256,384,1024 --fp16-storage --tune-cache .cache
```

如需准备完整 K 对照，给该命令以及后续推理命令都设置 `RK_NPU_MM_K_TILING=off`。两种配置分别保存与读取 profile；`--load-only` 可以确认是否命中，缺失时使用基线。

板端测试：

```sh
cmake -S native/rk_npu_matmul -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --target test_mm_tune test_mm_header_c -j4
build/native/test_mm_tune --cpu
python3 scripts/run_board.py --wait-lock --report results/float-tests-clocks.json -- \
  taskset -c 4-7 build/native/test_mm_tune --fp16
python3 scripts/run_board.py --wait-lock --report results/float-panel-clocks.json -- \
  taskset -c 4-7 build/native/test_mm_tune --panel-fp16
python3 scripts/run_board.py --wait-lock --report results/float-k-clocks.json -- \
  taskset -c 4-7 build/native/test_mm_tune --k-tiles
cargo test --release --locked fp16_linear_and_noncausal_attention_match_cpu -- --ignored --nocapture
```

通用调优的策略与缓存 API、公共 C 头可直接用于其他 C/C++ 调用方，Rust 仅通过公共 C ABI 接入。

低利用率的完整 K 对照、CBUF/task 几何与24层阶段计时见 [MFU 诊断](FLOAT_MM_MFU_DIAGNOSIS.md)。

## 扩展版 revision2 定频算子结果

FP16 在主板 RK3588、CPU4/mask `0xf0`、NPU mask `7`、大核2.4GHz、NPU1GHz、DDR2112MHz 下完成21个视觉形状的搜索。按用户要求，本轮跳过 BF16 搜索与 CPU 数值计算。每个 shape 分别对固定 M128 基线和上一版已选策略做3组同进程 ABBA，每条路径6个完整调用，取中位数。完整调用含 FP32 host IO、FP32 bias、pack/collection、同步和 submit/wait。

下表为本轮重测的固定基线→扩展版毫秒数，括号为加速比；merger 实际 M 为表头的四分之一。原始候选、正式样本、缓存与复核见 [扩展版证据](evidence/fp16_panel_autotune_20261018/README.md)。

| 投影 K×N | 逻辑 M256 | 逻辑 M384 | 逻辑 M1024 |
|---|---:|---:|---:|
| patch 1536×1024 | 2.363→1.007 (2.35×) | 3.570→1.567 (2.28×) | 9.542→4.009 (2.38×) |
| QKV 1024×3072 | 2.894→1.589 (1.82×) | 4.246→2.425 (1.75×) | 11.178→6.232 (1.79×) |
| Attention output 1024×1024 | 1.344→0.723 (1.86×) | 2.022→1.045 (1.93×) | 5.400→2.828 (1.91×) |
| MLP up 1024×4096 | 3.681→2.018 (1.82×) | 5.491→3.078 (1.78×) | 14.187→7.873 (1.80×) |
| MLP down 4096×1024 | 7.192→5.933 (1.21×) | 10.785→8.306 (1.30×) | 28.459→22.543 (1.26×) |
| merger FC1 4096×4096 | 10.704→4.922 (2.17×) | 11.023→7.220 (1.53×) | 22.607→19.874 (1.14×) |
| merger FC2 4096×2560 | 6.889→3.539 (1.95×) | 7.217→5.122 (1.41×) | 14.836→13.118 (1.13×) |

相对固定基线，21个形状的加速比为1.131–2.380×，几何平均1.699×。相对上一版已调优策略，同进程配对测量的加速比中位数1.483×、几何平均1.335×，范围0.996–1.708×。其中 M64 merger FC2 未观察到收益，新版慢0.36%；其他形状的额外收益见 [summary.json](evidence/fp16_panel_autotune_20261018/summary.json) 中的 `speedup_vs_previous`。

最终21个策略中，17个选中 panel A 或 D，20个选中融合 IO，2个选中显式 task M128；其余使用 CBUF 默认 task M。M96 merger FC2 选择分阶段 IO。布局、融合和 task M 都由完整调用实测决定。所有21组完整 FP32 输出相对固定基线均为 `max_abs_delta=0`。

## 完整 ViT、图文核对与 MFU

扩展版 Rust 进程只读命中全部21个 profile：`tune_cache_hits=21`、`tune_cache_misses=0`、`tune_calls=0`。红图与长方形蓝图的完整24层 ViT features，相对调优前保存的基线逐位一致。单红图、单蓝图和双图的固定 continuation 共10行、2,483,200个 logits 全部逐位一致，top1 相同、KL=0。

以下为加载好权重后，首次未命中特征缓存的单次完整 `Encoder::encode` 测量；包含 patch embedding、24层 Transformer、merger、CPU 处理、workspace 准备和同步，不含图片解码/resize、模型加载和语言模型。两档输入分辨率按宽×高表示。仅有单次 encoder 样本，暂不据此声称端到端配对加速比；M1024 本轮完成的是算子测量。

| 输入分辨率 | patch 数 M | merger 后 tokens | ViT 耗时 | 模型计算量 | 有效算力 | 按3 TFLOPS计 MFU |
|---|---:|---:|---:|---:|---:|---:|
| 256×256 | 256 | 64 | 704.4ms | 165.356 GFLOPs | 0.23475 TFLOPS | 7.825% |
| 384×256 | 384 | 96 | 1072.3ms | 252.866 GFLOPs | 0.23582 TFLOPS | 7.861% |

FMA 按2 FLOPs 计算；计入 patch Linear、24层全部 Linear、QK/PV 和两层 merger，忽略 norm/激活等少量 FLOPs。令 H=1024，计算式为 `F(M)=2*M*1536*H + 24*(24*M*H*H + 4*M*M*H) + 2*(M/4)*(4096*4096 + 4096*2560)`；`MFU=F/(encode_seconds*3e12)`。这是端到端编码口径，NPU 内核自身的利用率需另行计时。精确计算值见 [measurement.json](evidence/fp16_panel_autotune_20261018/measurement.json)。

扩展版的 Release fmt、全 target Clippy（`-D warnings`）、默认62项测试、FP16 Linear/NO_MASK Attention 板端测试通过。native FP16 调优/缓存/affinity/split 权重兼容性用例通过。新增576组 panel/task/IO dense CPU oracle 检查全部通过，覆盖 storage、FP32 host IO+bias、九种布局、task cap、N tile、M129 尾部填充与 K97 非整齐 K；其中108组专门覆盖 K97。公共 C 头编译通过。FP16 使用新增的 `rk_npu_mm_plan_create_tiled` 与 `rk_npu_mm_run_packed_b_f32`，既有 raw MM descriptor/options 和工厂 API 的 ABI 保持原样。

最终 validation、panel、performance、model 四份定频 runner 均 `exit_code=0`、`restore_errors=[]`，并逐项核对 CPU/NPU/DDR 最小/最大频率及 governor 恢复。最初搜索 runner 在21个形状完成后，因运行中的 shell 文件被修改而在后续步骤返回127；硬件恢复成功。原始搜索记录已保留，后续独立脚本补齐模型检查，并重新完成全部21组正式 ABBA，详见证据目录说明。扩展版本轮完整数组和冻结 Rust 二进制位于 `/home/zt/mnt/3090x4/tmp/floating-mm-panel-20261018/`。

## 初版 revision1 定频算子结果

主板 RK3588，CPU 4线程/mask `0xf0`、NPU mask `7`，大核2.4GHz、NPU1GHz、DDR2112MHz。以下为相同通用执行器的完整 FP32 host IO + FP32 bias 调用；固定基线使用 M128/native A/native C/Ntile0，调优默认不分 K。每个 shape 预热后做3组 ABBA，每条路径6个独立调用，取中位数；原始值保存在 [evidence/floating_mm_autotune_20261018/](evidence/floating_mm_autotune_20261018/)。

表内为基线→调优的毫秒数，括号内为加速比。merger 的实际 M 分别为64/96/256。

| 投影 K×N | 逻辑 M256 | 逻辑 M384 | 逻辑 M1024 |
|---|---:|---:|---:|
| patch 1536×1024 | 2.365→1.529 (1.55×) | 3.563→2.445 (1.46×) | 9.370→6.009 (1.56×) |
| QKV 1024×3072 | 2.810→2.389 (1.18×) | 4.194→3.548 (1.18×) | 11.001→9.278 (1.19×) |
| Attention output 1024×1024 | 1.340→1.246 (1.07×) | 2.019→1.665 (1.21×) | 5.305→4.588 (1.16×) |
| MLP up 1024×4096 | 3.611→2.974 (1.21×) | 5.422→4.406 (1.23×) | 14.212→11.564 (1.23×) |
| MLP down 4096×1024 | 7.109→6.906 (1.03×) | 10.720→9.881 (1.08×) | 28.184→26.775 (1.05×) |
| merger FC1 4096×4096 | 10.646→5.292 (2.01×) | 10.994→7.799 (1.41×) | 22.451→21.084 (1.06×) |
| merger FC2 4096×2560 | 6.848→3.489 (1.96×) | 7.146→5.151 (1.39×) | 14.768→13.991 (1.06×) |

原生 BF16 两组完整调用：M256/K1024/N3072 为3.436→2.993ms（1.15×），M129/K4096/N1024 为7.572→4.857ms（1.56×）。23个用例的完整 FP32 输出相对 M128 基线均为 `max_abs_delta=0`，独立 CPU 点积抽查也通过；调优过程中每个计时输出另经完整 synthetic oracle 检查。MLP down 收益较小，整体收益随形状变化。

21个 FP16 视觉 profile 在独立 Rust 进程中只读全部命中：`tune_cache_hits=21`、`tune_cache_misses=0`、`tune_calls=0`。红图 M256 与长方形蓝图 M384 的完整24层 ViT features，相对调优前保存的固定基线和本轮 M128 基线均逐位一致；相对同权重 FP32 reference 的 relative L2 为0.5928%和0.4278%。算子加速比仅描述静态 Linear 完整调用。

## 初版回归与证据

单图红、单图蓝、双图固定 continuation 共10行、2,483,200个图文 logits，调优前后逐位一致，top1 相同且 KL=0。features 与 logits 比较见 [summary.json](evidence/floating_mm_autotune_20261018/summary.json)；历史汇总入口为 `scripts/archive/fp16_vit_20261018/compare_float_autotune.py`，版本依赖见[实验归档](FP16_VIT_EXPERIMENT_ARCHIVE.md)。完整 FP32 数组、模型和受测二进制保留在共享目录 `/home/zt/mnt/3090x4/tmp/floating-mm-20261018/`。

Release 格式检查、全 target Clippy（`-D warnings`）、默认62项测试均通过，9项板端用例按默认设置忽略。另显式运行浮点缓存复用、视觉 Linear/NO_MASK Attention 两项板端 Rust 测试；C ABI 用例覆盖 FP16/BF16、mask1/2/4/3/7、M3/M129、缓存命中/失效、CPU/OpenMP affinity 恢复、split partition 不兼容拒绝和正确 split 权重的数值。raw MM 既有 FP16/BF16 回归与 BF16 超出 FP16 范围的精确值检查通过。公共 C 头编译通过。

两次正式 runner 均 `exit_code=0`、`restore_errors=[]`，并逐项核对 CPU/NPU/DDR 最小/最大频率与 governor 恢复。记录见 [validation-clocks.json](evidence/floating_mm_autotune_20261018/validation-clocks.json) 与 [performance-clocks.json](evidence/floating_mm_autotune_20261018/performance-clocks.json)。本轮使用独立 `CARGO_TARGET_DIR=target/float-autotune`；可运行二进制位于 `target/float-autotune/release/examples/`。

修改分布在父仓库的 Rust wrapper/FFI/example 和 `native/rk_npu_matmul` 子模块的公共 C API、实现、CMake 与测试中；既有视觉源码和其他未提交改动保留在当前工作树。
