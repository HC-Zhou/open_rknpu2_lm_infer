# ViT FP16 激活存储

2026-10-18。用户确认采用“FP16 张量，LayerNorm/softmax 内部保留 FP32 累加”。实现、显式调优和三分辨率完整 ViT 复测已完成；精度有可测偏移，不能宣称精度不变。原始记录见 [最终数据](evidence/vision_fp16_storage_rotary_20261018/summary.json)。

## 运行参数

默认使用FP16中间张量，FFN已改为NPU GEMM/bias/GELU融合，旧CPU近似GELU已删除。[融合实现与最新验证](NPU_FFN_GELU_RESEARCH.md)。下文性能表与精度消融为融合前冻结数据。需要数值对照时，通过模型参数切换ViT的中间张量存储精度：

```sh
open_rknpu2_lm_infer -m MODEL run --image IMAGE --vision-dtype fp32 -p 'Describe the image.'
# 默认值，也可显式指定：--vision-dtype fp16
```

`--vision-dtype`同样适用于`chat`、`serve`、`vision_probe`与`vision_profile`；服务在启动时选择，实例加载后固定。Rust调用方使用`ModelOptions::vision_dtype`，可选`VisionDtype::Fp16`（默认）或`VisionDtype::Fp32`。没有对应环境变量。

FP32模式使用FP32层间张量、LayerNorm、RoPE和残差计算；FFN仍使用融合GEMM/bias/GELU，并直接收集为FP32。普通Linear的bias相加和merger erf GELU使用FP32，NPU GEMM、Attention的QK/PV仍使用FP16；它不是下文独立PyTorch全FP32参考，也不复现已删除的CPU FFN。两种模式共用同一执行流程，分别使用HOST_STORAGE和HOST_F32的profile/常驻权重。缓存未命中时采用合法fallback，不在线调优；显式调优FP32 host profile时不带`float_tune --fp16-storage`。

此参数与`RK_NPU_MM_K_TILING`独立：两种张量模式的普通Linear默认都允许FP16 partial＋NPU FP16 Add tree；融合FFN UP固定完整K，非线性不能作用于未归约的partial；需要完整K对照时仍可设置`RK_NPU_MM_K_TILING=off`。`vision_dtype`会写入运行统计和probe/profile报告。

融合前参数切换的冻结验证：三种尺寸的默认FP16、手动FP32输出分别与各自冻结参考逐位一致；显式FP16和两种dtype的完整K对照也一致。两种host profile均21 hits、0 miss、0 search。记录见[参数验证](evidence/vision_dtype_20261018/README.md)。

## 已实现

- ViT 的 hidden、normalized、QKV/Q/K/V、Attention结果、MLP和merger中间结果统一为紧凑FP16位模式。
- `FloatBackend::fp16_storage` / `FloatLinear::run_half` 使用公共MM的HOST_STORAGE；库支持FP32 bias在相加后舍入回FP16，不创建整块FP32输出。
- `rk_npu_attention_f16_run_half` 复用同一Attention执行器与FP32 CPU归约，直接写FP16结果。
- CPU LayerNorm的均值/方差使用FP32；FFN GELU融合到NPU GEMM与bias，Rust直接收集FP16；merger erf GELU在Rust中计算。残差、位置和RoPE写回FP16。
- 图片patches在入口转换一次；最终features为兼容语言模型的embedding接口扩为FP32一次。开启trace时另为诊断导出FP32，不影响普通执行。
- RoPE按图片预计算FP32 sin/cos表，跨head和24层复用；CPU NEON转换/旋转后写回FP16。与首版标量FP16 RoPE的三档features和10行logits均逐位一致。
- 已按要求删除正常MM执行中的逐元素Inf/NaN扫描。显式调优仍在计时外用独立oracle验证输出。
- FP16调优revision7，host类型参与缓存身份；实验旧格式的兼容读取已删除，本机兼容profile在清理前已一次性转存。默认仍启用K tiling，固定FP16 partial + NPU FP16 Add tree；`RK_NPU_MM_K_TILING=off`仍是完整K精度对照入口。

当前 CPU 实现在 `src/ops/vision_half.rs` 与其子模块中，使用 Rust NEON/内联汇编，归约和 FMA 运算顺序显式保留；并行计算复用模型自身的 Rayon 线程池。两份曾用于 ViT 的 C++ kernel 及其 FFI 已删除。下表是迁移前的历史测量，Rust 迁移另以同一当前算子库的冻结二进制做 features 与定频性能配对，不与上游 Attention LUT 修改前的二进制混比。

## Rust CPU 实现迁移复核

2026-10-18，算子库两侧均为 `5b4e39c`，使用相同 HOST_STORAGE 缓存、CPU4/`0xf0`、NPU mask7、大核2.4GHz、NPU1GHz、DDR2112MHz。每档按 C++→Rust→Rust→C++ 执行四个独立进程；各进程冷编码一次、预热一次、正式三次，始终清空 features cache，以下取各路径六个完整暖态 encode 的中位数。两次同实现进程分别核对 features 一致，再比较两种实现。

| 输入宽×高 | C++ CPU kernel | Rust CPU kernel | 时延变化 | features |
|---|---:|---:|---:|---|
| 256×256 | 488.328 ms | 511.888 ms | +4.82% | 逐位一致 |
| 384×256 | 737.433 ms | 766.985 ms | +4.01% | 逐位一致 |
| 512×512 | 2222.394 ms | 2301.046 ms | +3.54% | 逐位一致 |

迁移消除了本轮新增的 ViT C++ 实现和对应 FFI，数值没有新增偏移，但存在上述性能回退。256² GELU 中位数由213.782ms增至230.728ms，仍是主要差项；两种实现的并行执行分别使用 OpenMP 与模型的 Rayon pool，尚未独立拆分编译器和线程调度的贡献。不能把这次迁移称为性能优化。

原始十二份分阶段记录、六样本数组及定频恢复信息见 [Rust 迁移记录](evidence/vision_rust_cpu_20261018/summary.json)。格式、全 target Release Clippy、FP16 CPU 五项回归，以及 FP16 Linear/Attention 板端回归通过；普通运行没有加入数值扫描。下面的性能、精度表保留迁移前各自受测版本的历史结果。

## 完整 ViT 性能

同一板锁、同一频率。每档分辨率按“原实现→仅删扫描→FP16→FP16→仅删扫描→原实现”顺序运行；每进程冷编码一次、预热一次、正式三次，每次清空feature cache并保留常驻权重。表中取两个独立进程的进程内中位数的中位数。encode包含patch、24层、merger和最终features扩为FP32，不含图片预处理、权重加载或语言模型。

| 输入宽×高 | patch M | 原FP32激活/含扫描 | FP32激活/仅删扫描 | FP16激活＋向量RoPE | 较原实现时延下降 | 3T口径MFU |
|---|---:|---:|---:|---:|---:|---:|
| 256×256 | 256 | 629.236ms | 574.577ms | 490.609ms | 22.03% | 11.23% |
| 384×256 | 384 | 966.807ms | 879.934ms | 748.705ms | 22.56% | 11.26% |
| 512×512 | 1024 | 2800.189ms | 2618.384ms | 2238.948ms | 20.04% | 11.00% |

FP16存储与RoPE改造相对“仅删扫描”另降低14.49%–14.91%时延。这里同时更换为HOST_STORAGE重新调优的profile，不能把整个收益归为数据类型这一项。三个尺寸共21个profile显式搜索耗时1401.65s，独立只读复核为21 hits、0 miss、0 search。正常推理没有在线调优。

256²最终累计分项：GELU213.797ms（43.6%）、Attention103.415ms（21.1%）、down56.011ms（11.4%）、up43.252ms（8.8%）、QKV34.153ms；两次LayerNorm合计4.132ms，QKV拆分＋RoPE8.935ms。down/up降至1.30倍。512²时GELU853.064ms、Attention735.066ms，分别占38.1%和32.8%。

RoPE首版使用逐元素Rust half转换，256²拆分＋RoPE约76.8ms，整图560.3ms；向量版将前者降至8.9ms。三分辨率features和三个图文case全部logits对照为逐位一致，因此该优化不引入额外数值偏移。

## 精度偏移

仅删扫描的输出与原FP32激活实现三档features均逐位一致。FP16激活相对该实现有如下偏移；相对L2的分母为左侧参考向量L2范数。

| 输入 | features相对L2 | cosine | 最大绝对差 |
|---|---:|---:|---:|
| 256×256红图 | 1.5560% | 0.99988038 | 0.0307922 |
| 384×256蓝图 | 1.0428% | 0.99994564 | 0.0220947 |
| 512×512红图 | 1.9909% | 0.99980179 | 0.0462646 |

以同权重PyTorch CUDA FP32为参考，关闭TF32/autocast、SDPA固定MATH，检查所有模块输出dtype，并补齐512²及CUDA FP16控制：

| 相对FP32参考的features相对L2 | 红图256² | 蓝图384×256 | 红图512² |
|---|---:|---:|---:|
| 旧FP32激活＋默认K tiling | 0.7463% | 0.7363% | 1.2262% |
| 当前FP16激活＋K tiling | 1.4419% | 0.8205% | 1.7849% |
| 当前FP16激活＋完整K | 1.1617% | 0.8744% | 1.7968% |
| PyTorch CUDA FP16控制 | 2.5080% | 0.9556% | 1.8887% |

当前默认红图cosine0.99989687，未达到此前0.9999的cosine门槛。旧实现这三组均更接近FP32；关闭K tiling只有红图明显改善。原始BF16发布权重扩为FP32计算，参考设置、原始checkpoint坐标对齐和逐层消融见[精度分析](VISION_FP16_PRECISION.md)及[严格FP32数据](evidence/vision_fp32_gold_20261018/README.md)。本轮精度结论仅比较ViT输出，避免混入文本端量化误差。

以下保留此前已完成的图文回归记录；包含文本量化执行的影响，不作为本轮ViT精度评价：

| 图文case，相同prompt/continuation | logits行数 | 相对旧FP32激活实现的logits相对L2 | 最大逐行KL(old‖new) | top1 |
|---|---:|---:|---:|---|
| red | 3 | 11.8435% | 0.0105113 | 3/3一致 |
| blue | 3 | 19.1922% | 0.1631465 | 3/3一致 |
| two | 4 | 15.2723% | 0.0795591 | 4/4一致 |

top1一致只覆盖这10行，不能代替任务准确率或说明分布不变。两种整模型运行均为tune_calls=0、weight_reload_calls=0。

`RK_NPU_MM_K_TILING=off`的FP16存储红图相对严格CUDA FP32参考L2为1.1617%、cosine0.99993458；启用/关闭之间相对L2为1.0693%。关掉K tiling可以做精度对照，不能消除FP16激活舍入本身。关闭时大部分shape走合法fallback、尚未重新搜索，不将该轮单次耗时与已调优默认路径比较。

FP16 GEMM输入不意味着整个Transformer块原本就按FP16保存。改造前bias相加、GELU输入和48次残差更新保留FP32；FP16模式在这些边界增加FP16舍入。真实权重逐Linear对照、同GEMM几何FP32图及逐层消融均已完成，见[精度分析](VISION_FP16_PRECISION.md)。默认仍为常规FP16图，手动`--vision-dtype fp32`可复核旧张量精度；局部FP32前缀只是诊断，不是运行配置。

## 验证

已通过：C公共头、native FP16 panel/task、168个K分块case（新增storage+bias）、Clippy全targets `-D warnings`、默认Rust测试及FP16 Linear/NO_MASK Attention板端回归。FP16 Linear和Attention输出与原FP32接口输出舍入到FP16后逐位一致，覆盖M3/M129、query尾块。LayerNorm的大值归约和GELU/残差舍入测试通过；向量RoPE新增严格逐位测试通过。当前lib测试47 passed/9 ignored，集成CPU测试13+1+5 passed；本次相关的FP16 NPU回归另以板锁显式运行。不运行BF16 CPU数值计算。最终18个完整ViT性能进程ioctl errors均为0，所有runner返回0且CPU/NPU/DDR恢复验证通过。

## down 诊断

删除扫描前，M256 down约3.95ms中：FP32→FP16约0.322ms、输入单线程Inf/NaN扫描0.960ms、分块打包0.594ms、GEMM1.158ms、NPU归约0.279ms、输出数值扫描0.278ms。输入处理约贡献down/up净差的85%。完整诊断和局部替换实验见 [down分项记录](evidence/fp16_down_stages_20261018/README.md)。

仅删扫描的独立复测：up约1.676ms，down约2.722ms；本轮测试输出哈希与原库一致。此结果不包含后续FP16激活存储改造。

## 执行记录

板端冻结目录：`/home/firefly/mnt/3090x4/tmp/vision-fp16-storage-20261018/`。板锁/定频入口为`scripts/run_board.py --wait-lock`。NPU1GHz、DDR2112MHz、大核2.4GHz、CPU4线程/mask0xf0、NPUmask7。

顺序脚本：`logs/fp16-vision-build.sh` → `fp16-vision-tune.sh` → `fp32-vision-reference-build.sh` → `fp16-vision-model.sh` → `fp16-vision-off-check.sh` → `fp16-vision-rotary-build.sh` → `fp16-vision-rotary-model.sh`。均已返回0。首版与最终版数据分别保存在[evidence/vision_fp16_storage_20261018](evidence/vision_fp16_storage_20261018/summary.json)和[evidence/vision_fp16_storage_rotary_20261018](evidence/vision_fp16_storage_rotary_20261018/summary.json)。

`logs/make-fp32-vision-reference.py` 在冻结目录创建构建专用源码副本，只将encoder/load恢复为原FP32激活图，其余使用相同当前库；参考example使用不同文件名，避免覆盖默认FP16二进制。`logs/finalize-fp16-vision.py --rotary` 汇总原始数据、恢复验证、性能和数值偏移，并断言向量/标量RoPE的features与logits均逐位一致。
