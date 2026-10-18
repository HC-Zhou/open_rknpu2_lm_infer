# ViT 激活精度与主流框架

2026-10-18。区分张量存储精度、算子内部累加精度和算子之间的舍入位置。FP16 matmul的输入/输出并不能决定整个Transformer块的精度。

后续更新：以下为融合GELU前的冻结精度实验。当前FFN默认使用NPU GEMM/bias/GELU融合，CPU近似GELU已删除；`--vision-dtype fp32`只恢复FP32张量存储，两种模式均融合FFN，不能复现下表的旧CPU FFN路径。[当前实现与精度](NPU_FFN_GELU_RESEARCH.md)。

## 严格FP32参考下的结果

本轮精度结论只比较ViT输出。参考使用同一C7视觉权重，全部参数和激活以PyTorch CUDA FP32计算；TF32/autocast关闭、matmul precision=highest、SDPA固定MATH。全部模块浮点输出dtype审计通过。输入patches与板端逐位一致。数据、设置和原始checkpoint坐标对齐见[FP32参考记录](evidence/vision_fp32_gold_20261018/README.md)。

| features相对FP32参考的L2 | 红图256² | 蓝图384×256 | 红图512² |
|---|---:|---:|---:|
| 旧NPU FP32激活＋默认K tiling | 0.7463% | 0.7363% | 1.2262% |
| 当前NPU FP16激活＋K tiling | 1.4419% | 0.8205% | 1.7849% |
| 当前NPU FP16激活＋完整K | 1.1617% | 0.8744% | 1.7968% |
| PyTorch CUDA FP16控制 | 2.5080% | 0.9556% | 1.8887% |

这三组上，旧实现更接近FP32参考。当前FP16路径并非因为离旧实现更远就被判更差；这里是独立FP32参考实际测出的距离。关闭K tiling没有一致改善。CUDA FP16控制也不是精度金标准，其参数和激活使用FP16、GEMM使用FP32累加、RoPE频率保持FP32，禁用低精度GEMM/SDPA归约。

同时运行原始未量化checkpoint的FP32 ViT，并在模型外旋转到C7坐标。两份checkpoint的24个block输出在三个尺寸上均逐位相同；merger FC2存在C7离线旋转后BF16保存的差别，两份FP32参考最终features距离约0.093%–0.095%。相对该原始参考，旧/新默认的排名也相同，详见summary.json。这里FP32指计算精度，发布的checkpoint权重本身存为BF16。

## 主流实现的源码对照

以下核对Qwen视觉实现，不据此断言框架的所有文本模型与后端均相同。源码固定到本次查询的提交，后续配置或优化可能改变执行路径。

| 框架 | 视觉激活/残差 | 依据 |
|---|---|---|
| vLLM | 随模型FP16/BF16 dtype；视觉block直接执行两次残差相加，没有专门转FP32保存残差 | [Qwen3_VisionBlock及dtype属性](https://github.com/vllm-project/vllm/blob/03ee2e134d6cc60c4cc84096c8492fcc971f0526/vllm/model_executor/models/qwen3_vl.py#L455) |
| SGLang | 输入转patch权重dtype，视觉block的两次原位残差更新保持该dtype | [Qwen3_VisionBlock](https://github.com/sgl-project/sglang/blob/8d02b69c1130d7ba161716bca0bce6537ce4b433/python/sglang/srt/models/qwen3_vl.py#L260) |
| llama.cpp | 常见GGML图中matmul输出显式为F32，后续视觉残差保持F32；F16权重不代表全部激活F16 | [ggml_mul_mat](https://github.com/ggml-org/llama.cpp/blob/797de63b2fa7396dcc8caca76834c699ac84e1c3/ggml/src/ggml.c#L3362)、[Qwen视觉图](https://github.com/ggml-org/llama.cpp/blob/797de63b2fa7396dcc8caca76834c699ac84e1c3/tools/mtmd/models/qwen3vl.cpp#L120) |

低精度存储不等于低精度累加。PyTorch的CUDA half GEMM helper默认compute_type为CUDA_R_32F，只有允许FP16 accumulation时改成16F；另有独立的reduced-precision reduction策略。[固定源码](https://github.com/pytorch/pytorch/blob/8fcaa153fa223ca9f7844513d86d4d30a0cf2533/aten/src/ATen/cuda/CUDABlas.cpp#L1076)。CUDA LayerNorm的Welford统计量为float、输出使用empty_like(input)，可以在FP32计算统计量后保存FP16/BF16结果。[固定源码](https://github.com/pytorch/pytorch/blob/8fcaa153fa223ca9f7844513d86d4d30a0cf2533/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L140)。不能由这些源码推断RK3588 NPU内部MAC累加寄存器的实际位宽。

当前选择“FP16张量＋FP32 norm/softmax归约”与vLLM/SGLang视觉图的存储方向一致。但NPU FP16 partial＋FP16 Add tree有自己的舍入位置，不能直接视为CUDA FP32累加GEMM的位级等价实现。

## 本次改造新增的舍入位置

旧图的GEMM已使用FP16输入和FP16结果，但收集后扩为FP32，再加FP32 bias；之后位置相加、GELU输入、RoPE输入及残差保持FP32，到下一次GEMM输入才再转FP16。新图在Linear加bias后直接写FP16，位置、非线性和每次残差也写FP16。

例如旧路径是`GELU(FP32(C_half)+bias32)`再转FP16输入down；新路径是`FP16(GELU(FP16(FP32(C_half)+bias32)))`。旧残差是连续FP32相加，新图24层共48次写回FP16。norm的FP32均值/方差无法恢复之前已经舍去的位数。

算子之后紧接另一个matmul时，原来就存在的FP16输入边界不一定新增误差；需要关注两次matmul之间存在位置/旋转/非线性/残差的路径，不能简单统计转换次数。

## 数值诊断方法

诊断使用冻结的当前FP16 profile、真实C7权重、红/蓝图。每个Linear额外创建相同M/N/K、layout、host M、task M、Ntile、Ktile的FP32 host workspace，比较FP16输出与FP32输出再舍入的每个元素；诊断workspace/全量trace只存在于独立构建副本，不加入主库或长期用户开关，诊断耗时不作为性能结果。

另恢复原FP32激活图，但强制读取当前HOST_STORAGE profile的几何和常驻K分区，仅改变host IO；加载和resident匹配失败直接报错。这样将“重新调优了几何”和“改变层间张量精度”分开。旧、新21个profile的Ktile核对全部相同。

在同一个FP16图的构建副本中进行四种额外对照：norm32只恢复旧FP32输入/输出临时值的LayerNorm算法，再转FP16；residual32在该norm控制组上仅保留FP32残差；gelu32在同一norm控制组上让两个up的GEMM/bias输出以FP32进入原GELU，再转FP16；residual-gelu32同时恢复这两项。其余张量、真实权重和GEMM几何保持相同。gelu32同时恢复原Rust FP32 GELU实现，因此不是只改变单一舍入指令的实验。

红/蓝各99次真实Linear，合计143,278,080个输出，FP16接口与相同几何FP32接口再舍入逐位一致。覆盖全部消融时为890次Linear、589,250,560个输出，全部mismatches=0。恢复原FP32激活图、仅使用新FP16 profile几何时，两张图的最终features均与旧实现逐位一致。由此排除本轮接口转换错误及重新调优导致的这两张图偏移。

下表以严格PyTorch FP32为参考，避免以“更接近旧实现”代替精度评价：

| 独立构建副本，features相对L2 | 红图 | 蓝图 |
|---|---:|---:|
| 当前FP16 | 1.4419% | 0.8205% |
| norm32控制 | 1.4788% | 0.8351% |
| norm32＋FP32残差 | 1.2891% | 0.8641% |
| norm32＋FP32 GELU输入/计算 | 1.4496% | 0.7687% |
| norm32＋FP32残差和GELU | 1.2206% | 0.6711% |

单独恢复残差或GELU并不保证每张图都更接近FP32；例如蓝图联合恢复后比旧实现的0.7363%更近。不同误差方向经过后续非线性和归一化会放大或抵消，不能依据单个距离做可加的贡献分摊。norm两种实现均用FP32归约，求和顺序改变后又写入FP16也会产生不同后续轨迹；红图norm32相对当前FP16的features距离为0.8453%，尽管两者到旧实现的距离接近。

红图当前FP16相对旧实现的层间距离：patch/position0.0291%、block0为0.0694%、block7为0.3094%、block15为0.8179%、block23为2.1679%，merger后1.5560%。该轨迹说明差异逐步累积，不单独证明某一算子是唯一主因。原始逐层数据与构建副本见[消融记录](evidence/vision_fp16_precision_20261018/summary.json)。局部混合精度只用于诊断；当时通过`--vision-dtype fp32`选择旧FP32张量路径，当前FFN融合后的行为见页首更新。

完整性能见[VISION_FP16_STORAGE.md](VISION_FP16_STORAGE.md)。这些诊断比较数值距离，不代表任务准确率；本轮不使用文本端的量化误差评价ViT。

## 误差位置的进一步验证

逐算子同输入oracle、每层gold输入重放、FP32后缀替换，以及真实NPU后续执行的前缀替换均已完成，详见[逐算子记录](evidence/vision_fp16_stage_precision_20261018/README.md)。末层主要放大早期差异；恢复入口与前两个block的旧FP32中间结果后，后续保持NPU FP16，红/蓝features相对FP32参考为0.7957%/0.7340%，而默认FP16为1.4419%/0.8205%。仅替换末层的FP32结果为1.4413%/0.8182%。

这些替换用于定位，不构成采用局部FP32前缀的依据。普通PyTorch FP16在同样回归图上的偏移也达到0.96%–2.51%。运行配置采用常规FP16中间张量；需要对照时通过`--vision-dtype fp32`整体恢复旧FP32张量路径。
