# ViT 严格 PyTorch FP32 参考

2026-10-18，仅比较视觉编码器输出，不加载文本decoder，不使用文本整数参考或图文logits评判本轮视觉精度。

同一C7视觉权重、同一图片patches。主参考的参数、激活和模块输出均为FP32，关闭TF32和autocast，matmul precision=highest，SDPA固定MATH。所有模块的浮点输出dtype审计通过；两项RoPE频率buffer保留原FP32值。原始发布权重存为BF16，本实验将其值扩为FP32计算。

下表均相对**同一C7权重的PyTorch FP32参考**，数值为features相对L2。NPU旧实现指FP32层间激活、NPU FP16 GEMM及默认K tiling；新实现均为FP16层间激活、FP32 norm/softmax归约。

| 输入宽×高 | NPU旧FP32激活 | NPU FP16＋K tiling | NPU FP16＋完整K | PyTorch CUDA FP16 |
|---|---:|---:|---:|---:|
| 红图256×256 | 0.7463% | 1.4419% | 1.1617% | 2.5080% |
| 蓝图384×256 | 0.7363% | 0.8205% | 0.8744% | 0.9556% |
| 红图512×512 | 1.2262% | 1.7849% | 1.7968% | 1.8887% |

旧实现三组均更接近FP32参考。关闭K tiling只在红图上改善，另外两组未改善。CUDA FP16控制使用FP16参数/张量、FP32 GEMM累加、禁用低精度GEMM归约及MATH SDPA低精度归约；它只是一个低精度控制，不是金标准，也不能把到FP32的距离解释为任务准确率。

另加载原始未量化checkpoint的ViT，在FP32计算后用C7的signed Hadamard 20×128旋转映射到相同坐标。C7与原始checkpoint的24个block输出在三张图上均逐位一致；只有merger FC2权重/偏置经离线旋转与BF16保存后不同。映射后的原始参考与C7 FP32参考的features相对L2为0.09328%/0.09337%/0.09536%。summary.json另列相对此原始参考的结果，避免混淆离线权重保存与本轮执行精度。

三张图的PyTorch patches与板端patches逐位相同。关闭K tiling的蓝图和512²探针经共享板锁执行，runner返回0、CPU/NPU/DDR恢复验证通过；它们未重新调优，只比较数值，不比较耗时。全部完整数组和trace位于`/home/zt/mnt/3090x4/tmp/vision-fp16-storage-20261018/fp32-gold/reference/`（服务器路径`/data007/zt/tmp/...`）。

入口：[vision_precision_reference.py](../../../scripts/qwen35/vision_precision_reference.py)。它复用固定Transformers revision的上游视觉定义；manifest.json记录计算精度、dtype审计和运行环境。summary.json记录L2、cosine、最大绝对差、逐层偏移及相同GEMM几何对照。范围只有三张数值回归图，不代表通用视觉任务准确率。
