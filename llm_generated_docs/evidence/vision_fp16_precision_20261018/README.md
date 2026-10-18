# ViT FP16 层间精度消融

仅做ViT数值诊断。原始层间数据与真实Linear检查见[summary.json](summary.json)，独立FP32参考下的最终评价见[精度报告](../../VISION_FP16_PRECISION.md)。

用相同GEMM几何、真实C7权重和相同输入，逐元素比较FP16接口与FP32接口再舍入；红/蓝当前默认各99次Linear合计143,278,080个输出全部逐位一致。五种消融合计890次Linear、589,250,560个输出，mismatches均为0。

恢复原FP32激活图、但强制读取新FP16 profile的几何与常驻K分区，两张图最终features与旧FP32激活实现逐位一致。两个source目录保存实际构建的encoder/floating源码，make-vision-precision-ablation.py和runner可复现隔离构建；额外分支不进入主库。

norm32控制两条路径的LayerNorm算法；residual32、gelu32、residual-gelu32分别恢复FP32残差、GELU输入/原GELU计算及两者。gelu32包含旧Rust GELU实现，不把它描述成仅改变一条转换指令。该目录到旧实现的距离用于描述轨迹；精度排序使用独立PyTorch FP32参考。

runner返回0，CPU/NPU/DDR的min/max/governor恢复验证通过。不运行BF16 CPU计算。完整trace/数组和冻结二进制留在`/home/zt/mnt/3090x4/tmp/vision-fp16-storage-20261018/precision/`，计时包含额外workspace与trace I/O，不用于性能结论。
