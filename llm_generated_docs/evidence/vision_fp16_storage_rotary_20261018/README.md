# ViT FP16 storage 最终配对结果

日期2026-10-18。完整说明见[FP16存储报告](../../VISION_FP16_STORAGE.md)，机器可读结果见[summary.json](summary.json)。

NPU1GHz、DDR2112MHz、CPU大核2.4GHz/4线程/mask0xf0，NPU允许mask7。所有runner经共享板锁执行，返回0，restore_errors为空，另核对恢复后的min/max/governor。BF16 CPU数值计算跳过。

三分辨率各6个独立进程，顺序为original/no-scan/fp16/fp16/no-scan/original。每进程冷encode一次、warmup一次、正式三次，正式调用清feature cache并保留权重。先取进程内中位数，再取同方案两个进程的中位数；不能把六个timed call当作六个独立进程样本。profile-*/profile.json保留原始样本和阶段值，ioctl.json保留完整进程的驱动计数；后者包含冷/热轮次，不与暖态总时延直接相减。

| 输入 | original | no-scan | fp16 | 相对original时延下降 | fp16 MFU/3T |
|---|---:|---:|---:|---:|---:|
| 256² | 629.236ms | 574.577ms | 490.609ms | 22.03% | 11.23% |
| 384×256 | 966.807ms | 879.934ms | 748.705ms | 22.56% | 11.26% |
| 512² | 2800.189ms | 2618.384ms | 2238.948ms | 20.04% | 11.00% |

original为冻结revision6、FP32激活、保留运行时数值扫描；no-scan恢复相同FP32激活图、使用当前revision7算子；fp16为当前FP16激活和向量RoPE。21个FP16 storage profile重新显式搜索，storage-load-only.json核对21 hits/0 miss/0 search。普通运行不在线搜索或重载W4权重。

original/no-scan三档features逐位相同；当前向量RoPE与首版标量FP16 RoPE的三档features和三个图文case全部logits逐位相同。FP16存储相对旧FP32激活仍有1.04%–1.99%的features相对L2偏移；图文10行top1一致，但最大逐行KL为0.1631465，不能宣称精度不变。关闭K tiling只验证数值，不比较未调优fallback的计时。

首版标量FP16数据另见[phase1 summary](../vision_fp16_storage_20261018/summary.json)。最终冻结二进制和完整输出位于板端`/home/firefly/mnt/3090x4/tmp/vision-fp16-storage-20261018/rotary/`，对应主机挂载`/home/zt/mnt/3090x4/tmp/vision-fp16-storage-20261018/rotary/`。binaries.sha256记录二进制身份。原始tuner log、21个shape及测试/runner记录已归档在本目录。

复核入口为`logs/finalize-fp16-vision.py --rotary`；它断言所有runner成功、恢复正确、特征缓存未命中、重复输出一致、转换前后数学对照、RoPE位级一致、全部profile命中和ioctl errors=0。
