# MLP up/down 当前搜索空间复查

2026-10-18，FP16 tuning revision6。针对256×256图片对应的 M256，只重新搜索 MLP up/down 两个 shape。使用独立空缓存，两个搜索均确认 `cache_hit=0`；原默认缓存保持不变。本轮未重新测完整 ViT。

此前 K tiling revision3 的 down 确实做过搜索，原始日志为 [pilot-256-4096-1024.log](../fp16_k_tile_autotune_20261018/pilot-256-4096-1024.log)，含1798条候选计时。随后正式性能/模型测试加载该缓存。revision6 清理验证也复用了该几何配置，没有重新搜索新增的均衡 task M 候选。

本轮 up 搜索379个粗筛候选、253个精测候选；down 搜索1381个粗筛候选、724个精测候选。完整输出均经过调优器独立 oracle 验证。两个新配置均只将 `task_m_tile=0` 改为128，其余选中参数与原配置相同：

| 项目 | up | down |
|---|---|---|
| M/K/N | 256/1024/4096 | 256/4096/1024 |
| host M tile / N tile | 256 / 1376 | 256 / 0 |
| A/C layout | panel8 / panel8 | normal / panel8 |
| K tile | 完整 K | 1024，4个 partial |
| host IO | fused | staged |
| 归约 | 无 | NPU FP16 Add tree |

冻结的当前 revision6 库同时执行原配置和新配置；固定 NPU 1GHz、DDR 2112MHz、大核2.4GHz、CPU4线程/0xf0、NPU允许mask7。每个 shape 测4个独立进程，每进程预热后8组 ABBA，每路径16个完整调用；表格先取各进程中位数，再取4个进程中位数。计时包含 FP32输入转换、pack/sync、GEMM、归约、输出收集及 bias，不含权重上传或 workspace 创建。

| 完整调用 | 原配置，本轮复测 | 独立空缓存重调后 | 时延下降 |
|---|---:|---:|---:|
| up | 1.9458ms | 1.9002ms | 2.35% |
| down | 4.0402ms | 3.9429ms | 2.41% |

新旧配置在本轮相同随机输入和权重上的完整输出逐位一致，最大绝对差为0；每配置另检验128个双精度 dot-product oracle 点。此项不代替真实模型精度验证。

down/up 仍为2.075倍；当前新增候选只带来小幅改善。两者 FLOPs 相同，但分块、归约、layout 和 host IO 路径不同。尚未对这组实际选中策略拆出 GEMM/归约/转换各自时间，不能将剩余差距精确归因于其中一项。

全部原始搜索日志、8份配对日志、频率与恢复记录和 [summary.json](summary.json) 保存在本目录。两个 runner 均返回0，恢复错误为空，min/max/governor恢复校验通过。冻结二进制哈希见 [binaries.sha256](binaries.sha256)，二进制和独立缓存保存在板端 `/home/firefly/mnt/3090x4/tmp/floating-mm-up-down-retune-20261018/`，共享库来自同级 `floating-mm-two-modes-20261018/`。

比较工具为 `scripts/diagnostics/fp16_tune_compare.cpp`，只读加载两个缓存，不搜索、不改写；参数为 `M K N PREVIOUS_CACHE FRESH_CACHE`。本目录 runner 脚本记录了搜索和配对测试命令。
