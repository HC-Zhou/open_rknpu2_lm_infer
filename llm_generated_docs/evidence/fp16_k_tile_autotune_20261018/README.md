# FP16 K 分块与独立 NPU reduce 证据

2026-10-18，RK3588 主板；CPU4/mask0xf0、2.4GHz，NPU mask7、1GHz，DDR2112MHz。FP16 revision3；BF16 CPU 计算与搜索跳过。

本组 K 分块方案有性能收益，但未通过整模型精度门槛，不作为默认视觉路径。21个形状中9个 K4096 形状选中独立 FP16 NPU Add 树；相对 revision2 完整 K 策略，算子加速比几何平均1.317×、最高2.380×。每档分辨率实际只有7类 Linear shape，24层复用同一形状的 profile。

| 输入宽×高 | 完整 K ViT | K 分块 ViT | 加速比 | K 分块 MFU，3T峰值 |
|---|---:|---:|---:|---:|
| 256×256 | 677.549ms | 624.846ms | 1.084× | 8.821% |
| 384×256 | 1028.649ms | 966.611ms | 1.064× | 8.720% |
| 512×512 | 3015.497ms | 2799.078ms | 1.077× | 8.797% |

完整 encode 的每次计时均清空 feature cache、复用常驻权重和 workspace；按 previous/tuned/tuned/previous 进程顺序比较，每进程3个正式样本，每条路径共6个热 workspace 样本。冷 encode 另存于原始 profile。

红图/蓝图 features 相对完整 K 的 L2 为0.706%/0.678%。10行固定 continuation 的 top1 全部相同，但红图首行 KL 为0.12353，双图首行 KL 为0.07373；整体 logits 相对 L2 为12.1%–17.8%。512×512 features cosine 为0.9998966，低于0.9999门槛。因此 `summary.json`、`measurement.json` 保持 `passed=false`，未放宽门槛。

- `summary.json`：完整调用 ABBA、revision2 策略配对、21个 load-only profile、features 与图文 logits 数值核对。
- `measurement.json`：完整 ViT 配对时延、阶段统计、驱动时间、MFU、失败门槛。
- `fp16-M-K-N.log`：21个完整搜索与正式算子样本；`pilot-256-4096-1024.log` 保存同配置 CPU/NPU 归约细测。该配置 CPU 归约5.711ms、NPU归约3.939ms。
- `native-k-tests.log`：112组 K 分块 CPU/NPU 归约与 dense CPU oracle 检查；`npu-reduce-probe.log`：FP16 Add 树独立精度检查。
- `resident/`：21组常驻权重 API 检查，构造时使用 M1、运行时使用真实 M；清空原始 host B 后，库内分区选择的输出仍与手工 packed B 逐位相同。
- `profile-*`：12个完整 ViT 进程的 profile、驱动计数；原始 features 留在共享目录。
- `validation-runner.log`：fmt、全 target Clippy、默认62项测试及显式 FP16 Linear/Attention 检查。
- `*-clocks.json`、`hardware-restored.json`：runner 均退出0、恢复错误为空，并核对 CPU/NPU/DDR min/max/governor 与运行前一致。

数值失败表示方案精度不符合当前 C7 门槛，独立记录于报告；硬件执行和恢复均成功。完整数组与冻结可执行文件位于 `/home/zt/mnt/3090x4/tmp/floating-mm-k-tile-20261018/`。后续 FP16 融合与 FP32 中间累加另行比较，不能混入本组结果。
