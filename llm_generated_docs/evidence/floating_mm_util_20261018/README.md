# FP16 完整 K 利用率诊断证据

主板 RK3588；CPU 4线程/mask0xf0、2.4GHz，NPU mask7、1GHz，DDR 2112MHz。FP16 revision2，完整 K，跳过 BF16 CPU 运算。

- `summary.json`：13个常驻设备 matmul 与三档完整 ViT 的结构化统计，由 `scripts/diagnostics/summarize_float_util.py` 生成。
- `mm-*.log`：每组3个 ABBA block、每路径6次完整调用；包含 device 和 host 两种口径。
- `mm-*-ioctl.json`：driver submit/sync/hardware elapsed 计数，全部 errors=0。
- `red/`、`blue/`、`red512/`：完整 encode 的冷样本、1次预热与3次正式样本、分阶段计时、驱动计数；每轮清空 feature cache，输出与首轮逐位相同。
- `fp-mm-util-build.log`、`fp-mm-util-*.sh`：构建与执行入口；`fp-mm-util-profile-clocks.json` 和 `hardware-restored.json` 保存定频及硬件恢复验证。

原始数组、512×512纯红输入、冻结可执行文件位于 `/home/zt/mnt/3090x4/tmp/floating-mm-util-20261018/`。结果解读见 [诊断说明](../../FLOAT_MM_MFU_DIAGNOSIS.md)。
