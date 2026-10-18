# FP16 revision2 自动调优证据

2026-10-18，主板 RK3588；CPU4/mask0xf0、大核2.4GHz、NPU mask7/1GHz、DDR2112MHz。本轮仅搜索、验证 FP16；BF16 保留初版结果。

- `summary.json`：完整调用 ABBA、与上一版已调优策略的配对比较、缓存读取、ViT features 与图文 logits 核对。生成入口为 `scripts/compare_float_autotune.py`。
- `fp16-M-K-N.log`：最终正常退出的正式样本，每个形状分别对固定 M128、上一版已调优策略做3组 ABBA；各路径6次完整调用。
- `search/`：首次搜索的全部候选和原始样本，含 coarse/final 阶段。最终计时另行运行，保留相同选中 profile。
- `cases.txt`：M/K/N 与上一版 m_tile/n_tile/A/C 参数；`load-only.json`：独立 Rust 进程读取的21个 profile，零 miss、零搜索。
- `panel-fp16-tests.log`：576组独立 dense CPU oracle 检查，其中108组 K97；`native-fp16-tests.log`：FP16 调优、缓存、affinity 和 split 权重兼容性回归。
- `validation-runner.log`：格式、Clippy、默认62项测试与显式 FP16 Linear/Attention 测试。`*-stats.json`、`*-encoder.json` 保存完整模型及视觉检查元数据。
- `measurement.json`：按用户给定3 TFLOPS峰值计算的端到端 MFU、FLOPs 口径及冻结 Rust 可执行文件 SHA256。
- `validation-clocks.json`、`panel-clocks.json`、`performance-clocks.json`、`model-clocks.json`：四个最终 runner 均退出0、无恢复错误；另核对 min/max/governor 与运行前一致。相应 `run-*.sh` 记录命令。

最初 `search-run-clocks.json` 的退出码127来自运行中的 shell 文件被修改，发生在21个算子搜索/计时全部完成之后；`search-script-error.log` 保留原错误。该次硬件正常恢复。后续独立脚本完成模型核对，并重新运行全部正式 ABBA，结果使用最终退出0的 runner。

ViT features 和图文 logits 的数值基线复用初版 `floating-mm-20261018` 保存的 FP32 数组；算子时延均在同进程重新配对测量。完整数组、权重及冻结二进制保留在共享目录 `/home/zt/mnt/3090x4/tmp/floating-mm-panel-20261018/`，不复制到 Git 工作树。细节见 [主说明](../../FLOAT_MM_AUTOTUNE.md)。
