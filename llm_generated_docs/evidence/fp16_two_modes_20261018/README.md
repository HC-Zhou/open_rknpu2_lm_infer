# Two retained FP16 MM configurations

2026-10-18。仅保留完整 K 和 FP16 K tiling + NPU FP16 Add tree；后者默认启用。设置 `RK_NPU_MM_K_TILING=off` 或 `0` 可关闭，作为精度/一致性对照。当前用法见 [自动调优说明](../../FLOAT_MM_AUTOTUNE.md)。

已删除实验版 FP32 partial/归约、CPU partial 归约、K 链 GEMM+EW 实现及其专用工具，移除算法选择字段和旧实验环境开关。保留正常 FP32 host IO 的转换/pack/collection 融合，以及 raw MM 原有能力。共享库符号检查确认被删的私有实现不存在。

FP16 profile revision6 只接受完整 K 或 FP16 NPU 归约，兼容读取合法旧策略，拒绝旧 CPU/FP32/fused 策略。普通推理保持只读缓存与合法回退。

验证结果：

- Release fmt、全 target Clippy `-D warnings`、62项默认 Rust 测试及显式 FP16 Linear/Attention 板端测试通过。
- 公共 C 头、native FP16 调优与112组 K 分块检查通过，覆盖 storage/FP32 host IO、M/K/N tail、N tile、task cap、mask1/7。
- 默认与禁用两种模式均命中21个旧 profile，零搜索；分别有9/0个 K tiled shape。
- 红/蓝图共四次完整 ViT features 检查：默认输出与旧 FP16 NPU reduce 路径逐位一致，关闭后与原完整 K 路径逐位一致。
- 新缓存写入/读取、选中策略的常驻权重复用和 host 源释放检查通过。
- 两份 runner 均返回0，频率限制/governor恢复通过，恢复错误为空。本轮没有计算 BF16 CPU oracle。

结构化结果、二进制指纹和恢复检查见 [summary.json](summary.json)。完整数组及匹配二进制保留在 `/home/zt/mnt/3090x4/tmp/floating-mm-two-modes-20261018/`。废弃实验的历史记录与外部源码快照仍可追溯，不属于当前运行配置。
