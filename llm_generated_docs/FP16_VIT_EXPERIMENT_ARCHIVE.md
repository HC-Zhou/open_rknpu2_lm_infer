# FP16 matmul 与 ViT 实验代码归档

日期：2026-10-18。源码保存在 [`scripts/archive/fp16_vit_20261018/`](../scripts/archive/fp16_vit_20261018/)，不参与 Cargo/CMake 构建。归档保留定位低利用率、K 分块归约、down/up 差异和精度误差的方法；普通推理中没有这些插桩或消融分支。

[`manifest.json`](../scripts/archive/fp16_vit_20261018/manifest.json) 记录每个文件的原路径、归档路径及迁移前后 SHA256。迁移仅修正三个 Python 文件的相邻文件查找和导入路径，其他实验源码原样保存。

| 文件 | 用途及依赖 |
|---|---|
| `fp16_device_probe.cpp` | 同一 raw MM plan 的常驻设备调用与 FP32 host IO 完整调用做 ABBA 对照；只测试完整 K，公共 ABI 与 half-bit helper 均来自算子库 |
| `npu_ioctl_profile.cpp` | 独立 LD_PRELOAD submit/sync 计时；依赖受测库的私有 UAPI，驱动硬件时间不能当作 MAC 忙周期 |
| `fp16_tune_compare.cpp` | 两份历史 FP32 host profile 的只读完整调用比较；旧缓存格式已不受当前库支持，K partition 也受 packed-B 接口约束 |
| `fp16_reduce_probe.cpp` | 最初的私有 FP16 Add tree 探针；当前公共回归已由算子库 `test_mm_tune --k-tiles` 覆盖 |
| `build_fp16_stage_trace.py`、`fp16_stage_trace.cpp/.h`、`fp16_stage_probe.cpp` | revision6 down/up 分阶段计时与转换/pack 消融；构建脚本依赖冻结的 `.cpp.original` 和同版本原生对象，不能用于当前库源码 |
| `summarize_float_util.py` | 历史设备/host 与完整 ViT MFU 汇总，按固定 red/blue/red512 目录读取记录 |
| `compare_float_autotune.py`、`summarize_float_k_tiling.py` | 旧调优日志、配对性能、features/logits 汇总；只适用于原实验目录及其字段/门槛 |
| `vision_stage_oracle.py`、`vision_prefix_inputs.py` | 同输入 FP32 oracle、层间误差传播、前缀替换和 FP32 后缀重放；保留当时 GPU/板端绝对路径，依赖冻结的 trace、checkpoint 和 CUDA 环境 |

历史脚本只用于检查原实验或借鉴实验方法，不表示它们已适配当前算子库。阶段插桩工具依赖的旧库、编译对象、模型、trace 和大数组仍保留在各原报告所列的共享 artifact 目录，本次没有复制构建产物或模型。缺失的旧脚本不会用近似实现补写。

当前维护的入口：

- `tests/tools/float_tune.rs`：显式 FP16/BF16 调优与 `--load-only` 复核；BF16 仅在显式指定时计算。
- `tests/tools/vision_probe.rs`：图片预处理、features、特征缓存和可选图文回归。
- `tests/tools/vision_profile.rs`：绕过特征缓存的完整 ViT 计时，保留分阶段结果。
- `scripts/qwen35/vision_reference.py`、`vision_precision_reference.py`：固定上游定义及独立 PyTorch FP32 ViT 参考。
- `scripts/qwen35/check_vision_positions.py`、`compare_vision.py`、`validate_vision_api.py`：坐标、输出与 API 检查。

当前数值模式见 [FP16 存储说明](VISION_FP16_STORAGE.md)，自动调优和缓存生命周期见 [浮点 MM 自动调优](FLOAT_MM_AUTOTUNE.md)。性能与精度表保留原测量时间和受测版本边界；上游 Attention LUT 更新后不要求与旧冻结二进制逐位一致。
