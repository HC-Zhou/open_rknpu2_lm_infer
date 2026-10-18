# GELU 与 Attention 诊断归档

2026-10-18。历史原型基于外层 `93d552e`、算子库 `5b4e39c`；已由正式 `rk_npu_mm_gelu` ABI 和 Rust FloatLinear 集成取代。这里的文件不参与 Cargo 目标、正常推理或 CI，包含失败/废弃方案，不能将运行成功视为数值正确。

- `npu_gelu_bridge.cpp`、`npu_gelu_bench.rs`、`npu_gelu_bench/`：独立 GELU、CPU 对照与动态库桥接。旧 CPU 近似 GELU 已从当前源码删除，复现实验需使用上述冻结版本，并恢复 Cargo example 声明。
- `npu_gelu_fused_bridge.cpp`、`npu_gelu_fused_snapshot.rs`：已测融合原型；每层独立分配与复制权重仅用于实验。正式实现复用常驻权重并共享工作区。
- `build_npu_gelu*_snapshot.py`、`npu_gelu_snapshot.rs`：历史源码快照生成器，依赖冻结版本的模块结构。原脚本中的 `tests/board/` 位置是归档前的位置；复现时把同目录探针恢复到该位置，或在隔离 checkout 中调整路径，不能直接对当前推理源码套用补丁。
- `npu_gelu_fused_check.py`、`npu_gelu_fused_vit.py`、`npu_gelu_fused_compare.py`：原型正确性、三路 ViT 和配对比较。正式回归使用算子库 `test_mm_gelu --board`、Rust hardware test 与普通 `vision_profile`。
- `npu_gelu_probe.py`、`npu_gelu_fp16_probe.py`：编码、尾部、直接 FP16 writer 和输出转换诊断。某些模式明确数值失败。
- `npu_gelu_fused_probe.cpp`：GEMM/标量 bias/逐通道 bias 连通性；第二个参数0为 Q13 writer，2为 FP32 carrier；1/3/4为格式诊断。
- `fused_n128_bridge.cpp`、`fused_row_collector.rs`：N tile128和逐行收集的历史对照组。
- `vision_attention_state.cpp`、`vision_attention_snapshot.rs`、`build_vision_attention_snapshot.py`：只读 Attention 路由探针，依赖当时私有 ABI。三种输入均走 LUT+PV 快路径，无 mask；generic query 内部仍按32行分块。stderr 同步日志会显著干扰耗时，探针时延不能用于性能比较。此次不修改 Attention。

板端探针须在独占锁和固定频率下执行。原始数值、样本和频率恢复记录见 `llm_generated_docs/evidence/npu_gelu_20261018/`；正式行为与结果见 `llm_generated_docs/NPU_FFN_GELU_RESEARCH.md`。保留失败模式用于解释 FP32 carrier 的选择，不提供运行时 CPU 回退或废弃后端选择。
