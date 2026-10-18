# ViT 的 FP16 matmul 接口与调优交接

**后续更新（2026-10-18）**：通用自动调优已落地；ViT现使用FP16激活张量、FP32 LayerNorm/softmax归约和默认FP16 K partial＋NPU FP16 Add tree。三档21个HOST_STORAGE shape已重新搜索；完整性能与精度偏移见[VISION_FP16_STORAGE.md](VISION_FP16_STORAGE.md)。下文是改造前固定FP32 host接口基线，保留为历史对照。

运行时默认`--vision-dtype fp16`，可手动`--vision-dtype fp32`切回FP32中间张量/CPU计算，NPU GEMM仍为FP16。Rust入口为`ModelOptions::vision_dtype`；通过参数选择，没有对应环境变量。

2026-10-18。FP16 通用调优由另一 agent 负责；本轮固定默认配置作为正确性基线。共享工作树的 wrapper 和算子库后续由另一 agent 更新；此处描述实测基线，不覆盖其新调优版本。

通用 FP16/BF16 调优接口与使用方式见 [FLOAT_MM_AUTOTUNE.md](FLOAT_MM_AUTOTUNE.md)。当前视觉 Linear 可按实际逻辑 M 读取布局、N tile 与 M 分块 profile；下文保留实测固定基线。

## C7 的实际静态 Linear

| 投影 | 每图逻辑 M | K | N | 层数/调用组 |
|---|---:|---:|---:|---:|
| patch Conv3d 展平 | 256–1024 | 1536 | 1024 | 1 |
| QKV | 256–1024 | 1024 | 3072 | 24 |
| Attention output | 256–1024 | 1024 | 1024 | 24 |
| MLP up | 256–1024 | 1024 | 4096 | 24 |
| MLP down | 256–1024 | 4096 | 1024 | 24 |
| merger FC1 | 64–256 | 4096 | 4096 | 1 |
| merger FC2 | 64–256 | 4096 | 2560 | 1 |

M 取决于保持比例的 resize 和32像素对齐，非方图不必是2的幂；当前每图最多1024 patch /256个视觉 token。当前最小面积65536；改变 processor 参数后应按实际 grid 建立 shape key。

本轮验证的固定FP16 wrapper（源码副本见 `evidence/qwen35_vision_20261018/baseline-source/src/npu/floating.rs`）将所有逻辑 M 分为128行物理调用，尾部补零；`FUSE_NONE`、native A(K8/M8)、native D(N8/M8)、`n_tile=0`、用户指定 core mask（默认7）。C7 BF16 权重转为FP16并常驻设备，按 K/N 共享计划和工作缓冲。输出收集为FP32后加FP32 bias，后续 LayerNorm、GELU、residual 使用FP32。Qwen3.5 decoder 的 W4/W8 调优不涉及该模块。

## 调优接口需要保持的边界

- 计时覆盖 FP32→FP16、pack A、cache sync、submit/wait、cache sync、unpack FP32、bias；权重上传和plan构建另列。
- shape/profile 与用户 core mask、ABI/调度版本和布局对应，不能从仅NPU耗时选择胜者。
- 同时考虑逻辑 M 与物理分块。当前 M128 只是基线；更大 batch 或更小尾块可以由通用机制选择。
- plan、packed weight 与workspace必须来自兼容配置；切换策略后不能沿用旧布局缓冲。
- split-K 是独立数值路径，有FP16 partial rounding；记录单算子误差，并复跑 ViT feature 与图文 logits 验证。
- 无因果 Attention 独立通过 `rk_npu_attention_f16_*` 调用，不属于静态 Linear 调优。每层先加载完整图像 K/V，再以≤128 query行执行；图像之间分开处理。

底层现有候选参数见 `rk_npu_matmul_f16.h`：A/D布局、N tile、core mask、split-K；固定基线没有FP16自动搜索/profile API；共享工作树后续开发中的新API应由调优agent记录与验证。既有经验见 `../lowlevel/docs/fp16_path_optimization_2026-07-20.md`，其中也明确列了 autotune TODO。ViT 的性能收益需要由新固定频率测量确认。

## 可复用验收入口

- `tests/tools/vision_probe.rs --image IMAGE --output DIR`：完整ViT、输出FP32 features、再次编码验证feature cache。
- `scripts/qwen35/vision_reference.py`：固定Transformers revision的原始ViT定义与C7权重；支持逐层feature对照。
- `npu::vision_tests::fp16_linear_and_noncausal_attention_match_cpu`：M3/M129尾块和全图Attention的板端CPU oracle。
- `tests/tools/vision_probe.rs --cases CASES.json --output DIR`：一次模型加载，多个图文prompt及固定continuation logits。
- `scripts/qwen35/validate_vision_api.py`：多图/历史/流式/换图/取消恢复；通过 `scripts/run_board.py` 在共享板锁内运行。

本轮数据入口见 [QWEN35_VISION_PROGRESS.md](QWEN35_VISION_PROGRESS.md)。单次带trace编码耗时只用于功能探针，不作为调优性能基线。
