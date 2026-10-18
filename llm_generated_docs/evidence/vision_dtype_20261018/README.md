# ViT 张量精度参数验证

2026-10-18。`--vision-dtype fp16|fp32`，默认`fp16`；Rust参数为`ModelOptions::vision_dtype`。选择FP32时，层间张量与CPU运算使用FP32，视觉NPU GEMM及Attention QK/PV仍使用FP16。两种模式共用同一份encoder执行流程，通过类型化storage实现调用对应算子；没有精度环境变量。

## 板端输出对照

| 验证 | 结果 |
|---|---|
| 默认FP16，256²/384×256/512² | 全部features与冻结FP16＋向量RoPE实现逐位一致 |
| `--vision-dtype fp32`，三种尺寸 | 全部features与冻结的旧FP32激活图逐位一致 |
| 显式`--vision-dtype fp16`，红图 | 与默认模式逐位一致 |
| `RK_NPU_MM_K_TILING=off`，两种dtype，红图 | 各自与冻结完整K路径逐位一致 |
| 两种host类型的21个profile只读加载 | 各21 hits / 0 miss / 0 search |
| feature cache | 每个probe一次真实编码、一次命中，输出逐位一致 |

probe报告的`vision_dtype`与参数一致。CLI测试覆盖默认值、run/chat/serve下的手动FP32参数和非法值拒绝。默认测试lib47 passed/9 ignored、CPU集成13+1+5 passed；另显式运行FP16 Linear/NO_MASK Attention板端回归1 passed。fmt、Release全targets Clippy `-D warnings`通过。不运行BF16 CPU数值计算，不比较文本logits。

所有板端工作在共享锁内执行；NPU1GHz、DDR2112MHz、CPU大核2.4GHz/4线程/mask0xf0。runner返回0，恢复错误为空，并核对CPU/NPU/DDR min/max/governor恢复。这里只验证功能与数值保持，不声称新的性能改善。

机器可读结果见[summary.json](summary.json)，原始runner、测试输出和profile报告位于本目录。冻结可执行文件与features位于`/home/zt/mnt/3090x4/tmp/vision-dtype-20261018/`，板端对应`/home/firefly/mnt/3090x4/tmp/vision-dtype-20261018/`。
