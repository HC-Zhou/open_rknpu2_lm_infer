# Attention LUT exponential

Stable Attention 默认使用 NPU LUT exponential 和 PINGPONG 提交。QK 后由 CPU
计算行最大值、应用 mask，NPU 执行 exponential 与 PV，最后以 FP32 修正 LUT
偏置并归一化。FP16 PV 幅值或提交任务数超限时自动使用 CPU softmax。

在启动进程前设置 `RK_NPU_ATTN_LUT_EXP=0` 恢复原有 CPU softmax，包括原有 KV
更新与截断行为。变量在 KV cache 创建时读取，适用于 CLI、TUI 和 HTTP 服务：

```sh
RK_NPU_ATTN_LUT_EXP=0 target/release/open_rknpu2_lm_infer \
  --model /path/to/model bench -p 512 -n 128
```

默认不需要设置环境变量。寄存器提交对照可使用
`RK_NPU_ATTN_LUT_SUBMIT=pc`；正常路径默认 `pingpong`。

## Qwen3-4B 结果

RK3588，CPU/NPU/DDR 固定为2.4/1/2.112 GHz，四个大核、三个 NPU 核，
ubatch128、KV tile512，Clang 22.1.3/O2。相同模型、相同构建切换 softmax 路径；
性能使用 CPU/LUT/LUT/CPU 顺序，每个进程预热一次、正式三次。

| 指标 | CPU softmax | NPU LUT + PINGPONG | 变化 |
| --- | ---: | ---: | ---: |
| prefill512 mean / ms | 2889.424 | 2841.546 | 吞吐 +1.68% |
| decode128 mean / ms | 12281.484 | 12283.904 | 吞吐 -0.02% |
| 整次进程 CPU 时间 mean / ms | 9353.747 | 8995.460 | CPU 工作量 -3.83% |
| WikiText-2 前缀 PPL | 14.885053 | 14.972756 | +0.5892% |

CPU 时间为所有线程累计值，含 reset/prepare；模型阶段 wall 计时不含这些准备。
PPL 使用 `Qwen3-4B-W4A8-Mixed-LMHeadW8`，32个独立512-token窗口、每窗清空 KV，
评分 target257–511，共8160个目标，不添加特殊 token 或聊天模板。
正常 GCC release 构建在不设置 LUT 环境变量时复验同一输入，全部8160个目标的
NLL 与显式开启的 Clang 路径逐值一致。
数值、配置与验证摘要见 [attention-lut.json](evidence/attention-lut.json)。

通用 GQA 的8种几何、query 行数1/3/31/32/64/128、三种 mask、KV grow、truncate、
替换后缀与 reset 通过。LUT 改变 FP16 Attention 的舍入，完整 logits 可与 CPU
路径不同；上述 PPL 是实际质量影响的前缀评测。

## 范围与内存

该结果对应上述 Qwen3 checkpoint 和输入，不代表其他 checkpoint 的 PPL 或下游任务质量。
每32个缓存 token、每个 V channel 额外保存4 bytes前缀和，另有当前累加和；
workspace 增加一个与 score 等大的 FP16 exponential 缓冲区。
