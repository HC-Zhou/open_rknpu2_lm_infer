# Qwen3.5-4B C7 与 Qwen3-4B 同板性能对比

日期：2026-10-17，Asia/Shanghai（UTC+08:00）。

本报告最近一次完整性能测量对应下文 SHA256 为 `90770d...e8876` 的重构前二进制。后续公共模块整理保留这些原始数据，重构后的数值回归单列于 [RUNTIME_MODULE_LAYOUT.md](RUNTIME_MODULE_LAYOUT.md)；本次目录/组件重组没有重跑完整性能矩阵。

## 测试协议

使用备用1 `firefly-rk3588-node01`（Linux 5.10.198、32 GB RAM），两款模型使用同一份受测二进制。CPU 4 线程、mask `0xf0`，NPU mask `7`；大核固定 2256000 kHz，NPU 固定 1 GHz。备用板没有 DMC devfreq，DDR 未受控；因此这里重新测量两款模型，不直接引用主板 README 的历史数字。

| 模型 | Checkpoint | 文本层数 | 词表 | Linear 位宽 |
|---|---|---:|---:|---|
| Qwen3.5-4B | `transfer/qwen35/models/C7` | 24 GDN + 8 Attention | 248320 | 197 W4 + 52 W8，含 W8 lm_head |
| Qwen3-4B | `Qwen3-4B-W4A8-Mixed-LMHeadW8` | 36 Attention | 151936 | 246 W4 + 7 W8，含 W8 lm_head |

模型根目录为交接项目 `w4a8_quant`。这比较上述两个真实 checkpoint 的完整 AR 推理速度，不代表相同网络算子的等量工作，也未启用 MTP。

统一 context4224、ubatch128、Attention tile512，分别使用独立缓存显式 `tune --batches 1,128`。正式进程只读缓存，检查 `tune_calls=0`、`weight_reload_calls=0`、`strategy_fallback_calls=0`。离线调优和模型加载耗时不计入吞吐。

每个进程测 `-p 0 -n 0 -pg 512,128 -pg 4096,128`，prefill 后保持状态连续 decode128；每个用例先预热1次、计时3次。主表采用修正后的无 head 口径，Qwen3.5 → Qwen3 配对测量，每个模型每个用例3个正式样本，报告描述性均值和样本标准差。输入采用 MT19937(42) 模各自词表；不同词表没有声称输入 token IDs 相同。早期包含末行一次 head 的结果只作为 legacy 记录，不并入主表。

prefill 和未计时的历史构建均调用独立 `LanguageModel::prefill`，不执行 final output norm / lm_head，并逐样本断言 `head_ms=0`；decode 每步执行完整 forward 和一次输出头。计时包含 CPU 操作、KV/GDN 状态更新、NPU 提交与同步，排除加载、准备、分词、采样、HTTP 和阶段统计读取。这里使用固定输入连续测量主干与 decode，不把它称为实际客户端请求延迟。

两个后端都在计时前检查：128-token 无 head prefill 后的下一 token logits，必须与普通 forward 建立同一前缀时逐值一致。阶段统计单列 Linear（含 fused finish）、CPU GDN mixer、Attention 调用、其他 CPU 和 head，避免把整轮 decode head 累计时间归入 prefill。

## 最近一次正式结果：C++ GDN 与批量 Attention

当前 release 二进制 SHA256 `90770d45181b40a619e2613b8f876f9f78111f3e397f15e462153e2c930e8876`。仍采用前述同板、同频率、每档预热1次/正式3次协议，两个模型均重新测量。prefill 不执行 final norm / lm_head。单位 tok/s，均值 ± 样本标准差。

| 用例 | Qwen3.5 C7 当前 | Qwen3-4B 同轮 | C7 相对变化 |
|---|---:|---:|---:|
| pp512 | 130.911 ± 0.151 | 180.740 ± 0.329 | -27.57% |
| tg128，前缀512 | 8.904 ± 0.010 | 9.928 ± 0.003 | -10.32% |
| pg512,128 | 34.998 ± 0.036 | 40.698 ± 0.007 | -14.01% |
| pp4096 | 124.487 ± 0.063 | 120.497 ± 0.111 | +3.31% |
| tg128，前缀4096 | 8.458 ± 0.010 | 8.020 ± 0.004 | +5.47% |
| pg4096,128 | 87.934 ± 0.052 | 84.559 ± 0.045 | +3.99% |

相对初版无 head C7，pp512由58.335升至130.911（2.244×），pp4096由49.780升至124.487（2.501×）。短前缀prefill仍比Qwen3慢27.57%；4096档prefill已快3.31%。decode主要受Linear/输出头影响，短前缀慢10.32%、长前缀快5.47%。

### 为什么此前 prefill 慢

在上一轮C++ GDN已优化、Attention尚为逐query调用时，pp4096阶段如下（秒）：

| 阶段 | C7：24 GDN + 8 Attention | Qwen3：36 Attention | 差值 |
|---|---:|---:|---:|
| Linear，含fused finish | 14.959 | 14.532 | +0.427 |
| 完整CPU GDN mixer | 12.486 | 0 | +12.486 |
| Attention完整调用 | 17.724 | 17.893 | −0.169 |
| 其他CPU | 3.042 | 1.474 | +1.568 |

C7多出的约14.3秒中，CPU GDN约占87%；另外8层Attention几乎与36层同耗时，是因为Qwen3.5路径遗留 `run_sequential_queries`，每次只提交1个query，而Dense路径已批量执行。该路径未修复早期chunk数值诊断，却保留了额外开销。现已统一为 `Attention::run`，每次最多128个query，并删除不再使用的逐query API。

修改后C7 pp4096的Attention从17.724降至2.861秒（约6.19×）；pp512从0.755降至0.157秒。相对仅优化GDN的上一版，pp512再快14.82%、pp4096再快46.53%。当前pp4096主要阶段为Linear14.790秒、完整CPU GDN12.263秒、Attention2.861秒、其他CPU2.984秒；core recurrence以外的卷积、norm、gate、scatter/检查等仍计入GDN mixer，不能用纯kernel微基准替代该时间。

### 精度与证据

当前32窗/8160个目标的PPL为 **8.578578429**，同窗口C7 FP32参考8.601783301（−0.269768%），通过1%调查线。批量Attention前后全部8160项NLL逐值相等，也与上一轮Rust四路SIMD版本逐项相等。两后端无head前缀的后续logits一致性检查通过；所有prefill样本 `head_ms=0`，正式进程tune/reload/strategy fallback均为0。

全套release测试54通过、7个既有硬件用例保持ignored；额外运行通用Attention原生CPU oracle通过（含Qwen3.5的16/4/256 geometry）。全target release Clippy `-D warnings`、format通过。4次runner（Attention oracle、两个模型性能、PPL）均正常退出，受控频率上下限/governor恢复。

当前证据入口：[qwen35_batch_attn_20261017/validation.json](evidence/qwen35_batch_attn_20261017/validation.json)，包含原始样本、逐token NLL、源码/二进制身份、构建测试与频率记录。仅优化C++ GDN的中间整模型结果见 [qwen35_cpp_e2e_20261017/](evidence/qwen35_cpp_e2e_20261017/)。独立GDN kernel复现与NPU实验数据见 [QWEN35_GDN_KERNEL_REPRO.md](QWEN35_GDN_KERNEL_REPRO.md)。

## 历史初版无 head 基线

初版无 head 基线已完成；下表为优化前结果，llama.cpp 内核对齐后的复测单列，避免混合不同二进制的样本。单位为 tok/s，均值 ± 样本标准差。

| 用例 | Qwen3.5 C7 初版 | Qwen3-4B | C7 相对变化 |
|---|---:|---:|---:|
| pp512 | 58.335 ± 0.004 | 182.287 ± 0.344 | −68.00% |
| tg128，前缀512 | 8.586 ± 0.003 | 9.919 ± 0.005 | −13.44% |
| pg512,128 | 27.022 ± 0.006 | 40.731 ± 0.004 | −33.66% |
| pp4096 | 49.780 ± 0.033 | 120.590 ± 0.122 | −58.72% |
| tg128，前缀4096 | 8.163 ± 0.002 | 8.018 ± 0.003 | +1.81% |
| pg4096,128 | 43.119 ± 0.022 | 84.599 ± 0.067 | −49.03% |

初版 pp512 的 CPU GDN mixer 为 5608.81 ms，占 8776.86 ms 的 63.9%；pp4096 为 45146.39 / 82281.57 ms，占 54.9%。这里的 mixer 包含卷积、激活、norm、状态处理和递推，不将其全部算作 recurrence kernel。两档 prefill 的 head 时间均为 0。

原始样本与受控频率恢复记录见 [evidence/qwen35_vs_qwen3_20261017/](evidence/qwen35_vs_qwen3_20261017/)。基线二进制 SHA256 为 `e38e8f58293da177a4ff371d5083a0511350c15a66b3fc011b763bfad105197c`，源码清单在同目录。

## Rust 四路 SIMD / 卷积优化后的历史复测

此轮已在用户要求集中算子优化之前完成，二进制 SHA256 `6e9e7c01050ab37ecca82deca18e20851b38e4555a980e9f8204bf4fb7fdb370`。之后切换独立 C++ GDN 模块并完成算子级测试；用户随后授权恢复端到端测试，当前 C++ 复测单独记录。下表只代表历史 Rust 工作树。

| 用例 | C7 Rust 优化版 | 同轮 Qwen3-4B | C7 相对变化 |
|---|---:|---:|---:|
| pp512 | 100.080 ± 0.573 | 181.320 ± 0.248 | −44.80% |
| tg128，前缀512 | 8.881 ± 0.001 | 9.938 ± 0.008 | −10.64% |
| pg512,128 | 32.772 ± 0.052 | 40.755 ± 0.038 | −19.59% |
| pp4096 | 78.181 ± 0.163 | 120.796 ± 0.058 | −35.28% |
| tg128，前缀4096 | 8.432 ± 0.001 | 8.016 ± 0.004 | +5.19% |
| pg4096,128 | 62.511 ± 0.103 | 84.688 ± 0.042 | −26.19% |

相对初版 C7，pp512 +71.56%、pp4096 +57.05%；完整 CPU mixer 从 5608.81 → 2039.21 ms、45146.39 → 15949.61 ms，decode128 mixer 从约2185 → 1677 ms。仍有 Linear、Attention 和其他 CPU 成本，不能用 mixer 加速比代替整模型收益。

同轮32窗 PPL 为 8.578578429，同窗口 FP32 参考8.601783301，相对差异 −0.269768%，在1%调查线内；该 PPL 也属于上述 Rust 二进制，不移用于后续 C++ 改动。原始报告、测试、构建与频率记录保存在 [evidence/qwen35_gdn_rust_alignment_20261017/](evidence/qwen35_gdn_rust_alignment_20261017/)。

C7 当前新增了 `model-mtp.safetensors` 中15个未量化 BF16 MTP tensor；清单声明 `source_values=unchanged`、`target_coordinate_alignment=required`。最初8个分片的SHA256仍逐一匹配首轮记录。两次性能进程均未启用MTP，新增文件没有改变被测文本路径。

## 复现

在备用1的构建目录执行，路径均为板端可读路径：

```sh
python3 scripts/qwen35/benchmark_models.py \
  --binary target/release/open_rknpu2_lm_infer \
  --qwen35 /path/to/w4a8_quant/transfer/qwen35/models/C7 \
  --qwen3 /path/to/w4a8_quant/Qwen3-4B-W4A8-Mixed-LMHeadW8 \
  --output /path/to/perf-comparison --paired

python3 scripts/qwen35/summarize_benchmark.py \
  --directory /path/to/perf-comparison --output /path/to/perf-comparison/comparison.json
```

上述脚本专用备用1的 2256000 kHz / DDR 不受控配置。已有对应缓存时可用 `--reuse-cache`；正式进程仍要求缓存完整。
