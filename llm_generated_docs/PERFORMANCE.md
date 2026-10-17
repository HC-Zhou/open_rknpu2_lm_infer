# RK3588 performance

现行 `bench` 的 prefill 与历史构建不执行 lm_head，decode 每步执行一次输出头；报告提供阶段计时和无 head 路径的继续解码一致性检查。本文历史表保留原测量口径（prefill 末行一次 head），不与新口径混算。Qwen3.5 / Qwen3 同板新测见 [QWEN35_VS_QWEN3_PERFORMANCE.md](QWEN35_VS_QWEN3_PERFORMANCE.md)。

## 三款模型的上下文测速

原厂 RKLLM 1.3.1 的完整对照、同为3800-token 历史的 decode128 实测及前缀 PPL 见 [SDK_COMPARISON.md](SDK_COMPARISON.md)。Qwen3 性能数据对应 `Qwen3-4B-W4A8-Mixed-LMHeadW8`，PPL 对应 `selection/models/C7-Rotate-QQQ`。

配置为 context8192、chunk64、CPU线程4、CPU mask `0xf0`、Linear NPU mask7、Attention tile512。Granite 使用 D64/R5 三核 Attention，MiniCPM5 使用 D128/R8 两核 Attention，Qwen3 使用 D128/R4 三核 Attention。各用例1次预热、3轮计时，单位 token/s，报告均值和样本标准差。

| 模型 | pp512 | pp1024 | pp4096 | decode128 @512 | decode128 @1024 | decode128 @4096 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 169.71 ± 0.13 | 157.03 ± 0.16 | 116.97 ± 0.05 | 10.497 ± 0.003 | 10.090 ± 0.004 | 8.223 ± 0.005 |
| Granite-4.2-3B | 182.48 ± 0.11 | 164.16 ± 0.24 | 110.31 ± 0.07 | 9.295 ± 0.012 | 9.054 ± 0.004 | 7.712 ± 0.007 |
| MiniCPM5-2B | 245.47 ± 0.28 | 219.87 ± 0.22 | 173.58 ± 0.03 | 13.151 ± 0.029 | 12.954 ± 0.005 | 11.546 ± 0.010 |

pp 从空 KV 开始；decode128 的历史 KV 深度分别为512、1024、4096，128步结束后深度增加128。所有结果包含完整同步模型调用和输出头，历史 prefill 在计时前完成。

各模型在固定频率下预备独立缓存；正式轮次零调优 miss。原始样本、固定输入、domain 与调优统计、时钟固定和恢复记录见 [context-performance.json](evidence/context-performance.json)。同一输入的64-token prefill 加两步 teacher-forced decode，Granite/MiniCPM5 调优前后的完整 logits 逐值相同，见 [tuned-model-logits.json](evidence/tuned-model-logits.json)。

在板上分别为每个模型运行，`/path/to/model-cache` 使用各自独立目录：

```sh
python3 scripts/run_board.py --wait-lock --report warm-clocks.json -- \
  target/release/open_rknpu2_lm_infer --model /path/to/model --ctx-size 8192 --ubatch-size 64 \
  --tune-cache /path/to/model-cache tune --batches 1,64 \
  --report tune.json
python3 scripts/run_board.py --wait-lock --report measured-clocks.json -- \
  target/release/open_rknpu2_lm_infer --model /path/to/model --ctx-size 8192 --ubatch-size 64 \
  --tune-cache /path/to/model-cache bench --suite contexts \
  --repetitions 3 --report measured.json
```

正式报告的 `stats.tune_calls` 与 `stats.strategy_fallback_calls` 应为0。`tune_cache_misses` 可能包含没有选用的 W8 分块候选，不能用它代替实际执行路径的检查。最大上下文容量至少4224，以容纳4096-token 历史与128步 decode。Granite 的 W8 8192×2560 down projection 使用2048/4096的 K 分块候选。

## 三款模型的前缀 PPL

WikiText-2 raw test 的前32个独立512-token窗口，context512、chunk64，评分零起始 target 位置257–511，共8160个目标；每窗重置 KV，不添加 BOS/EOS 或 chat 模板。

| 模型 | 同权重 FP32 Attention 参考 PPL | RK3588 PPL | PPL 变化 |
|---|---:|---:|---:|
| Qwen3-4B / C7-Rotate-QQQ | 12.5165 | 12.5317 | +0.12% |
| Granite-4.2-3B | 16.2676 | 16.0519 | -1.33% |
| MiniCPM5-2B | 13.7724 | 13.8073 | +0.25% |

参考使用相同量化权重、FP32 Attention 和 INT32 Linear。Qwen3 的 PPL 使用 `selection/models/C7-Rotate-QQQ`，性能采用同结构、同混合位宽分布的 `Qwen3-4B-W4A8-Mixed-LMHeadW8` 实测。逐窗 PPL 对照见 [model-ppl.json](evidence/model-ppl.json)。

## Qwen3 与 C++ native 对照

Qwen3-4B mixed W4/W8、W8 NPU lm_head、原生 FP16 GQA。配置为 context4096、chunk64、CPU线程4、CPU mask `0xf0`、NPU mask7、Attention tile512。

### 完整模型调用

短测按 C++ → Rust → Rust → C++ 执行，每个进程 warmup1、计时3轮，每个程序共6个样本。单位为 token/s，表中给出均值和样本标准差。

| 测试 | C++ native | Rust | Rust相对变化 |
|---|---:|---:|---:|
| pp512 | 166.09 ± 1.16 | 168.14 ± 7.63 | +1.23% |
| tg128，空KV | 10.777 ± 0.073 | 10.756 ± 0.046 | -0.19% |
| pp4096 | 118.25 ± 0.80 | 117.17 ± 0.79 | -0.91% |
| tg128，历史2048 | 9.325 ± 0.035 | 9.345 ± 0.005 | +0.22% |

完整原始样本见 [short-performance.json](evidence/short-performance.json)。全部样本计入统计，短测结果表明两个实现的平均速度接近。

pp4096 和历史2048 的 tg128 来自每个程序3个样本的配对长测；完整样本与同一 suite 的重复短用例见 [representative-performance.json](evidence/representative-performance.json)。

### 内存

上述短测的进程峰值 RSS：C++ 为 5.03 GiB，Rust 为 2.96 GiB，Rust 下降约41%。RSS 使用进程启动至测速结束的高水位，包含启动时的权重加载。

长测将 KV 扩到4096 后，累计峰值 RSS 为 C++ 5.03 GiB、Rust 3.35 GiB，Rust 下降约33%。

Rust 每个投影解包后回收已读取的 mmap 页，上传后释放解包缓冲。KV 在请求开始前按完整请求预算预留；容量保留供后续请求复用。IOMMU domain 仍按配置的最大上下文预算划分。

### 测量方法

固定 NPU1 GHz、DDR2.112 GHz、CPU4..7 2.4 GHz。每个程序使用各自真实 domain 布局对应的独立调优缓存，所有正式轮次为零调优 miss。`scripts/run_board.py` 持有共享板锁，并在命令结束后恢复频率。

输入为 `std::mt19937(42) % vocab_size` 对应的固定 token IDs，比较脚本检查输入相同。pp/tg 独立测量，每轮重置 KV；pp 每次只计算一次输出头，tg 每 token 调用一次完整模型。历史 KV 在计时前建立。

计时包括 NPU 提交与同步、CPU 运算、Attention、KV append 和输出头，排除加载、调优、warmup、预留、分词、采样与 HTTP。数据衡量完整模型 forward，不等于客户端端到端生成速度。

提供同一算子库构建的 C++ native baseline 后可复现对照：

```sh
python3 scripts/run_compare_board.py --model /path/to/model \
  --native-bin /path/to/llm_infer --rust-bin target/release/open_rknpu2_lm_infer \
  --cache /path/to/tuning --output results/comparison --mode short
```

`--mode representative` 测量 pp512、pp4096、空KV tg128 和带2048-token 历史的 tg128。已预备缓存时可用 `--reuse-cache`；正式轮次仍要求零 miss。

## 限制

吞吐结果对应上述三款 mixed checkpoint 与 RK3588。C++ native 对照和 RSS 数据对应 Qwen3-4B、context4096。RSS 和各对象 payload 使用不同统计口径；RSS 不替代完整设备与内核内存统计。其他模型格式、上下文和采样服务负载需要分别测量。

## ubatch128/192/256 后续小测

应用默认ubatch已改为128；[完整模型小测](UBATCH_SWEEP_20261015.md)记录128/192/256的吞吐和数值比较。上文历史chunk64数据及复现参数保留原值。
