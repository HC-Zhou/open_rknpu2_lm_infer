# Qwen3.5-4B：C7 与 RKLLM W8A8 的纯文本性能及 PPL

原生测试日期2026-10-19，RKLLM对照日期2026-10-20，Asia/Shanghai。本轮只测纯文本性能及精度。

## RKLLM 实测对照

同一块RK3588、相同固定频率、完全相同的真实文本token IDs，生成128个token。
下表每格为`prefill / decode`，单位均为token/s；倍率为C7 MTP1相对本次RKLLM实测。

| 输入token | RKLLM W8A8 P / D | C7 AR P / D | C7 MTP1 P / D | MTP1 P倍率 | MTP1 D倍率 |
|---|---:|---:|---:|---:|---:|
| 512 | 58.347 / 5.538 | 152.780 / 9.170 | 147.832 / 10.611 | 2.534× | 1.916× |
| 1024 | 57.844 / 5.416 | 152.548 / 9.111 | 148.166 / 10.379 | 2.561× | 1.917× |
| 3800 | 54.029 / 4.565 | 143.378 / 8.721 | 139.917 / 9.896 | 2.590× | 2.168× |

常驻模型完整执行时间，单位为秒；模型加载、输入分词和KV清空均不计入：

| 输入token | RKLLM | C7 AR | C7 MTP1 | RKLLM / MTP1时间比 |
|---|---:|---:|---:|---:|
| 512 | 31.724 | 17.211 | 15.450 | 2.053× |
| 1024 | 41.169 | 20.662 | 19.166 | 2.148× |
| 3800 | 98.169 | 41.080 | 40.012 | 2.454× |

RKLLM每档预热1次、正式3次，原生每模式每档正式2次；吞吐按固定token数除以正式耗时均值计算。
RKLLM使用SDK阶段计时，完整执行时间使用调用外侧单调时钟，均保存逐次样本和标准差。
两种模型的量化方案不同，性能倍率与下文PPL共同构成这组对照。

## 配置与输入

RK3588备用1，CPU4–7固定2256MHz、三个NPU核固定1GHz，DDR未受控。
主仓基于`88e2818`，native为`2b3b438`；包含删除辅助浮点投影的外部8行padding及最小M tile限制。
原生模型为`transfer/qwen35/models/C7`，混合W4/W8、共享W8输出头、ubatch128、context4096。
官方模型为`vendor_baseline/Qwen3.5-4B_w8a8_rk3588.rkllm`，5,541,501,724字节。
Toolkit和Runtime均为1.3.1，W8A8、normal、optimization_level=1、hybrid_rate=0、三NPU核、max_context=4096。
导出使用208条真实校准记录、每条最长512个token；转换参数记录于`vendor-conversion.json`。
SDK执行设置embed_flash=1、n_batch=1、CPU mask=0xf0；本板rknpu驱动为0.9.6，SDK日志提示建议0.9.7。
RKLLM初始化耗时40.662秒，单列保存，未进入性能表。
模型加载、KV重置、reserve和策略准备不计入P/D，setup单列；request包含setup、prefill、采样及完整生成，不含分词、模板渲染和HTTP。

输入取自已有WikiText-2 raw test语料的真实片段，使用chat模板要求详细分析，关闭thinking。
每个提示通过模型自己的TextCodec编码，精确为512、1024或3800个token。
没有随机token或占位token填充；`inputs.json`保存原始文字、完整模板和实际token IDs。
两条执行路径均使用greedy采样，temperature=0，实际生成128个token；第一个来自prefill，decode分子为其后的127个。
原生seed=42。RKLLM每次计时前调用`clear_kv_cache(False)`，实测每次SDK prefill_tokens均等于完整输入长度，generate_tokens均为127。

AR/MTP1顺序为`[AR,K1,K1,AR]`。每个配置块、每个输入先预热一次，再测量一次，
因此每档每模式有两个正式样本。重复的是同一greedy轨迹，不是独立语料样本。
比较使用正式LanguageModel、Sampler和speculative driver，包含完整draft/head/verify/repair，未使用固定continuation替代生成。
K3另测两个顺序配置块，每块每档同样预热一次、正式一次，并逐token对照上述AR输出；K3未与AR额外执行ABBA。

## 原生AR、MTP1与MTP3

吞吐由固定token数量除以两次耗时均值计算。Prefill栏为`ms / token/s`：

| 输入token | AR P | MTP1 P | MTP3 P |
|---|---:|---:|---:|
| 512 | 3351.221 / 152.780 | 3463.388 / 147.832 | 3420.812 / 149.672 |
| 1024 | 6712.651 / 152.548 | 6911.157 / 148.166 | 6897.414 / 148.461 |
| 3800 | 26503.442 / 143.378 | 27159.007 / 139.917 | 27067.734 / 140.389 |

Decode单位为token/s：

| 输入token | AR D | MTP1 D | MTP3 D | K1/AR | K1接受率 | K3接受率 |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 9.170 | 10.611 | 10.038 | 1.157× | 72.60% | 49.67% |
| 1024 | 9.111 | 10.379 | 9.599 | 1.139× | 69.33% | 46.54% |
| 3800 | 8.721 | 9.896 | 8.734 | 1.135× | 68.00% | 41.67% |

包含setup、prefill和生成的常驻模型执行时间，单位为秒：

| 输入token | AR | MTP1 | MTP3 | AR/K1时间比 |
|---|---:|---:|---:|---:|
| 512 | 17.211 | 15.450 | 16.091 | 1.114× |
| 1024 | 20.662 | 19.166 | 20.147 | 1.078× |
| 3800 | 41.080 | 40.012 | 41.631 | 1.027× |

这三条真实文本输入中K1均优于K3。3800输入下，K1减少decode时间，但前缀准备增加约0.66秒，完整执行收益小于decode吞吐收益。

正式样本的阶段耗时、decode标准差及draft/verify/repair分项见`summary-native.json`。
全部预热与正式样本共用逐token的AR参考，**同一输入的128个输出token IDs逐值一致**。

## PPL

本轮原生及RKLLM PPL使用完全相同的WikiText-2 raw test前32个独立512-token窗口，逐窗清KV，
对target位置257–511评分，共8160个目标，无chat模板及额外BOS/EOS。
原生NLL使用FP32 shifted exp和FP64 sum/log，RKLLM取完整logits后使用FP64 logsumexp；
两条路径的8160个目标token IDs逐项一致，逐token NLL重新汇总与各自保存的PPL一致。

| 执行路径 | PPL | 来源 |
|---|---:|---|
| 未量化原模型，FP32计算 | 8.146891 | 已有相同窗口参考 |
| C7量化权重，FP32整数参考 | 8.601783 | 已有相同窗口参考 |
| C7原生RK3588 | **8.579026** | 本轮实测 |
| 官方RKLLM W8A8 | **8.519662** | 本轮同板实测 |

原生相对C7 FP32执行参考的PPL差为-0.265%，
相对未量化原模型为+5.304%。
前者衡量运行时数值偏移，后者同时包含C7量化影响；不把两者混作同一精度结论。
RKLLM相对未量化原模型的PPL差为+4.576%，
C7原生相对RKLLM为+0.697%。
MTP的生成一致性按上述真实请求核对，不使用draft接受率代替精度检查。

## 调优与有效性

使用独立缓存，复用既有target小批量及MTP浮点策略，补齐3800输入最后88行的MTP FC/KV策略后再加载模型。
离线浮点候选沿用当前辅助decoder的N tile≤4096约束；正式执行链接生产库。
原生的全部预热/正式请求均**0策略miss、0在线搜索、0fallback、0权重reload**，缓存文件在测量前后不变。
649份实时采样均为CPU2256MHz/NPU1GHz、affinity4–7；最高bigcore1温度65.615℃。
K3另有395份实时频率样本，检查同样通过。四个原生NPU runner均退出0并恢复频率上下限及governor，restore_errors为空。

RKLLM性能测量有649份实时频率样本，均为CPU2256MHz/NPU1GHz，主线程affinity始终位于CPU4–7，
最高bigcore1温度68.384℃。性能和PPL两个runner均退出0，频率上下限及governor恢复检查通过。
RKLLM同一输入的预热和3个正式请求输出token IDs一致；C7的AR/K1/K3一致性如上单独核对。

## 测量边界

每档使用一条真实请求，重复测量同一greedy轨迹；它们反映这三条输入上的表现。
原生与RKLLM在相邻日期顺序测量，共用固定CPU/NPU频率；DDR未受控。
RKLLM为这次提供的W8A8导出配置，SDK推荐驱动版本与板端现有版本的差异已在上文记录。
完整执行时间包括各自调用边界内的CPU处理；本轮未测HTTP、多模态或模型加载后的首个冷请求延迟。

## 证据

证据目录为`llm_generated_docs/evidence/qwen35_text_pp512_1024_3800_tg128_20261019/`。
`native.json`保存全部输入/输出IDs、文字和逐轮统计；`inputs.json`保存真实提示；
`summary-native.json`保存原生汇总；`ppl/native.json`保存原生8160项NLL；各clocks文件保存恢复检查。
`vendor-perf.json`和`vendor-ppl.json`保存RKLLM全部请求、输出及NLL；`summary-comparison.json`保存对照汇总，
`vendor-conversion.json`和`vendor-tools/rkllm-runtime-version.json`记录导出参数与Runtime来源。
板端长期数据位于`/home/firefly/mnt/3090x4/tmp/qwen35-dev/text-pp512-1024-3800-tg128-20261019/`。
