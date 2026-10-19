# Qwen3.5 MTP 运行时实现与验收

日期：2026-10-19。默认关闭的MTP已完成本轮正确性与入口功能验证。固定K3的三轮ABBA已完成，阈值性能仍限于筛选实验。

目标：真实 Qwen3.5-4B C7 的 MTP 起草、批量 target 验证、接受前缀提交、正确采样与实际生成加速。实现不校准、不重新量化 checkpoint。原始 BF16 MTP 参数保留，运行时 FP16 GEMM 的宽 K 按列分块、FP32 累加；C7 显式坐标适配读取模型附带的 `mtp_coordinate_adapter.json`。

## 实现

- `src/model/qwen35/mtp/`：共享 embedding/head、embedding-first FC、标准 gated full-attention/FFN；使用主干最终 RMSNorm 后的 target hidden。移位为 `h[t] + embedding(x[t+1]) -> x[t+2]`。递归起草使用 MTP 自身归一化 hidden。
- `src/model/qwen35/transaction.rs`：target 验证不改 committed GDN；Attention 追加临时 KV，commit 接受前缀或 abort。`position()` 仅计入实际消费输入。
- `src/speculative/driver.rs`、`sampling.rs`：pending anchor、greedy draft 的 target sample-and-match、probabilistic draft 的真实 q/p 与 `min(1,p/q)`、拒绝后的 `(p-q)+` 和全接受 bonus；proposal history 独立。
- `engine.rs`：验证后先处理 EOS/stop/预算，再提交状态，最后发布流式事件；CLI/TUI/API 共用。启用参数为 `--spec-type draft-mtp`，默认关闭；默认 K3、greedy draft，阈值 0。
- 单层 MTP 的 teacher-forced prefix 只计算 FC、输入 norm 和 K/V 投影，省略无消费者的 Attention/FFN。GDN 全接受保留验证终态并交换提交，部分接受继续 compact replay；两项优化已通过整模型事务和 MTP oracle 复验。

## 命令与配置

模型目录需包含受支持的 Qwen3.5 target 权重、单层共享 embedding/head 的 MTP 权重，以及 tokenizer。旋转坐标的 C7 还需随模型提供的 `mtp_coordinate_adapter.json`；不能直接混用其他 checkpoint 的 adapter。

以下命令在板端运行，`MODEL` 和 `CACHE` 替换为本地模型和专用调优目录。先调优 target 的各个物理行数，再启用 MTP：

```sh
BIN=target/release/open_rknpu2_lm_infer
MODEL=/path/to/Qwen3.5-4B-C7
CACHE=/path/to/qwen35-c7-tune-cache

"$BIN" -m "$MODEL" --tune-cache "$CACHE" --ubatch-size 128 \
  tune --batches 1,2,4,8,16,32,64,128

# 固定 K3；--stats 在 stderr 输出计时、接受数及模型状态。
"$BIN" -m "$MODEL" --tune-cache "$CACHE" --spec-type draft-mtp --spec-draft-n-max 3 \
  run --conversation --temp 0 -n 128 --stats -p '请简要解释归并排序。'

# 可选置信度截短：低于阈值时丢弃当前提议并停止；阈值需按场景测量。
"$BIN" -m "$MODEL" --tune-cache "$CACHE" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  chat --temp 0 -n 128

"$BIN" -m "$MODEL" --tune-cache "$CACHE" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  serve --host 127.0.0.1 --port 8080
```

| 参数 | 默认 | 行为 |
|---|---|---|
| `--spec-type` | `none` | `draft-mtp` 启用 Qwen3.5 MTP，`none` 普通生成 |
| `--spec-draft-n-max` | `3` | 提议上限 0–7；0 关闭，非零必须小于 `--ubatch-size` |
| `--spec-draft-p-min` / `--draft-p-min` | `0` | 有限数 [0,1]；低于阈值的当前提议不进入验证，停止后续起草 |
| `--spec-draft-sampling` | `greedy` | `greedy` 起草 argmax、target 逐 token sample-and-match；`probabilistic` 起草采样并用 q/p 拒绝采样验证 |

起草行为对照 llama.cpp `common/speculative.cpp` 的 Qwen3.5 `draft-mtp` 路径：draft 不继承 target 的 repeat/presence/frequency、top-p 或 min-p；候选仅保留 top10。Greedy draft 置信度为 top10 的 temperature1 softmax 最大概率；probabilistic draft 只继承 target temperature，用该温度归一化候选。阈值与生成参数 `--min-p` 不同。Greedy draft 在 target logits 相同的前提下保持相同 seed 的 AR 输出；probabilistic draft 保持 target 分布，但不承诺相同 seed 的 AR token 序列。RNG 实现和浮点归一化不要求跨运行时逐位一致。

命名、默认选择与起草/截断/验证策略对照本地 llama.cpp `de3ebb93bd8828496faefb44ed45059ec7ed3ea5`（MTP 相关文件等同 upstream `a17e94d229fc7fed1567a799baebd11b771fe5af`）。本运行时仅提供 `none` / `draft-mtp`，最大七个提议，不提供独立 draft 模型或多序列并发。非零阈值要求启用 MTP。

上述参数对整个 `run/chat/serve` 实例生效，API 请求仍使用标准 sampling/stop/max_tokens 字段；`bench/eval/ppl/tune` 执行固定 token 路径，拒绝启用 MTP。MTP 性能实验使用 `cargo run --release --locked --example mtp_bench -- --help` 查看参数。无效范围、阈值组合和不支持的模型架构在打开设备/加载权重前报错。

生产命令不获取全局板锁。重复测速或硬件验证由 `scripts/run_board.py` 外层持锁和管理时钟。CLI 的 `--stats` 及服务日志报告 `mtp_draft_attempts`、`mtp_proposed`、`mtp_accepted`、`mtp_accepted_per_attempt`、`mtp_confidence_stops`，以及完整 draft/verify/repair 时间；接受率需同时说明分母，不能仅报告送验证候选的接受比例。

## 已验证

备用1：`firefly@192.168.10.211`，CPU4–7 固定 2256 MHz、NPU mask7 固定 1 GHz，DDR 未受控。通过 `scripts/run_board.py --wait-lock --cpu-khz 2256000 --skip-dmc` 持锁并恢复频率。

证据根目录为 `/home/zt/mnt/3090x4/tmp/qwen35-dev/mtp-20261018/`；板端前缀为 `/home/firefly/mnt/3090x4/`。以下标识保留测试当时的二进制/阶段身份，新增优化后必须重验。

| 检查 | 当前证据 |
|---|---|
| Rust 默认测试 | `cpu-softmax-20261019/default-tests.log`：65 lib、14 cpu_layers、1 openmp_startup、5 vision_input，共85 passed；13硬件/模型测试默认ignored；已包含删除资格接口后的最终源码 |
| 全 target release Clippy / fmt | `cpu-softmax-20261019/clippy.log` / `fmt.log`：全部通过 |
| 拒绝采样 | 小词表解析概率恒等式、100000 次随机分布检查、历史隔离通过 |
| streaming/stop/提交失败 | mock 生成器覆盖 K1/2/3/4/7、各接受边界、max_tokens0/1/2、EOS、跨 token stop，以及失败提交不发布候选 |
| target 整模型事务 | `integrated-tests.log`：恢复 batched Attention 后，8 个消费边界的 logits 及 commit 后下一行 logits 与串行 AR 逐值一致；重复 abort、预先取消通过；含 GDN 全接受终态交换 |
| MTP 同权重 FP32 oracle | `integrated-tests.log`：output NRMSE 0.0007358932，head input 0.0007268166，cached/full 逐值相等；门槛分别 0.005/0.005/0.001，未调整；KV-only teacher prefix 的下一步 draft 与完整前缀逐值相等 |
| 视觉共享算子 | `cpu-softmax-20261019/vision.log`：最终版本FP16 Linear M3/M129与非因果Attention对照CPU通过，Attention max_abs 1.43e-5 |
| 坐标变换 | target hidden/embedding 相对 dense PyTorch oracle 约 2.6e-7；默认测试另有独立 dense fixture |
| 真实CLI | 3例通过：stdin/参数输入、K3阈值0.75与AR文字相同、图片计数和stats |
| 真实文本API | greedy/probabilistic各13例通过：SSE/JSON、随机采样及惩罚、前缀复用、多轮、stop、预算、断流恢复；HTTP max_tokens=0按协议返回400 |
| 真实图文API / TUI | 9个API、5个TUI用例通过，覆盖图片替换、双图、图文历史、中文、reset及退出 |
| 旧模型 | Qwen3-4B-W4A8-Mixed-LMHeadW8普通CLI输出OK，exit0 |
| carrier修复阶段的prefill PPL | C7同一32窗口/8160目标：8.579026274；FP32参考8.601783301，差−0.26456%。此结果早于CPU分段修复；按用户要求，本轮未重测PPL |
| CPU softmax数值根因 | 真实C7跨1024时归一化越过FP16舍入边界；固定Q/K/V干预确认主因。native分段后16384个真实Attention输出逐位等于AR |
| native数值分段 | pingpong/split两种提交模式各45项板端回归通过，覆盖CPU short/long、V=64/65000、descriptor、preloaded future和回滚；正常CPU batch继续合批 |
| 修复后整模型跨界 | `cpu-softmax-20261019/model-lut0.log` / `model-lut1.log`：LUT开/关共18组R4/R8；verify及abort/部分/全提交后的next logits，合计28,308,480个FP32元素逐位等于AR |

参考生成入口为 `scripts/qwen35/mtp_reference.py`，使用交接项目的 Transformers MTP 定义，BF16 参数转为 FP32 计算。MTP 权重 SHA256 为 `e5fc485d419f2c554169c1fc441f57a1a45a50673bdb4e41d3dd0261b21abe7c`，来源与输出 hashes 记录在 `reference/manifest.json`。网络组织另外对照 llama.cpp upstream `a17e94d229fc7fed1567a799baebd11b771fe5af`；所读本地 `de3ebb93...` 的 Qwen3.5/MTP 相关文件与该 revision 无差异。

## 正式入口功能回归

`functional-20261019/` 保存完整命令、report、服务日志、TUI屏幕及clocks。生产子进程关闭继承的锁FD并移除bypass环境变量。功能回归期间曾并行执行Rust编译；这些请求耗时不作为性能样本。后面的固定K3性能表仍使用独立、无编译争用的ABBA记录。

回归入口为 `tests/board/mtp_cli.py`、`mtp_api.py`、`vision_api.py`、`vision_tui.py`；各自提供 `--help`。文本API回归不需要照片；其余图片用例要求 `--photos` 目录包含 `cats.png` 和 `bus.jpg`。TUI测试依赖见 `tests/board/requirements.txt`，可用 `--python-deps` 指定隔离安装目录。例如板端执行：

```sh
python3 scripts/run_board.py --wait-lock --cpu-khz 2256000 --skip-dmc \
  --report logs/mtp-api-clocks.json -- \
  python3 tests/board/mtp_api.py --binary "$BIN" --model "$MODEL" \
  --cache "$CACHE" --spec-type draft-mtp --spec-draft-n-max 3 \
  --output logs/mtp-api
```

追加 `--spec-draft-sampling probabilistic --spec-draft-p-min 0.5` 可验证拒绝采样路径。`verify_physical_rows` 计数表示Rust的power-of-two逻辑bucket总行数；具体NPU kernel的物理M见调优策略，二者不能直接等同。

模型层的`attention_verify_calls/rows`记录逻辑Attention验证调用。native可按布局或数值模式继续分段，这两个计数不表示物理submit次数；实际分段任务数由native run timings提供。旧报告中的batch/serial计数属于已删除的模型层fallback。

`cpu-softmax-20261019/verification.json`记录最终源码SHA256、18组整模型比较、90项native回归、默认测试和14份成功测试的时钟恢复检查。最初诊断脚本误用零行commit，以及视觉测试启动器的libclang/锁FD配置失败单列保存；修正启动脚本后的最终检查全部通过，未因此改动生产事务协议。

## 首轮真实生成诊断

固定两条提示，关闭 thinking，greedy，每条生成 64 token；每配置每提示预热一次、正式一次。这是定位成本的 pilot，未满足正式 ABBA 样本要求。调优缓存有大量 miss/fallback，不能作为完成调优后的性能结论。

| 提示 | AR decode tok/s | K1 MTP decode tok/s | 接受/提议 | token 对齐 |
|---|---:|---:|---:|---|
| 英文 TCP/UDP 解释 | 7.7900 | 7.7882 | 30/33 | 64/64 相同 |
| 中文归并排序/Python | 7.8022 | 6.7489 | 25/37 | 64/64 相同 |

原始记录为 `pilot-ar.json`、`pilot-mtp1.json`、对应日志和 clocks。decode 计时包含完整 draft、target、输出头、GDN replay、draft repair、采样；分子为首次 anchor 后的 63 个新增输出。加载时间不计入 decode；TTFT 单列并包含 MTP prompt prefill。

该阶段 K1 每轮均值：draft 约 56 ms、verify 143–144 ms、repair 约 44 ms。高接受率没有转化为完整吞吐收益；该阶段标为 `performance_status=NO_GAIN_PILOT`。它不代表下述新版本结果。

## 固定 K3：三轮 ABBA

同一 Qwen3.5-4B C7、相同四条提示，greedy、thinking off、context2048、chunk128、每条生成64 token。顺序为 `[AR,K3,K3,AR]` 重复三轮；每配置/提示先预热一次、正式一次，共48个正式样本，每提示每模式6个。接受率是实际接受的 proposal token / 实际提议 token，包含末轮因输出预算缩短的提议；同一 greedy 轨迹的重复测时不构成独立语料。

| 提示 | prompt tokens | AR tok/s | K3 tok/s | 提速 | 接受率 |
|---|---:|---:|---:|---:|---:|
| 英文 TCP/UDP | 25 | 9.225 | 11.658 | 1.264× | 63.08% |
| 中文归并排序/Python | 26 | 9.226 | 13.411 | 1.454× | 77.19% |
| 数学/单位换算 | 56 | 9.224 | 14.383 | 1.559× | 84.91% |
| 传感器长提示 | 1040 | 9.125 | 9.191 | 1.007× | 43.75% |

吞吐按总 decode token 数 / 总 decode wall time 汇总；分子为首 token 后的63个输出。四条提示相同生成量汇总：AR **9.200**、K3 **11.811 tok/s**，约 **1.284×**。本组最高为1.559×，没有测到2×。每提示的 tok/s 样本标准差：AR为0.004/0.007/0.009/0.007，K3为0.017/0.013/0.024/0.007。

| 提示 | AR TTFT ms | K3 TTFT ms | AR 完整请求 ms | K3 完整请求 ms |
|---|---:|---:|---:|---:|
| 英文 | 344.8 | 359.7 | 7174.1 | 5763.7 |
| 中文代码 | 353.8 | 369.4 | 7182.2 | 5067.1 |
| 数学 | 508.8 | 533.7 | 7338.8 | 4913.9 |
| 长提示 | 6879.8 | 7213.1 | 13783.9 | 14067.3 |

完整请求计时包括 reset/reserve/prepare、prefill和生成，排除加载、分词及HTTP/流式传输。长提示的 decode 收益约0.7%，但计入 prefill 后完整请求慢约2.0%。DDR 未受控，结论限定于本板和这四条固定提示。

全部48个输出序列与AR逐 token ID一致；target量化域的累计缓存 miss/fallback、推理在线 tune 和权重重载均为0。验证 Attention 全部使用 batch，串行 fallback计数为0。232份冻结调优文件在测试前后 SHA256不变。runner exit0、restore_errors=[]，受控频率界限/调速器恢复为 before。

每轮成本（draft/verify/repair）：英文91.27/142.90/8.44ms，中文92.70/143.17/7.90ms，数学91.04/143.13/5.46ms，长提示88.85/143.37/10.14ms。K3 的 draft 是至多三次起草的合计；输出头完整调用均值约20.8ms/次。共享W8头驻留权重636,692,480 bytes。

完整三提议轮次中，连续接受到第1/2/3个提议的比例分别为：英文90.5/57.1/38.1%，中文94.7/84.2/52.6%，数学88.2/82.4/82.4%，长提示80.8/38.5/11.5%。这些比例只统计完整K3轮，分母与上表的总接受率不同。

证据：`mtp-20261018/k3-abba-20261018T1248/` 下的 `samples.json`、`summary.json`、`manifest.json`、`clocks.json`、`run.log`；二进制 SHA256 `2260f1bbb9d84a2158b8366e7f4bac40f9dad48b94d58297f23733bb83fe8451`。汇总入口为 `scripts/qwen35/summarize_mtp.py --abba`。该阶段状态为 `MEASURED_GAIN_K3_LIMITED`，保留默认关闭。

## 置信度截短实验

实验入口为 `tests/tools/mtp_bench.rs` 的 `--spec-draft-p-min`（别名 `--draft-p-min`）、`--p-min-schedule` 和 `--keep-low-confidence-schedule`。下表是完整词表 temperature1 置信度的策略筛选记录；它与正式 llama.cpp top10 置信度阈值不同。测速工具的 `--draft-confidence-top-k 0` 选择完整词表，默认 10；“保留当前提议后停止”也仅在测速工具中显式选择。正式命令采用上面的 llama.cpp 策略。

两种策略：低置信度提议不进入验证，或保留已计算的当前提议并停止后续起草。判断发生在采样之前，不额外按抽中的 token 条件筛选，q/p 接受与 residual 分布保持原定义。验证行数和 bonus 使用实际保留的提议数，支持零提议。每轮记录实际 head 尝试数、送入验证的提议数、置信度和提前停止次数，完整成本仍含已计算但丢弃的 draft。

`confidence-unit.log`：61 lib +14 CPU层 +1 OpenMP +5视觉输入，共81 passed、13 ignored；新增测试覆盖置信度惩罚/greedy定义、RNG/历史不变，以及0/1/2/3提议下的提交和输出预算。

同频率、同四提示、同64-token生成，单进程依次测AR、固定K3、丢弃低置信提议0.5/0.75、保留当前提议后停止0.5/0.75；每配置/提示预热一次、正式一次，共24个正式样本。以下是策略筛选pilot，不是三轮ABBA；与上节固定K3的正式数据分开。

| 策略 | 英文 tok/s | 中文代码 tok/s | 数学 tok/s | 长提示 tok/s | 四类汇总 tok/s | 相对本轮固定K3 |
|---|---:|---:|---:|---:|---:|---:|
| AR | 9.198 | 9.199 | 9.179 | 9.106 | 9.170 | — |
| 固定K3 | 11.641 | 13.405 | 14.392 | 9.188 | 11.806 | 基准 |
| 丢弃，0.5 | 11.479 | 11.778 | 14.380 | 9.447 | 11.514 | −2.5% |
| 丢弃，0.75 | 10.362 | 10.847 | 13.092 | 8.429 | 10.424 | −11.7% |
| 保留后停止，0.5 | 12.178 | 12.682 | 14.093 | 10.454 | 12.211 | +3.4% |
| 保留后停止，0.75 | 12.191 | 12.978 | 14.122 | 10.387 | 12.264 | +3.9% |

保留后停止0.75相对固定K3：英文+4.7%、中文代码−3.2%、数学−1.9%、长提示+13.1%。本轮四类汇总相对AR约1.337×。长提示计入prefill后，从固定K3比AR慢2.0%，变为比AR快约3.2%；不能把13.1%的decode收益说成完整请求收益。

接受率提高可能仅是筛掉了已计算但不送验证的候选。中文代码的丢弃0.75：送验证候选接受率97.1%，但实际计算59个draft、仅送验证35个、接受34个；按实际计算数为57.6%，吞吐比固定K3低19.1%。固定K3是57次draft、44次接受。长提示的保留0.5则把实际draft从80次减到53次、接受34次，对照固定K3接受35次；没有计算后丢弃的提议。

全部24个序列逐token ID匹配AR；target量化缓存miss/fallback、在线调优和权重重载均为0。冻结缓存哈希不变，runner exit0、restore_errors=[]，受控时钟恢复。证据目录为 `mtp-20261018/minp-screen-20261018T1313/`，包含同名原始样本、汇总、manifest、verification和clocks。

该筛选中“保留当前提议后停止”改善低接受率长提示，但损失部分高接受率提示的吞吐。此策略和完整词表阈值仅保留在测速工具中，不能将其汇总收益归给正式 llama.cpp 对齐配置。正式阈值默认 0；更广语料、随机采样和 top10 阈值的配对性能验收尚未完成。

## batch 精度问题

原始 verify 的最终 logits NRMSE 为 0.0889029694。前三层 GDN 逐值一致，首个 full-attention 的微小差异经动态 A8 放大。独立算子审计证明 LUT 的 masked future carrier 在 FP16 PV 中参与舍入，后置 FP32 抵消不能恢复原值；不是该 fixture 的 batch M 几何问题。

CPU softmax另有独立的跨1024差异：短路径先归一化再转FP16，长路径先把exp转FP16、做PV，最后归一化；归一化越过舍入边界后并不等价。真实C7绕过旧判断时verify/next logits NRMSE为7.52%/5.55%。固定首层Attention Q/K/V后，单改exp近似不改变输出；恢复原归一化顺序消除了主要误差。

当前由native按每行AR的LUT/CPU、short/long、delayed-normalization和gain模式选择兼容分段，跨1024的R4仅拆3＋1，普通CPU/LUT继续batch。Graph使用可见前缀的V幅值界，未来高V不会改变早期行的数值模式；原V前缀回滚和carrier修复保留。专用公共资格接口、Rust封装和模型层serial fallback已删除。详情及原始干预记录见 [精度审计报告](QWEN35_BATCH_ATTENTION_PRECISION_AUDIT.md)。

## 未完成 gate

1. K1/2/3/4/7四提示筛选与K3三轮ABBA已完成；更广语料、长输出和更多历史深度尚未完成。
2. C7 离线调优已完成（M1/2/4/8/16/32/64/128）；置信度截短已完成单轮四提示筛选，更广语料和配对复验尚未完成。
3. CLI/TUI/API 的真实 MTP、随机采样、图文、多轮、断流恢复及旧模型CLI回归已通过。更广的模型/语料质量与长输出性能不在本轮验收范围。

MTP 的完整实现、正确性与性能验收分别记录。本轮支持作为默认关闭的实验功能提交；更广的语料与性能结论保持上述范围。
