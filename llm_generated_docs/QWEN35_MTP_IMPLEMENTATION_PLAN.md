# Qwen3.5 + MTP 实现计划

日期：2026-10-17，Asia/Shanghai（UTC+08:00）。

状态：已开始实现。分阶段进度与证据见 [QWEN35_MTP_PROGRESS.md](QWEN35_MTP_PROGRESS.md)；本文的完整功能与性能目标仍为规划，不表示全部实现或验收。

2026-10-17 验收补充：用户确认本轮真实 C7 文本主干以端到端 PPL 为整模型精度验收依据，小幅浮点漂移不阻塞；保留严格 logits 诊断及原阈值，CPU 算子和状态恢复测试仍按原门槛执行。32 窗板端结果见 [QWEN35_C7_BOARD_VALIDATION.md](QWEN35_C7_BOARD_VALIDATION.md)。

规划基线：主仓库 `e944563b3c34fa2ceb2bbc0df2751d73880679d6`，`native/rk_npu_matmul` 固定提交 `b402b3834946e43d36ea61752eec63d809628aab`。

## 1. 目标与工作边界

首个完整验收模型采用 Qwen3.5-4B dense，配置解析和实现按维度参数化。实现以下能力：

- Qwen3.5 混合 GDN / full-attention 文本主干，支持 prefill、decode、eval 和 PPL 所需的逐位置评分。
- 静态图像 ViT、多图文混排、M-RoPE，以及带图像历史的多轮生成。
- 模型自带 MTP 的 prompt 预填充、起草、target 批量验证、部分接受、修正 token 和 bonus token。
- CPU GDN 的状态提交与紧凑输入重放；后续以独立性能 gate 决定是否启用完整 ReplaySSM。
- greedy 与保持 target 采样分布的随机投机解码，接入 CLI、TUI 和现有 OpenAI 兼容 API。

硬件分工确定为：GDN 递推、causal convolution、门控和状态管理放在 CPU；大投影、FFN、可用的输出头走现有 NPU 后端。GDN 的 NPU 实现不列入主线。ViT 优先复用现有 Linear 和无因果 mask 的 Attention 公共接口，图像预处理在 CPU 完成。

**本任务不进行模型量化。** 校准、量化算法、位宽选择、量化模型导出与量化质量选型由用户安排的另一个 agent 负责。本计划只定义运行时消费接口、参考数据交接和同权重数值验收，不生成新的量化 checkpoint。

首版不扩展到 MoE 和视频输入。上下文先遵守当前原生 Attention 的 32768-token 上限；模型配置中的更长上下文不作为已支持能力。完整 ReplaySSM 不阻塞基础 MTP 交付，未完成时需明确标注。

## 2. 当前源码基础与需要扩展的部分

| 位置 | 当前能力 | 计划改动 |
|---|---|---|
| [`src/model.rs`](../src/model.rs) | `LanguageModel`、单位置 logits、reset/reserve/truncate、架构注册 | 新架构注册；多模态输入和可选投机能力；兼容旧模型默认行为 |
| [`src/model/dense.rs`](../src/model/dense.rs) | Qwen3/Llama/Granite 的标准 GQA、FFN、批量 NLL | 抽出确有复用价值的投影/输出头辅助代码；Qwen3.5 使用独立后端 |
| [`src/model/checkpoint.rs`](../src/model/checkpoint.rs) | safetensors、compressed-tensors W4/W8 | 嵌套配置、模块名映射、允许声明为浮点的模块、共享参数处理 |
| [`src/model/cpu.rs`](../src/model/cpu.rs) | RMSNorm、RoPE、SwiGLU 等 | Qwen3.5 数值语义、partial/interleaved M-RoPE；GDN/卷积独立模块 |
| [`src/npu.rs`](../src/npu.rs) | W4/W8、GQA KV、批量 bucket、显式离线调优 | 验证批量与 lm_head 工作区；ViT mask 模式；按需要接浮点 Linear |
| [`src/engine.rs`](../src/engine.rs) | 单 token 生成、文本前缀复用、取消和流式回压 | 多模态 prepared input；投机 driver；提交边界和缓存协议 |
| [`src/sampling.rs`](../src/sampling.rs) | 直接从 logits 返回 token，维护 penalties | 拆出分布处理；独立 proposal 历史；拒绝采样 |
| [`src/text.rs`](../src/text.rs)、[`src/api.rs`](../src/api.rs) | 文本模板、分词、stop 和文本请求 | 图像输入、placeholder 展开、原生模板对齐及流式验收 |
| [`src/bench.rs`](../src/bench.rs) | teacher-forced synthetic pp/tg/pg | 保留原语义；另加真实生成的投机 benchmark |

当前 `token_nll()` 已计算过多行输出头，可以复用其计算逻辑，但其接口不足以支持验证。生成路径还需要每个验证位置的 logits 和用于 MTP 的 hidden states。

已有底层 Attention 支持 `NO_MASK` / Boolean mask，ViT 不需要先创造一套新的 Attention 算子。Rust 侧仍要补全相应构造、生命周期和测试。

## 3. 与量化 agent 的交接

### 3.1 运行时输入约定

优先沿用项目已有的 compressed-tensors：per-channel 对称 W4/W8、per-token 对称动态 A8。运行时现有 A8 数值 ABI 保持一致，见 [MODELS.md](MODELS.md)。若外部产物需要新的量化格式，应先记录为额外适配工作，不默默改变这个 ABI。

需要交接以下信息；可以先提供配置、tensor 清单和小型 fixture，完整权重稍后到位。

| 内容 | 最低要求 |
|---|---|
| 模型来源 | 原始模型 ID、不可变 revision、转换后文件清单/校验和 |
| 配置 | 完整 `text_config` / `vision_config`、层类型、RoPE、MTP 层数、EOS、量化声明及实际存放位置 |
| 参数命名 | 优先保留官方 tensor 名称；有改名时给出显式映射 |
| 模块类型 | 标明各 Linear 为 W4、W8 或 BF16/F16/F32；不得因 ignore 规则静默漏掉必需模块 |
| MTP | 保留 `mtp.*` 参数；标明是否与主模型共享 embedding / lm_head，以及有无独立 embedding |
| 视觉 | 保留完整视觉参数和 processor 配置；注明保留浮点的模块 |
| 参考 | 固定 token IDs、图像原始输入/预处理产物、部分中间张量及 teacher-forced logits 的生成方法 |

当前通用 Dense 代码禁止“quantized lm_head + tied embedding”。Qwen3.5 原始配置包含 tied embedding，新后端需明确物理参数与逻辑共享关系：BF16 embedding 与独立量化输出头可以有不同存储表示；运行时不得仅凭 `tie_word_embeddings=true` 就忽略已提供的量化头。target 与 MTP 尽量共享同一份输出头权重，分别使用适当批量工作区。

### 3.2 可以先推进的工作

配置解析、CPU FP32 参考、GDN/卷积、replay 协议、合成权重模型、M-RoPE、采样数学测试均不依赖完整量化产物。NPU 整模型数值、真实 MTP 接受率和最终吞吐验收依赖交接后的真实模型。

参考 harness 可读取、解码已有量化权重以建立同权重 oracle；这属于运行时验证，不进行校准或重新量化。原始 BF16 模型与量化模型的质量比较由量化任务提供，本任务避免把模型量化误差归因于 replay 或 NPU 执行。

## 4. 模型后端与输入设计

### 4.1 独立 Qwen3.5 后端

拟新增 `src/model/qwen35/`，内部划分配置/加载、混合 decoder、GDN、卷积、M-RoPE、ViT、MTP、状态与 replay。共享 CPU 基础运算和 Linear 后端，不把现有 Dense 改为包含大量架构分支的单一实现。

注册完整模型 `qwen3_5`；对 `qwen3_5_text` 的独立文本配置按实际 checkpoint 结构支持。外层配置与 `text_config` 的字段来源、量化声明优先级必须显式处理。以 `layer_types` 驱动混合层，不写死层数和 3:1 顺序。

文本实现逐项核对 zero-centered RMSNorm 的 `1 + weight`、full-attention 的 Q/gate 拆分与 sigmoid、Q/K norm、partial rotary 的旋转范围，以及 GDN 的 QKV/Z/A/B 投影、卷积、L2 norm、衰减、beta 和输出 gated norm。M-RoPE 的纯文本退化情形也保留 partial/interleaved 配置语义，不直接套用原 Qwen3 的全维 RoPE。

新增 `LinearWeights` 一类的加载结果，区分现有量化矩阵和浮点 tensor。浮点小投影先有 CPU FP32 正确性路径；若 ViT/MTP 大层保留浮点，接入库已有浮点 matmul 公共 API，并单独验证其 BF16/F16 转换与误差。禁止运行时临时量化整个模型来绕过交接缺口。

CPU GDN 使用 FP32 canonical state。先以标量参考建立状态布局和计算顺序，再实现 NEON 与按 head 并行；prefill 与 decode 共用语义。可借鉴已有 `gdn_step.cpp` 的循环组织，但其模型特定门控、head 映射和 RKNN 构建依赖需要重新处理，不能直接作为 Qwen3.5 oracle。

### 4.2 明确的输入表示

拟引入 `PreparedInput`，包含 token IDs、真实 token 行数、视觉 embedding 插入位置、三轴 RoPE position IDs、视觉 grid/processor 标识及可用于前缀判断的输入身份。视觉特征可由推理 worker 在请求准备阶段延迟求值，避免 HTTP 线程直接调用 NPU。

必须区分：

- token/KV 的逻辑序列位置；
- temporal/height/width 的 RoPE 坐标和生成阶段 delta；
- NPU Linear 的物理 padding 行。

padding 不得推进 GDN、卷积、KV、采样历史或 usage。RoPE 缓存按实际坐标范围处理，不能把三轴位置直接用作 token 数组下标。

图像身份必须进入前缀匹配：相同 placeholder token IDs 加不同图像不能命中同一缓存。需要同时纳入图像内容、处理参数、grid、位置编码语义和模板结果。先支持 CLI 图像文件和 API 的内联图像 data URL；远程 URL 获取作为独立能力，首版给出明确的不支持错误。

拟定入口为可重复的 CLI `--image`、TUI `/image PATH` 绑定下一条消息，以及 API `text` / `image_url` content parts；参数和命令均为待实现设计。请求准备阶段按展开后的视觉 token 数检查历史加输出预算，再预留状态与 KV。单图 temporal patch 的构造、图像通道和 patch 顺序以 processor fixture 为准；图像特征缓存与原始图像历史分别设置容量上限。

### 4.3 对现有接口的最小扩展

保留现有 `forward()` 供文本普通解码和旧模型使用。新增 prepared-input prefill 的默认文本实现，以及默认关闭的可选 `SpeculativeModel` 能力。该能力至少定义以下操作，名称在 P0 固定：

| 操作 | 契约 |
|---|---|
| `prepare_speculation` | 预留窗口、准备工作区并完成 draft prompt prefill；返回可用能力 |
| `begin_round` | 记录本轮起始 target/draft 边界，建立可取消的验证事务 |
| `draft_step` | 使用已约定的 token/hidden/position 对齐产生 proposal logits；不直接发送输出 |
| `verify` | 返回真实验证行的逐位置 logits；模型保留需要的 target hidden states |
| `finish_round` | 以“被 target 消费的输入行数”提交 GDN、卷积、KV 和 draft 修复 |
| `abort_round` | 放弃临时状态，恢复轮次起点；若恢复失败，显式使会话失效并 reset |

engine/speculative driver 管理采样、接受判断、stop 和事件；模型管理架构内状态。MTP hidden state 可留在模型内部，避免通用 engine 依赖具体张量布局。接口必须在 `Box<dyn LanguageModel>` 下可用，默认方法保证旧模型不必实现投机逻辑。

## 5. 投机轮次的统一位置约定

采用 pending-anchor 约定，先写出小词表/小模型可执行 oracle，再实现真实 MTP。

1. 轮次开始时，target 已消费 C 个 token，保留相应 target hidden state；存在一个已采样、已计入输出但尚未被 target 消费的 anchor `a`。首次 anchor 来自 prompt 最后一行 logits。
2. MTP 从约定的 target seed hidden、anchor embedding 和位置开始，提出 K 个额外候选 `d1..dK`。hidden 使用 norm 前还是 norm 后、embedding 与位置的移位由 P0 参考锁定。
3. target 输入 `[a, d1, ..., dK]`，真实验证行数 R=K+1。row 0 的 logits 验证 d1；row j 的 logits 用于 d(j+1)；最后一行在全接受时提供 bonus 分布。
4. 接受 j 个 draft token 后，target 提交 `1+j` 个输入，消费位置变为 `C+1+j`。首个拒绝位置从相应 target/修正分布得到新 token；全接受则得到 bonus token。
5. 新 token 成为下一轮 pending anchor。旧 anchor 不重复发送；本轮常规新增输出为 j 个接受候选加一个修正/bonus token。终止条件可能缩短这个集合。

实现同时追踪“已输出位置”“target 已消费位置”“draft 已消费位置”。`position()` 表示 target 已消费位置，不表示累计发出的 token 数。前缀缓存只登记实际已消费输入；pending anchor 不得伪装为已存在 KV。

每轮根据剩余 context 与输出预算缩短 K。不能为了固定窗口越过 context 或生成上限；`max_tokens=0/1`、EOS、stop、只够消费 anchor 的情况都需单独覆盖。先暂存本轮已验证 token，通过 stop/EOS 处理确定提交边界，再完成模型提交和流式事件发送。不得把拒绝候选或终止边界后的候选写入采样历史或前缀缓存。

### 5.1 MTP 的 prompt prefill 与接受后修复

MTP 不是从空 draft KV 开始的独立无状态 head。prefill 阶段需要按已锁定的移位规则，利用 target hidden rows 与 embedding 建立 draft 前缀。按 chunk 消费这些 hidden rows，避免保存整个长 prompt 的全部 hidden states；其耗时计入 TTFT。

验证后的 draft KV 同步策略也必须显式定义。起草时递归使用 MTP hidden states，和使用已验证 target hidden states 的 teacher-forced 输入可能不同，不能默认将所有临时 draft KV 直接当作规范前缀。

第一版采用保守策略：回到本轮 draft 起始边界，使用已验证 target hidden rows，批量补齐本轮 anchor 与接受段。移位与边界按参考 fixture 验证，补齐成本单独计时。后续是否保留临时 draft KV 作为更便宜的 proposal 状态，是独立优化，必须保留实际 proposal 分布并重新测接受率。

无法证明 draft 恢复正确时，本请求降级为普通 target 解码，统计原因；不以清空 draft KV 后继续宣称有有效历史的方式恢复。

## 6. GDN、卷积与 replay 状态协议

### 6.1 第一版：紧凑输入日志与接受前缀重放

每层拥有 committed GDN state 和 committed convolution history。verify 使用共享临时状态区，不覆盖 committed state：

1. 该层开始验证时，把 committed state/history 装入临时区。
2. 对真实 R 行执行卷积和 GDN，产生所有输出，记录状态转移所需的归一化 key、value、decay/beta 等输入。记录必要的卷积前输入，以便恢复卷积历史。
3. 该层结束后临时区可供下一层复用；日志按层保留，等待最终接受边界。
4. 接受数确定后，只把前 `1+j` 行状态转移重放到 committed GDN；卷积历史从旧历史与这段真实输入恢复。主 attention KV 截到同一边界，选取该边界的 target hidden state。
5. 拒绝后缀日志全部丢弃。取消时 committed GDN/history 尚未变化，主/draft KV 恢复到轮次起点。

此路径不重跑 NPU Linear、FFN 或 ViT。只为恢复状态而重放时不需要缓存 query；后续若要重算输出，应另行定义 query 的需求。首先保存足够明确的原始状态更新输入，不提前引入依赖旧状态的压缩 delta。

共享 scratch 的前提是当前同步、单 worker 的执行顺序。持久 GDN/卷积状态和输入日志按层独立拥有；不得因为 scratch 可复用就覆盖尚未提交的层状态。提交所需容量预先分配；若多层提交中出现错误，整个请求 reset 并使前缀缓存失效，不暴露半提交会话。

内存预算分别记录 canonical state、临时状态区、状态更新日志、卷积日志、target/draft KV、hidden rows 和概率缓冲。4B 按 FP32 计算的主干 GDN canonical state 约为 `24 * 32 * 128 * 128 * 4 = 48 MiB`；逐位置额外保存四份完整状态会增加约 192 MiB。紧凑日志的实测占用需要包括卷积输入，不能只报告 key/value 日志大小。

### 6.2 CPU 计算组织

先实现逐 token FP32 oracle，再实现按 head 连续处理整个验证窗口的 NEON 版本，减少相同状态的重复装载。一个 128×128 FP32 head 为 64 KiB；按实际 CPU cache 和四大核分工选择循环顺序，用完整调用验证收益。

GDN prefill 首先使用同语义的 CPU batch recurrence。若长 prefill 明显成为瓶颈，再增加 CPU chunk 算法；它必须与 recurrent 路径比较最终状态及后续 decode，不能只比较 chunk 输出。FP32 门控、归一化、softplus/exp、head 扩展顺序由 oracle 固定。

### 6.3 后续：完整 ReplaySSM

在相同状态接口下增加 checkpoint + 输入 ring buffer 的实现，研究 output-only 与定期 flush。只有已提交历史可以合入不可回退 checkpoint；验证窗口跨 ring 边界前，先处理已提交历史并留足临时空间，不能把待接受候选提前 flush。

GDN 的 delta rule 含依赖状态的修正项；完整 ReplaySSM 的缓存表示需要按其算法独立验证，不能把 Mamba 的输入累加公式直接套用。FP32 checkpoint 为默认；改变计算顺序引入的误差需要做长序列状态与 logits 检查。

CPU 上先和已经按 head 批处理的普通 recurrence 比较。是否减少实际 DDR 流量或加快模型，以板端测量为准；GPU serving 数据仅作为算法来源，不作为 RK3588 预期收益。

### 6.4 跨请求前缀缓存

短投机窗口 rollback 与任意历史 prefix truncate 是两种能力。首个 Qwen3.5 后端可在无法恢复更短前缀时返回 `truncate=false`，由引擎 reset/recompute，确保正确性。

后续加入有内存预算的 hybrid checkpoint：记录对应位置的 GDN/history、RoPE 上下文及 draft 同步信息，并核对其对应的 target/draft KV 是否仍有效。需要截到非 checkpoint 位置时，从最近的可用较早边界重放目标前缀。报告实际复用和重算量，不把整个文本公共前缀都计为 cached tokens。checkpoint 数量必须有上限，不能每个 prompt chunk 永久增加一份 48 MiB 状态。

## 7. NPU 小批量执行与调优

实际验证长度为 R=K+1，不能只测整数幂长度。例如 K=1/2/4 对应逻辑 M=2/3/5，当前 Linear 分别进入物理 bucket 2/4/8。

- 最低覆盖逻辑 M=1/2/3/4/5/7/8，区分实际行数和物理执行行数；端到端测试 K=1/2/3/4/7，先完成 K=1。
- 独立检查 target Linear、MTP Linear、target/MTP 共享 lm_head 和 Attention 的完整调用，包含量化/布局转换、同步与输出收集。
- 当前量化头构造将 chunk 设为 1，需审查其 workspace、容量和多行 prepare；不能默认 head 已自然覆盖验证批量。
- 显式 `tune` 准备所有实际需要的 bucket 和 prefill 档位。正常请求只加载策略，不触发在线搜索、权重 reload 或反复建立档位。
- `retain_workspaces` 必须保留 prefill、普通 decode 和 speculative verification 所需集合，避免调用 prepare 后删掉下一阶段马上要用的工作区。
- 模型权重常驻，target/MTP 共享不可变输出头时不重复上传；仍需按 domain 计算所有权重、KV 和 workspace 预算。

大矩阵合并只作用于 checkpoint 中可兼容的位宽/量化语义。GDN、卷积及 CPU 操作仅处理真实 R 行，所有 padding 均留在 Linear 封装内。

## 8. 采样与服务语义

第一阶段 greedy 对每个位置应用正确历史下的 penalties，再比较 target argmax。所有接受长度 0..K 都要强制构造，不能仅靠自然生成“碰巧覆盖”。

完整随机采样将 `Sampler` 拆分为 logits processors / 分布构造 / 抽样 / 历史提交。保存实际 proposal 的 q，target 的 p 使用同一候选前缀下的 temperature、top-k/top-p/min-p 与 penalties。接受概率为 `min(1, p(d)/q(d))`；拒绝时使用归一化的 `(p-q)+`，全接受时从最后 target 分布生成 bonus。补齐零概率、浮点残差和稳定求和处理。

proposal 的临时历史与 committed history 分开。随机数流的使用顺序由实现固定并可复现，不要求与普通随机解码在相同 seed 下逐 token 相同；验收依据为分布等价。greedy 则要求开关投机后输出 token 一致。

stream、thinking、tools、EOS、stop、usage、取消和回压均复用现有公共路径。验证 token 不直接计入 completion tokens；不能继续用“decode 调用次数/耗时”充当投机生成 token/s。取消可以 reset 整个请求缓存，但下一请求必须干净可用。

## 9. 阶段与验收 gate

下表为熟悉项目的单人工作日粗估，不含量化工作和等待模型/板卡的时间。代码阶段开始前确认当前 checkout；不要把本规划基线 SHA 当作未来实现时的最新 HEAD。

| 阶段 | 工作与交付 | 验收 gate | 粗估 |
|---|---|---|---|
| P0 | 固定模型/上游 reference revision、配置与量化交接；小模型和 teacher-forced fixtures；冻结位置协议 | G0：模块清单、MTP seed/shift/prefill、state layout 和输入语义可复验 | 1–2 天 |
| P1 | Qwen3.5 文本后端；FP32 GDN/卷积、full-attention gate、norm、partial RoPE；普通 decode/eval | G1：分层与整模型 oracle 通过；不同 chunk 与连续 decode 的状态一致 | 4–6 天 |
| P2 | CPU NEON、共享验证 scratch、紧凑日志、commit/abort/replay；假 proposal 驱动验证 | G2：0..K 接受边界、连续拒绝、取消和跨层状态恢复全部通过 | 3–5 天 |
| P3 | MTP 加载、prompt prefill、seed hidden、逐步起草和 draft KV 修复 | G3：固定 prefix 的 MTP logits/位置与参考一致，修复后的下一步可复验 | 3–5 天 |
| P4 | greedy 投机 driver、K+1 行验证、小批量 lm_head、CLI/统计；首次真实 A/B | G4：投机开关 token 对齐，正确性与吞吐分别出报告，明确接受率和成本 | 4–7 天 |
| P5 | ViT、原生图像预处理、M-RoPE、混合 embedding、CLI/API 多图输入 | G5：processor/grid/视觉输出对齐；图文生成与带图历史回滚通过 | 5–8 天 |
| P6 | 随机拒绝采样、全部 sampling 参数、流式终止与取消 | G6：小词表解析概率测试通过，真实模型功能与随机分布检查通过 | 3–5 天 |
| P7 | 有预算的 hybrid prefix checkpoints、多轮缓存和服务恢复 | G7：命中/未命中同结果；不同图像不误命中；缓存边界与恢复完整 | 3–5 天 |
| P8 | 正式调优、性能矩阵、旧模型回归、打包和用户文档 | G8：功能与数值全通过；输出性能状态和全部恢复记录 | 3–6 天 |
| P9 | 可选完整 ReplaySSM：output-only、ring、flush、CPU 优化 | G9：相对 P8 正确性通过且有额外完整调用收益，才进入默认候选 | 4–8 天起 |

P0–P4 为最短文本 greedy MTP 路线，约 3–5 人周；P0–P8 的完整首版约 6–10 人周。P9 独立评估。估算不包含任何速度承诺。

P5 的预处理 fixtures 和 M-RoPE 坐标测试可以提前准备，但图文 + MTP 集成须通过 G4、G5 后验收。随机采样支持可以先在纯文本完成数学验证，再用图文输入回归。完整交付要求 P0–P8 都完成；不能把文本 greedy demo 标为全功能支持。

## 10. 验证矩阵与证据

### 10.1 CPU、状态与参考

- 标量 FP32 GDN 与 NEON：包括非零初始状态、长衰减、不同 head 配置、长度 1/2/3/4/5/7/8/31/32/33/127/128/129。
- prefill 一次执行、不同 chunk 分割、逐 token 执行：比较每层卷积历史、GDN 最终状态以及后续 teacher-forced logits。
- replay：逐一比较接受边界 0..K；这里 j=0 仍须消费已确认 anchor。强制全接受、首位拒绝、中间拒绝、交替模式和重复拒绝。
- 对每个恢复边界，用从空状态重新执行同一已提交前缀的结果做 oracle；仅比较 position 整数不能证明状态正确。
- gate 参数、Q/K L2 norm、zero-centered RMSNorm、partial/interleaved M-RoPE、视觉坐标和位置 delta 分别使用固定 fixtures。
- MTP 特别覆盖长 prompt prefill、chunk 接缝、接受后补齐、下一轮 seed 与最后一个可用 context 位置。
- 取消注入在 draft、各验证层、接受判断后、提交/事件发送边界；失败后新请求与全新会话比较。

逐值相等的 gate 用于相同运算路径的复制/恢复；存在浮点重排的比较先采用 `max_abs <= 1e-5 + 1e-4 * max_abs(reference)` 且 relative L2 <= 1e-4 作为 FP32 小型算子的初始门槛。NPU/FP16 整模型用同权重量化 oracle，P0 先固定逐层阈值和 NRMSE 门槛，再开始优化；不能看到结果后为通过而放宽。PPL 同权重 reference 的相对变化先以 1% 为调查线，超出需定位误差，不能据此自动改变量化方案。

greedy 投机开关要求验收集逐 token 一致；若批量路径在接近并列的 logits 上改变 argmax，记录完整 logits 和 margin，先解决路径差异或显式回退该不稳定验证位置，再通过 G4。保留两个比较：投机开关一致性与独立 reference 数值一致性。

### 10.2 多模态与 API

覆盖单图、多图、宽高不同/需要 resize 的图、图文混排、同图复用、不同图但相同文本、带图历史追加纯文本、再次追加图像。对齐解码 RGB、resize/normalize、patch 顺序、grid、placeholder 数量、merger 输出和三轴坐标。预处理 resize 的坐标约定、插值和 antialias 以固定 reference 为准。

API 覆盖非流式、SSE、usage、thinking、tools、EOS、跨 token stop、`max_tokens` 边界、客户端断流、回压和排队后的下一请求。先保证 CLI/API 可用；TUI 复用 engine 并完成文本与带图历史交互回归，不把图像终端显示列为验收项。

### 10.3 有效的性能比较

保留现有 synthetic teacher-forced bench，新增真实生成 bench。随机 token 的 teacher forcing 不衡量 MTP 接受率；accept-all 注入仅用于成本上限探测，单独标为 synthetic。

| 维度 | 最低覆盖 |
|---|---|
| 执行方式 | 普通 AR；紧凑 replay 的 greedy MTP；完整随机 MTP；P9 完成后另列完整 ReplaySSM |
| 验证窗口 | K=1/2/3/4/7，报告实际 R 和物理 M，按 context/输出预算裁剪 |
| 历史/输出 | 历史约 512/2048/4096，生成 128 与 512 token；提前 EOS 时报告实际长度 |
| 内容 | 中文聊天、代码、数学/推理、工具格式；另列单图/多图问答 |
| 缓存 | 冷前缀与命中前缀分开，计入实际 draft prefill/repair 成本 |
| 诊断 | low-batch Linear/head/Attention、CPU GDN、replay 和完整生成分别计时 |

使用相同模型 revision、量化权重、prompt、采样参数、CPU/NPU mask 和固定频率。greedy 做配对 ABBA；至少完成三个 ABBA block，保留每次 run 与 prompt 的原始值，不把同一 run 内的 token 当成独立样本。每个配置预热，缓存建立和显式 tune 在正式计时前完成。

真实生成报告包含从 prefill 完成到最后 token 的 decode wall time，以及含预处理/ViT/target 与 draft prefill 的 TTFT 和请求总时延。记录 target verify、draft、draft repair、CPU GDN、replay、sampling、同步及服务开销；分项若嵌套必须标注，不将重叠耗时相加。

新增统计至少包括：`draft_proposed`、`draft_accepted`、`accepted_prefix_histogram`、`spec_rounds`、`verify_input_rows`、`verify_physical_rows`、`target_consumed_tokens`、`emitted_tokens`、`replay_rows`、`draft_repair_rows`、`fallback_reason`、各阶段耗时、各类 buffer 峰值和现有调优/reload 计数。吞吐以实际新增输出 token 数为分子，包含拒绝与恢复的完整成本。

### 10.4 性能 gate 与停止规则

G4 首次真实 A/B 即检查低批量优势是否能覆盖 draft/head/replay 成本。若无收益，仍保留已正确的普通 Qwen3.5 后端，先分析成本，再决定是否继续优化窗口和状态实现。

默认开启 MTP 的候选 gate：预先固定的纯文本 greedy 验收集上，完整 decode speedup 几何均值至少 1.20×，且各预先固定类别的平均吞吐回退不超过 5%；图文、随机采样及缓存命中场景分别报告，未达标的配置保持显式启用。窗口由配置/离线结果选择，不在普通请求中跑在线搜索。

完整 ReplaySSM 在相同输入与窗口下，相对紧凑 replay 的 decode 至少再提升 5%，且超过样本波动、状态/长序列数值通过，才考虑替换默认实现。未通过可记录 `NO_GAIN`，不影响已完成的基础 MTP 功能。上述 20%/5% 是本项目拟定决策门槛，不是预测结果。

数值、恢复或图像前缀身份检查失败时，停止性能推广并修复；不通过降低 GDN state 精度或省略同步掩盖问题。量化权重缺失仅标记对应真实模型 gate 待验证，其余 CPU/协议工作可继续。

## 11. 执行、回归与交付组织

实现按阶段保持可 review 的范围：先模型加载/普通前向，再状态事务，再 MTP，随后多模态与完整采样。依赖公共算子库的变更在该库单独维护并更新固定提交，主仓库不复制一套私有算子实现。

代码阶段遵守 [CONTRIBUTING.md](CONTRIBUTING.md)。在适合的平台运行以下检查，涉及真实 NPU 的用例在 RK3588 上持锁执行：

```sh
rtk cargo fmt --check
rtk cargo clippy --all-targets --release --locked -- -D warnings
rtk cargo test --release --locked
```

沿用 `scripts/run_board.py --wait-lock` 统一板锁、固定频率与恢复；正式证据必须包含时钟前后快照、退出码和恢复错误记录。共享库改动后重建依赖可执行文件，避免测到 stale binary。重跑原有三种模型的相关数值/API 回归，旧 `bench` 的 pp/tg/pg 语义保持原定义。

所有新增设计、报告和小型证据放在 `llm_generated_docs/`。建议后续使用 `evidence/qwen35_mtp/` 保存 fixture 清单、数值报告、逐轮状态账本、原始性能样本和时钟记录；模型权重、大 tensor dump 与构建产物不加入源码提交。

每阶段维护四个独立状态：`implementation_status`、`correctness_status`、`board_status`、`performance_status`。不能用编译通过代替板端正确性，不能用 high acceptance 或单个 Linear 加速代替整模型吞吐。

## 12. 实现前需冻结的项目与资料来源

P0 需要冻结以下项目，均可通过源码/fixture 和模型交接解决，无需先开展量化工作：

1. 首个完整 checkpoint 的不可变 revision、参数命名、浮点层清单与 lm_head 共享策略。
2. MTP target seed hidden 的 norm 位置、移位、prompt prefill 和接受后补齐的精确定义。
3. GDN state 的矩阵方向、共享 key 到 value head 的映射、卷积布局及 norm/gate 的数值顺序。
4. 图像 processor 的尺寸/插值/patch 语义和 M-RoPE 坐标；初始图像 token 预算。
5. 初始窗口上限、状态日志/前缀 checkpoint 内存预算，以及验证阈值与性能验收集。

外部资料在 2026-10-17 查阅。以下分支链接用于定位；实现时须记录实际使用的不可变 revision，不能仅保存 `main` URL：

- [Qwen3.5-4B 官方配置](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)：用于配置维度和能力盘点，不能据此宣称已支持其全部上下文长度。
- [Qwen3.5-4B 官方 tensor 索引](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/model.safetensors.index.json)：用于文本、视觉和 `mtp.*` 参数清单交接。
- [Transformers Qwen3.5 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)：用于数值语义与多模态参考；MTP 另行验证。
- [vLLM Qwen3.5 MTP 实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)：用于 MTP 网络与输入语义参考，不照搬其 GPU 调度。
- [ReplaySSM 作者算法说明](https://dao-lab.ai/blog/2026/replayssm/) 与 [研究实现](https://github.com/Johnny-Liou/ReplaySSM)：用于完整 ReplaySSM 的 GDN 算法定位。紧凑输入重放作为本项目第一版方案独立验收；GPU 性能数字不作为板端结论。

完成标准：P0–P8 的 gate 和证据齐备，文档明确实际支持范围、降级行为及性能状态；P9 单列为已完成、无收益或未实施。模型量化的完成与质量结论由另一个任务独立交付。
