# Qwen3.5 batch Attention 精度审计

日期：2026-10-19。算子仓库基线：`c3bff0252574a2b7c17b82e7f95132a77c67deab`。

默认 LUT Attention 的 masked future carrier 是本次 AR/batch logits 分歧的已验证起点。将未来 carrier 在 FP16 PV **之前**归零，并按每行可见前缀抵消 carrier 后，可以保留 batched QK/PV，匹配原有 AR 数值。真实 layer 3 的 8 行输出共 32,768 个 FP32 元素与旧 AR 逐值相等；跨 32/512 边界和 32K 上下文的算子回归也通过。

当前 native 工作树还修复了独立的 CPU softmax 跨1024问题：由算子内部按每行 AR 的数值模式分段，保留正常 CPU/LUT batch。过渡期的 `rk_npu_attention_f16_can_batch_causal` 公共接口、Rust封装与模型层逐行fallback已删除。精确回滚V幅值元数据与尾带同步保留。固定K3四提示ABBA属于此前版本，结果和测量边界见 [MTP运行时报告](QWEN35_MTP_RUNTIME.md)。

## 证据与定位

本机证据根目录：

`/home/zt/mnt/3090x4/tmp/qwen35-dev/mtp-20261018/batch-attention-audit/`

板端对应目录将 `/home/zt/` 换为 `/home/firefly/`。全部 NPU 调用均通过 `scripts/run_board.py --wait-lock --cpu-khz 2256000 --skip-dmc`，NPU mask7；最终 `final-v2`/`split` 回归固定在 CPU4–7，DMC 不受控。

| 比较 | 结果 |
|---|---|
| 原 trace，position5，layer0–2 全部已保存阶段 | 逐值一致 |
| layer3 prompt Q/K/V，以及 position5 第一行 Q/K/V | 逐值一致 |
| 原库 LUT1，同一 rows1 query、span32，cache6 vs cache13 | 4095/4096 不同；NRMSE 0.00036145145242，max_abs 0.00246524810791 |
| 原库 LUT1，同一 cache13，rows1 vs batch8 第一行 | 逐值一致 |
| 原库 LUT1，将完全 masked 的未来 V 改为 0 / -V | 第一行输出改变 |
| 原库 LUT1，cache13 truncate6 vs 原 cache6 | 逐值一致 |
| 原库 LUT0，同一短 fixture 的上述比较 | 全部逐值一致 |
| 原库 LUT1，serial8 vs batch8 | 28669/32768 不同；NRMSE 0.000403608422914 |
| 修正库 LUT1，batch8 vs 原库 serial8 | 32768/32768 逐值一致 |

这些对照固定 query、可见 K/V、物理计算长度和 NPU 核数。只有预加载的未来 V 改变时，原 LUT 输出就改变；同一 cache 内改变 batch M 却不改变第一行。故当前真实 fixture 的起点是 future carrier 的抵消舍入，不能归因于 M 变化或 GDN 事务状态。

原 trace 的 `attn` 是 sigmoid gate 后输出，NRMSE 为 0.0002979724206413913，max_abs 为 0.00024139881134033203。用原 AR 的 gated/raw 比值估计 gate，再作用于独立 probe 的 batch raw 输出，能预测原 batch trace，max_abs 仅 1.4901161193847656e-08。gate 是由输出反推，未单独捕获；该结果与 carrier 起点一致。

原始数据：`probe.log`、`full/probe.log`、`trace-audit.json`；FP32 输出在相应目录的 `.f32` 文件。复核脚本：

- [audit_batch_attention.cpp](../scripts/qwen35/audit_batch_attention.cpp)：真实 Q/K/V 与公共 Attention C ABI 的区分实验。
- [audit_batch_attention_trace.py](../scripts/qwen35/audit_batch_attention_trace.py)：原 trace 与动态 A8 的数值传播。
- [run_batch_attention_audit.py](../scripts/qwen35/run_batch_attention_audit.py)：旧 AR、新 batch 和非因果路径的严格比较。

## 普通 prefill 与验收边界

该缺陷适用于公共 causal LUT 路径，包括普通文本 prefill；不以 MTP 是否启用为条件。某层最后一行没有未来 token，但同层更早行的误差仍可能通过后续层的 K/V、GDN 与动态 A8 传播到最后的 logits。修复位于 native 公共算子，覆盖普通 prefill 和 target verify；非因果 ViT 不经过此 causal 分支。

历史32窗口PPL的8.578578与FP32参考8.601783相差约−0.27%，通过当时的1%调查线；该质量门槛不能证明 masked future 不影响输出。“批量化前后8160项NLL逐值相同”比较的是使用同一native内核的实现，不是独立的因果性证明。历史严格logits和分块检查保留了失败，因此不能把旧PPL通过解释为批量/AR已严格一致。

native回归 `masked_future_case` 固定可见Q/K/V，分别预加载正常、零值及反号未来V，要求输出与仅加载可见前缀逐值相等；另有batch与真实逐token append/run的逐值对照。普通prefill的模型PPL及性能记录必须保留库版本身份，旧库结果不直接归给本修复。

## 具体算术机制

源码定位以未修改的算子文件为准：

| 位置 | 行为 |
|---|---|
| `src/npu/attention.rs:303–322` | 一次把整个输入 batch 的 K/V append 到 cache |
| `native/rk_npu_matmul/src/rk_npu_attention_f16_internal.h:123–154` | stable cache 默认启用 LUT，并保留 V 的逐 token 累计量、每 32 token checkpoint |
| `native/rk_npu_matmul/src/rk_npu_attention_f16_lut.h:11–26` | LUT 正编码 bias=16，gain=255/256；输出 carrier `δ=16/32768=1/2048` |
| `native/rk_npu_matmul/src/rk_npu_attention_f16.cpp:645–656` | masked score 写成 -16，仍经 LUT 得到正 carrier |
| 同文件 `133–151` | `[E@V,E@1]` 用 FP16 PV partial 保存 |
| 同文件 `679–689` | FP32 finish 才减 `δ*Vsum`、`δ*valid_length` |

对本例的单个 PV block，令 `L_j` 为可见 token 的实际 FP16 LUT 输出，`H` 表示原生 PV 的 FP16 partial 舍入，位置 t=5。旧 AR 仅有 6 个有效 KV：

```text
N_AR = H(sum(j<=t, L_j * V_j)) - δ * sum_fp32(j<=t, V_j)
Z_AR = H(sum(j<=t, L_j))       - δ * (t+1)
```

batch 预先放入 13 个有效 KV；未来 7 个 token 的 mask 不会产生零 LUT 输出：

```text
N_batch = H(sum(j<=t, L_j * V_j) + δ*sum(t<j<13, V_j))
          - δ * sum_fp32(j<13, V_j)
Z_batch = H(sum(j<=t, L_j) + δ*7) - δ*13
```

真实数域中未来项可以消去；经过 FP16 partial 舍入后，`H(a+b)-b` 通常不等于 `H(a)`。未来 V 因此能改变 masked query 的输出。**只改 finish 的 prefix Vsum/length 而不清零 PV 前的未来 carrier，会留下真实未来项，不能作为修正。**

LUT 的每个表项至少保留 bias16。仅把 masked score 从 -16 换成 -17，尚不能据此保证硬件输出零；需要独立验证 DPU 的下溢/区间边界行为。当前已验证修正使用 LUT 后显式清零。

## 动态 A8 如何放大

W4A8 的逐行 scale 是 `max(abs(x))/127`，整数 code 使用 ties-away-from-zero：`rk_npu_w4a8_quant.cpp:31–40,80–82`。真实 layer3 的第一行如下：

| 阶段 | NRMSE | A8 scale：AR → batch | A8 code 改变数 |
|---|---:|---:|---:|
| attn | 0.0002979724 | 0.00980183389 → 0.00980373472 | 0/4096 |
| o | 0.0001939259 | — | — |
| post norm | 0.0000746472 | 0.02427140996 → 0.02426966839 | 12/2560 |
| FFN middle | 0.0033768757 | 0.00885709841 → 0.00884211157 | 78/9216 |
| hidden | 0.0084414293 | — | — |

表中 A8 只统计实际量化 consumer 的输入；o 和 hidden 还要经过残差/归一化，并不直接以表中原值量化。最初 o-proj 的输入 codes 仍相同，但输入 scale 已改变，所以输出随 scale 改变。残差和归一化之后，12 个 FFN 输入 code 跨越量化边界；后续非线性及量化继续放大。原完整 trace 最后一层 hidden NRMSE 0.0932333、最终 logits NRMSE 0.0889029694，与这条起点和传播链吻合。

## 保留 batch 的修正

carrier 独立补丁：证据根目录 `causal-carrier.patch`，仅修改 `src/rk_npu_attention_f16.cpp`；已在 `native/rk_npu_matmul` 根目录运行 `git apply --check` 通过。

1. 只有 Graph 的 `lut && mask_mode==CAUSAL` 路径将原组合 LUT/PV 阶段拆为 LUT 和 PV 两次 submit。
2. LUT 完成后，exponent FROM_DEVICE；将每行 `j>q_start+r` 的 FP16 exponent 置零；TO_DEVICE；再提交 batched PV。
3. finish 仅抵消该行可见 KV 的 carrier，分子使用相同前缀的 Vsum，分母使用 `q_start+r+1`。

普通 AR 保持原 pipeline：现有 `Workspace::geometry` 在 `rows==1 && prefix==cache.length` 时，先把 Graph 的 mask 改为 `NO_MASK`（原文件 721–722 行）。补丁判断 **Graph cfg**，因此终点 AR 不走新增 submit/sync；相关说明已经写入补丁注释。BOOLEAN、NO_MASK、fixed-shift 和 CPU exp 路径保持原分支。

前缀计算不随 context 长度增长。每个 KV head 从现有 `value_prefix[floor(q_start/32)]` 开始，最多回读 31 个历史 V，沿原 token 顺序累加 R 个 query 的 V。该顺序等于旧 `Cache::add_value_sum`，没有以浮点减法回退后缀。Q heads 共享所属 KV head 的前缀；缓冲复用 Graph 的 `softmax_scratch`。

复杂度为 `O(Hkv*(31+R)*D)`，暂存为 `R*D*sizeof(float)`，每个 KV head复用。C7 R8/D256 仅需 8 KiB 暂存，前缀加法上界为 `4*(31+8)*256=39936` 次。

## 回归覆盖

`final-v2/comparison.json`：10 个位置、120 项严格比较全部 PASS：

```text
q_start = 5, 27, 31, 253, 255, 509, 511, 1021, 4093, 32759
```

position5 使用原始真实 layer3 Q/K/V；其他位置周期展开这些 K/V，属于边界合成 fixture，不能代表整模型长上下文精度。Hq16/Hkv4/D256，R8，kv_tile512，mask7。

- LUT1：旧库 serial8 vs 新库 batch8，以及旧/新 serial8 均逐值相等。
- LUT0：旧/新 batch8、旧/新 serial8 均逐值相等。
- LUT0/1：NO_MASK，以及 BOOLEAN 的部分可见、全隐藏、全可见输出，旧/新均逐值相等。
- 每次调用重复运行逐值相同；`q_start=-1` 在各 mask mode 下维持 `ERR_PARAM`；causal start 超出 cache 也维持 `ERR_PARAM`，错误调用不改输出。
- `split/comparison.json`：`RK_NPU_ATTN_LUT_SUBMIT=split` 下 start0/5 同样通过，覆盖零前缀和 LUT 独立加载模式。

`final-v2/clocks.json` 的 exit_code=0、restore_errors=[]，before/restored 时钟字段全部一致。runner 耗时 67.84 s，仅用于正确性。`split/clocks.json` 同样记录恢复信息。

此轮carrier修复尚未处理独立的CPU边界：LUT0在span<=1024使用短exp/normalized路径，在>1024使用长exp/delayed路径。start1021的旧实现serial8/batch8有12277个不同元素，NRMSE 0.000177102192229、max_abs 0.00147581100464。CPU路径没有LUT carrier泄漏。下文通过真实Q/K/V隔离两种算术变化，并修复分段策略；上面的120项比较保留其原版本身份。

本次实际 13-token decoder trace 的 8 个 full-attention 层 max(abs(V)) 最高为 51.7743，低于 tile512 LUT 选择上界约 63.9048。该 fixture 使用 LUT；V 幅值或 descriptor 数触发的 CPU fallback 仍需保持独立精度边界。

## 过渡期的资格查询（已删除）

初版用只读公共接口检查整批和各AR前缀的LUT资格，Rust根据返回值选择batch或逐行append/run。关闭LUT、高V或descriptor不足时全部串行，因此正确但保守。`eligibility/all-probe.log`的14项检查记录了该过渡实现的零ioctl和无状态修改；这些是历史证据，不再是当前接口。

当前调用方统一使用普通append/run，数值模式选择由native负责。对应旧诊断源码已移除，回归集中在native的`tests/test_attention_f16_causal.cpp`。

## 整模型反例（CPU修复前）

2026-10-19，在真实C7与同一份运行时中，通过链接器 `--wrap=rk_npu_attention_f16_can_batch_causal` 切换正常查询和直接返回允许batch。未修改native数值实现、target权重或输入。输入取固定长提示的1040个真实token IDs；每轮验证4行，即K3的anchor+3proposals。每个对照先重置并建立相同前缀，比较完整4行logits，以及提交4个输入后的下一行logits。该实验不执行draft网络，不作为MTP生成速度测试。

| LUT | 前缀长度 | 保留查询：logits/next NRMSE | 强制batch：logits NRMSE | 强制batch：next NRMSE |
|---|---:|---:|---:|---:|
| 开启 | 1021 | 0 / 0 | 0 | 0 |
| 关闭 | 1016 | 0 / 0 | 0 | 0 |
| 关闭 | 1021 | 0 / 0 | 0.0751631 | 0.0554600 |
| 关闭 | 1024 | 0 / 0 | 0 | 0 |

CPU softmax跨界的1021用例，强制batch后993280个验证logits中993279个改变，最大绝对差1.0085311；下一行248320个logits全部改变，最大绝对差0.58617866。4行与下一行的argmax仍相同，不能据此认为采样分布相同。保留查询时，8个Attention层均选逐行，每个元素与AR一致。

这个反例说明当时不能直接删除判断：返回0虽保守，但跨界确实改变target分布。1016和1024两个CPU对照均能batch，表明不应把全部CPU路径永久串行。以下修复保持各行的AR算法，由native选择兼容分段。

默认LUT下的真实CLI文本、图片与本次1021对照均未触发串行fallback。关闭LUT是库已支持的数值验证配置；上述反例不代表默认C7每次验证都有问题。公共接口与native调度的组织方式可独立调整，但仅移动或绕过检查不能消除此数值差异。

证据为 `mtp-20261018/gate-necessity-20261019/` 下的 `probe.rs`、`run.sh`、`long-prompt.tokens`、`lut0.log`、`lut1.log` 和对应clocks。两个runner均exit0、restore_errors=[]，受控频率上下限及governor恢复。

## CPU softmax根因与native分段修复

真实C7、prefix1021、R4，在首个FullAttention层捕获compact FP16 Q/K/V和FP32输出。AR与batch的全部Q以及各可见前缀K/V逐位相等。固定这些输入，独立native replay逐位复现16384个batch输出；各干预版本的raw QK也逐位相等。

前三行的可见长度为1022/1023/1024，AR的物理span均为1024；整批span为1056，提前走long路径。第四行可见长度1025，本来就应该走long路径。

```text
short: E=exp6(score-max); P=FP16(E/sum_fp32(E)); O=PV(P)/PV(P,1)
long:  E=exp5(score-max); U=FP16(E);           O=PV(U)/PV(U,1)
```

归一化移动到了FP16舍入边界另一侧，FP16(E/Z)与FP16(E)后除并不等价。softmax在CPU执行仍会受到后续FP16输入和PV partial舍入影响；这与LUT未来carrier缺陷独立。

| 干预 | 前三行相对AR | 第四行相对AR |
|---|---|---|
| 原long路径 | NRMSE 0.000352/0.000308/0.000338 | 逐位相等 |
| 只把long exp换为exp6 | 全部输出仍逐位等于原batch | 逐位相等 |
| 只恢复PV前归一化 | 第1行仅2/4096不同，NRMSE 2.38e-6；第2/3行逐位相等 | NRMSE 0.000320 |
| 同时恢复exp6和PV前归一化 | 三行均逐位相等 | NRMSE 0.000320 |
| 整批强制完整short实现 | 三行均逐位相等 | NRMSE 0.000320 |

因此归一化顺序是主因，exp近似只留下更小差异。不能把整批统一改成short来修复：这会改变原本正确的第四行AR行为。单层约0.03%的起始误差经过后续动态A8和网络运算，得到上文整模型verify约7.52%的误差。

当前实现：

- `Workspace::causal_rows`比较候选batch和各行AR的LUT资格、实际descriptor/split配置、short/long分支、每KV head的delayed-normalization与half-probability模式，只合并兼容行。1021/R4拆成3行short＋1行long。
- `Graph`持有本次run的前缀V幅值界；每次bind均更新，即使几何和buffer地址未变。未来高V不会改变较早行的LUT/CPU、归一化或gain选择；cache.length/value_total不作临时修改。
- 前缀幅值从每32行checkpoint继续读取最多31个历史V，再按新行顺序延伸。低于所有切换阈值的全cache界直接复用；正常LUT/CPU batch不扫描历史V。规划计时计入prepare，total包括全部调用。
- 删除公共资格接口、Rust封装、模型层serial fallback和过渡期专用probe。模型统计改为`attention_verify_calls/rows`，计数为逻辑调用，不能解释成native物理分段数。公共query的tasks为基础几何估计，实际分段任务见run timings。

固定真实Q/K/V的修复版4行共16384个输出与AR全部逐位相等。native45项板端回归在pingpong和split两种提交模式下均通过，共90项；包含未来V=64/65000、LUT/CPU与gain切换、32K descriptor、1/2/3核、GQA32+1、bulk load/append、reserve及truncate/abort。

| LUT0，R4 | native batch实际tasks | 四次AR tasks | 输出 |
|---|---:|---:|---|
| prefix1016 | 12 | 48 | 逐位相等，保留batch |
| prefix1021 | 28 | 52 | 逐位相等，3＋1分段 |
| prefix1024 | 16 | 64 | 逐位相等，保留batch |

证据为`mtp-20261018/cpu-softmax-20261019/`：`capture.rs/capture.log`、五组`attention-*.cpp`及`raw-score.f16/pv-input.f16/output.f32`、`isolation.json`、`native.log`和clocks。该轮只做数值修复与针对性验证，没有扩展普通prefill PPL或性能测量。

完整运行时复验：`model-lut0.log`与`model-lut1.log`均为最终native数值分段实现，不再使用链接器绕过资格接口。每种LUT设置各9组：prefix1016/R4与1024/R4全提交，prefix1021/R4的abort/消费1/3/4行，prefix1021/R8的abort/消费7/8行。每组对照全部verify logits，以及真实abort或提交后的下一行logits；18组、36项比较、28,308,480个FP32元素全部逐位等于AR。ctx2048、chunk128、同一冻结调优cache与1040个固定真实token IDs。abort使用独立`abort_verification()`，不将消费0行传给提交接口。

最终Rust默认85 passed/13 ignored、fmt与全target release Clippy通过。额外FP16 Linear M3/M129、非因果Attention与FP16输出对照通过，Attention相对CPU max_abs为1.43e-5。14份成功测试clocks均exit0、restore_errors=[]，受控频率上下限及调速器恢复；汇总和最终源码hash见`verification.json`。模型回归期间并行编译默认测试，所有耗时只作为执行记录，不作为性能证据。

## 高 V 回滚元数据

原 `Cache::truncate` 保留历史 `value_absmax`。未来 V=64 被截掉后，LUT 资格仍为 false，重放首行有 4096/4096 个元素改变，NRMSE 0.00039465977235（`eligibility/v2-probe.log`）。这是数值模式状态的事务回滚问题，单靠逐行 fallback 不能避免。

`causal-prefix-absmax.patch` 添加每 32 token、每个物理 KV head 的 `uint16_t` prefix absmax checkpoint。load/append 按每行真实 V 扫描并保存 checkpoint，不能用整个 append 的最终 max 代替中途 max。truncate 从 `floor(length/32)` checkpoint 加最多 31 行 retained V 恢复精确 bound，包含 LUT 被禁用时的 CPU 路径。

checkpoint 空间为 `(capacity/32+1)*Hkv*2` bytes，C7/32K 为 8,200 bytes；恢复成本 `O(Hkv*31*D)`，不随历史长度增加。reserve 保留已有 checkpoint，reload 重置，truncate 后替换 append 按精确保留 bound 延续。

未来高 V 的 partial commit 与 abort 覆盖 start=0/5/27/31/509/511，包含 checkpoint 和 KV tile 边界。所有情形重新获得 LUT 资格，partial replay 与 abort 后重新 decode 均逐值匹配从未见高 V 的 AR（`eligibility/all-probe.log` 的六项 `ROLLBACK_CHECK`）。

## 尾带同步与性能 gate

carrier 独立补丁为 causal LUT 增加一轮 submit 和 exponent FROM/TO 各一轮完整张量 sync。C7 R8 的 physical_M=32；每方向同步字节是 `4*32*span*2`，32K 时为 8 MiB。普通终点 AR 没有这些新增操作。

独立 `causal-tail-sync.patch` 已实现 native K8 尾带同步：每 head 从 `floor((q_start+1)/8)*8` 到 span，仅处理有输出的实际行；范围按 64-byte cache line 对齐，FROM_DEVICE 保证相同 cache line 内未修改的可见概率来自本轮设备输出。普通 GQA `span=align32(q_start+R)`，尾带长度上界 R+38，与 context 无关。32K 的本例仅需 2 KiB/方向（所有 heads 合计），代价是 2*Hkv 次小范围 sync，替代两次整张量 sync。后置 NPU mask 或已验证的 LUT 零下溢编码可继续省去 CPU exponent 同步。

尾带/资格/prefix absmax 组合构建与 full-sync carrier 构建的十位置、120 项严格比较全部通过（`eligibility/trace/comparison.json`），含真实 p5、跨 32/512、32K、NO_MASK/BOOLEAN 和 LUT0/1；padding M 另由上述资格 probe 覆盖。`eligibility/all-clocks.json` 和 `eligibility/trace/clocks.json` 均 exit0、restore_errors=[]，before/restored 时钟字段一致。完整 trace 回归耗时 67.80 s，仅用于正确性。

carrier修复阶段的集成回归：native `tests/test_attention_f16_causal.cpp --board` 共34例通过，覆盖1/2/3核、GQA32+1分段、预加载未来/零尾、bulk load/append checkpoint 和 reserve/truncate。当时默认CTest仅运行`--cpu`的资格查询检查，C header编译通过；当前已改为普通run参数检查，上述专用接口不再存在。Rust tuned chunk128 的八个整模型提交边界和后续 logits 共17项零差异比较，重复 abort/预取消通过，记录为 `mtp-20261018/final-tests.log`。固定频率的 K3 生成另作完整 ABBA 测量，不使用算子回归耗时推断吞吐。

## 修复开销的隔离测量

本节A/B只覆盖carrier、尾带同步与前缀回滚修复，测量早于CPU数值分段重构；这些数字不代表新增planner的开销。本轮没有重跑完整MTP性能或普通prefill测速。

2026-10-19，使用相同 `g++ -O2 -std=c++17 -fopenmp -march=armv8.2-a+fp16` 编译修复前后的 Attention 和 cache 两个编译单元，其余链接同一冻结静态库。CPU4–7固定2256MHz、NPU固定1GHz，DDR未受控。每个几何三轮 `[before,after,after,before]`，每进程预热8次、采样40次；共180个进程记录、7200次完整 Attention run 调用。计时含 query pack、QK、softmax、PV、同步和输出处理，排除初始 KV load、建图及进程启动。

| q_start | rows | 修复前 ms | 修复后 ms | 增量 ms |
|---:|---:|---:|---:|---:|
| 5 | 4 | 0.151 | 0.227 | +0.076 |
| 511 | 4 | 0.254 | 0.470 | +0.216 |
| 1021 | 4 | 0.377 | 0.575 | +0.198 |
| 4093 | 4 | 1.246 | 1.430 | +0.183 |
| 32759 | 4 | 9.170 | 9.330 | +0.160 |

R4 对应 K3 的 anchor+3 proposals。单层增加约0.08–0.22ms，短上下文的相对增幅较大；1K/4K/32K测点分别约52.4%/14.7%/1.75%。C7八个 Attention 层按这些测点估计增加约1.3–1.7ms/轮，相对此前约143ms的完整 verify 约1%；这是跨算子到整模型的估算，不是完整模型修复前后的 A/B。

普通终点 AR 不增加 submit/sync。R1 在1K/4K/32K测点变化均小于1µs；p5/511的短调用实测分别增加约39/29µs，差异集中于 submit 段，不能将“未增加提交”写成所有深度零耗时变化。R8 增量约0.11–0.21ms。只有p5读取真实短 trace；其余深度由相同 K/V 周期展开，代表算子几何，不能代表真实长上下文整模型行为。

测量开始前已结束 Rust/native 编译，正式测量期间未运行模型或其他编译任务。runner exit0、restore_errors=[]，受控频率上下限和 governor 恢复。原始CSV、分块记录、汇总、同参数构建源码与 clocks 在 `mtp-20261018/attention-fix-latency-20261019/isolated/`。

## 复现与产物

统一应用候选为证据目录 `eligibility/native-batch-attention-all.patch`：277 行、四个文件，基于算子 `c3bff02`。该组合补丁已在 clean native 根执行 `git apply --check`，并在独立副本实际应用，得到的四个文件逐字节等于已编译测试的源码。

需要分开审查时按以下顺序应用；组合版已经包含四者：

1. `causal-carrier.patch`
2. `eligibility/causal-batch-eligibility.patch`
3. `eligibility/causal-tail-sync.patch`
4. `eligibility/causal-prefix-absmax.patch`

资格/回滚回归二进制：`eligibility/probe-all`；trace 回归二进制：`eligibility/probe-all-trace`。两者均链接独立修改的 Attention 与 cache 编译单元以及冻结的 `librk_npu_matmul-before.a`，不依赖后续主库重编译。最终摘要、源码及二进制 SHA256 记录在 `eligibility/final-hashes.sha256`。

证据目录保留 `attention_before.cpp`、`attention_causal.cpp`、`librk_npu_matmul-before.a`、`probe.cpp`、`probe-full`、`probe-causal`、`causal-carrier.patch`，以及全部 `.f32`、日志、时钟记录。

`hashes.sha256` 记录原始源码/静态库/probe；`reviewed-hashes.sha256` 记录最终注释补齐后的源码与 patch。重新编译后的 `probe-causal-reviewed` 与已验证 `probe-causal` 二进制 SHA256 相同：`437c2a35925fd854b9eb1ca0d98733280109e3b2cd2b0f4fb881ac39a5056539`。

板端命令（`audit` 为上述板端证据目录）：

```bash
python3 /tmp/qwen35-mtp-runtime/scripts/run_board.py \
  --wait-lock --cpu-khz 2256000 --skip-dmc \
  --report "$audit/reproduce/clocks.json" -- taskset -c 4-7 \
  python3 "$audit/run_audit.py" \
  --before "$audit/probe-full" --after "$audit/probe-causal" \
  --trace /home/firefly/mnt/3090x4/tmp/qwen35-dev/mtp-20261018/trace \
  --output "$audit/reproduce"
```

构建使用 `g++ -O2 -std=c++17 -fopenmp -pthread -march=armv8.2-a+fp16 -DRK_NPU_ENABLE_TRACE=0`，include 原库 `include/` 与 `src/`；before 链接 `probe.cpp + librk_npu_matmul-before.a -ldl`，after 在静态库之前加入 `attention_causal.cpp`，只覆盖该编译单元，不修改主库。
