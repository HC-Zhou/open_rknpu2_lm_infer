# GDN kernel 独立复现与 NPU 实验交接

日期：2026-10-17，Asia/Shanghai。仅覆盖 recurrence kernel，不依赖 C7 权重、Rust runtime 或整模型。

## 入口与数据

- CPU C++ 模块：[`native/cpu_kernels/`](../native/cpu_kernels/)，C ABI 见 [`gdn.h`](../native/cpu_kernels/gdn.h)。CPU保留FP32 canonical state；`gdn.cpp` 管 head/window 和状态复制，`vec.h` 管通用 NEON，`block.h` 管 Dk128 的四行共享 Q/K 加载。
- 独立测试数据：[`evidence/qwen35_gdn_kernel_20261017/`](evidence/qwen35_gdn_kernel_20261017/)，8个计时用例加3个非均匀gate正确性用例；每个都是独立非零初态窗口，计时用例token数为1/2/4/7/8/32/128/512。
- [`manifest.json`](evidence/qwen35_gdn_kernel_20261017/manifest.json) 记录布局、范围、每个文件和原始 ggml 输入/输出的 SHA256；[`cpu-baseline.json`](evidence/qwen35_gdn_kernel_20261017/cpu-baseline.json) 保留1/4线程的原始7个计时样本。
- 完整 CPU 优化过程、实际 ggml 对照与构建身份见 [QWEN35_GDN_LLAMA_ALIGNMENT.md](QWEN35_GDN_LLAMA_ALIGNMENT.md)。上游 ggml 基点 `a17e94d229fc7fed1567a799baebd11b771fe5af`，本地 checkout `de3ebb93bd8828496faefb44ed45059ec7ed3ea5`。
- 数据目录内还附了 `reference_cpu/`、`reference_runner/` 和 `check_case.py` 的源码快照及哈希。可以只拷贝这个目录，用 `cmake -S reference_runner -B /tmp/gdn-cpu` 编译；无需恢复整个未提交工作树。

主包是C7形状的确定性合成kernel输入，非真实checkpoint activation；`g=-0.07`、`beta=0.37`，非均匀gate的额外C7形状用例为 `stress-0001/0007/0128.npz`，覆盖 `g=0/-1e-6/-0.07/-1/-10/-90/-inf` 和 `beta=0/1/0.01/0.37/0.99`，同时随token/head变化。数据取自实际 ggml CPU op 的已测输入/输出，没有用本实现重算后替换 oracle。源格式中 Q/K 已展开为32个 head；打包时检查每对 head 逐值相等，再无损压缩为16个 head。C++ ARM kernel 与参考的8个窗口、输出及最终 state 均逐值相等。独立驱动在 x86_64 的标量 fallback 也通过数值门槛，记录为 `standalone-host-validation.json`。

当前主线代码已按通用算子/层重组；上述路径是新的维护入口。证据目录中的 `reference_cpu/`、`reference_runner/`、manifest 及11个NPZ保持原始快照和哈希，其中旧 C ABI 名称 `qwen35_gdn_*` 仍可独立编译。新主线 ABI 为 `rknpu2_gdn_*`，状态/输入/输出布局不变。

后续 CPU M≤8 优化见 [QWEN35_GDN_SMALL_BATCH_CPU_OPTIMIZATION.md](QWEN35_GDN_SMALL_BATCH_CPU_OPTIMIZATION.md)：普通 forward 原地更新，verify 保留工作副本，短窗口独立分块。这里的oracle包及历史单层基线保持原始快照，NPU对接的C ABI和FP32布局保持一致。

## 数学与布局

固定 C7 形状 `Hk=16, Hv=32, Dk=Dv=128`，所有数组均为 C contiguous、little-endian FP32：

| NPZ key | shape | 含义 |
|---|---|---|
| `q`, `k` | `[T,16,128]` | 已做L2 norm，尚未应用query scale |
| `v` | `[T,32,128]` | post-convolution/SiLU value |
| `g` | `[T,32]` | **log decay**，kernel 内取一次 `exp(g)` |
| `beta` | `[T,32]` | 已做sigmoid的更新系数 |
| `state_initial` | `[32,128,128]` | `[value_head,value_channel,key_channel]` 非零初态 |
| `output` | `[T,32,128]` | ggml reference的core输出，尚未做output gated RMSNorm |
| `state_final` | `[32,128,128]` | 最后一个真实token之后的state |

每个 value head `h` 使用 **`kh=h//2`** 的 Q/K。不能直接使用 ggml op 未展开输入时的 `h%16` 映射；当前 safetensors 的连续 repeat-interleave 顺序不同。

对每个 head/token，令转置状态 `M` 的行是 value channel，列是 key channel：

```text
M *= exp(g)
delta = (v - M @ k) * beta
M += delta[:, None] * k[None, :]
output = (M @ q) / sqrt(128)
```

状态更新先 decay、再prediction、再outer correction。`q/k` 不重复归一化，`beta` 不重复sigmoid，query scale只应用一次。只有真实token行进入状态；硬件padding不推进state。此处不含卷积、QKV/Z/A/B投影、output gate/norm和输出投影。

现有CPU调查线为 `max_abs <= 1e-5 + 1e-4*max_abs(reference)`、relative L2 <= `1e-4`，输出与最终state分别检查。额外的小维度非均匀gate/非零状态/矩形layout fixture保存在 `transformers-small-fixtures.json`，来自固定 Transformers revision `09281b8d56b08199ce9f397c694f38853cc45278`；不依赖本C++实现。

## 独立运行 CPU reference

在仓库根目录执行，不需要初始化或构建 NPU 子模块：

```sh
cmake -S scripts/gdn/cpu_reference -B /tmp/gdn-cpu \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/gdn-cpu -j4
python3 scripts/gdn/check_case.py \
  --case llm_generated_docs/evidence/qwen35_gdn_kernel_20261017/tokens-0128.npz \
  --runner /tmp/gdn-cpu/gdn_kernel_runner --threads 4
```

脚本依赖NumPy；逐项核对core输出和最终state。AArch64使用NEON，其他主机使用FP32标量fallback，允许合理的reduction舍入差。

## 对接另一个 NPU kernel

直接从 NPZ 读取各输入。将 NPU 结果按 **`output.ravel()` 然后 `state_final.ravel()`** 写成一个 little-endian FP32 文件，例如 `npu-output.f32`：

```sh
python3 scripts/gdn/check_case.py \
  --case llm_generated_docs/evidence/qwen35_gdn_kernel_20261017/tokens-0128.npz \
  --actual /path/to/npu-output.f32
```

最小接入可以只做方形D128和C7 head数。矩形/短value row的CPU oracle在 `tests/cpu_layers/gdn.rs`，后续扩展时可复用。replay只需要state更新，不需要query/output；单层事务与部分接受边界在 `src/layers/gated_delta/state.rs` / `tests/cpu_layers/state.rs`，与kernel执行独立。

CPU完整调用基线：备用1，4个大核/`0xf0`，2256 MHz，FP32 state；token1 **0.133325 ms**、token128 **8.448469 ms**、token512 **33.955851 ms**。包含真实Rust接口的输入检查、FFI、每个worker复制自己的初态和token-major输出，预热2次、每档7个样本。相应实际ggml为0.197835/12.152299/48.626169 ms。性能结果不要混合device-only与包含pack/传输/同步/下游输出的完整调用；NPU常驻state时另列状态所有权与提交成本。

备用1的端到端作业使用 `/tmp/rknpu_lowlevel_submit.lock`。另一个Agent上板时请沿用 `scripts/run_board.py --wait-lock --cpu-khz 2256000 --skip-dmc`，该板没有DMC devfreq；也可先在其他板做模块实验。原始大文件和运行现场保留在备用1 `/home/firefly/mnt/3090x4/tmp/qwen35-dev/gdn-cpp-20261017/`，本机对应 `/home/zt/mnt/3090x4/tmp/qwen35-dev/gdn-cpp-20261017/`。

重新生成数据包：

```sh
python3 scripts/gdn/export_cases.py \
  --source /path/to/gdn-cpp-20261017/final-t4 \
  --stress-source /path/to/gdn-kernel-handoff \
  --output /path/to/gdn-kernel-cases
```
