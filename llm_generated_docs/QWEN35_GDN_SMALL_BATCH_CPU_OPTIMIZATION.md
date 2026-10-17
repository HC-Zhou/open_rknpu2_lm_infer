# CPU GDN 短窗口优化

2026-10-17；基线 `ff70cf9`，备用1。本轮覆盖 CPU M≤8，未改 NPU 实现或模型权重。

## 实现

- 普通 `LayerState.forward` 直接原地更新 FP32 canonical state，省去每窗口24层共48 MiB的初态复制；`verify` 继续使用独立工作状态，接受前缀仍用 replay 提交。卷积历史在成功后提交。
- 内部 `ValidatedProjected` 类型连接输入检查与 mixer，投影的形状/有限值只检查一次。GDN update 的数值检查保留在原地更新之前；后续输出 norm 的形状与 epsilon 由权重构造及工作缓冲保证。
- 新的 [`native/cpu_kernels/small_batch.h`](../native/cpu_kernels/small_batch.h) 以4个 value row为一块（2 KiB），连续处理短窗口，复用局部状态。多线程、Dk128、Dv为4的倍数、M=1..8启用；单线程和其他形状沿用原 kernel。
- C ABI、FP32状态布局、head映射与每行浮点算序保持一致。验证路径保留连续初态复制。

## 测量

Release，Rust 1.95（opt-level3、thin LTO、codegen-units1），C++ `-O3 -DNDEBUG -mtune=cortex-a76`，未启用fast-math。CPU4/`0xf0`，2256 MHz，OpenMP passive/零自旋；共享板锁，未控制DMC。新旧程序使用同一份 benchmark 源码及输入，按 A→B→B→A 执行。每个程序预热2次、每档7个计时样本；每侧两次程序运行，汇总14个描述性样本。计时用单调时钟，I/O/输入构造不计入。完整mixer的每个样本前还会reset并预跑一遍，准备非零状态/历史，此过程不计时。

主结果是 **普通forward的24层完整 CPU mixer**：包含投影检查、Conv+SiLU、Q/K归一化、gate、recurrence、输出 gated norm及历史提交；不含投影矩阵乘、Attention、lm_head或NPU。各层有独立 canonical state，共48 MiB；投影输入和权重使用确定性C7形状合成值，权重对象在层间共享。表中是组件重放，不能直接换算成整模型吞吐或MTP verify的提升。

| M | 优化前 ms/24层 | 优化后 ms/24层 | 延迟下降 |
|---:|---:|---:|---:|
| 1 | 10.977 | 9.521 | 13.26% |
| 2 | 14.078 | 10.961 | 22.14% |
| 3 | 16.591 | 12.921 | 22.12% |
| 4 | 18.806 | 15.028 | 20.09% |
| 5 | 21.924 | 17.724 | 19.16% |
| 6 | 24.243 | 19.674 | 18.85% |
| 7 | 27.008 | 22.582 | 16.39% |
| 8 | 28.880 | 24.425 | 15.43% |

另用固定 ggml fixture 测量实际 Rust recurrence 接口：包含检查、初态复制、24层 canonical轮转、共享scratch和状态交换。其输入为已有 oracle 包的归一化Q/K、非零初态，输出与最终state逐值检查。

| M | 优化前 ms/24层 | 优化后 ms/24层 | 延迟下降 |
|---:|---:|---:|---:|
| 1 | 6.403 | 6.293 | 1.72% |
| 2 | 7.441 | 6.971 | 6.32% |
| 4 | 10.219 | 9.625 | 5.81% |
| 7 | 14.666 | 14.356 | 2.11% |
| 8 | 16.085 | 15.814 | 1.68% |
| 128 | 203.359 | 202.751 | 0.30% |

M128沿用原kernel，差异小于基线样本波动，不计作收益。

单线程的一层调用作为辅助检查，结果单独保存在原始JSON中；不与24层轮转口径混合。

## 验证与证据

13项 `cpu_layers` Release测试通过：包含固定参考、M=1..9、矩形/尾部回退、非均匀/极端gate、原地/复制/replay一致性、分块等价、部分接受和历史提交。新增用例还验证有限投影产生无效update时，在卷积预处理之后返回错误，canonical state/history/position仍保持原值，下一次正常调用与重算一致。

C7 recurrence窗口1/2/4/7/8/128，在1/4线程下的输出与最终state均与固定ggml oracle逐值一致。频率上下限/governor恢复检查通过。Release Clippy全target（`-D warnings`）、格式及diff检查通过。2026-10-18提交前完整Release回归56项通过、7项忽略，并在共享锁和固定频率下复跑。C7整模型固定129-token prompt及4-token continuation产生5行、1,241,600个logits，与重构前保存的参考逐位一致（`commit-validation.json`）；频率恢复成功。本轮未重测整模型吞吐或PPL。

原始样本、两次程序/侧的比较、测试和构建日志保存在 [evidence/qwen35_gdn_small_batch_20261017/](evidence/qwen35_gdn_small_batch_20261017/)，入口为 `summary.json`；大型输入和实测二进制保留在 `/home/zt/mnt/3090x4/tmp/qwen35-dev/gdn-small-20261017/`（板端挂载前缀为 `/home/firefly/mnt/3090x4/`）。

## 复现入口

- [`examples/gdn_small_bench.rs`](../examples/gdn_small_bench.rs)：Rust recurrence完整调用，默认24层；`--mixer` 使用 [`mixer.rs`](../examples/gdn_small_bench/mixer.rs) 测完整CPU层，覆盖 M=1..8。
- [`scripts/gdn/bench_small_batch.py`](../scripts/gdn/bench_small_batch.py)：C ABI共享库的ABBA比较，可分别测复制/原地/output-free replay。
- 固定数值参考来自 [既有GDN oracle包](evidence/qwen35_gdn_kernel_20261017/)。本轮的 `fixtures-manifest.json` 记录原始NPZ及紧凑FP32文件布局。

板端构建并运行完整CPU mixer，不需要checkpoint或输入文件：

```sh
cargo build --release --locked --example gdn_small_bench
python3 scripts/run_board.py --wait-lock --cpu-khz 2256000 --skip-dmc \
  --report /tmp/gdn-small-clocks.json -- taskset 0xf0 \
  target/release/examples/gdn_small_bench /tmp /tmp/gdn-small.json \
  --mixer --layers 24 --threads 4 --rows 1,2,3,4,5,6,7,8
```

recurrence模式读取 `tokens-NNNN.input.f32`（依次为压缩head后的`q,k,v,g,beta,state_initial`）和 `tokens-NNNN.expected.f32`（`output,state_final`），各数组直接按NPZ中的C连续布局保存为little-endian FP32。将文件目录作为第一个参数，去掉 `--mixer`，使用 `--rows 1,2,4,7,8,128` 即可复现；该模式首个窗口必须与oracle逐值一致才计时。
