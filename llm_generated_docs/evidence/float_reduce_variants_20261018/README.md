# Floating MM reduction variants, 2026-10-18

See [analysis and comparison](../../FLOAT_MM_REDUCTION_COMPARISON.md) for results and limits.
The complete six-path performance/precision table is in [results](../../FLOAT_MM_REDUCTION_RESULTS.md).

- `microbench.json`: complete-call ABBA samples for FP16 fused, FP16 partial/CPU sum, FP32 partial/NPU tree and FP32 partial/CPU sum. Each file preserves its own paired revision3 FP16 NPU-tree reference. These are operator measurements.
- `fused-fp16/`: full-channel feedback checks and raw fused parameter scans; the first pilot failed to compile before NPU execution, then was corrected.
- `fp32-primitives/`: true FP32 writer and Add-tree checks, library dense checks, raw parameter scans. `marker*.log` are failed fused FP32 feedback probes, retained as failures. They are not valid candidates or performance results.
- `ablation/`: one process-level independent/fused/fused/independent group for each fixed geometry and core mask. Each process performs two initial calls and twelve timed calls; repeated calls are not independent trials. The ioctl summaries include all fourteen calls.
- `variants.json`: model precision, available complete ViT timing samples, and fixed-geometry fusion comparisons. Experimental precision failures are recorded rather than excluded. The historical analysis tool is included in the external revision5 source snapshot.
- `default-load-only.json`: current revision4 library reads all21 revision2 complete-K profiles with21hits and zero searches.
- `revision5/`: balanced-task refinement, all21 legacy complete-K/FP32 profile hits with zero search, and the fused/CPU variants'14-profile searches, model metadata and timing samples. A full red-image run confirmed bitwise FP32 feature preservation when reusing revision4 profiles and resident-weight markers.
- `runners/`: commands, outputs, exits and fixed-clock restoration records. `hardware-restored.json` checks frequency limits/governors and restore errors. Numeric probe failures and the initial compile failure remain nonzero exits.

Raw arrays and frozen current binaries are retained at `/home/zt/mnt/3090x4/tmp/floating-mm-variants-20261018/`. Primitive/fused binaries and their matching shared libraries are retained in `floating-mm-fp32-accum-20261018/` and `floating-mm-fused-k-20261018/`. Current Rust examples link the native library statically. Tests and benchmarks do not compute BF16.

The current runtime retains only default-on FP16 K tiling with NPU FP16 reduce, or full K via RK_NPU_MM_K_TILING=off. This directory records the earlier experiments. Current usage is in [the API guide](../../FLOAT_MM_AUTOTUNE.md); normal inference remains load-only with legal fallback. A successful primitive check does not imply model precision passes. FP32 fused feedback is disabled after a half-channel numerical failure; no NPU submission failure occurred in these probes.

The removed experimental implementations and diagnostic tools are preserved outside the repository at `/home/zt/mnt/3090x4/tmp/floating-mm-variants-20261018/experimental-source-revision5/`. Historical script paths below refer to that snapshot, not current runnable tools.
