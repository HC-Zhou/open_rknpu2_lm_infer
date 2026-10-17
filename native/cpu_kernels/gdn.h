#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Rust validates all dimensions, buffer lengths and values before this ABI.
// State: [value_head, value_channel, key_channel]; other tensors token-major.
// Q/K use contiguous HF repeat-interleave heads. Null query/output means replay.
// initial_state may equal state; otherwise the worker copies its own head.
struct rknpu2_gdn_params {
    size_t rows, key_heads, value_heads, key_dim, value_dim, threads;
    const float * initial_state;
    float * state;
    const float * query;
    const float * key;
    const float * value;
    const float * log_decay;
    const float * beta;
    float * output;
};

void rknpu2_gdn_run(const struct rknpu2_gdn_params * params);
// Exact four-pass ggml loop, retained for paired module benchmarks.
void rknpu2_gdn_run_unfused(const struct rknpu2_gdn_params * params);

#ifdef __cplusplus
}
#endif
