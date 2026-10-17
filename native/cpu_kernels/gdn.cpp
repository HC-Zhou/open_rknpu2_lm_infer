// Adapted from ggml_compute_forward_gated_delta_net_one_chunk, llama.cpp
// a17e94d229fc7fed1567a799baebd11b771fe5af, MIT license.
// Copyright (c) 2023-2026 The ggml authors.
// Full license: ../../llm_generated_docs/THIRD_PARTY_NOTICES.md.
#include "gdn.h"
#include "vec.h"
#include "block.h"
#include "small_batch.h"
#include <cstring>
#include <vector>

namespace {
template<bool fused, bool output>
void run(const rknpu2_gdn_params & p) {
    using namespace rknpu2_cpu_kernels;
    const size_t kw = p.key_heads * p.key_dim;
    const size_t vw = p.value_heads * p.value_dim;
    const size_t head_size = p.key_dim * p.value_dim;
    const float scaling = 1.0f / std::sqrt(static_cast<float>(p.key_dim));
    #pragma omp parallel num_threads(p.threads)
    {
        // ggml uses per-worker delta scratch, never per-row allocations.
        std::vector<float> delta(fused ? 0 : p.value_dim);
        #pragma omp for schedule(static)
        for (size_t h = 0; h < p.value_heads; ++h) {
            float * s = p.state + h * head_size;
            const float * initial = p.initial_state + h * head_size;
            if (s != initial) std::memcpy(s, initial, head_size * sizeof(float));
            const size_t kh = h / (p.value_heads / p.key_heads);
            for (size_t t = 0; t < p.rows; ++t) {
                const float * k = p.key + t * kw + kh * p.key_dim;
                const float * q = output ? p.query + t * kw + kh * p.key_dim : nullptr;
                const float * v = p.value + t * vw + h * p.value_dim;
                float * out = output ? p.output + t * vw + h * p.value_dim : nullptr;
                const float decay = std::exp(p.log_decay[t * p.value_heads + h]);
                const float beta = p.beta[t * p.value_heads + h];
                if constexpr (fused) {
                    size_t j = 0;
#if defined(__aarch64__)
                    if (p.key_dim == 128) {
                        constexpr size_t block_rows = 4;
                        for (; j + block_rows <= p.value_dim; j += block_rows)
                            update_block<block_rows, output>(s + j*128, k, decay, beta, v + j, q,
                                output ? out + j : nullptr, scaling);
                    }
#endif
                    for (; j < p.value_dim; ++j) {
                        const float result = update_row<output>(p.key_dim, s + j*p.key_dim, k, decay, beta, v[j], q);
                        if constexpr (output) out[j] = result * scaling;
                    }
                } else {
                    // Literal ggml loop, with rectangular dimensions and HF
                    // head mapping supplied by this adapter.
                    scale(head_size, s, decay);
                    for (size_t j = 0; j < p.value_dim; ++j)
                        delta[j] = (v[j] - dot(p.key_dim, s + j*p.key_dim, k)) * beta;
                    for (size_t j = 0; j < p.value_dim; ++j)
                        mad(p.key_dim, s + j*p.key_dim, k, delta[j]);
                    if constexpr (output) {
                        for (size_t j = 0; j < p.value_dim; ++j)
                            out[j] = dot(p.key_dim, s + j*p.key_dim, q) * scaling;
                    }
                }
            }
        }
    }
}
} // namespace

extern "C" void rknpu2_gdn_run(const rknpu2_gdn_params * p) {
#if defined(__aarch64__)
    // One worker does not benefit from the short-window loop interchange.
    if (p->threads > 1 && p->rows >= 1 && p->rows <= 8 && p->key_dim == 128 && p->value_dim % 4 == 0) {
        if (p->output) rknpu2_cpu_kernels::run_small_batch<true>(*p);
        else rknpu2_cpu_kernels::run_small_batch<false>(*p);
        return;
    }
#endif
    if (p->output) run<true, true>(*p);
    else run<true, false>(*p);
}

extern "C" void rknpu2_gdn_run_unfused(const rknpu2_gdn_params * p) {
    if (p->output) run<false, true>(*p);
    else run<false, false>(*p);
}
