#pragma once
#include "gdn.h"
#include "block.h"
#include <cstring>

namespace rknpu2_cpu_kernels {
#if defined(__aarch64__)
// A short window reuses a 2 KiB value tile instead of scanning a whole head
// between tokens. Keep the contiguous copy for verification's separate state.
template<bool output>
inline void run_small_batch(const rknpu2_gdn_params & p) {
    const size_t kw = p.key_heads * 128, vw = p.value_heads * p.value_dim;
    const float scaling = 1.0f / std::sqrt(128.0f);
    #pragma omp parallel for num_threads(p.threads) schedule(static)
    for (size_t h = 0; h < p.value_heads; ++h) {
        const size_t kh = h / (p.value_heads / p.key_heads);
        const size_t head = h * p.value_dim * 128;
        if (p.state != p.initial_state)
            std::memcpy(p.state + head, p.initial_state + head, p.value_dim * 128 * sizeof(float));
        float decay[8];
        for (size_t t = 0; t < p.rows; ++t)
            decay[t] = std::exp(p.log_decay[t * p.value_heads + h]);
        for (size_t j = 0; j < p.value_dim; j += 4) {
            float * state = p.state + head + j * 128;
            for (size_t t = 0; t < p.rows; ++t) {
                const float * q = output ? p.query + t * kw + kh * 128 : nullptr;
                float * out = output ? p.output + t * vw + h * p.value_dim + j : nullptr;
                update_block<4, output>(state, p.key + t * kw + kh * 128,
                    decay[t], p.beta[t * p.value_heads + h],
                    p.value + t * vw + h * p.value_dim + j, q, out, scaling);
            }
        }
    }
}
#endif
} // namespace rknpu2_cpu_kernels
