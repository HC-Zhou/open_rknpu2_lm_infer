// Calls the actual pinned ggml CPU op; no reimplementation of its recurrence.
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>
#ifdef GDN_HAVE_LOCAL_CPU
#include "gdn.h"
#endif

int main(int argc, char ** argv) {
    if (argc != 7 && argc != 8) {
        std::fprintf(stderr, "usage: gdn_reference INPUT OUTPUT ROWS THREADS LOOPS SAMPLES [ggml|fused|unfused]\n");
        return 2;
    }
    constexpr int d = 128, h = 32;
    const int rows = std::atoi(argv[3]), threads = std::atoi(argv[4]);
    const int loops = std::atoi(argv[5]), samples = std::atoi(argv[6]);
    if (rows < 1 || threads < 1 || loops < 1 || samples < 1) return 2;
    // Explicitly expanded Q/K preserve HF's repeat_interleave head mapping,
    // independently of ggml's modulo-broadcast convention.
    ggml_init_params init{128u * 1024 * 1024, nullptr, false};
    ggml_context * ctx = ggml_init(init);
    if (!ctx) throw std::runtime_error("ggml_init failed");
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, h, rows, 1);
    ggml_tensor * k = ggml_dup_tensor(ctx, q);
    ggml_tensor * v = ggml_dup_tensor(ctx, q);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, h, rows, 1);
    ggml_tensor * beta = ggml_dup_tensor(ctx, g);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, d, h, 1);
    std::ifstream input(argv[1], std::ios::binary);
    for (ggml_tensor * tensor : {q, k, v, g, beta, state}) {
        input.read(static_cast<char *>(tensor->data), ggml_nbytes(tensor));
        if (!input) throw std::runtime_error("short input");
    }
    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_threadpool_params tp = ggml_threadpool_params_default(threads);
    tp.poll = 0; // Same passive worker policy as the Rust/Rayon benchmark.
    ggml_threadpool_t pool = ggml_threadpool_new(&tp);
    ggml_cplan plan = ggml_graph_plan(graph, threads, pool);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    const auto run = [&] {
#ifdef GDN_HAVE_LOCAL_CPU
        if (argc == 8 && std::string(argv[7]) != "ggml") {
            rknpu2_gdn_params p{static_cast<size_t>(rows), h, h, d, d, static_cast<size_t>(threads),
                static_cast<const float *>(state->data), static_cast<float *>(out->data) + d*h*rows,
                static_cast<const float *>(q->data), static_cast<const float *>(k->data),
                static_cast<const float *>(v->data), static_cast<const float *>(g->data),
                static_cast<const float *>(beta->data), static_cast<float *>(out->data)};
            if (std::string(argv[7]) == "unfused") rknpu2_gdn_run_unfused(&p);
            else rknpu2_gdn_run(&p);
            return;
        }
#endif
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("ggml_graph_compute failed");
    };
    run();
    run();
    std::printf("{\"milliseconds\":[");
    for (int sample = 0; sample < samples; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < loops; ++i) run();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() / loops;
        std::printf("%s%.9f", sample ? "," : "", ms);
    }
    std::printf("]}\n");
    std::ofstream output(argv[2], std::ios::binary);
    output.write(static_cast<const char *>(out->data), ggml_nbytes(out));
    if (!output) throw std::runtime_error("output write failed");
    ggml_threadpool_free(pool);
    ggml_free(ctx);
}
