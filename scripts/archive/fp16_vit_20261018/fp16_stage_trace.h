#pragma once
// Diagnostic snapshot builds only; never included by the production library.
#include <cstdint>
#include <ctime>

namespace fp16_trace {
enum Stage { idle, narrow, pack, sync_in, gemm, reduce, sync_out, unpack, collect,
             fused_input, fused_output, narrow_validate, collect_validate, count };
struct Submit {
    int stage;
    uint32_t mask, tasks, core_tasks[3];
    uint64_t wall_ns, hardware_ns;
};
struct State {
    bool enabled;
    int current;
    uint64_t ns[count], sync_ns[count], sync_bytes[count], sync_calls[count];
    uint32_t submit_count, errors;
    Submit submits[64];
};
extern thread_local State state;
inline uint64_t now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec)*1000000000ull+t.tv_nsec;
}
struct Scope {
    int stage, previous;
    uint64_t start;
    explicit Scope(int value) : stage(value), previous(state.current), start(state.enabled ? now() : 0) {
        if (start) state.current = value;
    }
    ~Scope() {
        if (start) { state.ns[stage] += now()-start; state.current = previous; }
    }
};
template<class F> auto measured(int stage, F&& fn) { Scope scope(stage); return fn(); }
}
#define FP_STAGE(stage, ...) fp16_trace::measured(fp16_trace::stage,[&]() { return (__VA_ARGS__); })
extern "C" void fp16_trace_reset();
extern "C" void fp16_trace_finish(fp16_trace::State*);
