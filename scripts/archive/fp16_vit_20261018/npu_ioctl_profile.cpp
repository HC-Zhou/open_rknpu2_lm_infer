// Diagnostic-only LD_PRELOAD collector; uses the library's exact driver ABI.
#include "rk_npu_internal.h"
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dlfcn.h>
#include <sys/ioctl.h>

namespace {
std::atomic<uint64_t> submits{}, submit_ns{}, hardware_ns{}, syncs{}, sync_ns{}, errors{};
std::atomic<uint64_t> masks[8]{};
uint64_t now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000ull + t.tv_nsec;
}
__attribute__((destructor)) void report() {
    const char* path = std::getenv("NPU_IOCTL_PROFILE_FILE");
    FILE* f = path ? std::fopen(path, "w") : stderr;
    if (!f) return;
    std::fprintf(f, "{\"submits\":%llu,\"submit_wall_us\":%.3f,\"hardware_us\":%.3f,"
        "\"syncs\":%llu,\"sync_wall_us\":%.3f,\"errors\":%llu,\"core_mask_counts\":[",
        (unsigned long long)submits.load(), submit_ns.load()/1000., hardware_ns.load()/1000.,
        (unsigned long long)syncs.load(), sync_ns.load()/1000., (unsigned long long)errors.load());
    for (int i = 0; i < 8; ++i) std::fprintf(f, "%s%llu", i ? "," : "", (unsigned long long)masks[i].load());
    std::fprintf(f, "]}\n");
    if (path) std::fclose(f);
}
}
extern "C" int ioctl(int fd, unsigned long request, ...) noexcept {
    using Ioctl = int (*)(int, unsigned long, ...);
    static auto real_ioctl = reinterpret_cast<Ioctl>(dlsym(RTLD_NEXT, "ioctl"));
    va_list ap; va_start(ap, request); void* arg = va_arg(ap, void*); va_end(ap);
    if (request != IOCTL_SUBMIT && request != IOCTL_MEM_SYNC) return real_ioctl(fd, request, arg);
    const auto start = now();
    const int rc = real_ioctl(fd, request, arg);
    const int saved_errno = errno;
    const auto elapsed = now() - start;
    if (request == IOCTL_SUBMIT) {
        submits.fetch_add(1, std::memory_order_relaxed);
        submit_ns.fetch_add(elapsed, std::memory_order_relaxed);
        auto* s = static_cast<rknpu_submit*>(arg);
        if (rc >= 0 && s && s->hw_elapse_time > 0)
            hardware_ns.fetch_add(s->hw_elapse_time, std::memory_order_relaxed);
        if (s && s->core_mask < 8) masks[s->core_mask].fetch_add(1, std::memory_order_relaxed);
    } else {
        syncs.fetch_add(1, std::memory_order_relaxed);
        sync_ns.fetch_add(elapsed, std::memory_order_relaxed);
    }
    if (rc < 0) errors.fetch_add(1, std::memory_order_relaxed);
    errno = saved_errno;
    return rc;
}
