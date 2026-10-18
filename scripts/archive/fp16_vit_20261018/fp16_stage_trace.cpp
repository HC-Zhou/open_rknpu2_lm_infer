#include "fp16_stage_trace.h"
#include "rk_npu_internal.h"
#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <dlfcn.h>
#include <sys/ioctl.h>

thread_local fp16_trace::State fp16_trace::state{};
extern "C" void fp16_trace_reset() { fp16_trace::state = {}; fp16_trace::state.enabled = true; }
extern "C" void fp16_trace_finish(fp16_trace::State* out) {
    fp16_trace::state.enabled = false;
    *out = fp16_trace::state;
}
extern "C" int ioctl(int fd, unsigned long request, ...) noexcept {
    using Ioctl = int (*)(int,unsigned long,...);
    static auto real_ioctl = reinterpret_cast<Ioctl>(dlsym(RTLD_NEXT,"ioctl"));
    va_list ap; va_start(ap,request); void* arg=va_arg(ap,void*); va_end(ap);
    auto& trace = fp16_trace::state;
    if (!trace.enabled || (request != IOCTL_SUBMIT && request != IOCTL_MEM_SYNC)) return real_ioctl(fd,request,arg);
    const auto start = fp16_trace::now();
    const int rc = real_ioctl(fd,request,arg), saved_errno = errno;
    const auto elapsed = fp16_trace::now()-start;
    if (rc < 0) ++trace.errors;
    if (request == IOCTL_SUBMIT) {
        auto* s = static_cast<rknpu_submit*>(arg);
        if (trace.submit_count < 64) {
            auto& item = trace.submits[trace.submit_count++];
            item.stage=trace.current; item.mask=s->core_mask; item.tasks=s->task_number;
            item.wall_ns=elapsed; item.hardware_ns=rc >= 0 && s->hw_elapse_time > 0 ? s->hw_elapse_time : 0;
            int logical=0;
            const int cores=__builtin_popcount(s->core_mask);
            for (int physical=0; physical<3; ++physical) if (s->core_mask & (1u<<physical)) {
                const int slot=cores==3 ? physical+2 : cores==2 ? logical : physical;
                item.core_tasks[physical]=s->subcore_task[slot].task_number;
                ++logical;
            }
        }
    } else {
        trace.sync_ns[trace.current]+=elapsed;
        ++trace.sync_calls[trace.current];
        trace.sync_bytes[trace.current]+=static_cast<rknpu_mem_sync*>(arg)->size;
    }
    errno=saved_errno;
    return rc;
}
