// Compare resident device execution with fused FP32 host IO, same raw plan.
#include "rk_npu_mm.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(#x)); } while (0)
using Clock = std::chrono::steady_clock;
namespace bits = rknpu2_matmul_open::bits;
struct Device {
    rk_npu_ctx* ctx = rk_npu_open(nullptr);
    rk_npu_iommu_domain* domain = ctx ? rk_npu_iommu_domain_create(ctx, 12) : nullptr;
    Device() { CHECK(ctx && domain); }
    ~Device() { rk_npu_iommu_domain_free(domain); rk_npu_close(ctx); }
};
struct Memory {
    Device& device;
    rk_npu_mem mem{};
    Memory(Device& d, uint64_t bytes) : device(d) { CHECK(!rk_npu_mem_alloc(d.domain, bytes, RK_NPU_MEM_DATA_DEFAULT, &mem)); }
    ~Memory() { rk_npu_mem_free(device.ctx, &mem); }
};
int main(int argc, char** argv) {
    try {
        CHECK(argc == 8);
        const int m = std::stoi(argv[1]), k = std::stoi(argv[2]), n = std::stoi(argv[3]);
        rk_npu_mm_desc d; rk_npu_mm_desc_init(&d, m, n, k, 1);
        rk_npu_mm_options o; rk_npu_mm_options_init(&o);
        o.a_layout = static_cast<rk_npu_mm_layout>(std::stoi(argv[4]));
        o.c_layout = static_cast<rk_npu_mm_layout>(std::stoi(argv[5]));
        o.n_tile = std::stoi(argv[6]); o.split_k = 0; o.timeout_ms = 2000;
        rk_npu_mm_plan* raw_plan = nullptr;
        CHECK(!rk_npu_mm_plan_create_tiled(&d, &o, std::stoi(argv[7]), &raw_plan));
        std::unique_ptr<rk_npu_mm_plan, decltype(&rk_npu_mm_plan_free)> plan(raw_plan, rk_npu_mm_plan_free);
        rk_npu_mm_info info{}; CHECK(!rk_npu_mm_plan_get_info(plan.get(), &info));
        Device dev;
        Memory a(dev, info.device.input_bytes), b(dev, info.device.weight_bytes), c(dev, info.device.output_bytes);
        std::vector<uint16_t> ah(size_t(m)*k, bits::float_to_half(0.03125f));
        std::vector<uint16_t> bh(size_t(k)*n, bits::float_to_half(0.03125f));
        std::vector<uint16_t> ch(size_t(m)*n);
        CHECK(!rk_npu_mm_pack_a(plan.get(), ah.data(), ah.size()*2, &a.mem));
        CHECK(!rk_npu_mm_pack_b(plan.get(), bh.data(), bh.size()*2, &b.mem));
        for (auto* mem : {&a.mem, &b.mem, &c.mem}) CHECK(!rk_npu_mem_sync(dev.ctx, mem, RK_NPU_SYNC_TO_DEVICE));
        rk_npu_mm_workspace *raw_device = nullptr, *raw_host = nullptr;
        CHECK(!rk_npu_mm_workspace_create(dev.domain, plan.get(), RK_NPU_MM_DEVICE_ONLY, &raw_device));
        std::unique_ptr<rk_npu_mm_workspace, decltype(&rk_npu_mm_workspace_free)> device(raw_device, rk_npu_mm_workspace_free);
        CHECK(!rk_npu_mm_workspace_create(dev.domain, plan.get(), RK_NPU_MM_HOST_PACKED_B, &raw_host));
        std::unique_ptr<rk_npu_mm_workspace, decltype(&rk_npu_mm_workspace_free)> host(raw_host, rk_npu_mm_workspace_free);
        rk_npu_mm_packed_b* raw_b = nullptr;
        CHECK(!rk_npu_mm_packed_b_create(dev.domain, plan.get(), bh.data(), bh.size()*2, &raw_b));
        std::unique_ptr<rk_npu_mm_packed_b, decltype(&rk_npu_mm_packed_b_free)> weight(raw_b, rk_npu_mm_packed_b_free);
        std::vector<float> af(size_t(m)*k, 0.03125f), cf(size_t(m)*n);
        auto run = [&](int kind) {
            const int rc = kind ? rk_npu_mm_run_packed_b_f32(host.get(), af.data(), weight.get(), nullptr, cf.data(), m)
                                : rk_npu_mm_run_device(device.get(), &a.mem, &b.mem, &c.mem);
            CHECK(!rc);
        };
        run(0); run(1);
        std::vector<double> dt, ht;
        for (int block=0; block<3; ++block) for (int kind : {0,1,1,0}) {
            auto start = Clock::now(); run(kind);
            const double us = std::chrono::duration<double,std::micro>(Clock::now()-start).count();
            (kind ? ht : dt).push_back(us);
        }
        CHECK(!rk_npu_mem_sync(dev.ctx, &c.mem, RK_NPU_SYNC_FROM_DEVICE));
        CHECK(!rk_npu_mm_unpack_c(plan.get(), &c.mem, ch.data(), ch.size()*2));
        const float expected = k/1024.f;
        for (size_t j=0; j<128; ++j) {
            const size_t at = (j*7919+37)%ch.size();
            CHECK(bits::half_to_float(ch[at]) == expected && cf[at] == expected);
        }
        std::printf("PROBE {\"m\":%d,\"k\":%d,\"n\":%d,\"a_layout\":%d,\"c_layout\":%d,\"n_tile\":%d,\"task_m_cap\":%s,\"tasks\":%d,\"actual_mask\":%u,\"flops\":%.0f,\"device_us\":[",m,k,n,int(o.a_layout),int(o.c_layout),o.n_tile,argv[7],info.device.num_tasks,info.actual_npu_core_mask,2.*m*k*n);
        for (size_t j=0;j<dt.size();++j) std::printf("%s%.3f",j?",":"",dt[j]);
        std::printf("],\"host_us\":[");
        for (size_t j=0;j<ht.size();++j) std::printf("%s%.3f",j?",":"",ht[j]);
        std::printf("],\"oracle\":\"PASS\"}\n");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
