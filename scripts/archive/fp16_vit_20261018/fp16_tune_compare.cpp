// Read-only comparison of two FP16 complete-call tuning caches.
#include "rk_npu_mm_tune.h"
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

int main(int argc, char** argv) {
    try {
        CHECK(argc == 6);
        const int m = std::stoi(argv[1]), k = std::stoi(argv[2]), n = std::stoi(argv[3]);
        std::unique_ptr<rk_npu_ctx, decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr), rk_npu_close);
        CHECK(ctx);
        std::unique_ptr<rk_npu_iommu_domain, decltype(&rk_npu_iommu_domain_free)> domain(
            rk_npu_iommu_domain_create(ctx.get(), 12), rk_npu_iommu_domain_free);
        CHECK(domain);
        rk_npu_mm_autotune_config config;
        rk_npu_mm_autotune_config_init(&config, m, n, k);
        config.host_type = RK_NPU_MM_HOST_F32;
        config.with_bias = 1;
        rk_npu_mm_strategy strategy[2]{};
        for (int i = 0; i < 2; ++i) {
            CHECK(!rk_npu_mm_autotune_cache_load(ctx.get(), &config, argv[4+i], &strategy[i], nullptr));
            const auto& s = strategy[i];
            std::printf("STRATEGY {\"path\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,"
                "\"revision\":%u,\"m_tile\":%d,\"n_tile\":%d,\"a_layout\":%d,\"c_layout\":%d,"
                "\"task_m_tile\":%d,\"fused_io\":%d,\"k_tile\":%d,\"cpu_mask\":%llu,\"cpu_threads\":%d}\n",
                i ? "fresh" : "previous",m,k,n,s.tuning_revision,s.m_tile,s.options.n_tile,
                int(s.options.a_layout),int(s.options.c_layout),s.task_m_tile,s.fused_io,s.k_tile,
                static_cast<unsigned long long>(s.cpu_core_mask),s.cpu_threads);
        }
        std::vector<float> a(size_t(m)*k), bias(n), c(size_t(m)*n);
        std::vector<uint16_t> b(size_t(k)*n);
        uint32_t rng = 0x31415926;
        auto random = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
        for (auto& x : a) x = (int(random()%127)-63)/256.f;
        for (auto& x : b) x = bits::float_to_half((int(random()%127)-63)/512.f);
        for (auto& x : bias) x = (int(random()%63)-31)/2048.f;
        using Work = std::unique_ptr<rk_npu_mm_tuned_workspace, decltype(&rk_npu_mm_tuned_workspace_free)>;
        using Weight = std::unique_ptr<rk_npu_mm_packed_b, decltype(&rk_npu_mm_packed_b_free)>;
        std::vector<Work> work;
        std::vector<Weight> weight;
        for (const auto& s : strategy) {
            rk_npu_mm_tuned_workspace* w = nullptr;
            CHECK(!rk_npu_mm_tuned_workspace_create(domain.get(), &s, &w));
            work.emplace_back(w, rk_npu_mm_tuned_workspace_free);
            rk_npu_mm_packed_b* packed = nullptr;
            CHECK(!rk_npu_mm_tuned_packed_b_create(domain.get(), &s, b.data(), b.size()*2, &packed));
            weight.emplace_back(packed, rk_npu_mm_packed_b_free);
        }
        auto run = [&](int kind) {
            CHECK(!rk_npu_mm_tuned_run(work[kind].get(), a.data(), a.size()*4,
                weight[kind].get(), bias.data(), c.data(), c.size()*4));
        };
        std::vector<float> previous;
        double max_delta = 0;
        for (int kind = 0; kind < 2; ++kind) {
            run(kind);
            double max_error = 0;
            for (size_t j=0; j<128; ++j) {
                const size_t index = (j*7919+37)%c.size();
                const size_t row = index/n, col = index%n;
                double ref = bias[col];
                for (int q=0; q<k; ++q) ref += double(a[row*k+q])*bits::half_to_float(b[size_t(q)*n+col]);
                const double err = std::abs(c[index]-ref);
                CHECK(std::isfinite(c[index]) && err <= 0.003+0.005*std::abs(ref));
                max_error = std::max(max_error, err);
            }
            if (!kind) previous = c;
            else for (size_t j=0; j<c.size(); ++j) {
                CHECK(std::isfinite(c[j]) && std::abs(c[j]-previous[j]) < 0.006);
                max_delta = std::max(max_delta, double(std::abs(c[j]-previous[j])));
            }
            std::printf("ORACLE path=%s checks=128 max_abs=%.9g PASS\n", kind ? "fresh" : "previous", max_error);
        }
        // Same process, same rounded data, resident weights/workspaces, complete host calls.
        for (int repeat=0; repeat<3; ++repeat) for (int kind : {0,1}) run(kind);
        std::vector<double> samples[2];
        for (int block=0; block<8; ++block) for (int kind : {0,1,1,0}) {
            const auto start = Clock::now();
            run(kind);
            const double us = std::chrono::duration<double,std::micro>(Clock::now()-start).count();
            samples[kind].push_back(us);
            std::printf("ABBA block=%d path=%s us=%.3f\n", block,kind ? "fresh" : "previous",us);
        }
        std::printf("RESULT {\"m\":%d,\"k\":%d,\"n\":%d,\"max_abs_delta\":%.9g,",m,k,n,max_delta);
        for (int kind=0; kind<2; ++kind) {
            std::printf("\"%s_us\":[",kind ? "fresh" : "previous");
            for (size_t j=0; j<samples[kind].size(); ++j) std::printf("%s%.3f",j ? "," : "",samples[kind][j]);
            std::printf("]%s",kind ? "}\n" : ",");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n",e.what());
        return 1;
    }
}
