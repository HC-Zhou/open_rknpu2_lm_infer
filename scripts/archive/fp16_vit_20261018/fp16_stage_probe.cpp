// Complete-call stage probe; runs with either frozen or instrumented library.
#include "rk_npu_mm_tune.h"
#include "rk_npu_half_bits.h"
#include "fp16_stage_trace.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(#x)); } while (0)
namespace bits = rknpu2_matmul_open::bits;
using Work = std::unique_ptr<rk_npu_mm_tuned_workspace, decltype(&rk_npu_mm_tuned_workspace_free)>;
using Weight = std::unique_ptr<rk_npu_mm_packed_b, decltype(&rk_npu_mm_packed_b_free)>;
struct Data {
    int m, k, n;
    std::vector<float> a, bias, c;
    std::vector<uint16_t> b;
    Data(int rows, int inner, int cols) : m(rows), k(inner), n(cols), a(size_t(m)*k), bias(n), c(size_t(m)*n), b(size_t(k)*n) {
        uint32_t rng=0x31415926;
        auto random=[&]() { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; };
        for (auto& x:a) x=(int(random()%127)-63)/256.f;
        for (auto& x:b) x=bits::float_to_half((int(random()%127)-63)/512.f);
        for (auto& x:bias) x=(int(random()%63)-31)/2048.f;
    }
};
struct Case {
    std::string name;
    rk_npu_mm_strategy s;
    Data* data;
    Work work{nullptr,rk_npu_mm_tuned_workspace_free};
    Weight weight{nullptr,rk_npu_mm_packed_b_free};
    std::vector<float> expected;
    Case(const char* label, rk_npu_mm_strategy strategy, Data& values, rk_npu_iommu_domain* domain)
        : name(label), s(strategy), data(&values) {
        rk_npu_mm_tuned_workspace* w=nullptr;
        CHECK(!rk_npu_mm_tuned_workspace_create(domain,&s,&w)); work.reset(w);
        rk_npu_mm_packed_b* b=nullptr;
        CHECK(!rk_npu_mm_tuned_packed_b_create(domain,&s,data->b.data(),data->b.size()*2,&b)); weight.reset(b);
        rk_npu_mm_desc desc; rk_npu_mm_desc_init(&desc,s.m_tile,s.N,s.K,1);
        rk_npu_mm_plan* raw=nullptr;
        CHECK(!rk_npu_mm_plan_create_partitioned(&desc,&s.options,s.task_m_tile,s.k_tile,&raw));
        std::unique_ptr<rk_npu_mm_plan,decltype(&rk_npu_mm_plan_free)> plan(raw,rk_npu_mm_plan_free);
        rk_npu_mm_info info{}; CHECK(!rk_npu_mm_plan_get_info(plan.get(),&info));
        std::printf("PLAN {\"case\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,\"m_tile\":%d,\"k_tile\":%d,"
            "\"a_layout\":%d,\"c_layout\":%d,\"n_tile\":%d,\"task_m_tile\":%d,\"fused_io\":%d,"
            "\"tasks\":%d,\"mask\":%u,\"split_count\":%d,\"input_bytes\":%llu,\"weight_bytes\":%llu,\"output_bytes\":%llu}\n",
            name.c_str(),s.M,s.K,s.N,s.m_tile,s.k_tile,int(s.options.a_layout),int(s.options.c_layout),s.options.n_tile,
            s.task_m_tile,s.fused_io,info.device.num_tasks,info.actual_npu_core_mask,info.split_count,
            (unsigned long long)info.device.input_bytes,(unsigned long long)info.device.weight_bytes,
            (unsigned long long)info.device.output_bytes);
    }
    void run() {
        CHECK(!rk_npu_mm_tuned_run(work.get(),data->a.data(),data->a.size()*4,weight.get(),
            data->bias.data(),data->c.data(),data->c.size()*4));
    }
    void verify() {
        run();
        double max_error=0;
        for (size_t j=0;j<128;++j) {
            const size_t at=(j*7919+37)%data->c.size(), row=at/s.N, col=at%s.N;
            double ref=data->bias[col];
            for(int q=0;q<s.K;++q) ref+=double(data->a[row*s.K+q])*bits::half_to_float(data->b[size_t(q)*s.N+col]);
            const auto err=std::abs(data->c[at]-ref);
            CHECK(std::isfinite(data->c[at]) && err<=0.003+0.005*std::abs(ref));
            max_error=std::max(max_error,err);
        }
        expected=data->c;
        uint64_t hash=1469598103934665603ull;
        const auto* bytes=reinterpret_cast<const unsigned char*>(expected.data());
        for (size_t j=0;j<expected.size()*4;++j) { hash^=bytes[j]; hash*=1099511628211ull; }
        std::printf("ORACLE {\"case\":\"%s\",\"checks\":128,\"max_abs\":%.9g,\"hash\":\"%016llx\"}\n",
            name.c_str(),max_error,(unsigned long long)hash);
    }
};
int main(int argc,char** argv) {
    try {
        CHECK(argc==2 || argc==3);
        const int m=argc==3 ? std::stoi(argv[2]) : 256;
        std::unique_ptr<rk_npu_ctx,decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr),rk_npu_close); CHECK(ctx);
        std::unique_ptr<rk_npu_iommu_domain,decltype(&rk_npu_iommu_domain_free)> domain(
            rk_npu_iommu_domain_create(ctx.get(),12),rk_npu_iommu_domain_free); CHECK(domain);
        Data up(m,1024,4096), down(m,4096,1024);
        auto load=[&](Data& data) {
            rk_npu_mm_autotune_config c; rk_npu_mm_autotune_config_init(&c,data.m,data.n,data.k);
            c.host_type=RK_NPU_MM_HOST_F32; c.with_bias=1;
            rk_npu_mm_strategy s{};
            CHECK(!rk_npu_mm_autotune_cache_load(ctx.get(),&c,argv[1],&s,nullptr)); return s;
        };
        const auto us=load(up), ds=load(down);
        std::vector<std::unique_ptr<Case>> cases;
        auto add=[&](const char* name,rk_npu_mm_strategy s,Data& data) {
            cases.push_back(std::make_unique<Case>(name,s,data,domain.get()));
        };
        add("up_selected",us,up); add("down_selected",ds,down);
        auto s=us; s.fused_io=0; add("up_staged",s,up);
        s=ds; s.task_m_tile=88; add("down_cap88",s,down);
        s=ds; s.options.n_tile=352; add("down_n352",s,down);
        s=ds; s.k_tile=0; s.task_m_tile=0; s.options.n_tile=512; s.fused_io=1;
        add("down_full_k_reference",s,down);
        for (auto& c:cases) c->verify();
        for (int repeat=0;repeat<3;++repeat) for(auto& c:cases) c->run();
        using Reset=void(*)(); using Finish=void(*)(fp16_trace::State*);
        const auto reset=reinterpret_cast<Reset>(dlsym(RTLD_DEFAULT,"fp16_trace_reset"));
        const auto finish=reinterpret_cast<Finish>(dlsym(RTLD_DEFAULT,"fp16_trace_finish"));
        CHECK(bool(reset)==bool(finish));
        for (int block=0;block<8;++block) for(int direction:{0,1}) for(size_t j=0;j<cases.size();++j) {
            auto& c=*cases[direction ? cases.size()-1-j : j];
            fp16_trace::State trace{};
            if(reset)reset();
            const auto start=std::chrono::steady_clock::now();
            c.run();
            const double total=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
            if(finish)finish(&trace);
            CHECK(!trace.errors && !std::memcmp(c.expected.data(),c.data->c.data(),c.expected.size()*4));
            std::printf("SAMPLE {\"case\":\"%s\",\"block\":%d,\"direction\":%d,\"profiled\":%s,\"total_us\":%.3f,\"stage_us\":[",
                c.name.c_str(),block,direction,reset ? "true" : "false",total);
            for(int stage=0;stage<fp16_trace::count;++stage)std::printf("%s%.3f",stage ? "," : "",trace.ns[stage]/1000.);
            std::printf("],\"submits\":[");
            for(uint32_t at=0;at<trace.submit_count;++at) {
                const auto& item=trace.submits[at];
                std::printf("%s{\"stage\":%d,\"mask\":%u,\"tasks\":%u,\"core_tasks\":[%u,%u,%u],\"wall_us\":%.3f,\"hardware_us\":%.3f}",
                    at ? "," : "",item.stage,item.mask,item.tasks,item.core_tasks[0],item.core_tasks[1],item.core_tasks[2],item.wall_ns/1000.,item.hardware_ns/1000.);
            }
            std::printf("],\"sync_bytes\":[");
            for(int stage=0;stage<fp16_trace::count;++stage)std::printf("%s%llu",stage ? "," : "",(unsigned long long)trace.sync_bytes[stage]);
            std::printf("]}\n");
        }
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
