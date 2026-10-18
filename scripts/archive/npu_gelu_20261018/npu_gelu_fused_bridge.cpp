// Experimental GEMM + channel bias + GELU. Rust collects the FP32 carrier.
#include "rk_npu_internal.h"
#include "rk_npu_matmul_f16.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using Clock=std::chrono::steady_clock;
thread_local std::string error;
void check(int rc) { if(rc) throw std::runtime_error(rk_npu_strerror(rc)); }
double us(Clock::time_point a,Clock::time_point b) { return std::chrono::duration<double,std::micro>(b-a).count(); }
uint32_t get(const std::vector<uint64_t>& body,uint64_t target,uint32_t reg) {
    for(auto word:body) if(word>>48==target && uint16_t(word)==reg) return uint32_t(word>>16);
    throw std::runtime_error("register missing");
}
void set(std::vector<uint64_t>& body,uint64_t target,uint32_t reg,uint32_t value) {
    for(auto& word:body) if(word>>48==target && uint16_t(word)==reg) { word=E(target,reg,value); return; }
    throw std::runtime_error("register missing");
}
std::vector<uint64_t> lut() {
    std::vector<uint64_t> result;
    for(int bank=0;bank<2;++bank) {
        result.push_back(E(T_DPU,0x4100,uint32_t(2|bank)<<16));
        for(int i=0;i<=512;++i) {
            const double x=double(i+(bank?0:-512))/128.;
            const double y=.5*x*(1.+std::tanh(.7978845608028654*(x+.044715*x*x*x)));
            int code=std::clamp(int(std::nearbyint(y*8192.)),-32768,32767);
            if(!bank && !i) code=0;
            result.push_back(E(T_DPU,0x4104,uint32_t(code)));
        }
    }
    return result;
}
struct Work {
    rk_npu_ctx* ctx;
    rk_npu_mem input{},output{},commands{},tasks{};
    rk_npu_matmul_f16_config cfg{};
    uint32_t starts[3]{},counts[3]{},mask=7;
    int count=0;
    explicit Work(rk_npu_ctx* c):ctx(c) {}
    ~Work() { for(auto* mem:{&tasks,&commands,&output,&input}) rk_npu_mem_free(ctx,mem); }
};
struct Probe {
    rk_npu_ctx* ctx=nullptr;
    rk_npu_iommu_domain* domain=nullptr;
    rk_npu_mem weight{},bias{};
    int k=0,n=0;
    std::unique_ptr<Work> work;
    ~Probe() {
        work.reset();
        if(ctx) { rk_npu_mem_free(ctx,&bias); rk_npu_mem_free(ctx,&weight); }
        rk_npu_iommu_domain_free(domain); rk_npu_close(ctx);
    }
    void prepare_weights(int channels,int inner,const uint16_t* w,const float* b) {
        n=channels;k=inner;
        if(!w || !b || n<32 || n%32 || k<32 || k%32) throw std::runtime_error("weight shape");
        ctx=rk_npu_open(nullptr);if(!ctx)throw std::runtime_error("open NPU");
        domain=rk_npu_iommu_domain_create(ctx,15);if(!domain)throw std::runtime_error("domain15");
        rk_npu_matmul_f16_config cfg{};rk_npu_matmul_f16_config_init(&cfg,8,n,k,RK_NPU_FUSE_NONE);
        rk_npu_matmul_sizes sizes{};check(rk_npu_matmul_f16_query(&cfg,&sizes));
        check(rk_npu_mem_alloc(domain,sizes.weight_bytes,RK_NPU_MEM_DATA_DEFAULT,&weight));
        check(rk_npu_mem_alloc(domain,n*4,RK_NPU_MEM_DATA_DEFAULT,&bias));
        check(rk_npu_matmul_f16_pack_b(&cfg,w,&weight));
        std::memcpy(bias.vaddr,b,n*4);
        check(rk_npu_mem_sync(ctx,&weight,RK_NPU_SYNC_TO_DEVICE));
        check(rk_npu_mem_sync(ctx,&bias,RK_NPU_SYNC_TO_DEVICE));
    }
    void prepare(int rows) {
        if(work && work->cfg.M==rows)return;
        if(rows<8 || rows>1024 || rows%8)throw std::runtime_error("rows outside probe envelope");
        auto p=std::make_unique<Work>(ctx);
        auto& cfg=p->cfg;rk_npu_matmul_f16_config_init(&cfg,rows,n,k,RK_NPU_FUSE_NONE);
        cfg.a_layout=RK_NPU_F16_A_LAYOUT_PANEL8;cfg.d_layout=RK_NPU_F16_D_LAYOUT_PANEL8;
        cfg.core_mask=1;cfg.timeout_ms=500;
        cfg.n_tile=n>=1376?1376:0;
        rk_npu_matmul_sizes sizes{};check(rk_npu_matmul_f16_query(&cfg,&sizes));
        check(rk_npu_mem_alloc(domain,sizes.input_bytes,RK_NPU_MEM_DATA_DEFAULT,&p->input));
        check(rk_npu_mem_alloc(domain,sizes.output_bytes*2+4096,RK_NPU_MEM_DATA_DEFAULT,&p->output));
        std::memset(p->output.vaddr,0xa5,p->output.size);
        check(rk_npu_mem_sync(ctx,&p->output,RK_NPU_SYNC_TO_DEVICE));
        std::vector<rknpu2_matmul_open::detail::RegisterTask> source;
        check(rknpu2_matmul_open::detail::emit_f16_batch_tasks(domain,1,&cfg,&p->input,&weight,&p->output,source));
        const int cores=std::min(3,int(source.size()));p->mask=cores==3?7:cores==2?3:1;
        std::vector<std::vector<uint64_t>> bodies;std::vector<int> bases;
        int at=0,first=0;
        for(int core=0;core<cores;++core) {
            p->starts[core]=first;p->counts[core]=source.size()/cores+(core<int(source.size()%cores));
            for(int task=0;task<int(p->counts[core]);++task,++first) {
                auto body=std::move(source[first].body);
                const int n0=(get(body,T_CNA,R_CNA_DCOMP_ADDR0)-weight.dma_addr)/(k*2);
                const uint32_t shape=get(body,T_CORE,R_CORE_DATAOUT_SIZE_0);
                const uint32_t channels=get(body,T_DPU,R_DATA_CUBE_CHANNEL)&0xffff;
                const uint32_t offset=get(body,T_DPU,R_DST_BASE_ADDR)-p->output.dma_addr;
                set(body,T_DPU,R_DST_BASE_ADDR,uint32_t(p->output.dma_addr+offset*2));
                set(body,T_DPU,R_DATA_FORMAT,0xb4000005);
                set(body,T_DPU,R_BS_CFG,0x20150);set(body,T_DPU,R_BS_OW_CFG,0x36e);
                set(body,T_DPU,R_BN_CFG,0x20040);set(body,T_DPU,0x4064,0x80000000);
                set(body,T_DPU,0x4068,0x6c000000);set(body,T_DPU,R_EW_CFG,0x302);
                set(body,T_DPU,R_OUT_CVT_SCALE,1);set(body,T_DPU,0x4088,0);
                const uint32_t notch=(n/4-1)*8;
                set(body,T_DPU,R_DATA_CUBE_NOTCH,(notch<<16)|notch);
                set(body,T_DPU,R_SURFACE_ADD,get(body,T_DPU,R_SURFACE_ADD)*2);
                const std::pair<uint32_t,uint32_t> fields[]={
                    {0x4108,0x68},{0x410c,0x50500},{0x4110,uint32_t(-16384)},{0x4114,0},
                    {0x4118,0},{0x411c,16384},{0x4120,0},{0x4124,0},{0x4128,16384u<<16},{0x412c,13u<<5}};
                for(auto [reg,value]:fields)set(body,T_DPU,reg,value);
                const std::pair<uint32_t,uint32_t> rdma[]={
                    {0x5004,14},{0x500c,shape&0xffff},{0x5010,shape>>16},{0x5014,channels},
                    {0x5018,0},{0x501c,2},{0x5020,uint32_t(bias.dma_addr+n0*4)},
                    {0x5028,0},{0x502c,0},{0x5034,1},{0x5038,0},{0x5040,16},
                    {0x5044,0x2f8b0},{0x5048,0},{0x504c,0},{0x5064,0},{0x5068,0x01010101},{0x506c,0}};
                for(auto [reg,value]:rdma)body.push_back(E(T_RDMA,reg,value));
                if(!task) { auto table=lut();table.insert(table.end(),body.begin(),body.end());body=std::move(table); }
                bases.push_back(at);at=align_up(at+int(body.size())+4,2);bodies.push_back(std::move(body));
            }
        }
        p->count=bodies.size();
        check(rk_npu_mem_alloc(domain,at*8,RK_NPU_MEM_NON_CACHEABLE,&p->commands));
        check(rk_npu_mem_alloc(domain,p->count*sizeof(rknpu_task),RK_NPU_MEM_KERNEL_MAPPING,&p->tasks));
        std::memset(p->tasks.vaddr,0,p->tasks.size);
        for(int core=0;core<cores;++core) {
            const int begin=p->starts[core],end=begin+p->counts[core];
            rknpu2_matmul_open::detail::write_chain(static_cast<uint64_t*>(p->commands.vaddr),
                static_cast<rknpu_task*>(p->tasks.vaddr)+begin,p->commands.dma_addr,
                {bodies.begin()+begin,bodies.begin()+end},{bases.begin()+begin,bases.begin()+end},{0x1d,0,0x1d});
        }
        work=std::move(p);
    }
    const float* run(int rows,const uint16_t* input,double* times) {
        prepare(rows);auto& p=*work;
        const auto t0=Clock::now();check(rk_npu_matmul_f16_pack_a(&p.cfg,input,&p.input));
        const auto t1=Clock::now();check(rk_npu_mem_sync(ctx,&p.input,RK_NPU_SYNC_TO_DEVICE));
        const auto t2=Clock::now();int64_t hardware=0;
        check(rknpu2_matmul_open::detail::do_submit_multicore(ctx->fd,p.tasks.obj_addr,p.count,p.mask,p.starts,p.counts,15,500,&hardware));
        const auto t3=Clock::now();check(rk_npu_mem_sync(ctx,&p.output,RK_NPU_SYNC_FROM_DEVICE));
        const auto t4=Clock::now();
        if(times) { times[0]=us(t0,t1);times[1]=us(t1,t2);times[2]=us(t2,t3);times[3]=us(t3,t4);times[4]=hardware/1000.; }
        return static_cast<const float*>(p.output.vaddr);
    }
};
}
extern "C" void* npu_gelu_fused_create(int n,int k,const uint16_t* weight,const float* bias) {
    try { auto p=std::make_unique<Probe>();p->prepare_weights(n,k,weight,bias);return p.release(); }
    catch(const std::exception& e) { error=e.what();return nullptr; }
}
extern "C" void npu_gelu_fused_free(void* p) { delete static_cast<Probe*>(p); }
extern "C" const char* npu_gelu_fused_error() { return error.c_str(); }
extern "C" const float* npu_gelu_fused_run(void* p,int rows,const uint16_t* input,double* times) {
    try { return static_cast<Probe*>(p)->run(rows,input,times); }
    catch(const std::exception& e) { error=e.what();return nullptr; }
}
extern "C" int npu_gelu_fused_guard(void* pointer) {
    auto& p=*static_cast<Probe*>(pointer);if(!p.work)return 0;
    const auto& w=*p.work;const auto* guard=static_cast<const uint8_t*>(w.output.vaddr)+size_t(w.cfg.M)*p.n*4;
    return std::all_of(guard,guard+4096,[](uint8_t x){return x==0xa5;});
}
