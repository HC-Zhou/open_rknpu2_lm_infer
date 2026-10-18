// Experimental NPU command/workspace bridge for the Rust GELU benchmark.
// Not linked into normal inference. CPU work here is tensor staging only.
#include "rk_npu_attention_f16_lut.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

using namespace rknpu2_matmul_open;
using Clock = std::chrono::steady_clock;
namespace {
thread_local std::string last_error;
void check(int rc) { if (rc) throw std::runtime_error(rk_npu_strerror(rc)); }
double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double,std::micro>(b-a).count(); }
void patch(std::vector<uint64_t>& body, uint64_t target, uint32_t reg, uint32_t value) {
    for (auto& word:body) if ((word>>48)==target && uint16_t(word)==reg) { word=E(target,reg,value); return; }
    throw std::runtime_error("register not present");
}
std::vector<uint64_t> table() {
    std::vector<uint64_t> words;
    for (int bank=0;bank<2;++bank) {
        words.push_back(E(T_DPU,0x4100,uint32_t(2|bank)<<16));
        for(int i=0;i<=512;++i) {
            const double x=double(i+(bank?0:-512))/128.;
            const double y=.5*x*(1.+std::tanh(.7978845608028654*(x+.044715*x*x*x)));
            const int value=std::clamp(int(std::nearbyint(y*8192.)),-32768,32767);
            words.push_back(E(T_DPU,0x4104,uint32_t(value)));
        }
    }
    return words;
}
std::vector<uint64_t> body(int width,int channels,uint64_t a,uint64_t c,bool load) {
    auto v=attention::lut_exp_regs(8,a,c,false);
    const std::pair<uint32_t,uint32_t> dp[]={
        {0x4010,0x28000002},{0x4024,uint32_t(width*16)},
        {0x4030,uint32_t(width-1)},{0x403c,uint32_t(((channels-1)<<16)|(channels-1))},
        {0x4058,uint32_t(channels-1)},{0x405c,uint32_t(width-1)},
        {0x4068,0x6c000000},{0x4084,1},{0x4088,0},{0x40c0,uint32_t(width*16)},
        {0x4110,uint32_t(-16384)},{0x4114,0},{0x4118,0},{0x411c,16384}};
    for(auto [reg,value]:dp)patch(v,T_DPU,reg,value);
    for(auto [reg,value]:{std::pair<uint32_t,uint32_t>{0x500c,uint32_t(width-1)},
            {0x5014,uint32_t(channels-1)},{0x5040,uint32_t(width*16)}})patch(v,T_RDMA,reg,value);
    if(load) { auto t=table();t.insert(t.end(),v.begin(),v.end());return t; }
    return v;
}
struct Probe {
    rk_npu_ctx* ctx=nullptr;
    rk_npu_iommu_domain* domain=nullptr;
    rk_npu_mem input{},output{},commands{},tasks{};
    size_t n=0;
    uint32_t mask=1,starts[3]{},counts[3]{};
    int task_count=0;
    ~Probe() {
        if(ctx)for(auto* mem:{&tasks,&commands,&output,&input})rk_npu_mem_free(ctx,mem);
        rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);
    }
    void prepare(size_t count,uint32_t allowed) {
        if(!count || count%32 || count>4*1024*1024 ||
           (allowed!=1 && allowed!=2 && allowed!=4 && allowed!=3 && allowed!=7))
            throw std::runtime_error("probe count/mask outside validated envelope");
        n=count;mask=allowed;
        const size_t unit=n%4096==0?4096:32,total_units=n/unit;
        if(__builtin_popcount(mask)>int(total_units))mask=total_units>=2?3:1;
        ctx=rk_npu_open(nullptr);if(!ctx)throw std::runtime_error("open NPU");
        domain=rk_npu_iommu_domain_create(ctx,15);if(!domain)throw std::runtime_error("NPU domain15");
        check(rk_npu_mem_alloc(domain,n*2,RK_NPU_MEM_DATA_DEFAULT,&input));
        check(rk_npu_mem_alloc(domain,n*2+4096,RK_NPU_MEM_DATA_DEFAULT,&output));
        std::memset(output.vaddr,0xa5,output.size);check(rk_npu_mem_sync(ctx,&output,RK_NPU_SYNC_TO_DEVICE));
        std::vector<std::vector<uint64_t>> bodies;
        std::vector<int> bases;
        int qwords=0;size_t at=0;
        const int cores=__builtin_popcount(mask);
        for(int core=0;core<cores;++core) {
            starts[core]=bodies.size();
            size_t remaining=(total_units/cores+size_t(core<int(total_units%cores)))*unit;
            bool first=true;
            while(remaining) {
                const int width=remaining>=4096?128:1;
                const int channels=width==128?int(std::min<size_t>(4096,remaining/4096*32)):int(std::min<size_t>(4096,remaining));
                bases.push_back(qwords);
                bodies.push_back(body(width,channels,input.dma_addr+at*2,output.dma_addr+at*2,first));
                qwords=align_up(qwords+int(bodies.back().size())+4,2);
                at+=size_t(width)*channels;remaining-=size_t(width)*channels;first=false;
            }
            counts[core]=bodies.size()-starts[core];
        }
        task_count=bodies.size();
        check(rk_npu_mem_alloc(domain,qwords*8,RK_NPU_MEM_NON_CACHEABLE,&commands));
        check(rk_npu_mem_alloc(domain,task_count*sizeof(rknpu_task),RK_NPU_MEM_KERNEL_MAPPING,&tasks));
        std::memset(tasks.vaddr,0,tasks.size);
        for(int core=0;core<cores;++core) {
            const int first=starts[core],last=first+counts[core];
            detail::write_chain(static_cast<uint64_t*>(commands.vaddr),static_cast<rknpu_task*>(tasks.vaddr)+first,
                commands.dma_addr,{bodies.begin()+first,bodies.begin()+last},{bases.begin()+first,bases.begin()+last},{0x18,4,0x18});
        }
        for(auto* mem:{&commands,&tasks})if(mem->flags&RK_NPU_MEM_F_CACHEABLE)
            check(rk_npu_mem_sync(ctx,mem,RK_NPU_SYNC_TO_DEVICE));
    }
    void run(const uint16_t* source,uint16_t* destination,double* times) {
        const auto t0=Clock::now();
        auto* packed=static_cast<uint16_t*>(input.vaddr);
#pragma omp parallel for schedule(static) num_threads(4) if(n >= (1u<<18))
        for(size_t i=0;i<n;i+=8) {
#if defined(__aarch64__)
            const auto x=vld1q_u16(source+i);
            vst1q_u16(packed+i,vorrq_u16(vandq_u16(x,vdupq_n_u16(0x8000)),
                vminq_u16(vandq_u16(x,vdupq_n_u16(0x7fff)),vdupq_n_u16(0x4400))));
#else
            for(size_t j=i;j<i+8;++j)packed[j]=(source[j]&0x8000)|std::min<uint16_t>(source[j]&0x7fff,0x4400);
#endif
        }
        const auto t1=Clock::now();check(rk_npu_mem_sync(ctx,&input,RK_NPU_SYNC_TO_DEVICE));
        const auto t2=Clock::now();int64_t hardware=0;
        check(detail::do_submit_multicore(ctx->fd,tasks.obj_addr,task_count,mask,starts,counts,15,500,&hardware));
        const auto t3=Clock::now();check(rk_npu_mem_sync(ctx,&output,RK_NPU_SYNC_FROM_DEVICE));
        const auto t4=Clock::now();
        const auto* q=static_cast<const int16_t*>(output.vaddr);
#pragma omp parallel for schedule(static) num_threads(4) if(n >= (1u<<18))
        for(size_t i=0;i<n;i+=8) {
#if defined(__aarch64__)
            const auto raw=vld1q_s16(q+i);
            const auto lo=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(raw))),1.f/8192);
            const auto hi=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(raw))),1.f/8192);
            auto y=vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(lo),vcvt_f16_f32(hi)));
            const auto original=vld1q_u16(source+i),magnitude=vandq_u16(original,vdupq_n_u16(0x7fff));
            const auto sign=vandq_u16(original,vdupq_n_u16(0x8000));
            const auto tail=vbslq_u16(vceqq_u16(sign,vdupq_n_u16(0)),original,vdupq_n_u16(0x8000));
            y=vbslq_u16(vcgtq_u16(magnitude,vdupq_n_u16(0x4400)),tail,y);
            y=vorrq_u16(y,vandq_u16(sign,vceqq_u16(vandq_u16(y,vdupq_n_u16(0x7fff)),vdupq_n_u16(0))));
            vst1q_u16(destination+i,y);
#else
            for(size_t j=i;j<i+8;++j) {
                const auto x=source[j];auto y=bits::float_to_half(q[j]/8192.f);
                if((x&0x7fff)>0x4400)y=x&0x8000?0x8000:x;
                if(!(y&0x7fff))y|=x&0x8000;destination[j]=y;
            }
#endif
        }
        const auto t5=Clock::now();
        if(times) { times[0]=us(t0,t1);times[1]=us(t1,t2);times[2]=us(t2,t3);times[3]=us(t3,t4);times[4]=us(t4,t5);times[5]=hardware/1000.; }
    }
};
}
extern "C" void* npu_gelu_probe_create(size_t n,uint32_t mask) {
    try { auto p=std::make_unique<Probe>();p->prepare(n,mask);return p.release(); }
    catch(const std::exception& e){last_error=e.what();return nullptr;}
}
extern "C" void npu_gelu_probe_free(void* pointer) { delete static_cast<Probe*>(pointer); }
extern "C" const char* npu_gelu_probe_error() { return last_error.c_str(); }
extern "C" int npu_gelu_probe_run(void* pointer,const uint16_t* a,uint16_t* c,double* times) {
    try { static_cast<Probe*>(pointer)->run(a,c,times);return 0; }
    catch(const std::exception& e){last_error=e.what();return -1;}
}
extern "C" int npu_gelu_probe_guard(void* pointer) {
    const auto* p=static_cast<Probe*>(pointer);const auto* bytes=static_cast<const uint8_t*>(p->output.vaddr)+p->n*2;
    return std::all_of(bytes,bytes+4096,[](uint8_t x){return x==0xa5;});
}
