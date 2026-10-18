// Bounded GEMM -> DPU LUT connectivity probe. No public ABI or inference changes.
#include "rk_npu_float_backend.h"
#include "rk_npu_internal.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace rknpu2_matmul_open;
namespace {
void check(int rc) { if (rc) throw std::runtime_error(rk_npu_strerror(rc)); }
struct Device {
    rk_npu_ctx* ctx = rk_npu_open(nullptr);
    rk_npu_iommu_domain* domain = ctx ? rk_npu_iommu_domain_create(ctx, 15) : nullptr;
    std::vector<rk_npu_mem> memory;
    ~Device() {
        for (auto& m : memory) rk_npu_mem_free(ctx, &m);
        rk_npu_iommu_domain_free(domain); rk_npu_close(ctx);
    }
    rk_npu_mem alloc(size_t bytes, uint32_t flags) {
        if (!domain) throw std::runtime_error("open NPU domain15");
        rk_npu_mem m{}; check(rk_npu_mem_alloc(domain, bytes, flags, &m));
        memory.push_back(m); return m;
    }
};
void patch(std::vector<uint64_t>& words, uint32_t reg, uint32_t value) {
    for (auto& word : words) if (word >> 48 == T_DPU && uint16_t(word) == reg) {
        word = E(T_DPU, reg, value); return;
    }
    throw std::runtime_error("DPU register missing");
}
double gelu(double x) {
    return .5*x*(1.+std::tanh(.7978845608028654*(x+.044715*x*x*x)));
}
std::vector<uint64_t> table() {
    std::vector<uint64_t> words;
    for (int bank=0; bank<2; ++bank) {
        words.push_back(E(T_DPU, 0x4100, uint32_t(2|bank)<<16));
        for (int i=0; i<=512; ++i) {
            const double x=double(i+(bank?0:-512))/128.;
            const int code=std::clamp(int(std::nearbyint(gelu(x)*8192.)), -32768, 32767);
            words.push_back(E(T_DPU, 0x4104, uint32_t(code)));
        }
    }
    return words;
}
std::vector<uint64_t> convert(int m, int n, uint64_t input, uint64_t output, bool serial) {
    std::vector<uint64_t> v;
    const std::pair<uint32_t,uint32_t> dp[]={
        {0x4004,14},{0x400c,0x1e5},{0x4010,serial?0x5400000du:0x54000005u},{0x4014,0},
        {0x4020,uint32_t(output)},{0x4024,uint32_t(m*16)},{0x4030,uint32_t(m-1)},
        {0x4034,0},{0x4038,0},{0x403c,uint32_t(((n-1)<<16)|(n-1))},
        {0x4040,0x42},{0x4044,0},{0x4048,0x08000000},{0x404c,0},{0x4050,2},{0x4054,0},
        {0x4058,uint32_t(n-1)},{0x405c,uint32_t(m-1)},
        {0x4060,0x53},{0x4064,0},{0x4068,0},{0x406c,0},
        {0x4070,0x383},{0x4074,0},{0x4078,1},{0x407c,0},{0x4080,0},
        {0x4084,0x10001},{0x4088,0},{0x40c0,uint32_t(m*32)},{0x40c4,0}};
    for(auto [reg,value]:dp) v.push_back(E(T_DPU,reg,value));
    for(uint32_t reg=0x4090; reg<=0x40ac; reg+=4) v.push_back(E(T_DPU,reg,0));
    for(uint32_t reg=0x4100; reg<=0x412c; reg+=4) v.push_back(E(T_DPU,reg,0));
    const std::pair<uint32_t,uint32_t> rd[]={
        {0x5004,14},{0x500c,uint32_t(m-1)},{0x5010,0},{0x5014,uint32_t(n-1)},
        {0x5018,uint32_t(input)},{0x501c,0},{0x5020,0},{0x5028,0},{0x502c,0},
        {0x5034,1},{0x5038,0},{0x5040,16},{0x5044,0x2f8a1},{0x5048,0},{0x504c,0},
        {0x5064,0},{0x5068,0x01010101},{0x506c,0}};
    for(auto [reg,value]:rd) v.push_back(E(T_RDMA,reg,value));
    return v;
}
void probe(int m, int n, int k, int bias_mode, int fp32_proc = 0, bool tails = false) {
    Device d;
    rk_npu_matmul_f16_config cfg{};
    rk_npu_matmul_f16_config_init(&cfg, m, n, k, RK_NPU_FUSE_NONE);
    cfg.a_layout=RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8;
    cfg.d_layout=RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8;
    rk_npu_matmul_sizes sizes{}; check(rk_npu_matmul_f16_query(&cfg, &sizes));
    auto a=d.alloc(sizes.input_bytes, RK_NPU_MEM_DATA_DEFAULT);
    auto b=d.alloc(sizes.weight_bytes, RK_NPU_MEM_DATA_DEFAULT);
    auto c=d.alloc(sizes.output_bytes*2+4096, RK_NPU_MEM_DATA_DEFAULT);
    auto final=d.alloc(sizes.output_bytes*2+4096, RK_NPU_MEM_DATA_DEFAULT);
    auto bias=d.alloc(n*sizeof(float), RK_NPU_MEM_DATA_DEFAULT);
    auto commands=d.alloc(16384, RK_NPU_MEM_NON_CACHEABLE);
    auto tasks=d.alloc(sizeof(rknpu_task)*2, RK_NPU_MEM_KERNEL_MAPPING);
    std::vector<uint16_t> input(m*k), weight(k*n);
    for (int row=0; row<m; ++row) for (int col=0; col<k; ++col)
        input[row*k+col]=bits::float_to_half(float((row*k+col)%97-48)/16.f);
    if (tails) {
        const float anchors[]={-65504,-100,-32,-16.125,-8,-4,-2,-1,-.1,-.001,0,.001,.1,1,2,4,8,16.125,32,100,65504};
        for (size_t i=0; i<std::size(anchors); ++i) input[i]=bits::float_to_half(anchors[i]);
    }
    for (int col=0; col<n; ++col) weight[(col%k)*n+col]=bits::float_to_half(1.f);
    auto* biases=static_cast<float*>(bias.vaddr);
    for (int col=0; col<n; ++col) biases[col]=float(col%5-2)/8.f;
    check(rk_npu_matmul_f16_pack_a(&cfg, input.data(), &a));
    check(rk_npu_matmul_f16_pack_b(&cfg, weight.data(), &b));
    check(rk_npu_mem_sync(d.ctx, &a, RK_NPU_SYNC_TO_DEVICE));
    check(rk_npu_mem_sync(d.ctx, &b, RK_NPU_SYNC_TO_DEVICE));
    check(rk_npu_mem_sync(d.ctx, &bias, RK_NPU_SYNC_TO_DEVICE));
    for (int mode : {0, 1, 0}) {
        std::memset(c.vaddr, 0xa5, c.size);
        std::memset(final.vaddr, 0xa5, final.size);
        check(rk_npu_mem_sync(d.ctx, &c, RK_NPU_SYNC_TO_DEVICE));
        check(rk_npu_mem_sync(d.ctx, &final, RK_NPU_SYNC_TO_DEVICE));
        std::vector<uint64_t> body;
        detail::float_native_regs(body, m, n, k, a.dma_addr, b.dma_addr,
                                  c.dma_addr, detail::FloatPrecision::F16);
        if (mode) {
            if (bias_mode) {
                patch(body, 0x4040, bias_mode==1?0x20050:0x20150);
                patch(body, 0x4044, bias_mode==1?0x3e000000:0);
            }
            patch(body, 0x4010, 0x28000002);
            patch(body, 0x4060, 0x20040); patch(body, 0x4064, 0x80000000);
            patch(body, 0x4068, 0x6c000000); patch(body, 0x4070, 0x302);
            patch(body, 0x4084, 1); patch(body, 0x4088, 0);
            patch(body, 0x4108, 0x68); patch(body, 0x410c, 0x50500);
            patch(body, 0x4110, uint32_t(-16384)); patch(body, 0x4114, 0);
            patch(body, 0x4118, 0); patch(body, 0x411c, 16384);
            if (fp32_proc) {
                patch(body, 0x4010, (2u<<29)|(5u<<26)|5u);
                patch(body, 0x4084, 0x10001); patch(body, 0x4088, 13u<<12);
                if (fp32_proc>=2) {
                    patch(body, 0x4010, (5u<<29)|(5u<<26)|5u);
                    patch(body, 0x4084, 1); patch(body, 0x4088, 0);
                    patch(body, 0x4050, 0x36e); patch(body, 0x40c0, uint32_t(m*64));
                    patch(body, 0x4128, 16384u<<16); patch(body, 0x412c, 13u<<5);
                }
            }
            auto lut=table(); lut.insert(lut.end(), body.begin(), body.end()); body=std::move(lut);
            if (bias_mode==2) {
                const std::pair<uint32_t,uint32_t> rdma[]={
                    {0x5004,14},{0x500c,uint32_t(m-1)},{0x5010,0},{0x5014,uint32_t(n-1)},
                    {0x5018,0},{0x501c,2},{0x5020,uint32_t(bias.dma_addr)},
                    {0x5028,0},{0x502c,0},{0x5034,1},{0x5038,0},{0x5040,16},
                    {0x5044,fp32_proc?0x2f8b0u:0x17850u},{0x5048,0},{0x504c,0},{0x5064,0},{0x5068,0x01010101},{0x506c,0}};
                for (auto [reg,value]:rdma) body.push_back(E(T_RDMA,reg,value));
            }
        }
        std::memset(tasks.vaddr, 0, tasks.size);
        const uint32_t enable=mode && bias_mode==2?0x1d:0xd;
        const bool chained=mode && fp32_proc>=3;
        std::vector<std::vector<uint64_t>> bodies{body}; std::vector<int> bases{0};
        if (chained) { bases.push_back(align_up(int(body.size())+4,2)); bodies.push_back(convert(m,n,c.dma_addr,final.dma_addr,fp32_proc==4)); }
        auto* tk=static_cast<rknpu_task*>(tasks.vaddr);
        auto* cmd=static_cast<uint64_t*>(commands.vaddr);
        detail::write_chain(cmd, tk, commands.dma_addr, bodies, bases, {enable, 0, enable});
        if (chained) {
            cmd[bases[1]+bodies[1].size()+3]=E(T_PC,8,0x18);
            tk[1].enable_mask=0x18; tk[1].op_idx=4;
        }
        uint32_t starts[3]{}, counts[3]{uint32_t(bodies.size())};
        check(detail::do_submit_multicore(d.ctx->fd, tasks.obj_addr, bodies.size(), 1,
                                         starts, counts, 15, 500));
        check(rk_npu_mem_sync(d.ctx, &c, RK_NPU_SYNC_FROM_DEVICE));
        check(rk_npu_mem_sync(d.ctx, &final, RK_NPU_SYNC_FROM_DEVICE));
        const auto* raw=static_cast<const uint16_t*>(chained?final.vaddr:c.vaddr);
        double sum_error=0, sum_reference=0, max_abs=0;
        for (int row=0; row<m; ++row) for (int col=0; col<n; ++col) {
            double x=bits::half_to_float(input[row*k+col%k]);
            if (mode) x+=bias_mode==1?.125:bias_mode==2?biases[col]:0.;
            const double ref=mode?gelu(x):x;
            const uint16_t code=raw[(col/8*m+row)*8+col%8];
            const double got=mode && fp32_proc==2
                ? static_cast<const float*>(c.vaddr)[(col/4*m+row)*4+col%4]/8192.
                : mode && !fp32_proc?int16_t(code)/8192.:bits::half_to_float(code);
            const double error=std::abs(got-ref);
            if (!std::isfinite(got) || error>(mode?.002+.0005*std::abs(ref):0.)) {
                std::cerr << "row=" << row << " col=" << col << " got=" << got
                          << " expected=" << ref << " raw=" << code << '\n';
                if (chained) {
                    for (int j=0; j<n; ++j) std::cerr << "lane " << j << " before "
                        << static_cast<const float*>(c.vaddr)[(j/4*m)*4+j%4]/8192.
                        << " after " << bits::half_to_float(raw[(j/8*m)*8+j%8]) << '\n';
                }
                throw std::runtime_error("GEMM/LUT output mismatch");
            }
            sum_error+=error*error; sum_reference+=ref*ref; max_abs=std::max(max_abs,error);
        }
        const auto* guard=static_cast<const uint8_t*>(c.vaddr)+sizes.output_bytes*(mode && fp32_proc>=2?2:1);
        if (!std::all_of(guard, guard+4096, [](uint8_t x){return x==0xa5;}))
            throw std::runtime_error("output guard changed");
        const auto* final_guard=static_cast<const uint8_t*>(final.vaddr)+sizes.output_bytes;
        if (chained && !std::all_of(final_guard, final_guard+4096, [](uint8_t x){return x==0xa5;}))
            throw std::runtime_error("converted output guard changed");
        std::cout << "{\"M\":" << m << ",\"N\":" << n << ",\"K\":" << k
                  << ",\"fused\":" << mode << ",\"bias_mode\":" << bias_mode << ",\"max_abs\":" << max_abs
                  << ",\"output_mode\":" << fp32_proc << ",\"tails\":" << tails
                  << ",\"relative_l2\":" << std::sqrt(sum_error/sum_reference)
                  << ",\"guard_ok\":true}\n";
    }
}
}
int main(int argc, char** argv) {
    if (!std::getenv("LLM_INFER_LOCK_FD")) { std::cerr << "run under scripts/run_board.py\n"; return 2; }
    try {
        const int bias=argc>1?std::stoi(argv[1]):0;
        const int fp32_proc=argc>2?std::stoi(argv[2]):0;
        const bool tails=argc>3 && std::stoi(argv[3])==1;
        if (bias<0 || bias>2) throw std::runtime_error("bias mode must be 0, 1 (scalar), or 2 (channel)");
        probe(16,32,32,bias,fp32_proc,tails); probe(32,64,32,bias,fp32_proc,tails); probe(128,128,32,bias,fp32_proc,tails);
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
