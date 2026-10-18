#include "rk_npu_float_reduce.h"
#include "rk_npu_internal.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace bits=rknpu2_matmul_open::bits;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while(0)
int main(int argc,char** argv) {
    rk_npu_ctx* ctx=nullptr;rk_npu_iommu_domain* domain=nullptr;rk_npu_mem partial{};
    try {
        const int count=argc>1?std::stoi(argv[1]):2;
        const int elements=argc>2?std::stoi(argv[2]):128*4096;
        ctx=rk_npu_open(nullptr);CHECK(ctx);domain=rk_npu_iommu_domain_create(ctx,12);CHECK(domain);
        CHECK(!rk_npu_mem_alloc(domain,uint64_t(count)*elements*2,RK_NPU_MEM_DATA_DEFAULT,&partial));
        std::vector<uint16_t> data(size_t(count)*elements),expected(elements);
        uint32_t random=0x12345678;
        for(auto& x:data) { random^=random<<13;random^=random>>17;random^=random<<5;x=bits::float_to_half((int(random%127)-63)/128.f); }
        auto current=data;int live=count;
        while(live>1) {
            for(int pair=0;pair<(live+1)/2;++pair) for(int e=0;e<elements;++e) {
                const float a=bits::half_to_float(current[size_t(pair*2)*elements+e]);
                const float b=pair*2+1<live?bits::half_to_float(current[size_t(pair*2+1)*elements+e]):0.f;
                current[size_t(pair)*elements+e]=bits::float_to_half(a+b);
            }
            live=(live+1)/2;
        }
        std::copy(current.begin(),current.begin()+elements,expected.begin());
        {
            rknpu2_matmul_open::detail::FloatReduce reduce;
            const int rc=reduce.prepare(domain,partial,count,uint64_t(elements)*2,7,500);
            if(rc)std::fprintf(stderr,"prepare rc=%d dma=%llx bytes=%llu\n",rc,(unsigned long long)partial.dma_addr,(unsigned long long)partial.size);
            CHECK(!rc);
            std::copy(data.begin(),data.end(),static_cast<uint16_t*>(partial.vaddr));
            CHECK(!rk_npu_mem_sync(ctx,&partial,RK_NPU_SYNC_TO_DEVICE));
            CHECK(!reduce.run());
            CHECK(!rk_npu_mem_sync(ctx,&reduce.output(),RK_NPU_SYNC_FROM_DEVICE));
            const auto* out=static_cast<const uint16_t*>(reduce.output().vaddr);
            double delta=0;int unequal=0;
            for(int e=0;e<elements;++e) {
                CHECK(std::isfinite(bits::half_to_float(out[e])));
                delta=std::max(delta,double(std::abs(bits::half_to_float(out[e])-bits::half_to_float(expected[e]))));
                unequal+=out[e]!=expected[e];
            }
            std::printf("FP16 NPU tree reduce partials=%d elements=%d unequal=%d max_abs=%.9g\n",count,elements,unequal,delta);
            if(delta)for(int e=0;e<8;++e)std::printf("sample %d got=%g expected=%g\n",e,bits::half_to_float(out[e]),bits::half_to_float(expected[e]));
            CHECK(delta==0);
        }
        rk_npu_mem_free(ctx,&partial);rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what());return 1; }
}
