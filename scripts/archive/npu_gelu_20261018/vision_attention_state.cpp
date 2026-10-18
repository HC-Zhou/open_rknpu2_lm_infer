// Read-only inspection of the exact native Attention revision used by the probe.
#include "rk_npu_attention_f16_internal.h"
#include <algorithm>

struct AttentionState {
    int lut,cache_lut,can_lut,mask_mode,flags,query_rows,physical_rows,span,blocks;
    int matmul_tasks,tasks,heads,head_dim,kv_tile,core_mask,lut_split;
    int stage_tasks[3];
    float value_max,lut_value_limit;
};
extern "C" int vision_attention_state(const rk_npu_attention_f16_cache* c,
    const rk_npu_attention_f16_workspace* w,AttentionState* out) {
    if(!c || !w || !w->graph || !out)return 1;
    const auto& g=*w->graph;
    *out={int(g.lut),int(c->use_lut()),int(g.can_lut(*c)),int(g.cfg.mask_mode),int(g.cfg.flags),
          g.rows,g.m,g.span,g.blocks,g.matmul_tasks,g.tasks,g.cfg.query_heads,g.shape.key_dim,
          g.cfg.kv_tile,int(rknpu2_matmul_open::attention::core_mask(g.cfg)),int(g.lut_split),{},0,
          32752.f/(g.cfg.kv_tile*1.001f)};
    for(int hi=0;hi<c->h();++hi)
        out->value_max=std::max(out->value_max,rknpu2_matmul_open::attention::widen(c->value_absmax[hi]));
    for(int stage=0;stage<3;++stage)for(int core=0;core<3;++core)
        out->stage_tasks[stage]+=g.stage_count[stage][core];
    return 0;
}
