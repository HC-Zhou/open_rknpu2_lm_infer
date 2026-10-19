#include "rk_npu_attention_f16.h"
#include "rk_npu_half_bits.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int qh = 16, kh = 4, dim = 256, prompt = 5, batch = 8;
void check(int rc) {
    if (rc) throw std::runtime_error("NPU API status " + std::to_string(rc));
}
std::vector<float> read(const std::string& path, size_t size) {
    std::vector<float> out(size);
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(out.data()), size * sizeof(float)))
        throw std::runtime_error("cannot read " + path);
    return out;
}
uint16_t half(float x) {
    return rknpu2_matmul_open::bits::float_to_half(x);
}
struct Fixture {
    std::vector<float> q, k, v;
    int start;
    explicit Fixture(const std::string& root, int position) : start(position) {
        q = read(root + "/verify/p5-l3-query.f32", batch * qh * dim);
        auto pk = read(root + "/verify/p0-l3-key.f32", prompt * kh * dim);
        auto pv = read(root + "/verify/p0-l3-value.f32", prompt * kh * dim);
        auto nk = read(root + "/verify/p5-l3-key.f32", batch * kh * dim);
        auto nv = read(root + "/verify/p5-l3-value.f32", batch * kh * dim);
        k = std::move(pk); k.insert(k.end(), nk.begin(), nk.end());
        v = std::move(pv); v.insert(v.end(), nv.begin(), nv.end());
        if (start != prompt) {
            auto old_k = k, old_v = v;
            k.resize(size_t(start + batch) * kh * dim); v.resize(k.size());
            for (int r = 0; r < start + batch; ++r) for (int h = 0; h < kh; ++h)
                for (int d = 0; d < dim; ++d) {
                    const size_t i = (size_t(r) * kh + h) * dim + d;
                    const size_t j = (size_t(r % (prompt + batch)) * kh + h) * dim + d;
                    k[i] = old_k[j]; v[i] = old_v[j];
                }
        }
    }
    std::vector<uint16_t> kv(const std::vector<float>& x, int length, int future = 1) const {
        std::vector<uint16_t> out(size_t(kh) * length * dim);
        for (int h = 0; h < kh; ++h) for (int r = 0; r < length; ++r)
            for (int d = 0; d < dim; ++d)
                out[(size_t(h) * length + r) * dim + d] =
                    half(x[(size_t(r) * kh + h) * dim + d] * (r > start ? future : 1));
        return out;
    }
    std::vector<uint16_t> query(int rows, int offset = 0) const {
        std::vector<uint16_t> out(size_t(qh) * rows * dim);
        for (int h = 0; h < qh; ++h) for (int r = 0; r < rows; ++r)
            for (int d = 0; d < dim; ++d)
                out[(size_t(h) * rows + r) * dim + d] = half(q[(size_t(r + offset) * qh + h) * dim + d]);
        return out;
    }
};
std::vector<float> first_row(const std::vector<float>& x, int rows) {
    std::vector<float> out(qh * dim);
    for (int h = 0; h < qh; ++h)
        std::copy_n(x.data() + size_t(h) * rows * dim, dim, out.data() + h * dim);
    return out;
}
void compare(const std::string& name, const std::vector<float>& a, const std::vector<float>& b) {
    double error = 0, energy = 0, maximum = 0;
    int different = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - b[i];
        error += d * d; energy += double(a[i]) * a[i];
        maximum = std::max(maximum, std::abs(d)); different += a[i] != b[i];
    }
    std::printf("COMPARE %s different=%d nrmse=%.12g max_abs=%.12g\n",
        name.c_str(), different, std::sqrt(error / std::max(energy, 1e-30)), maximum);
}
struct Npu {
    rk_npu_ctx* ctx = rk_npu_open(nullptr);
    rk_npu_iommu_domain* domain = ctx ? rk_npu_iommu_domain_create(ctx, 14) : nullptr;
    Npu() { if (!ctx || !domain) throw std::runtime_error("cannot create NPU context/domain"); }
    ~Npu() { rk_npu_iommu_domain_free(domain); rk_npu_close(ctx); }
    std::vector<float> run(const Fixture& f, bool lut, int rows, int length, int future = 1,
        bool truncate = false, int offset = 0, std::vector<float>* full_output = nullptr,
        rk_npu_attention_f16_mask_mode mode = RK_NPU_ATTENTION_F16_CAUSAL, int mask_pattern = 0) {
        setenv("RK_NPU_ATTN_LUT_EXP", lut ? "1" : "0", 1);
        rk_npu_attention_f16_config c;
        rk_npu_attention_f16_config_init(&c);
        c.query_heads = qh; c.kv_heads = kh; c.head_dim = dim;
        c.mask_mode = mode;
        c.flags = RK_NPU_ATTENTION_F16_STABLE_SOFTMAX;
        c.initial_capacity = c.max_capacity = ((f.start + batch + 31) / 32) * 32;
        c.core_mask = 7; c.kv_tile = 512;
        rk_npu_attention_f16_cache* cache = nullptr;
        rk_npu_attention_f16_workspace* workspace = nullptr;
        check(rk_npu_attention_f16_cache_create(domain, &c, &cache));
        check(rk_npu_attention_f16_prepare(domain, &c, &workspace));
        auto k = f.kv(f.k, length), v = f.kv(f.v, length, future), q = f.query(rows, offset);
        check(rk_npu_attention_f16_cache_load(ctx, cache, k.data(), v.data(), length));
        if (truncate) check(rk_npu_attention_f16_cache_truncate(cache, f.start + 1));
        std::vector<float> out(qh * rows * dim), repeat(out.size());
        std::vector<uint8_t> visible(size_t(rows) * length);
        for (int r = 0; r < rows; ++r) for (int j = 0; j < length; ++j)
            visible[size_t(r) * length + j] = mask_pattern == 1 ? 0 : mask_pattern == 2 ? 1 : (j + r) % 3 != 1;
        rk_npu_attention_f16_boolean_mask mask{};
        rk_npu_attention_f16_boolean_mask_init(&mask, visible.data(), visible.size(), 1, rows, length);
        const auto* active = mode == RK_NPU_ATTENTION_F16_BOOLEAN_MASK ? &mask : nullptr;
        rk_npu_attention_f16_timings t{};
        check(rk_npu_attention_f16_run(ctx, workspace, cache, q.data(), rows, f.start + offset, out.data(), &t, active));
        check(rk_npu_attention_f16_run(ctx, workspace, cache, q.data(), rows, f.start + offset, repeat.data(), nullptr, active));
        if (out != repeat) throw std::runtime_error("repeat differs");
        if (rk_npu_attention_f16_run(ctx, workspace, cache, q.data(), rows, -1, repeat.data(), nullptr, active) != RK_NPU_ERR_PARAM)
            throw std::runtime_error("negative query start accepted");
        if (mode == RK_NPU_ATTENTION_F16_CAUSAL &&
            rk_npu_attention_f16_run(ctx, workspace, cache, q.data(), rows, length, repeat.data(), nullptr, active) != RK_NPU_ERR_PARAM)
            throw std::runtime_error("causal query beyond cache accepted");
        if (out != repeat) throw std::runtime_error("invalid query changed output");
        std::printf("CASE lut=%d mode=%d pattern=%d rows=%d start=%d cache=%d future=%d truncate=%d span=%d tasks=%d total_us=%.3f repeat_exact=1 invalid_start_rejected=1\n",
            int(lut), int(mode), mask_pattern, rows, f.start + offset, length, future, int(truncate), t.compute_length, t.tasks, t.total_us);
        rk_npu_attention_f16_workspace_free(workspace);
        rk_npu_attention_f16_cache_free(cache);
        if (full_output) *full_output = out;
        return first_row(out, rows);
    }
};
void save(const std::string& path, const std::vector<float>& x) {
    std::ofstream file(path, std::ios::binary);
    if (!file.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float)))
        throw std::runtime_error("cannot save " + path);
}
} // namespace
int main(int argc, char** argv) try {
    if (argc != 3 && argc != 4) throw std::runtime_error("usage: audit_batch_attention TRACE_ROOT OUTPUT_DIR [START]");
    const int start = argc == 4 ? std::stoi(argv[3]) : prompt;
    if (start < 0 || start + batch > 32768) throw std::runtime_error("invalid start");
    const Fixture f(argv[1], start);
    std::printf("FIXTURE start=%d real_trace=%d\n", start, int(start == prompt));
    Npu npu;
    for (bool lut : {true, false}) {
        const auto one6 = npu.run(f, lut, 1, start + 1);
        const auto one13 = npu.run(f, lut, 1, start + batch);
        std::vector<float> batch_output;
        const auto eight13 = npu.run(f, lut, batch, start + batch, 1, false, 0, &batch_output);
        const auto zero13 = npu.run(f, lut, 1, start + batch, 0);
        const auto negative13 = npu.run(f, lut, 1, start + batch, -1);
        const auto truncated = npu.run(f, lut, 1, start + batch, 1, true);
        const std::string prefix = "lut" + std::to_string(lut);
        compare(prefix + "_same_M_cache6_vs13", one6, one13);
        compare(prefix + "_same_cache_M1_vs8", one13, eight13);
        compare(prefix + "_same_M_future_zero", one13, zero13);
        compare(prefix + "_same_M_future_negative", one13, negative13);
        compare(prefix + "_cache6_vs_truncate6", one6, truncated);
        save(std::string(argv[2]) + "/" + prefix + "-one6.f32", one6);
        save(std::string(argv[2]) + "/" + prefix + "-one13.f32", one13);
        save(std::string(argv[2]) + "/" + prefix + "-eight13.f32", eight13);
        std::vector<float> serial(batch_output.size());
        for (int r = 0; r < batch; ++r) {
            const auto row = npu.run(f, lut, 1, start + r + 1, 1, false, r);
            for (int h = 0; h < qh; ++h)
                std::copy_n(row.data() + h * dim, dim, serial.data() + (size_t(h) * batch + r) * dim);
        }
        compare(prefix + "_serial8_vs_batch8", serial, batch_output);
        save(std::string(argv[2]) + "/" + prefix + "-serial-all.f32", serial);
        save(std::string(argv[2]) + "/" + prefix + "-batch-all.f32", batch_output);
        for (auto mode : {RK_NPU_ATTENTION_F16_NO_MASK, RK_NPU_ATTENTION_F16_BOOLEAN_MASK}) {
            for (int pattern = 0; pattern < (mode == RK_NPU_ATTENTION_F16_BOOLEAN_MASK ? 3 : 1); ++pattern) {
                std::vector<float> output;
                npu.run(f, lut, batch, start + batch, 1, false, 0, &output, mode, pattern);
                save(std::string(argv[2]) + "/" + prefix + "-mode" + std::to_string(mode) + "-pattern" + std::to_string(pattern) + ".f32", output);
            }
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "ATTENTION AUDIT FAIL: %s\n", e.what());
    return 1;
}
