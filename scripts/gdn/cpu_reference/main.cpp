// Standalone recurrence reference; no Rust runtime, model weights or NPU needed.
#include "gdn.h"
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5) return 2; // INPUT.f32 OUTPUT.f32 TOKENS THREADS
    const int tokens = std::atoi(argv[3]), threads = std::atoi(argv[4]);
    if (tokens < 1 || threads < 1 || threads > 4) return 2;
    const size_t t = static_cast<size_t>(tokens);
    std::vector<float> q(t*16*128), k(q.size()), v(t*32*128);
    std::vector<float> g(t*32), beta(g.size()), initial(32*128*128), state(initial.size());
    std::vector<float> output(v.size());
    std::ifstream input(argv[1], std::ios::binary);
    for (auto * data : {&q, &k, &v, &g, &beta, &initial}) {
        input.read(reinterpret_cast<char *>(data->data()), data->size()*sizeof(float));
        if (!input) throw std::runtime_error("short compact input");
    }
    rknpu2_gdn_params params{t, 16, 32, 128, 128, static_cast<size_t>(threads),
        initial.data(), state.data(), q.data(), k.data(), v.data(), g.data(), beta.data(), output.data()};
    rknpu2_gdn_run(&params);
    std::ofstream result(argv[2], std::ios::binary);
    result.write(reinterpret_cast<const char *>(output.data()), output.size()*sizeof(float));
    result.write(reinterpret_cast<const char *>(state.data()), state.size()*sizeof(float));
    if (!result) throw std::runtime_error("output write failed");
}
