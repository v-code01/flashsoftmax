// Measure the accuracy and cost of the three softmax variants over a grid of sequence lengths and
// score magnitudes, with several seeds per cell for confidence intervals. Data is generated in-process
// with a deterministic PRNG so the benchmark is self-contained and reproducible. For each (length,
// scale, seed) it emits the max-abs error of each f32 variant against the double-precision (f64) reference (or
// "nan" if the variant produced a non-finite result) and the minimum-over-reps latency of the two safe
// variants.
//
// Usage: bench REPS
// Output: JSON lines {len, scale, seed, twopass_err, online_err, naive_err|null,
//                     twopass_ns, online_ns}
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "softmax.hpp"

namespace {
using Clock = std::chrono::steady_clock;

// splitmix64: a small deterministic PRNG, so results do not depend on <random> implementation details.
uint64_t g_state;
uint64_t next_u64() {
    uint64_t z = (g_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
// Standard-normal via Box-Muller, scaled.
float normal(float scale) {
    double u1 = (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    double u2 = (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    if (u1 < 1e-300) u1 = 1e-300;
    double r = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    return static_cast<float>(r * scale);
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

template <typename F>
double min_ns(F&& fn, int reps) {
    double best = 1e300;
    volatile float sink = 0.0f;
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        std::vector<float> out;
        fn(out);
        auto t1 = Clock::now();
        sink += out.empty() ? 0.0f : out[0];
        best = std::fmin(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    (void)sink;
    return best;
}
}  // namespace

int main(int argc, char** argv) {
    const int reps = (argc > 1) ? std::atoi(argv[1]) : 200;
    const std::vector<int> lengths = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    const std::vector<float> scales = {1.0f, 10.0f, 30.0f};
    const int seeds = 8;

    std::string out;
    for (int L : lengths) {
        for (float scale : scales) {
            for (int seed = 0; seed < seeds; ++seed) {
                g_state = 0xD1B54A32D192ED03ULL + static_cast<uint64_t>(seed) * 0x100000001B3ULL +
                          static_cast<uint64_t>(L) * 1000003ULL +
                          static_cast<uint64_t>(scale) * 2654435761ULL;
                std::vector<float> x(static_cast<size_t>(L));
                for (float& v : x) v = normal(scale);

                std::vector<double> ref;
                fsm::softmax_reference(x, ref);
                std::vector<float> tp, on, na;
                fsm::softmax_twopass(x, tp);
                fsm::softmax_online(x, on);
                fsm::softmax_naive(x, na);

                double tp_err = fsm::max_abs_error(tp, ref);
                double on_err = fsm::max_abs_error(on, ref);
                bool na_ok = all_finite(na);
                double na_err = na_ok ? fsm::max_abs_error(na, ref) : 0.0;

                double tp_ns = min_ns([&](std::vector<float>& o) { fsm::softmax_twopass(x, o); }, reps);
                double on_ns = min_ns([&](std::vector<float>& o) { fsm::softmax_online(x, o); }, reps);

                char na_field[32];
                if (na_ok)
                    std::snprintf(na_field, sizeof(na_field), "%.6e", na_err);
                else
                    std::snprintf(na_field, sizeof(na_field), "null");
                char row[256];
                std::snprintf(row, sizeof(row),
                              "{\"len\": %d, \"scale\": %.0f, \"seed\": %d, \"twopass_err\": %.6e, "
                              "\"online_err\": %.6e, \"naive_err\": %s, \"twopass_ns\": %.1f, "
                              "\"online_ns\": %.1f}\n",
                              L, static_cast<double>(scale), seed, tp_err, on_err, na_field,
                              tp_ns, on_ns);
                out += row;
            }
        }
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
