// Correctness tests for the softmax variants. They check the invariants the study rests on: all
// variants normalize to 1, two-pass and online track the long-double reference closely, online and
// two-pass agree, and naive (no max subtraction) overflows on large scores while the safe variants do
// not.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "softmax.hpp"

namespace {
int g_checks = 0;
void check(bool cond, const char* what) {
    ++g_checks;
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

double sum(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += x;
    return s;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

void test_uniform() {
    std::vector<float> x(4, 0.0f), out;
    fsm::softmax_twopass(x, out);
    for (float p : out) check(std::fabs(p - 0.25f) < 1e-6f, "twopass uniform -> 0.25");
    fsm::softmax_online(x, out);
    for (float p : out) check(std::fabs(p - 0.25f) < 1e-6f, "online uniform -> 0.25");
    fsm::softmax_naive(x, out);
    for (float p : out) check(std::fabs(p - 0.25f) < 1e-6f, "naive uniform -> 0.25");
}

void test_normalizes_and_matches_reference() {
    std::vector<float> x = {1.0f, -2.0f, 3.5f, 0.0f, -1.25f, 4.0f, 2.2f}, out;
    std::vector<long double> ref;
    fsm::softmax_reference(x, ref);
    fsm::softmax_twopass(x, out);
    check(std::fabs(sum(out) - 1.0) < 1e-6, "twopass sums to 1");
    check(fsm::max_abs_error(out, ref) < 1e-6, "twopass matches reference");
    std::vector<float> outo;
    fsm::softmax_online(x, outo);
    check(std::fabs(sum(outo) - 1.0) < 1e-6, "online sums to 1");
    check(fsm::max_abs_error(outo, ref) < 1e-6, "online matches reference");
    // online and two-pass agree closely with each other
    double diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) diff = std::fmax(diff, std::fabs(out[i] - outo[i]));
    check(diff < 1e-6, "online agrees with two-pass");
}

void test_naive_overflows_safe_does_not() {
    // Large scores: exp overflows f32 (exp(100) is inf) unless the max is subtracted.
    std::vector<float> x = {100.0f, 90.0f, 95.0f, 99.0f}, out;
    fsm::softmax_naive(x, out);
    check(!all_finite(out), "naive overflows on large scores");
    fsm::softmax_twopass(x, out);
    check(all_finite(out) && std::fabs(sum(out) - 1.0) < 1e-6, "twopass stays finite on large scores");
    fsm::softmax_online(x, out);
    check(all_finite(out) && std::fabs(sum(out) - 1.0) < 1e-6, "online stays finite on large scores");
}

void test_online_denominator_positive() {
    std::vector<float> x = {-5.0f, 0.0f, 5.0f, 2.0f};
    fsm::Online s = fsm::online_reduce(x);
    check(s.d > 0.0f && std::isfinite(s.d), "online denominator is positive and finite");
    check(std::fabs(s.m - 5.0f) < 1e-6f, "online max is the true max");
}

}  // namespace

int main() {
    test_uniform();
    test_normalizes_and_matches_reference();
    test_naive_overflows_safe_does_not();
    test_online_denominator_positive();
    std::printf("OK: %d checks passed\n", g_checks);
    return 0;
}
