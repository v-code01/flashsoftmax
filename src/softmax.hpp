// Three ways to compute a softmax over a vector of scores, plus a high-precision reference. The point
// of interest is the "online" (flash-attention style) softmax, which folds the maximum and the
// denominator sum into a single streaming pass with an incremental rescale, instead of the textbook
// two-pass (one pass for the max, one for the exponentials). Flash attention relies on exactly this
// streaming reduction to avoid materializing the full score row. This header lets a study measure what
// the streaming trick costs numerically and in passes over the data.
//
//   softmax_naive     - exp(x_i) / sum(exp) with NO max subtraction. Overflows to inf/NaN once the
//                       scores are large; the classic reason max subtraction exists.
//   softmax_twopass   - m = max(x); e_i = exp(x_i - m); p = e / sum(e). Numerically safe, two passes.
//   softmax_online    - a single streaming pass maintaining (m, d): on each score, m' = max(m, x_i),
//                       d = d * exp(m - m') + exp(x_i - m'). Algebraically identical to two-pass, but
//                       the repeated rescale d*exp(m-m') accumulates rounding as the sequence grows.
//   softmax_reference - the same two-pass computation in double: the oracle to measure error.
//
// All f32 variants write normalized probabilities into `out`. The online variant additionally returns
// its streaming (m, d) so a caller can check the denominator directly.
#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <cmath>
#include <cstddef>
#include <vector>

namespace fsm {

struct Online {
    float m;  // running maximum
    float d;  // running denominator sum, expressed relative to m
};

// Naive: no max subtraction. Included precisely to show where it breaks.
inline void softmax_naive(const std::vector<float>& x, std::vector<float>& out) {
    out.resize(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float e = std::exp(x[i]);
        out[i] = e;
        sum += e;
    }
    for (float& v : out) v /= sum;
}

// Two-pass with max subtraction: the numerically safe textbook form.
inline void softmax_twopass(const std::vector<float>& x, std::vector<float>& out) {
    out.resize(x.size());
    float m = -INFINITY;
    for (float v : x) m = std::fmax(m, v);
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float e = std::exp(x[i] - m);
        out[i] = e;
        sum += e;
    }
    for (float& v : out) v /= sum;
}

// Online (flash-attention) reduction: a single streaming pass for (m, d). Returns the streaming state;
// probabilities are then emitted with a normalization pass using the final (m, d).
inline Online online_reduce(const std::vector<float>& x) {
    float m = -INFINITY;
    float d = 0.0f;
    for (float v : x) {
        float mnew = std::fmax(m, v);
        // exp(m - mnew) is <= 1 and rescales the accumulated denominator into the new frame.
        d = d * std::exp(m - mnew) + std::exp(v - mnew);
        m = mnew;
    }
    return Online{m, d};
}

inline void softmax_online(const std::vector<float>& x, std::vector<float>& out) {
    out.resize(x.size());
    Online s = online_reduce(x);
    for (size_t i = 0; i < x.size(); ++i) out[i] = std::exp(x[i] - s.m) / s.d;
}

// High-precision reference (double, two-pass). The oracle for measuring f32 error.
inline void softmax_reference(const std::vector<float>& x, std::vector<double>& out) {
    out.resize(x.size());
    double m = -INFINITY;
    for (float v : x) m = std::fmax(m, static_cast<double>(v));
    double sum = 0.0L;
    for (size_t i = 0; i < x.size(); ++i) {
        double e = std::exp(static_cast<double>(x[i]) - m);
        out[i] = e;
        sum += e;
    }
    for (double& v : out) v /= sum;
}

// Maximum absolute error of an f32 probability vector against the double-precision (f64) reference.
inline double max_abs_error(const std::vector<float>& p, const std::vector<double>& ref) {
    double e = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        double d = std::fabs(static_cast<double>(p[i]) - static_cast<double>(ref[i]));
        e = std::fmax(e, d);
    }
    return e;
}

}  // namespace fsm

#endif  // SOFTMAX_HPP
