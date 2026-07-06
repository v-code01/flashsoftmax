# Adversarial review: flashsoftmax

A skeptic's pass over the claims, and why each survives.

## "Everyone knows online softmax equals two-pass - this is textbook, not a finding."
The algebraic identity is textbook; the numerical behavior over 32k sequential rescales is not something
most people have measured, and the naive intuition (O(L) dependent rounding steps must drift) points the
other way. The study's own pilot pointed that way too. Measuring that the error ratio stays at ~1 out to
32768 - that the flash rescale accumulates nothing - is the contribution, and it is the opposite of what
a careless analysis (or the pilot) would conclude.

## "You disclosed a pilot that disagreed - so your method is unreliable."
The disclosure is the point. A quick NumPy pilot mixed float32 and float64 and produced a spurious
length trend; the rigorous instrument uses a double-precision (f64) reference and a pure-f32 online path
and shows no trend. Reporting the misleading pilot and why it was wrong is what keeps the final claim
honest, rather
than quietly dropping it.

## "Claiming flash attention's softmax is 2.6-2.9x slower is misleading - flash attention is fast."
The claim is carefully scoped: the *standalone* online softmax is 2.6-2.9x slower because it recomputes
exponentials (~3L versus two-pass's L). The study explicitly says this is not a claim that flash
attention is slow - its win is
memory and fusion with the value matmul, which a standalone softmax cannot exhibit, and the measured
slowdown is an upper bound on the isolated-softmax cost. The finding corrects the "single pass = faster"
intuition without maligning flash attention.

## "A double-precision reference is not exact - your errors are against an approximation."
The reference is an IEEE double (53-bit mantissa) versus f32's 24, giving 29 extra bits (~5e8x) of
headroom over the f32 error being measured - ample and unbiased for comparing two f32 methods. (On Apple
Silicon `long double` is the same IEEE binary64 as `double`, so the code uses `double` explicitly and
claims no more precision than it has.) Both methods are measured against the same reference, so any
residual bias cancels in the ratio.

## "Gaussian scores are unrealistic; real attention logits differ."
The two claims that matter do not depend on the distribution. Max-subtraction conditioning and the
rescale factor being <= 1 are algebraic; the overflow onset depends only on the maximum score exceeding
~88, which any distribution with large enough scores reaches. The scale sweep (1, 10, 30) spans from
diffuse to near-one-hot, bracketing real attention.

## "Minimum-over-reps latency hides real-world variance."
Minimum is the standard estimator for intrinsic cost - it removes scheduler and contention noise, which
would only add to both methods. The ~2.6-2.9x ratio is consistent across every length above the
timer-resolution floor, which a noise artifact would not be.

## "verify.py just echoes analyze.py."
verify.py re-reads the raw per-cell results and recomputes the error ratios, the latency ratio, and the
overflow trend with its own medians, sharing no code with analyze.py. It re-derives every asserted
verdict.

## Pre-registration honesty
All five predictions were committed before the final run, including the two informative ones (P3 free, P4
not-faster) that cut against common intuition, and the disclosure of the misleading pilot. PREREG.md is
unedited.
