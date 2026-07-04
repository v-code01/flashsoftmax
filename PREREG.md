# Pre-registration: flashsoftmax

Committed to git BEFORE the final benchmark and analysis. Not edited afterward.

## What is measured

Three ways to compute a softmax over a vector of scores, against a long-double reference:

- **naive** - `exp(x_i) / sum(exp(x_i))`, no max subtraction.
- **two-pass** - subtract the max, then exponentiate and sum. The safe textbook form.
- **online** - the flash-attention streaming reduction: a single pass maintaining a running maximum `m`
  and running denominator `d`, rescaling `d` by `exp(m_old - m_new)` on each element.

For a grid of sequence lengths L (32 .. 32768) and logit scales (1, 10, 30), 8 seeds per cell, we record
each f32 variant's maximum absolute error against the reference (or a non-finite flag), and the latency
of the two safe variants. The question is what the flash-attention online reduction costs relative to
the textbook two-pass: does its O(L) sequential rescaling accumulate error as the sequence grows, and is
being "single pass" actually faster?

**Pilot note (honest disclosure):** a quick NumPy pilot suggested the online error *grew* with sequence
length. That pilot mixed float32 and float64 in a way the rigorous C++ instrument (with a long-double
oracle) does not, so I treat it as suspect and test the question properly below.

## Predictions

**P1 - Naive overflows, length-dependently.** Naive softmax produces non-finite output once a score
exceeds the f32 exp limit (~88), and because a longer sequence has a larger maximum, the overflow
fraction at a fixed large scale increases with sequence length. *Falsifier:* naive stays finite at large
scale, or the overflow fraction does not grow with length.

**P2 - The safe variants are always safe.** Two-pass and online stay finite and normalized at every
length and scale. *Falsifier:* either produces a non-finite or non-normalized result.

**P3 - The online rescale is numerically free (no length drift).** Contrary to the pilot hint and to the
naive intuition that O(L) sequential rescales must accumulate, the online error tracks two-pass to within
a small constant factor (ratio <= 2) at every length including 32768, with no upward trend. *Falsifier:*
the online/two-pass error ratio rises with length or exceeds 2.

**P4 - Single-pass online is NOT faster standalone.** Despite being "single pass," online recomputes
exponentials (one in the streaming reduce, one in normalization) versus two-pass's single exp pass, so
standalone it is at least as slow as two-pass. Its real benefit is fusion with the value matmul in full
attention, which a standalone softmax cannot show. *Falsifier:* online median latency below two-pass.

**P5 - Error is set by scale, not length.** The error magnitude is governed by logit scale, is
non-monotonic in scale (largest at moderate scale, smallest near one-hot), and is essentially
independent of sequence length. *Falsifier:* error tracks length rather than scale.

## Commitment

P3 and P4 are the informative predictions: P3 says the flash trick is numerically free (against a pilot
that hinted otherwise), and P4 says it is not a standalone speed win (against the "single pass = faster"
intuition). Both are reported as-is whatever the data shows.
