# flashsoftmax: what the online (flash-attention) softmax reduction actually costs

Flash attention avoids materializing the full attention-score row by computing the softmax in a single
streaming pass: it keeps a running maximum and a running denominator, and rescales the denominator by
`exp(m_old - m_new)` every time a larger score arrives. The textbook alternative is two passes - one to
find the maximum, one to exponentiate and sum. This measures what the online reduction costs relative to
two-pass, in accuracy and in standalone speed, against a long-double reference over sequence lengths from
32 to 32768 and three logit scales, 8 seeds per cell.

## Pre-registration

Five predictions were committed to git (`PREREG.md`) before the final run: (P1) naive softmax overflows,
length-dependently; (P2) the safe variants are always safe; (P3) the online rescale is numerically free
with no length drift; (P4) single-pass online is not faster standalone; (P5) error is set by scale, not
length. **All five held.** PREREG.md also records, honestly, that a quick NumPy pilot had suggested the
online error grows with length - the rigorous C++ instrument (long-double oracle) shows it does not, and
the pilot was a float32/64 artifact.

## Results

**1. The online rescale is numerically free - no length drift.** Online-vs-two-pass max-abs-error ratio:

```
 length     scale=1   scale=10   scale=30
     256       1.02      1.00       1.00
    4096       1.05      0.99       1.00
   16384       0.98      1.01       1.00
   32768       1.01      1.01       1.00
```

The online error tracks two-pass to within a small factor at every length (worst 1.19; the scale=1
scatter is because errors there sit near the f32 floor of ~1e-9), with no upward trend out to 32768. The
rescale factor `exp(m_old - m_new)` is always <= 1 and well-conditioned, so the O(L) sequential rescales
do not accumulate. **The flash-attention softmax is numerically free even at long context.**

**2. But single-pass online is ~2.8x slower standalone.**

```
 length    two-pass ns   online ns   online / two-pass
    256           833       2333            2.80x
   4096          6230      17500            2.81x
  32768         50750     142000            2.80x
```

"Single pass" does not mean fewer operations: online does an exponential in the streaming reduce pass
*and* in the normalization pass (~2L exp calls) while two-pass does L (its max pass is exp-free). So the
online form is ~2.8x slower as a standalone softmax, constant across length. Its payoff is elsewhere -
fusing with the value matmul and never storing the score row, which a standalone softmax benchmark
cannot show. This is worth stating plainly: flash attention's softmax is a memory/fusion win, not a
faster softmax.

**3. Naive softmax overflows, and the onset moves to shorter sequences as scale rises.**

```
 length    scale=10   scale=30
     64         0%        12%
    256         0%        38%
   1024         0%        88%
   4096         0%       100%
  32768         0%       100%
```

Naive softmax overflows f32 once any score exceeds ~88 (`exp(88)` is inf). A longer sequence has a larger
maximum, so at scale 30 the overflow fraction climbs from a few seeds at short length to every seed by a
few thousand tokens. Max subtraction is mandatory, and increasingly so with context length.

**4. Error is set by logit scale, not length** - and non-monotonically: the two-pass median error is
4.2e-9 at scale 1, 1.7e-7 at scale 10, and back down to 2.5e-8 at scale 30 (the distribution approaches
one-hot, so few terms carry weight). Length barely moves it.

## The one-line finding

The flash-attention online softmax is numerically identical to the safe two-pass form at every context
length up to 32768 - the streaming rescale accumulates nothing - but it is not a faster softmax in
isolation (it recomputes exponentials, ~2.8x slower); its value is memory and fusion inside full
attention. Meanwhile the naive no-subtraction softmax overflows, the more readily the longer the
sequence.

## Reproduce

```
./reproduce.sh 400        # build, benchmark (self-contained seeded data), analyze, verify
./scripts/gate.sh         # C++ -Werror build + tests, ruff, mypy, ASCII, leak scan, independent verify
```

The benchmark generates its own scores with a deterministic PRNG, so it needs no model or data files.
`tools/verify.py` recomputes the error ratios, the latency ratio, and the overflow trend from the raw
per-cell results with its own arithmetic, sharing no code with `analyze.py`.

## Limitations and falsifiers

- One machine, one compiler at `-O3 -march=native`, f32 activations with a long-double oracle. Absolute
  nanoseconds are machine-specific; the accuracy equivalence, the ~2.8x standalone slowdown, and the
  overflow trend are not.
- The online variant here still does a normalization pass to emit a probability vector. In real flash
  attention that pass is fused into the value accumulation, so the "extra exp" is amortized differently;
  the standalone slowdown measured here is an upper bound on the isolated-softmax cost, and the study
  says so rather than claiming flash attention is slow.
- Scores are Gaussian; real attention logits are not exactly Gaussian, but the max-subtraction and
  rescale conditioning arguments do not depend on the distribution.
- **Falsifier (the informative one):** if the online/two-pass error ratio had risen with length, the
  streaming rescale would accumulate; it does not (ratio ~1 out to 32768), refuting the pilot hint.

MIT licensed. The oracle is a long-double reference softmax; error is exact max-abs deviation; latency is
minimum-over-reps monotonic-clock time. No LLM judgement.
