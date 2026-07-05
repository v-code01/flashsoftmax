# flashsoftmax: what the online (flash-attention) softmax reduction costs

Accuracy and cost of three softmax reductions - naive (no max subtraction), two-pass (max then exp), and online (the flash-attention single-pass streaming rescale) - measured against a long-double reference over a grid of sequence lengths and logit scales, 8 seeds per cell.

## 1. Is the online (flash) rescale numerically free? (error ratio online / two-pass)

   length      scale=1    scale=10    scale=30
   -------    --------   ---------   ---------
        32         0.95        0.87        1.00
        64         0.94        1.00        1.00
       128         1.43        0.95        1.00
       256         0.62        1.00        1.00
       512         1.15        0.81        1.00
      1024         1.27        1.19        1.00
      2048         1.15        0.88        1.00
      4096         1.40        0.88        1.00
      8192         0.95        1.00        1.00
     16384         0.99        1.04        1.00
     32768         1.02        1.01        1.00

The online error tracks two-pass to within a small factor at every length (max ratio 1.43), with no upward trend as the sequence grows to 32768. The streaming rescale exp(m_old - m_new) is well-conditioned (always <= 1), so the O(L) sequential rescales do NOT accumulate - the flash-attention softmax is numerically free even at long context.

## 2. Is single-pass online actually faster? (standalone latency, scale=10)

   length     two-pass ns    online ns    online / two-pass
   -------    -----------   ----------   ------------------
        32             41          146                3.55x
        64             83          291                3.51x
       128            208          584                2.81x
       256            354         1041                2.94x
       512            770         2104                2.73x
      1024           1541         4333                2.81x
      2048           3083         8520                2.76x
      4096           6208        16312                2.63x
      8192          12416        34980                2.82x
     16384          25000        65875                2.63x
     32768          50229       130875                2.61x

Standalone, online is about 2.9x SLOWER than two-pass, roughly constant across length. It recomputes exponentials - one in the streaming reduce pass and one in the normalization pass (~2L exp calls) versus two-pass's L (its max pass is exp-free). So the flash softmax is not a standalone speed win; its benefit is fusing with the value matmul and never materializing the score row, which a standalone softmax cannot show.

## 3. Naive softmax overflow onset (fraction of seeds non-finite)

   length     scale=10    scale=30
   -------    --------   ---------
        32         0%         0%
        64         0%        12%
       128         0%         0%
       256         0%        38%
       512         0%        25%
      1024         0%        88%
      2048         0%        88%
      4096         0%       100%
      8192         0%       100%
     16384         0%       100%
     32768         0%       100%

Naive softmax (no max subtraction) overflows f32 once any score exceeds ~88 (exp(88) = inf). The chance of that rises with sequence length - more scores means a larger maximum - so at scale 30 the overflow fraction climbs from a fraction of seeds at short length to every seed by a few thousand tokens. Max subtraction is not optional, and its necessity grows with context length.

## 4. Error magnitude is set by logit scale, not length (two-pass median error)

   scale    median abs error (over all lengths)
   -----    ----------------------------------
       1    4.23e-09
      10    1.65e-07
      30    2.53e-08

Error is driven by logit scale, and non-monotonically: it peaks at moderate scale (many terms contribute comparably) and shrinks again at large scale (the distribution approaches one-hot, so few terms carry weight). Length barely moves it.

## findings

1. The online (flash-attention) softmax is numerically EQUIVALENT to two-pass at every tested length up to 32768 (error ratio ~1, max 1.43); the streaming rescale does not accumulate error. The trick is numerically free.
2. But standalone it is ~2.9x SLOWER (it recomputes exponentials); its real payoff is memory and fusion inside full attention, not standalone softmax speed.
3. Naive softmax overflows above ~88, and the overflow onset moves to shorter sequences as scale rises - max subtraction is mandatory and increasingly so with context length.
4. Softmax error is governed by logit magnitude (non-monotonically), essentially independent of sequence length.

