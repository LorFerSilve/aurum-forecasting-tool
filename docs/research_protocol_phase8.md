# Researchprotocol — fase 8, compact multi-timeframe neural core

**Protocol:** `phase8-v1`

## Status and scope

This protocol freezes the first neural challenger before any phase-8 development result is
observed. It may use only the existing 2020–2024 development history. The final holdout from
2025-01-01 UTC remains technically forbidden.

Phase 8 does **not** replace the v0.2 price-only champions by default. The neural model is a
challenger and must beat the frozen phase-7 reference on identical outer test samples before
it can be admitted as a research champion. No phase-8 outcome activates paper or live trading.

## Frozen reference

The benchmark references the already validated phase-7 run:

- run: `20260907T014255645674Z-b2afe281`;
- protocol: `phase7-v2`;
- release: `v0.2.0`;
- champion mapping: `configs/phase7_champion.yaml`.

Phase 8 verifies the digest of every small phase-7 reference artifact it actually reads. For
each outer fold it also requires an exact test `sample_id` digest match with phase 7. The
50 GB phase-7 directory is therefore a provenance source, not training input.

## Architecture budget

The first challenger is intentionally compact:

- separate single-layer GRU encoder for each selected timeframe;
- availability-aware gated fusion of encoder embeddings;
- one three-class direction head;
- one direct arithmetic-return head;
- non-negative range and realized-volatility auxiliary heads;
- hard parameter budget of 150,000 parameters.

A Transformer is explicitly deferred. Phase 8 first answers whether a small recurrent neural
core adds value at all.

### Selected timeframes

`phase8-v1` uses `1min`, `3min`, `15min`, and `1h`. The `3h` source is excluded
from this first neural baseline because phase 7 measured only about 15.9% alignment
availability. `5min` and `30min` are also deferred to avoid starting with redundant
branches.

Sequence lengths are frozen at 60×1min, 20×3min, 8×15min and 4×1h candles. Each sequence is
oldest-to-newest and consists only of fully closed, contiguous candles. A source sequence may
be absent for a sample; it is never forward-filled across a gap. The fusion mask excludes an
unavailable branch. Every sample must retain at least one available configured branch and the
3min encoder is part of every horizon policy.

Short horizons 3–15m use `1min/3min/15min`; horizons 30–180m use `3min/15min/1h`.

## Per-candle inputs

Each encoder receives only six causal price-derived values:

1. one-candle log return in bps;
2. candle body in bps;
3. candle range in bps;
4. upper wick in bps;
5. lower wick in bps;
6. close position within high/low.

No label, future price, ask, spread, tick count, calibration statistic or outer-fold statistic
enters a sequence.

## Training contract

- mini-batches may shuffle samples, but timestep order inside every sequence is immutable;
- median imputation, mean and scale are fitted **only on training rows**;
- continuous targets are normalized only with training-target scales;
- direction starts with ordinary cross-entropy; class weighting/focal loss is not enabled in
  the v1 baseline;
- return/range/volatility use Huber losses on train-normalized targets;
- loss weights are frozen before the benchmark;
- AdamW, gradient clipping and non-finite checks are mandatory;
- early stopping uses inner-validation macro-F1 only;
- the outer calibration quarter remains reserved and unused;
- two frozen seeds are evaluated; final outer predictions average the two independently
  trained seed models;
- the final seed models are refit for the median inner-selected epoch count, so outer test data
  never selects epochs;
- mixed precision is allowed only on CUDA; deterministic algorithms remain enabled;
- CUDA AMP uses dynamic GradScaler loss scaling. A non-finite gradient caused by
  temporary loss-scale overflow skips that optimizer update and reduces the scale;
  non-finite unscaled/full-precision gradients remain a hard failure;
- before expensive fitting, the formal run validates all phase-7 comparison
  artifacts and outer sample digests and executes real-data device smoke fits for
  both the short- and long-horizon timeframe sets.

## Evaluation contract

The outer years, label definitions, 181-minute training-side gap, execution assumptions,
policy grid and base/stress costs are inherited from phase 6/7. Horizons remain
3, 6, 9, 12, 15, 30, 60 and 180 minutes.

For every horizon/fold the neural outer test digest must exactly equal the frozen phase-7 test
digest. The benchmark reports individual seed metrics and the two-seed ensemble.

## Promotion gate

A neural challenger receives **predictive admission** only if, versus the frozen phase-7
champion for that horizon:

- mean macro-F1 is strictly better;
- worst-fold macro-F1 is not worse;
- mean multiclass Brier is not worse;
- mean log loss is not worse.

Because the neural model is more complex, improving only one headline metric while degrading
probability quality is insufficient.

Economic promotion additionally requires the inherited phase-6 conditions:

- at least 20 trades in the worst fold;
- positive worst-fold net bps;
- positive mean stress net bps;
- worst-fold net bps no worse than the phase-7 fallback.

Even a candidate that passes both gates is only a development result for review. Phase 8 does
not authorize trading.

## Verification requirements

Before the full benchmark is accepted:

- future candle mutation cannot change older sequences;
- an unclosed higher-timeframe candle is never consumed;
- tensor shapes and availability masks are invariant-tested;
- train-only normalization is proven against mutated validation data;
- the model can deliberately overfit a tiny synthetic dataset;
- gradients remain finite and are clipped;
- same seed + same CPU input reproduces predictions;
- every head produces finite structurally valid output;
- parameter count remains inside budget;
- training and inference times are recorded;
- source/config/data/checkpoint provenance is persisted;
- the final holdout remains closed;
- the formal run starts only from a clean committed Git state.

## Exit decision

If the neural challenger does not pass the predictive gate, the v0.2 phase-7 champion remains
the fallback and the neural model may continue only as a research variant into phase 9. A
negative phase-8 result is valid and preferable to promoting complexity without evidence.
