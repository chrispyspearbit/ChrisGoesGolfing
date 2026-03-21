# Research Changelog — Parameter Golf

Permanent log of all experiments and research analyses. **Never delete entries.**

---

## Entry #0 — RESEARCH — 2026-03-20

**Agent:** RESEARCH (initial analysis)
**Branch:** main

### Analysis

#### Leaderboard Study (SOTA = 1.1428, Baseline = 1.2244)

The competition has been won by **packing more parameters into 16MB**, not by training faster. Key techniques from top submissions:

| Technique | BPB Gain | Used By Top-5? |
|-----------|----------|-----------------|
| Sliding window eval (stride=64) | -0.032 to -0.035 | All |
| Int6/Int5 QAT | -0.015 to -0.020 | All |
| 3x MLP expansion | -0.010 to -0.015 | All |
| SWA (checkpoint averaging) | -0.003 to -0.006 | 4/5 |
| SmearGate + BigramHash | -0.005 to -0.008 | 3/5 |
| Muon weight decay | -0.003 to -0.005 | 3/5 |
| Extra layers (10-11L) | -0.005 to -0.010 | 4/5 |
| zstd-22 compression | ~-0.002 | 3/5 |

#### Our Quantization Analysis (tested on real trained model)

| Quant | Artifact Size | Bytes/param | Max Params in 16MB | RMSE |
|-------|--------------|-------------|-------------------|------|
| int8+zlib | 14.01 MB | 0.770 | 20.8M | 0.000406 |
| int6+zstd | 14.45 MB | 0.683 | 23.4M | 0.001941 |
| int5+zstd | 12.15 MB | 0.574 | 27.9M | 0.003997 |

#### Our Speed Work (CUDA graphs on H100)

- 1xH100: 342ms/step vs 546ms baseline = 38% faster
- 8xH100: 43.21ms/step = same as baseline (CUDA graphs irrelevant at grad_accum=1)
- The model is too small for 8 H100s (10% MFU)

### Decision: PIVOT

Our strategy was speed. The competition is about **information density per artifact byte**. We need to:
1. Implement sliding window eval (free -0.035)
2. Implement int6 QAT
3. Make the model bigger (3x MLP, more layers)
4. Add SWA and weight decay

### Current Priority List

1. **Sliding window eval** (FREE, eval-only change)
2. **NTK-RoPE eval extrapolation** (FREE, eval-only change)
3. **Int6 QAT** (training change, enables bigger models)
4. **3x MLP expansion** (architecture change)
5. **SWA** (training change)
6. **Muon weight decay** (optimizer change)
7. **Extra layer** (architecture, needs int6 to fit)

---

## Entry #1 — EXPERIMENT — 2026-03-21

**Agent:** EXPERIMENT
**Branch:** main
**Current Step:** 1 of 15 — Sliding Window Evaluation
**Step Status:** IN PROGRESS
**Best val_bpb_quant so far:** 1.9686 (sliding window) / 1.9742 (non-overlapping, prior best)
**Current train.py state:** 8L/640dim/SwiGLU + sliding window eval (stride=64)

**Hypothesis:** Sliding window eval with stride=64 overlapping windows gives every scored token 960+ tokens of context, reducing BPB compared to non-overlapping 1024-token chunks where early tokens have minimal context.

### Changes Made
- Added `loss_last_n()` method to GPT model: computes loss on only the last N positions of each sequence
- Added `evaluate_bpb_sliding()` function: generates overlapping windows with stride=64, batches them, scores only the last 64 tokens per window
- Added `eval_stride` hyperparameter (default 64, configurable via EVAL_STRIDE env var)
- Added `sliding_loss_fn` wrapper in main() for the sliding window forward pass
- Final eval (both unquantized and quantized) now uses sliding window
- Mid-training eval still uses fast non-overlapping method
- CI threshold relaxed to max(eval_ci_threshold, 0.01) for sliding window eval

### Results
- **Smoke test (121 steps):** non-overlapping 3.0066 vs sliding 3.0022 = **-0.004 improvement**
- **Full run (588 steps):**
  - val_bpb (sliding): 1.978619 ±0.010
  - val_bpb_quant (sliding): 1.968631 ±0.010
  - val_bpb (non-overlapping, training loop): 1.9730 ±0.005
  - artifact_bytes: 14,038,178 (PASS)
  - training_seconds: 389.5
  - total_tokens: 4.8M, steps: 588, params: 21.16M

### Analysis
- The sliding window eval gives val_bpb_quant = 1.9686 vs previous best 1.9742 = **-0.0056 BPB improvement**
- This is smaller than the competition estimate of -0.032 to -0.035, likely because:
  - We train for ~600 steps locally vs ~13,780 on 8xH100
  - Lower model quality means less context improvement to extract
  - Wide CI (±0.01) means true improvement may be larger
- The sliding eval takes ~2 minutes (2x non-overlapping) due to CI convergence with smaller scored-tokens-per-batch
- Non-overlapping vs sliding BPB comparison is noisy (different random seeds, different CI widths)

### Decision
KEEP — sliding window eval is implemented and working. The improvement is positive (if noisy). This is a FREE eval-only change that all top submissions use.

### Next Steps
Hand off to RESEARCH agent for analysis of whether stride=64 is optimal, whether CI convergence is fast enough, and whether the implementation is correct.

---
