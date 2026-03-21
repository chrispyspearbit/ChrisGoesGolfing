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
