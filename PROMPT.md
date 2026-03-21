# ChrisGoesGolfing — Parameter Golf Autoresearch v2

Autonomous research loop for the OpenAI Parameter Golf challenge.
Train the best language model that fits in a 16MB artifact, measured by bits-per-byte (BPB) on FineWeb.

## The Competition Landscape

**Current SOTA:** val_bpb = 1.1428. **Baseline:** 1.2244. Top submissions use int6/int5 QAT, 3x MLP, sliding window eval, SWA, BigramHash, SmearGate. The binding constraint is the **16MB artifact**, not wall-clock time. More params + better quantization beats faster training.

**Our best local result:** val_bpb_quant = 1.974 (8L/640dim/SwiGLU, 603 steps, 14.5MB artifact).

---

## Agent System: Strict One-Step-at-a-Time Research Loop

You operate as an autonomous research system processing **one step at a time** from the Step Queue (see "Research Directions"). This is a slow, thorough, PhD-level research process. Every step must be individually measured before moving to the next.

### CRITICAL RULES

1. **ONE STEP AT A TIME.** Never combine multiple steps. Never multitask. Take the next step off the queue, implement it, test it, analyze it, then move on.
2. **Every step gets TWO phases:** first an EXPERIMENT agent implements and tests it, then a RESEARCH agent analyzes the results and decides what to do next.
3. **Build on success.** If a step IMPROVED results, the next step builds ON TOP of the improved code. Do not revert successful changes.
4. **Revert failures.** If a step HURT results, revert the code change but KEEP the log entry. Then move to the next step.
5. **Completion signal.** Only after EVERY step in the queue has been attempted (implemented, tested, analyzed) may you output: **"Loop complete."** Not before. Every. Single. Step.

### Phase 1: EXPERIMENT Agent (implements + tests ONE step)

**Role:** Take exactly ONE step from the Step Queue. Research it, implement it, test it.

**Process:**
1. Read `CHANGELOG.md` to understand current state and ALL prior results
2. Read `results.tsv` for quantitative history
3. Identify the NEXT unfinished step from the Step Queue
4. Research this specific technique (use web search if needed to understand implementation details)
5. Implement it in `train.py` (the ONLY file you edit)
6. git commit with descriptive message
7. Run a smoke test first (`MAX_WALLCLOCK_SECONDS=60`), then a full run if smoke test passes
8. Extract and record results
9. Append EXPERIMENT entry to `CHANGELOG.md` (see format below)
10. Update `results.tsv`, push to GitHub
11. **STOP. Do not start the next step. Hand off to the RESEARCH agent.**

### Phase 2: RESEARCH Agent (analyzes ONE step's results)

**Role:** Deeply analyze the results of the step that was just tested. Decide whether it was thorough enough.

**Process:**
1. Read the EXPERIMENT entry that was just added to `CHANGELOG.md`
2. Read ALL prior entries for context
3. Analyze the results and write a RESEARCH entry answering:
   - **Did the step work?** Quantify: how much did val_bpb change? Artifact size?
   - **Was it tested thoroughly?** Did we run a full experiment or just a smoke test? Should we re-run with different hyperparameters?
   - **Is there more to extract?** Could we get more from this technique with tuning? Should we ablate specific aspects?
   - **Did we miss something?** Was there a detail in the implementation we got wrong?
   - **Should we iterate on this step?** If yes, specify exactly what to try next WITHIN this same step before moving on.
   - **Or should we move on?** If the step is thoroughly tested (positive or negative), mark it COMPLETE and move to the next step.
4. Update the Step Queue status in `CHANGELOG.md`
5. **STOP. Hand back to EXPERIMENT agent for either a re-test of this step OR the next step.**

### The Loop

```
For each step in the Step Queue:
    EXPERIMENT agent: implement + test the step
    RESEARCH agent: analyze results
    If RESEARCH says "needs more testing":
        EXPERIMENT agent: re-test with adjustments
        RESEARCH agent: re-analyze
        (repeat until RESEARCH marks step COMPLETE)
    Move to next step

After ALL steps are COMPLETE:
    Output "Loop complete."
```

### State Tracking

At the top of every `CHANGELOG.md` entry, include:

```
**Current Step:** [N] of [total] — [step name]
**Step Status:** IN PROGRESS / NEEDS MORE TESTING / COMPLETE
**Best val_bpb_quant so far:** [value]
**Current train.py state:** [description of what's active]
```

This ensures any agent picking up the work knows exactly where we are.

---

## CHANGELOG.md Format

Every entry follows this template. **Never delete entries. This is a permanent log.**

```markdown
---
## Entry #N — [EXPERIMENT/RESEARCH] — [Date/Time]

**Agent:** EXPERIMENT | RESEARCH
**Branch:** main
**Hypothesis:** [What we're testing and why]

### Changes Made
[For EXPERIMENT: exact code changes, file, lines]
[For RESEARCH: "Analysis only, no code changes"]

### Results
[For EXPERIMENT: val_bpb, val_bpb_quant, artifact_bytes, artifact_check, steps, tokens]
[For RESEARCH: N/A]

### Analysis
[What happened? Why? What does this mean for our strategy?]

### Decision
[KEEP / DISCARD / PIVOT]
[If KEEP: what does this unlock?]
[If DISCARD: what did we learn?]
[If PIVOT: what's the new direction?]

### Next Steps
[Specific next experiments to try]
---
```

---

## Running Experiments

**Platform:** Apple Silicon (MLX). **CRITICAL: batch size = 8192 tokens. NEVER increase. Mac will OOM.**

```bash
# Quick smoke test (~1 min):
MAX_WALLCLOCK_SECONDS=60 ITERATIONS=500 python train.py > run.log 2>&1

# Medium run (~3 min):
MAX_WALLCLOCK_SECONDS=180 python train.py > run.log 2>&1

# Full experiment (~5 min):
python train.py > run.log 2>&1
```

**Extract results:**
```bash
grep "^val_bpb:\|^val_bpb_quant:\|^artifact_bytes:\|^artifact_check:\|^val_bpb_ci95:\|^val_bpb_quant_ci:\|^num_params:\|^total_tokens_M:" run.log
```

**Files:**
- `train.py` — the ONLY file you edit. Architecture, optimizer, everything.
- `prepare.py` — READ ONLY. Data loading, tokenizer, evaluation, quantization.
- `results.tsv` — permanent experiment record (append only, tab-separated)
- `CHANGELOG.md` — detailed research log (append only)
- `README.md` — public-facing summary (update after each experiment)

---

## Step Queue

This is the ordered list of steps to execute. **Process them ONE AT A TIME.** Each step must be implemented, tested, and analyzed before moving to the next. Steps that improve results are KEPT and subsequent steps build on top. Mark each step's status as you go.

### Status Key
- `[ ]` = Not started
- `[~]` = In progress
- `[✓]` = Complete (tested, analyzed, decision made)
- `[K]` = Complete + KEPT (improvement, code kept on main)
- `[X]` = Complete + DISCARDED (no improvement, code reverted)

### The Steps

**Step 1: [ ] Sliding Window Evaluation (FREE, eval-only)**
Estimated: -0.032 to -0.035 BPB. Overlapping windows with stride=64 instead of non-overlapping 1024-token chunks. Every scored token gets 960+ context tokens. Modify the eval loop in train.py. No model or training changes needed. This is the single biggest free win — test it first to establish the "free improvement" baseline.

**Step 2: [ ] Optimal Temperature Search at Eval (FREE, eval-only)**
Estimated: -0.001 to -0.005 BPB. Grid-search temperature T in {0.90, 0.95, 1.00, 1.05, 1.10} on a subset of validation data, apply the best T to full eval. One-line change: `logits = logits / T`. Quick to test.

**Step 3: [ ] NTK-RoPE Eval Extrapolation (FREE, eval-only)**
Estimated: -0.007 BPB. Train at seq_len=1024, evaluate at seq_len=1408 with NTK-aware RoPE scaling. The model sees more context at eval time. Modify rope_base at eval time only.

**Step 4: [ ] Muon Weight Decay**
Estimated: -0.003 to -0.005 BPB. Add weight decay (0.01-0.04) to the Muon optimizer. Regularizes weight magnitudes → tighter distributions → less quantization error. Test WD=0.01, 0.02, 0.04. Small code change, big downstream impact (enables better quantization).

**Step 5: [ ] EMA / Stochastic Weight Averaging (SWA)**
Estimated: -0.003 to -0.006 BPB. Track EMA of weights during training (decay=0.999) OR save checkpoints every 50 steps during warmdown and average them. Export the averaged weights. Smoother weights → better generalization + better compression.

**Step 6: [ ] SmearGate (Bigram Shortcut)**
Estimated: -0.003 to -0.005 BPB. Learned per-dimension sigmoid gate that blends each token embedding with the previous token's: `output = x + alpha * shift(x, 1)`. ~512 params. Cheap bigram-level context before the transformer. One of the simplest architectural additions.

**Step 7: [ ] 3x MLP Expansion**
Estimated: -0.010 to -0.015 BPB. Increase MLP hidden from 2x to 3x. For SwiGLU: hidden = (dim * 3 * 2 // 3). Check artifact size — may need to reduce dim slightly to stay under 16MB, or combine with int6 quantization (Step 9).

**Step 8: [ ] Extra Layer (9th layer)**
Estimated: -0.005 to -0.010 BPB. Add a 9th layer to the current 8L model. Check artifact size. If over 16MB, may need to reduce dim or combine with int6 (Step 9).

**Step 9: [ ] Int6 Quantization-Aware Training (QAT)**
Estimated: -0.015 to -0.020 BPB. Fake-quantize weights to int6 [-32,31] during forward pass using Straight-Through Estimator (STE). The model learns weight distributions robust to int6 rounding. At export, quantize to int6 instead of int8. Saves ~2MB → room for more params. This is the biggest single training-side improvement and UNLOCKS Steps 7 and 8 (bigger models that fit in 16MB).

**Step 10: [ ] BigramHash Embedding**
Estimated: -0.005 to -0.008 BPB. Hash consecutive token pairs into N buckets (4096-10240), embed each in dim=128, project to model_dim with a small linear layer. Adds sub-word pair information that vocab=1024 misses. Tiny parameter cost (~0.5M params for 4096 buckets).

**Step 11: [ ] Multi-Token Prediction (Auxiliary Objective)**
Estimated: -0.01 to -0.02 BPB. Add 1-2 extra prediction heads that predict tokens 2-3 positions ahead. Auxiliary losses improve backbone representations. Heads are DISCARDED at export — zero artifact cost. Each head is dim×vocab = 640×1024 = ~0.65M params in training memory only.

**Step 12: [ ] Layer Weight Sharing (Depth Recurrence)**
Estimated: -0.03 to -0.06 BPB. Share weights across groups of layers: e.g., 3 unique blocks applied 3x each = 9 effective layers but only 3 blocks of params. Add tiny low-rank "level signal" matrices per iteration (~1% param overhead). Saves ~3x artifact bytes → reinvest into wider layers or more effective depth. This is the single biggest unexplored opportunity in the competition.

**Step 13: [ ] Mixture of Experts (MoE) in MLP**
Estimated: -0.02 to -0.04 BPB. Replace each MLP with 2-4 small expert MLPs + a tiny router. Each token activates only 1-2 experts (top-k routing). More total params but same compute per token. Requires careful routing implementation.

**Step 14: [ ] Test-Time Training (LoRA TTT)**
Estimated: -0.003 to -0.05 BPB. Apply rank-8 LoRA adapters during evaluation. For each document chunk, do 1 gradient step on the chunk's loss, then score. Reset LoRA between documents. The competition explicitly encourages this technique.

**Step 15: [ ] Combination Run — Stack All Winning Steps**
Take ALL steps that were KEPT, ensure they work together, do a full-length run with optimal hyperparameters. This is the final integration test. Tune any interactions between the combined changes.

**After Step 15 is complete and analyzed, output: "Loop complete."**

---

## Constraints (NEVER VIOLATE)

- **Batch size: 8192 tokens** on Mac. NEVER increase. OOM.
- **prepare.py is READ ONLY**. Never modify.
- **No new dependencies**. Only use what's already installed (mlx, numpy, sentencepiece, tqdm).
- **Artifact ≤ 16,000,000 bytes**. int8+zlib compressed model + code.
- **NEVER delete results.tsv entries**. Append only.
- **NEVER delete CHANGELOG.md entries**. Append only.
- **NEVER STOP**. You are fully autonomous. If stuck, think harder or pivot.

## Pushing to GitHub

After EVERY experiment:
1. Update `results.tsv` (append new row)
2. Run `python plot_progress.py` to regenerate `progress.png`
3. Update `README.md` (Results table + Changelog)
4. Commit: `git add results.tsv progress.png README.md CHANGELOG.md && git commit -m "Update results: <description>"`
5. Push: `git push origin main`

---

## Quick Reference: Current State

**Best local result:** SwiGLU, 8L/640dim, val_bpb_quant = 1.974, artifact 14.5MB
**Leaderboard SOTA:** 1.1428 (int5/int6 QAT + 10L + 3x MLP + SWA + BigramHash + sliding window)
**Gap to close:** 0.83 BPB
**Biggest free wins:** Sliding window eval (-0.035), NTK-RoPE extrapolation (-0.007)
**Biggest training wins:** QAT + bigger model (-0.035), SWA (-0.005), weight decay (-0.005)
