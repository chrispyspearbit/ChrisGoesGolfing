# ChrisGoesGolfing — Parameter Golf Autoresearch

This is an autonomous research loop for the OpenAI Parameter Golf challenge.
Train the best language model that fits in a 16MB artifact, measured by bits-per-byte (BPB) on FineWeb.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar18`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading, tokenizer, evaluation, quantization. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, hyperparameters, training loop.
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains data shards and `./data/tokenizers/` contains the tokenizer. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs locally on Apple Silicon (MLX). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python train.py`.

For faster iteration during exploration, you can reduce the time budget:
- Quick smoke test: `MAX_WALLCLOCK_SECONDS=60 ITERATIONS=500 TRAIN_BATCH_TOKENS=8192 python train.py`
- Medium run: `MAX_WALLCLOCK_SECONDS=180 python train.py`
- Full experiment: `python train.py` (5 minutes, default)

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, number of layers, attention heads, MLP width, activation functions, skip connections, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric. (Note: `train.py` uses `evaluate_bpb_strided`, a fast strided eval with 95% CI early-exit that wraps the same loss function. This is fine — it produces statistically equivalent results much faster.)

**The goal is twofold:**
1. **Lowest val_bpb** — the primary metric. Lower is better.
2. **Artifact must fit in 16MB** — the int8+zlib compressed model + code must be ≤ 16,000,000 bytes.

The artifact size is checked automatically. If `artifact_check` shows FAIL, your model is too large. You'll need to reduce parameters, adjust quantization-friendliness, or find a more compact architecture.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.234567
val_bpb_ci95:     ±0.003210
val_bpb_quant:    1.245678
val_bpb_quant_ci: ±0.003150
artifact_bytes:   15800000
artifact_check:   PASS (15800000/16000000)
model_bytes:      15750000
code_bytes:       50000
training_seconds: 300.1
total_tokens_M:   499.6
num_steps:        953
num_params:       5000000
num_layers:       9
model_dim:        512
```

Key metrics to extract:

```
grep "^val_bpb:\|^val_bpb_quant:\|^artifact_bytes:\|^artifact_check:\|^val_bpb_ci95:\|^val_bpb_quant_ci:" run.log
```

**val_bpb** is the pre-quantization score. **val_bpb_quant** is the post-quantization roundtrip score (this is the official Parameter Golf metric). **artifact_bytes** must be ≤ 16,000,000. The **ci95** lines show the 95% confidence interval half-width on each BPB estimate (strided eval with early-exit — speeds up eval ~8x).

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	iterations	val_bpb	val_bpb_quant	artifact_bytes	status	description
```

1. git commit hash (short, 7 chars)
2. iterations run (e.g. 200, 500, 953)
3. val_bpb achieved (pre-quant, e.g. 1.234567) — use 0.000000 for crashes
4. val_bpb_quant achieved (post-quant roundtrip) — use 0.000000 for crashes
5. artifact_bytes (e.g. 15800000) — use 0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	iterations	val_bpb	val_bpb_quant	artifact_bytes	status	description
a1b2c3d	953	1.234567	1.245678	15800000	keep	baseline
b2c3d4e	953	1.220000	1.231000	15900000	keep	increase matrix_lr to 0.06
c3d4e5f	953	1.250000	1.261000	15800000	discard	switch to GELU activation
d4e5f6g	0	0.000000	0.000000	0	crash	double model width (too slow)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar18`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^val_bpb_quant:\|^artifact_bytes:\|^artifact_check:\|^val_bpb_ci95:\|^val_bpb_quant_ci:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Record the results in `results.tsv`
8. If val_bpb_quant improved (lower) AND artifact_check is PASS, you "advance" the branch, keeping the git commit
9. If val_bpb_quant is equal or worse, or artifact is too large, you git reset back to where you started
10. **After every experiment** (keep or discard), update and push to GitHub (see "Pushing to GitHub" below)

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck, you can rewind but do this sparingly.

## Pushing to GitHub

After EVERY experiment (whether kept, discarded, or crashed), you must update the remote so progress is visible on GitHub:

1. **Update `results.tsv`** — **APPEND** the new row (this file is tracked in git). **NEVER delete or overwrite existing rows in results.tsv. This is the permanent record of all experiments. Only add new rows at the end.**
2. **Regenerate the progress graph**: `python plot_progress.py` — this reads `results.tsv` and writes `progress.png`.
3. **Update `README.md`** — update the Results table and Changelog section:
   - The **Results table** is a markdown table with columns: #, Commit, val_bpb_quant, Artifact, Status, Description. Add a row for the new experiment. Show artifact size in MB (e.g. "15.3 MB").
   - The **Changelog** section lists only kept experiments in reverse chronological order, with the best marked. Format: `- **#N** \`commit\` — description → **val_bpb_quant**`
   - Update the **"Current best"** line below the table.
4. **Commit the updates**: `git add results.tsv progress.png README.md && git commit -m "Update results: <short description>"`
5. **Push to GitHub**: `git push --force origin HEAD`

This ensures anyone watching the repo on GitHub can see a live, growing record of all experiments with a visual progress graph.

**Timeout**: Each experiment should take ~5 minutes total (+ a couple minutes for startup, warmup, and eval — eval uses strided sampling with CI early-exit so it's fast). If a run exceeds 12 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo), fix and re-run. If the idea itself is broken, just log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?". The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Ideas to explore

Here are some directions worth investigating (not exhaustive):

- **Architecture**: layer count vs width tradeoff, GQA head ratios, MLP expansion factor, different activation functions (GELU, SwiGLU), skip connection strategies
- **Optimizer**: learning rates for each parameter group, momentum schedules, warmdown fraction, weight decay
- **Training**: batch size, sequence length, gradient accumulation steps
- **Quantization-aware**: choices that compress better under int8+zlib (smoother weight distributions, fewer outliers)
- **Model size**: finding the sweet spot where more parameters improve BPB but still fit in 16MB after compression
- **Tokenizer interaction**: the 1024 vocab is fixed but architecture choices interact with it
