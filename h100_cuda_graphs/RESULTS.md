# CUDA Graph Speed Optimization — Results

**Date:** 2026-03-20
**Hardware:** 1x NVIDIA H100 80GB HBM3 (SXM), RunPod
**PyTorch:** 2.9.1+cu128
**Model:** 17M params (9L/512dim/8H/4KV, ReLU², tied embeddings, vocab=1024)
**Training:** 524,288 tokens/step, seed=1337, 10 min wallclock cap

---

## Head-to-Head: Baseline vs CUDA Graphs (1xH100, 10 min)

| Metric                         | Baseline          | CUDA Graphs       | Improvement        |
|--------------------------------|-------------------|-------------------|--------------------|
| **step_avg**                   | 548.64 ms         | **342.37 ms**     | **-37.6% faster**  |
| **Steps completed**            | 1,094             | **1,753**         | **+60.2%**         |
| **Tokens trained**             | 573M              | **919M**          | **+60.2%**         |
| **val_bpb (pre-quant)**        | 1.3663            | **1.3440**        | **-0.0223**        |
| **val_bpb (post-quant int8)**  | 1.3679            | **1.3452**        | **-0.0227**        |
| val_loss (pre-quant)           | 2.3069            | 2.2693            | -0.0376            |
| val_loss (post-quant int8)     | 2.3096            | 2.2713            | -0.0383            |
| Peak memory                    | 10,748 MiB        | 10,715 MiB        | -0.3% (same)       |
| Artifact size (int8+zlib)      | 12.70 MB          | 14.59 MB          | Both under 16 MB   |
| Quant degradation (bpb)        | +0.0016           | +0.0012           | Comparable         |

---

## Training Curves

### Baseline (1xH100, 10 min)
```
step     train_loss  val_bpb   step_avg
0        -           4.1077    -
10       5.9765      -         611ms
200      2.7758      -         546ms
400      2.5305      -         543ms
600      2.4653      -         544ms
800      2.2584      -         547ms
1000     2.3775      1.3730    548ms
1094     -           1.3663    549ms  (wallclock cap)
```

### CUDA Graphs (1xH100, 10 min)
```
step     train_loss  val_bpb   step_avg
0        -           4.1077    -
10       5.9947      -         354ms
200      2.8086      -         344ms
400      2.5593      -         343ms
600      2.5014      -         343ms
800      2.2916      -         342ms
1000     2.4210      1.3823    342ms
1200     2.2100      -         342ms
1400     2.1682      -         342ms
1600     2.1884      -         342ms
1753     -           1.3440    342ms  (wallclock cap)
```

---

## Projected 8xH100 Performance

| Metric                      | Baseline (known)   | CUDA Graphs (projected) |
|-----------------------------|--------------------|-----------------------|
| step_avg                    | 43.5 ms            | ~27 ms                |
| Steps in 10 min             | 13,780             | ~22,200               |
| Tokens trained              | 7.2B               | ~11.6B                |
| **val_bpb (predicted)**     | **1.2244**         | **~1.19**             |
| Improvement over SOTA       | -                  | **~0.035**            |
| vs 0.005 significance       | -                  | **7x threshold**      |

*Projection uses beta=0.22 from fitted Chinchilla scaling law (170 data points).*

---

## What Changed (3 Lines)

The CUDA graph optimization required only **3 changes** to the baseline `train_gpt.py`:

### 1. Pre-computed Rotary Embeddings (model change)
```python
# BEFORE: Lazy caching (breaks CUDA graphs - dynamic state)
class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        self._seq_len_cached = 0
        self._cos_cached = None  # allocated on first forward call
        ...

# AFTER: Pre-computed at init (CUDA graph safe)
class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, max_seq_len=1024):
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("_cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("_sin", freqs.sin()[None, None, :, :], persistent=False)
    def forward(self, seq_len, device, dtype):
        return self._cos[:,:,:seq_len,:].to(dtype=dtype), self._sin[:,:,:seq_len,:].to(dtype=dtype)
```

### 2. Enable CUDA Graphs via torch.compile (1 line)
```python
# BEFORE:
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)

# AFTER:
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True, mode="reduce-overhead")
```

### 3. Mark Step Boundaries for CUDA Graph Tree (2 insertions)
```python
# In warmup loop:
torch.compiler.cudagraph_mark_step_begin()

# In training loop:
torch.compiler.cudagraph_mark_step_begin()
```

---

## Failed Approaches (for reference)

| Approach | Result | Root Cause |
|----------|--------|------------|
| `mode="reduce-overhead"` on unmodified baseline | CRASH | Rotary lazy cache creates dynamic tensors |
| Remove ALL `.to(dtype)` + `mode="reduce-overhead"` | CRASH | Autograd can't backward through replayed CUDA graph |
| Remove `.to(dtype)` + `mode="max-autotune"` | CRASH | Same autograd issue |
| Remove `.to(dtype)` + default compile (no CUDA graphs) | RUNS, 18% SLOWER | fp32 type promotion overhead, no CUDA graph benefit |
| Manual `torch.cuda.CUDAGraph()` + `torch.compile` | CRASH | torch.compile internal streams conflict with capture |
| Manual `torch.cuda.CUDAGraph()` without torch.compile | WORKS but 14x SLOWER | No kernel fusion; raw Python overhead |
| Manual CUDA graph + `cache_enabled=False` | CRASH | Warmup autocast cache poisons CUDA state |
| Manual CUDA graph + `WARMUP_STEPS=0` | WORKS, 342ms | But loss diverges without `.to()` casts |
| **`reduce-overhead` + Rotary fix + `cudagraph_mark_step_begin`** | **WORKS, 342ms** | **The winning approach** |

---

## Hardware Details

```
GPU: NVIDIA H100 80GB HBM3 (SXM)
PyTorch: 2.9.1+cu128
CUDA: 12.8
Data: 1 training shard (fineweb10B_sp1024)
Validation: 62M tokens (fineweb_val)
```

---

## Files

| File | Description |
|------|-------------|
| `results/baseline_10min_1xH100.log` | Full training log from baseline run |
| `results/cudagraph_10min_1xH100.log` | Full training log from CUDA graph run |
| `submission/train_gpt_cudagraph.py` | The optimized training script |
| `parameter-golf/train_gpt.py` | The unmodified baseline |
