"""
ChrisGoesGolfing — Parameter Golf training script (MLX, Apple Silicon).
This is the ONLY file the agent modifies. Everything else is in prepare.py.

Usage: python train.py > run.log 2>&1
"""
from __future__ import annotations

import csv
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from prepare import (
    TIME_BUDGET,
    ARTIFACT_SIZE_LIMIT,
    COMPUTE_DTYPE,
    DEFAULT_DATA_PATH,
    DEFAULT_TOKENIZER_PATH,
    TokenLoader,
    load_validation_tokens,
    build_sentencepiece_luts,
    validate_dataset_tokenizer_pair,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    compress_artifact,
)

# ==============================================================================
# HYPERPARAMETERS — edit these to experiment
# ==============================================================================

class Hyperparameters:
    # Data / tokenizer
    data_path: str = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", DEFAULT_TOKENIZER_PATH)
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4())[:8])
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 65_536))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 8_192))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", TIME_BUDGET))

    # Model architecture
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 8))
    model_dim: int = int(os.environ.get("MODEL_DIM", 640))
    num_heads: int = int(os.environ.get("NUM_HEADS", 10))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 1))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    activation: str = os.environ.get("ACTIVATION", "swiglu")  # swiglu, relu2, gelu

    # Optimizer
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.05))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Strided eval: 95% CI early-exit
    eval_ci_threshold: float = float(os.environ.get("EVAL_CI_THRESHOLD", 0.005))
    eval_min_batches: int = int(os.environ.get("EVAL_MIN_BATCHES", 30))

    # Sliding window eval: overlapping windows with stride for better context
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", 64))  # 0 = disabled (use non-overlapping)

    # Scaling study: periodic eval for intermediate data points
    scaling_eval_every: int = int(os.environ.get("SCALING_EVAL_EVERY", 0))  # 0 = off
    scaling_csv: str = os.environ.get("SCALING_CSV", "")  # path to append scaling data

    # μP: set MUP_BASE_DIM to enable width-scaled LR/init transfer
    mup_base_dim: int = int(os.environ.get("MUP_BASE_DIM", 0))  # 0 = disabled

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self):
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self):
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self):
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step, elapsed_ms):
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g, steps, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def token_chunks(total_tokens, seq_len, max_chunk_tokens):
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(accum, grads_tree, scale):
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ==============================================================================
# MODEL
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, init_scale=1.0):
        super().__init__()
        self.weight = (nn.Linear(in_dim, out_dim, bias=False).weight * init_scale).astype(mx.float32)

    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x):
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, activation="swiglu"):
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            # SwiGLU: 3 projections but smaller hidden to match param count
            hidden = (dim * mlp_mult * 2 // 3 + 15) // 16 * 16
            self.gate = CastedLinear(dim, hidden)
            self.up = CastedLinear(dim, hidden)
            self.proj = CastedLinear(hidden, dim)
        elif activation == "relu2":
            # ReLU²: 2 projections (standard FFN)
            hidden = (dim * mlp_mult + 15) // 16 * 16
            self.up = CastedLinear(dim, hidden)
            self.proj = CastedLinear(hidden, dim)
        elif activation == "gelu":
            hidden = (dim * mlp_mult + 15) // 16 * 16
            self.up = CastedLinear(dim, hidden)
            self.proj = CastedLinear(hidden, dim)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x):
        if self.activation == "swiglu":
            return self.proj(nn.silu(self.gate(x)) * self.up(x))
        elif self.activation == "relu2":
            h = self.up(x)
            return self.proj(nn.relu(h) * nn.relu(h))
        elif self.activation == "gelu":
            return self.proj(nn.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, activation="swiglu"):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult, activation=activation)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x, x0):
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, dim, num_heads, num_kv_heads, mlp_mult,
                 logit_chunk_tokens, logit_softcap, rope_base, tied_embed_init_std, qk_gain_init,
                 mup_width_mult=1.0, activation="swiglu"):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.mup_width_mult = mup_width_mult  # 1.0 = no μP scaling

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, activation=activation)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits):
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids):
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        # μP: scale output logits by 1/width_mult so gradients are width-invariant
        output_scale = 1.0 / self.mup_width_mult
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T * output_scale
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T * output_scale
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)


# ==============================================================================
# OPTIMIZERS
# ==============================================================================

class Muon:
    def __init__(self, keys, params, args):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params, grads, step, lr_mul):
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)


class SplitOptimizers:
    def __init__(self, model, args, mup_lr_scale=1.0):
        self.args = args
        self.mup_lr_scale = mup_lr_scale  # base_dim / model_dim when μP is on
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2
            and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k == "skip_weights" or (k.startswith("blocks.") and (
                p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
        ]
        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,  # embedding LR unchanged in μP
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps, bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps, bias_correction=True,
        )

    def step(self, model, grads_tree, step, lr_mul):
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        # μP: scale matrix LR by base_dim/model_dim
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul * self.mup_lr_scale))
        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul  # embedding LR unchanged in μP
        updated.update(self.adam_embed.apply_gradients(
            {self.embed_key: grads[self.embed_key]},
            {self.embed_key: params[self.embed_key]},
        ))
        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))
        model.update(tree_unflatten(list(updated.items())))


# ==============================================================================
# TRAINING HELPERS
# ==============================================================================

def loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad):
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def _compute_per_position_loss(model, input_ids, target_ids):
    """Compute per-position cross-entropy loss (unreduced). Returns shape [batch*seq_len]."""
    x = model(input_ids).reshape(-1, model.tok_emb.weight.shape[1])
    y = target_ids.reshape(-1)
    output_scale = 1.0 / model.mup_width_mult
    logits_proj = x @ model.tok_emb.weight.astype(x.dtype).T * output_scale
    logits = model.softcap(logits_proj)
    return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")


def evaluate_bpb_sliding_window(model, val_tokens, seq_len, stride,
                                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                                batch_size=8, desc="eval (sliding)"):
    """
    Sliding window evaluation: overlapping windows with small stride.
    Each scored token gets (seq_len - stride) context tokens.
    Much more accurate than non-overlapping eval.
    """
    n_tokens = val_tokens.size - 1  # last token is only used as target
    # Generate window start positions
    starts = list(range(0, n_tokens - seq_len + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] + seq_len < n_tokens:
        starts.append(n_tokens - seq_len)

    total_loss_sum = 0.0
    total_tokens_scored = 0
    total_bytes = 0.0
    n_batches = math.ceil(len(starts) / batch_size)

    pbar = tqdm(total=n_batches, desc=desc, leave=False)
    for b in range(n_batches):
        batch_starts = starts[b * batch_size : (b + 1) * batch_size]
        bsz = len(batch_starts)

        # Gather windows: [bsz, seq_len]
        offsets = np.arange(seq_len)
        x_np = np.stack([val_tokens[s + offsets] for s in batch_starts])
        y_np = np.stack([val_tokens[s + 1 + offsets] for s in batch_starts])

        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)

        per_pos_loss = _compute_per_position_loss(model, x, y)
        mx.eval(per_pos_loss)
        per_pos_loss_np = np.array(per_pos_loss, dtype=np.float64).reshape(bsz, seq_len)

        for j in range(bsz):
            global_idx = b * batch_size + j
            # First window: score all positions; rest: only last stride positions
            score_start = 0 if global_idx == 0 else (seq_len - stride)
            scored_losses = per_pos_loss_np[j, score_start:]
            scored_x = x_np[j, score_start:]
            scored_y = y_np[j, score_start:]

            # Byte counting
            bytes_arr = base_bytes_lut[scored_y].astype(np.int16, copy=True)
            bytes_arr += (
                has_leading_space_lut[scored_y] & ~is_boundary_token_lut[scored_x]
            ).astype(np.int16, copy=False)

            total_loss_sum += float(scored_losses.sum())
            total_tokens_scored += len(scored_losses)
            total_bytes += float(bytes_arr.astype(np.float64).sum())

        if (b + 1) % 10 == 0 or b == n_batches - 1:
            running_bpb = (total_loss_sum / total_tokens_scored / math.log(2.0)) * (total_tokens_scored / total_bytes)
            pbar.set_postfix_str(f"bpb={running_bpb:.4f}")
        pbar.update(1)

    pbar.close()
    val_loss = total_loss_sum / total_tokens_scored
    val_bpb = (val_loss / math.log(2.0)) * (total_tokens_scored / total_bytes)
    return val_loss, val_bpb, total_tokens_scored


def evaluate_bpb_strided(loss_fn, val_tokens, seq_len, val_batch_tokens,
                         base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                         ci_threshold=0.005, min_batches=30, desc="eval"):
    """
    Strided evaluation with 95% CI early-exit.
    Shuffles sequence order (random seed each call) for decorrelated sampling,
    then stops as soon as the 95% CI half-width on BPB drops below ci_threshold.
    Returns (val_loss, val_bpb, ci_half_width, n_batches_used, total_batches).
    """
    val_batch_seqs = val_batch_tokens // seq_len
    total_seqs = (val_tokens.size - 1) // seq_len
    total_batches = math.ceil(total_seqs / val_batch_seqs)

    # Shuffle with a fresh random seed each call
    rng = np.random.RandomState()
    seq_indices = np.arange(total_seqs)
    rng.shuffle(seq_indices)

    # Precompute token offsets for vectorized gather
    offsets = np.arange(seq_len)

    # Running accumulators
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    batch_bpbs = []
    ci_half = float("inf")

    pbar = tqdm(total=total_batches, desc=desc, leave=False)

    for batch_idx in range(total_batches):
        batch_start = batch_idx * val_batch_seqs
        batch_end = min(batch_start + val_batch_seqs, total_seqs)
        batch_seq_ids = seq_indices[batch_start:batch_end]

        # Vectorized gather of non-contiguous sequences
        x_starts = batch_seq_ids[:, None] * seq_len + offsets[None, :]
        x_np = val_tokens[x_starts]
        y_np = val_tokens[x_starts + 1]

        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)

        chunk_token_count = float(y.size)
        batch_loss = float(loss_fn(x, y).item())

        # Byte counting
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        batch_bytes = float(bytes_np.astype(np.float64).sum())

        # Accumulate
        total_loss_sum += batch_loss * chunk_token_count
        total_tokens += chunk_token_count
        total_bytes += batch_bytes

        # Per-batch BPB for CI
        batch_bpb = (batch_loss / math.log(2.0)) * (chunk_token_count / batch_bytes)
        batch_bpbs.append(batch_bpb)

        # Compute running BPB and CI
        n = len(batch_bpbs)
        running_bpb = (total_loss_sum / total_tokens / math.log(2.0)) * (total_tokens / total_bytes)
        if n >= 2:
            arr = np.array(batch_bpbs)
            ci_half = 1.96 * arr.std(ddof=1) / math.sqrt(n)
            pbar.set_postfix_str(f"bpb={running_bpb:.4f} ci95=±{ci_half:.4f}")
        else:
            pbar.set_postfix_str(f"bpb={running_bpb:.4f}")

        pbar.update(1)

        # Check 95% CI after minimum batches
        if n >= min_batches and ci_half < ci_threshold:
            pbar.close()
            val_loss = total_loss_sum / total_tokens
            return val_loss, running_bpb, ci_half, n, total_batches

    # Exhausted all batches
    pbar.close()
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    ci_half = 1.96 * np.array(batch_bpbs).std(ddof=1) / math.sqrt(len(batch_bpbs))
    return val_loss, val_bpb, ci_half, len(batch_bpbs), total_batches


def clip_grad_tree(grads_tree, max_norm):
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(np.array(grad.astype(mx.float32), dtype=np.float32, copy=False)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main():
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"

    def log(msg, console=True):
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    # Tokenizer setup
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")

    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path, args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

    # Model setup
    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # μP: compute width multiplier and LR scale
    mup_width_mult = 1.0
    mup_lr_scale = 1.0
    if args.mup_base_dim > 0:
        mup_width_mult = args.model_dim / args.mup_base_dim
        mup_lr_scale = args.mup_base_dim / args.model_dim  # hidden LR ∝ 1/width
        log(f"muP enabled: base_dim={args.mup_base_dim} width_mult={mup_width_mult:.2f} lr_scale={mup_lr_scale:.4f}")

    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        mup_width_mult=mup_width_mult,
        activation=args.activation,
    )
    opt = SplitOptimizers(model, args, mup_lr_scale=mup_lr_scale)

    compiled_loss_raw = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    _eval_pbar = None
    def compiled_loss(x, y):
        loss = compiled_loss_raw(x, y)
        mx.eval(loss)
        if _eval_pbar is not None:
            _eval_pbar.update(1)
        return loss
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    # FLOPs per training step: 6 * N * batch_tokens (fwd + bwd ≈ 3x fwd, each fwd ≈ 2*N*tokens)
    flops_per_step = 6 * n_params * args.train_batch_tokens
    log(f"run_id:{args.run_id}")
    log(f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len}")
    log(f"flops_per_step:{flops_per_step:.3e}")
    log(f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} "
        f"grad_accum_steps:{args.grad_accum_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.0f}")
    log(f"time_budget:{TIME_BUDGET}s artifact_limit:{ARTIFACT_SIZE_LIMIT} bytes")

    # Scaling CSV: append (run_id, step, N, D, C, wall_s, val_bpb) for scaling law fitting
    scaling_csv_path = Path(args.scaling_csv) if args.scaling_csv else None
    if scaling_csv_path:
        scaling_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not scaling_csv_path.exists():
            with scaling_csv_path.open("w", newline="") as f:
                csv.writer(f).writerow([
                    "run_id", "step", "n_params", "tokens_seen", "flops",
                    "wall_seconds", "val_bpb", "model_dim", "num_layers",
                ])

    # Warmup (compile MLX graphs)
    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime eval graph
        val_batch_tokens = args.val_batch_size
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        warm_val_loss = compiled_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # Training loop
    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0

    def _log_scaling_point(step_num, vbpb, wall_s):
        """Append a scaling data point to the CSV if enabled."""
        if scaling_csv_path is None:
            return
        tokens_seen = step_num * args.train_batch_tokens
        flops = step_num * flops_per_step
        with scaling_csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                args.run_id, step_num, n_params, tokens_seen, f"{flops:.6e}",
                f"{wall_s:.1f}", f"{vbpb:.6f}", args.model_dim, args.num_layers,
            ])

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        do_scaling_eval = (args.scaling_eval_every > 0 and step > 0
                           and step % args.scaling_eval_every == 0 and not last_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0) or do_scaling_eval:
            val_batch_tokens = args.val_batch_size
            val_loss, val_bpb, ci_half, n_used, n_total = evaluate_bpb_strided(
                compiled_loss, val_tokens, args.train_seq_len, val_batch_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                ci_threshold=args.eval_ci_threshold, min_batches=args.eval_min_batches,
                desc="eval",
            )
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            if step % 25 == 0 or last_step or do_scaling_eval:
                log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"bpb_ci95:±{ci_half:.4f} eval_batches:{n_used}/{n_total} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms")
            _log_scaling_point(step, val_bpb, train_time_ms / 1000.0)
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}")
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # Final serialization + quantized roundtrip eval
    t_end = time.perf_counter()
    total_seconds = train_time_ms / 1000.0

    flat_state = {k: v for k, v in tree_flatten(model.state)}
    quant_blob, artifact_bytes, code_bytes, model_bytes, quant_stats = compress_artifact(flat_state, __file__)
    quant_path = out_dir / f"{args.run_id}_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)

    # Roundtrip eval: load dequantized model and re-evaluate
    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    val_batch_tokens = args.val_batch_size
    q_val_loss, q_val_bpb, q_ci_half, q_n_used, q_n_total = evaluate_bpb_strided(
        compiled_loss, val_tokens, args.train_seq_len, val_batch_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        ci_threshold=args.eval_ci_threshold, min_batches=args.eval_min_batches,
        desc="eval (quantized)",
    )

    fits = "PASS" if artifact_bytes <= ARTIFACT_SIZE_LIMIT else "FAIL"

    log(f"eval (quantized): bpb={q_val_bpb:.4f} ci95=±{q_ci_half:.4f} batches={q_n_used}/{q_n_total}")

    # Sliding window eval (if enabled)
    sw_val_bpb = 0.0
    sw_q_val_bpb = 0.0
    if args.eval_stride > 0:
        # Re-load unquantized model for sliding window eval
        model.update(tree_unflatten(list(flat_state.items())))
        _, sw_val_bpb, sw_n_scored = evaluate_bpb_sliding_window(
            model, val_tokens, args.train_seq_len, args.eval_stride,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            desc="eval (sliding window)",
        )
        log(f"eval (sliding window): bpb={sw_val_bpb:.4f} tokens_scored={sw_n_scored}")

        # Also eval quantized model with sliding window
        model.update(tree_unflatten(list(quant_flat.items())))
        _, sw_q_val_bpb, sw_q_n_scored = evaluate_bpb_sliding_window(
            model, val_tokens, args.train_seq_len, args.eval_stride,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            desc="eval (quantized, sliding window)",
        )
        log(f"eval (quantized, sliding window): bpb={sw_q_val_bpb:.4f} tokens_scored={sw_q_n_scored}")

    # Print summary in grep-friendly format
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"val_bpb_ci95:     ±{ci_half:.6f}")
    print(f"val_bpb_quant:    {q_val_bpb:.6f}")
    print(f"val_bpb_quant_ci: ±{q_ci_half:.6f}")
    if args.eval_stride > 0:
        print(f"val_bpb_sw:       {sw_val_bpb:.6f}")
        print(f"val_bpb_quant_sw: {sw_q_val_bpb:.6f}")
        print(f"eval_stride:      {args.eval_stride}")
    print(f"artifact_bytes:   {artifact_bytes}")
    print(f"artifact_check:   {fits} ({artifact_bytes}/{ARTIFACT_SIZE_LIMIT})")
    print(f"model_bytes:      {model_bytes}")
    print(f"code_bytes:       {code_bytes}")
    print(f"training_seconds: {total_seconds:.1f}")
    print(f"total_flops:      {step * flops_per_step:.6e}")
    print(f"total_tokens_M:   {step * args.train_batch_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {n_params}")
    print(f"num_layers:       {args.num_layers}")
    print(f"model_dim:        {args.model_dim}")


if __name__ == "__main__":
    main()
