"""
ChrisGoesGolfing — Parameter Golf training script (MLX, Apple Silicon).
This is the ONLY file the agent modifies. Everything else is in prepare.py.

Usage: python train.py > run.log 2>&1
"""
from __future__ import annotations

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
    evaluate_bpb,
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
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", TIME_BUDGET))

    # Model architecture
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 640))
    num_heads: int = int(os.environ.get("NUM_HEADS", 10))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 5))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))

    # Optimizer
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

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
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x):
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
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
                 logit_chunk_tokens, logit_softcap, rope_base, tied_embed_init_std, qk_gain_init):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
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
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
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
    def __init__(self, model, args):
        self.args = args
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
            learning_rate=args.tied_embed_lr,
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
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))
        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
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
    )
    opt = SplitOptimizers(model, args)

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
    log(f"run_id:{args.run_id}")
    log(f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len}")
    log(f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} "
        f"grad_accum_steps:{args.grad_accum_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.0f}")
    log(f"time_budget:{TIME_BUDGET}s artifact_limit:{ARTIFACT_SIZE_LIMIT} bytes")

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
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
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

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            val_batch_tokens = args.val_batch_size // args.grad_accum_steps
            val_batch_seqs = val_batch_tokens // args.train_seq_len
            total_val_batches = math.ceil((val_tokens.size - 1) // args.train_seq_len / max(val_batch_seqs, 1))
            _eval_pbar = tqdm(total=total_val_batches, desc="eval", leave=False)
            val_loss, val_bpb = evaluate_bpb(
                compiled_loss, val_tokens, args.train_seq_len, val_batch_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            _eval_pbar.close()
            _eval_pbar = None
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            if step % 25 == 0 or last_step:
                log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms")
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
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_val_batches = math.ceil((val_tokens.size - 1) // args.train_seq_len / max(val_batch_seqs, 1))
    _eval_pbar = tqdm(total=total_val_batches, desc="eval (quantized)", leave=False)
    q_val_loss, q_val_bpb = evaluate_bpb(
        compiled_loss, val_tokens, args.train_seq_len, val_batch_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    _eval_pbar.close()
    _eval_pbar = None

    fits = "PASS" if artifact_bytes <= ARTIFACT_SIZE_LIMIT else "FAIL"

    # Print summary in grep-friendly format
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"val_bpb_quant:    {q_val_bpb:.6f}")
    print(f"artifact_bytes:   {artifact_bytes}")
    print(f"artifact_check:   {fits} ({artifact_bytes}/{ARTIFACT_SIZE_LIMIT})")
    print(f"model_bytes:      {model_bytes}")
    print(f"code_bytes:       {code_bytes}")
    print(f"training_seconds: {total_seconds:.1f}")
    print(f"total_tokens_M:   {step * args.train_batch_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {n_params}")
    print(f"num_layers:       {args.num_layers}")
    print(f"model_dim:        {args.model_dim}")


if __name__ == "__main__":
    main()
