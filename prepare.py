"""
Fixed infrastructure for ChrisGoesGolfing Parameter Golf experiments.
Downloads data, provides tokenizer/dataloader, evaluation, and quantization.

Usage:
    python prepare.py                        # download default 10 training shards
    python prepare.py --train-shards 1       # smoke test with 1 shard
    python prepare.py --train-shards 80      # full dataset (8B tokens)

Data and tokenizer are stored in ./data/ relative to this file.

DO NOT MODIFY THIS FILE. The agent only modifies train.py.
"""

import glob
import json
import math
import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300            # training time budget in seconds (5 minutes per experiment)
ARTIFACT_SIZE_LIMIT = 16_000_000  # 16MB decimal, the Parameter Golf cap
DEFAULT_SEQ_LEN = 1024       # default sequence length
DEFAULT_VOCAB_SIZE = 1024    # default vocabulary size
DEFAULT_VARIANT = "sp1024"

# Paths relative to this file
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
TOKENIZERS_DIR = DATA_DIR / "tokenizers"
DEFAULT_DATA_PATH = str(DATASETS_DIR / "fineweb10B_sp1024")
DEFAULT_TOKENIZER_PATH = str(TOKENIZERS_DIR / "fineweb_1024_bpe.model")

COMPUTE_DTYPE = mx.bfloat16

# Quantization constants
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_shard(path: Path) -> np.ndarray:
    """Load a binary shard in the Parameter Golf format."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


class TokenStream:
    """Infinite streaming iterator over binary shard files."""

    def __init__(self, pattern, log_fn=None, dataset_name=""):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    """Wraps TokenStream into (input, target) batches for MLX."""

    def __init__(self, pattern, log_fn=None, dataset_name=""):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens, seq_len):
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def load_validation_tokens(pattern, seq_len):
    """Load all validation shard tokens into a contiguous array."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(f) for f in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


# ---------------------------------------------------------------------------
# Tokenizer / BPB evaluation utilities
# ---------------------------------------------------------------------------

def build_sentencepiece_luts(sp, vocab_size):
    """Build lookup tables for BPB (bits-per-byte) calculation."""
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path, tokenizer_path):
    """Validate that dataset and tokenizer are compatible via manifest."""
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_bpb(loss_fn, val_tokens, seq_len, val_batch_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """
    Compute val_loss and val_bpb (bits per byte) — the Parameter Golf metric.
    loss_fn: callable(x, y) -> scalar cross-entropy loss (mean reduction)
    Returns (val_loss, val_bpb).
    """
    val_batch_seqs = val_batch_tokens // seq_len
    total_seqs = (val_tokens.size - 1) // seq_len
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_seq_start in range(0, total_seqs, val_batch_seqs):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * seq_len
        raw_end = batch_seq_end * seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        total_loss = total_loss + loss_fn(x, y).astype(mx.float32) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
    total_loss = total_loss / total_tokens
    mx.eval(total_loss)
    val_loss = float(total_loss.item())
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


# ---------------------------------------------------------------------------
# Quantization (int8 + zlib) for 16MB artifact constraint
# ---------------------------------------------------------------------------

def _np_float32(arr):
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name, arr, passthrough_orig_dtypes):
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr):
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(flat_state):
    """Quantize model state dict to int8 for compression. Returns (quant_obj, stats)."""
    quantized = {}
    scales = {}
    dtypes = {}
    passthrough = {}
    passthrough_orig_dtypes = {}
    qmeta = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)

    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj):
    """Dequantize int8 state dict back to MLX arrays."""
    out = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def compress_artifact(flat_state, code_path=None):
    """
    Quantize, compress, and return artifact size.
    Returns (quant_blob, artifact_bytes, code_bytes, quant_stats).
    """
    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    model_bytes = len(quant_blob)
    code_bytes = 0
    if code_path is not None:
        code_bytes = Path(code_path).stat().st_size
    artifact_bytes = model_bytes + code_bytes
    return quant_blob, artifact_bytes, code_bytes, model_bytes, quant_stats


# ---------------------------------------------------------------------------
# Data download (one-time setup)
# ---------------------------------------------------------------------------

def download_data(variant="sp1024", train_shards=10):
    """Download FineWeb shards and tokenizer from HuggingFace."""
    from huggingface_hub import hf_hub_download

    REPO_ID = "willdepueoai/parameter-golf"
    REMOTE_ROOT_PREFIX = "datasets"

    def dataset_dir_for_variant(name):
        if name == "byte260":
            return "fineweb10B_byte260"
        if name.startswith("sp") and name[2:].isdigit():
            return f"fineweb10B_{name}"
        raise ValueError(f"unsupported variant {name!r}")

    def local_path_for_remote(relative_path):
        remote_path = Path(relative_path)
        if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
            remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
        if remote_path.parts[:1] == ("datasets",):
            return DATASETS_DIR.joinpath(*remote_path.parts[1:])
        if remote_path.parts[:1] == ("tokenizers",):
            return TOKENIZERS_DIR.joinpath(*remote_path.parts[1:])
        return DATA_DIR / remote_path

    def get_file(relative_path):
        import shutil
        destination = local_path_for_remote(relative_path)
        if destination.exists():
            return
        if destination.is_symlink():
            destination.unlink()
        remote_path = Path(relative_path)
        cached_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=remote_path.name,
                subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
                repo_type="dataset",
            )
        )
        cached_source = cached_path.resolve(strict=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(cached_source, destination)
        except OSError:
            shutil.copy2(cached_source, destination)
        print(f"  Downloaded {destination.name}")

    dataset_dir = dataset_dir_for_variant(variant)

    # Download manifest
    print("Downloading manifest...")
    get_file(f"{REMOTE_ROOT_PREFIX}/manifest.json")
    manifest_path = local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None)
    if dataset_entry is None:
        raise ValueError(f"dataset {dataset_dir} not found in manifest")

    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train"))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val"))
    if train_shards > max_train_shards:
        raise ValueError(f"{variant} only has {max_train_shards} training shards, requested {train_shards}")

    # Download tokenizer
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry:
        for key in ("model_path", "vocab_path", "path"):
            value = tokenizer_entry.get(key)
            if value:
                print(f"Downloading tokenizer: {value}")
                get_file(f"{REMOTE_ROOT_PREFIX}/{value}")

    # Download validation shards
    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{dataset_dir}"
    print(f"Downloading {val_shards} validation shards...")
    for i in range(val_shards):
        get_file(f"{dataset_prefix}/fineweb_val_{i:06d}.bin")

    # Download training shards
    print(f"Downloading {train_shards} training shards...")
    for i in range(train_shards):
        get_file(f"{dataset_prefix}/fineweb_train_{i:06d}.bin")

    print(f"\nData ready at {DATASETS_DIR / dataset_dir}")
    print(f"Tokenizer ready at {TOKENIZERS_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data for ChrisGoesGolfing experiments")
    parser.add_argument("--train-shards", type=int, default=10,
                        help="Number of training shards to download (default: 10)")
    parser.add_argument("--variant", default="sp1024",
                        help="Tokenizer variant (default: sp1024)")
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")
    download_data(variant=args.variant, train_shards=args.train_shards)
    print("\nDone! Ready to train.")
