# ChrisGoesGolfing

Auto-iterative Parameter Golf research loop, adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) challenge.

An AI agent autonomously iterates on a small GPT model, trying to minimize bits-per-byte (BPB) on FineWeb while keeping the artifact under 16MB (int8+zlib compressed).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare.py            # downloads FineWeb data + tokenizer
```

## Usage

The agent reads `program.md` for instructions, then runs the RALPH loop:

1. Modify `train.py` (model architecture, hyperparameters, optimizer)
2. Commit changes
3. Run `python train.py > run.log 2>&1`
4. Parse results: `grep "^val_bpb:\|^artifact_bytes:" run.log`
5. Keep commit if val_bpb improved, else git reset
6. Repeat forever

## Files

- `prepare.py` — Fixed: data download, tokenizer, evaluation, quantization. **Do not modify.**
- `train.py` — Editable: model architecture, optimizer, hyperparameters, training loop.
- `program.md` — Agent instructions for the autonomous research loop.
- `analysis.ipynb` — Notebook for analyzing experiment results.
