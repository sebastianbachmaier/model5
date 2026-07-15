# ~0.5B Llama-style LM — pretrain, SFT, inference from scratch

A from-scratch (no `AutoModel`) implementation of a small Llama-style decoder-only
transformer: pretraining on FineWeb-Edu, instruction/tool-use SFT, and KV-cached
chat inference. Built for a single node of 8x H100 80GB, but every script also runs
on 1-2 GPUs (or CPU, for a quick correctness smoke test).

## Files

| File               | Purpose                                                              |
|--------------------|-----------------------------------------------------------------------|
| `model.py`         | The transformer itself: RMSNorm, RoPE, GQA attention, SwiGLU, KV cache |
| `data_prep.py`      | Streams FineWeb-Edu, tokenizes, packs into flat `uint16` shards        |
| `data.py`           | Memory-mapped shard reader, yields `(x, y)` windows, DDP-disjoint      |
| `train.py`          | Pretraining loop (DDP, bf16, `torch.compile`, cosine LR, grad accum)   |
| `sft.py`            | Instruction + tool-use fine-tuning with assistant-only loss masking    |
| `chat_template.py`  | The one shared chat template used by both `sft.py` and `generate.py`  |
| `generate.py`       | KV-cache autoregressive generation + chat mode + tool-call parsing    |

## Architecture

Llama-style decoder-only transformer, default config ≈ **506M parameters**:

- `dim=1280`, `n_layers=26`, `n_heads=20`, `n_kv_heads=5` (4x grouped-query attention)
- `max_seq_len=2048`, `vocab_size=32768`, RoPE `theta=10000`
- Pre-norm RMSNorm (computed in fp32), no biases anywhere
- SwiGLU FFN with hidden dim rounded to a multiple of 256 (3584 for `dim=1280`)
- Attention via `F.scaled_dot_product_attention(..., is_causal=True)` (flash-attention path)
- Token embedding tied with `lm_head`
- Scaled init: `wo`/`w2` (the two "output" projections per block) use
  `std = 0.02 / sqrt(2 * n_layers)`; everything else uses `std = 0.02`

Run `python model.py` to print the parameter count for any config:

```
total parameters: 506,333,440 (0.506B)
```

## ⚠️ Tokenizer / vocab-size note

The spec asks to reuse `meta-llama/Llama-3.2-1B`'s tokenizer *and* assert the vocab
fits in `uint16`, *and* use `vocab_size=32768` for the model. In reality the
Llama-3.2 tokenizer has **~128,256** tokens, which satisfies neither `vocab_size=32768`
nor the `uint16` (max 65,535) constraint. `data_prep.py` will fail loudly at
`assert_vocab_fits_uint16` if you point it at that tokenizer as-is. To actually run
this end-to-end you have two options:

1. **Use/train a smaller-vocab tokenizer** (e.g. a 32k BPE tokenizer) — then
   everything here works unmodified.
2. **Keep the Llama-3.2 tokenizer** and switch the on-disk dtype from `uint16` to
   `uint32` in `data_prep.py`/`data.py` (a one-line dtype change in both files),
   and pass `--vocab_size 128256` (or larger, rounded up) to `train.py`.

Everything downstream (`data.py`, `train.py`, `sft.py`, `generate.py`) reads
`vocab_size` from `data/.../meta.json` / the checkpoint's saved config, so once you
pick an option it's consistent everywhere automatically.

## 1. Data prep

```bash
python data_prep.py \
    --tokenizer meta-llama/Llama-3.2-1B \
    --out_dir data/fineweb_edu \
    --target_tokens 55_000_000_000 \
    --shard_size 100_000_000 \
    --num_workers 96
```

Streams `HuggingFaceFW/fineweb-edu` (`sample-100BT`), tokenizes with the HF fast
tokenizer, appends one EOS per document, and packs everything end-to-end into
`fineweb_edu_XXXXXX.bin` shards (~100M tokens / ~200MB each at uint16) plus a
`meta.json` recording vocab size / tokenizer / shard count. ~55B tokens is ~550
shards, roughly ~110GB on disk. Tokenizing 55B tokens with 96 CPU workers takes
on the order of half a day to a day depending on network/CPU (fineweb-edu documents
average a few hundred tokens; throughput is dominated by BPE encode + streaming
download bandwidth, not disk).

## 2. Pretraining

```bash
# smoke test on 1 GPU
python train.py --data_dir data/fineweb_edu --micro_bsz 8 --grad_accum 4 \
    --total_tokens 100_000_000 --log_interval 1 --ckpt_interval 50

# full run, 8x H100
torchrun --standalone --nproc_per_node=8 train.py \
    --data_dir data/fineweb_edu \
    --micro_bsz 16 --grad_accum 8 \
    --total_tokens 55_000_000_000 \
    --max_lr 4e-4 --min_lr 4e-5 --warmup_steps 2000 \
    --out_dir checkpoints/pretrain
```

With `micro_bsz=16`, `seq_len=2048`, `grad_accum=8`, `world_size=8`:
`tokens/step = 16 * 2048 * 8 * 8 = 2,097,152` (~2.1M, in the target 2-3M range).
55B tokens / 2.1M tokens-per-step ≈ **26,200 steps**.

**Rough compute estimate** (6·N·D FLOPs approximation, N=506M params, D=55B tokens):
`6 * 5.06e8 * 5.5e10 ≈ 1.7e20` FLOPs. At a (conservative, achievable-with-compile)
~400 TFLOP/s sustained per H100 in bf16, 8 GPUs ⇒ ~3.2 PFLOP/s aggregate ⇒
**~15 hours** wall clock for the full 55B-token run (higher MFU / larger batch
can bring this down; lower MFU or smaller batch will push it up — treat as an
order-of-magnitude estimate, not a guarantee).

Checkpoints save `model` + `optimizer` + `step` + `config` and are fully
resumable with `--resume path/to/ckpt.pt`.

## 3. Instruction + tool-use SFT

Prepare a JSONL file of `{"messages": [...]}` conversations (roles: `system`,
`user`, `assistant`, `tool`; assistant turns may include `tool_calls`). Suggested
mix (concatenate JSONL files in these rough proportions before training):

- **~60% general instruction data**: OpenHermes-2.5, Tulu-3 SFT mixture
- **~40% function-calling data**: Glaive-function-calling-v2,
  Hermes-Function-Calling, ToolACE, xLAM function-calling

```bash
torchrun --standalone --nproc_per_node=8 sft.py \
    --init_checkpoint checkpoints/pretrain/ckpt_final.pt \
    --tokenizer meta-llama/Llama-3.2-1B \
    --data data/sft/train.jsonl \
    --epochs 3 --lr 1e-5 --warmup_steps 100 \
    --out_dir checkpoints/sft
```

`sft.py` adds `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|eot|>` as
new special tokens, grows the (tied) embedding matrix to match, and masks the
loss (`-100`) on every token except assistant-turn tokens (including inline
`<tool_call>{...}</tool_call>` JSON), so the model only ever learns to *predict*
assistant output, never to reproduce system/user/tool text. SFT over a few hundred
thousand conversations on 8x H100 typically finishes in well under an hour; it is
tiny compared to pretraining.

## 4. Inference / chat

```bash
# raw next-token completion
python generate.py --checkpoint checkpoints/sft/sft_final.pt --prompt "Once upon a time"

# one-shot chat turn with a system prompt
python generate.py --checkpoint checkpoints/sft/sft_final.pt --chat \
    --system "You are a helpful assistant with access to tools." \
    --user "What's the weather in Paris?"

# interactive multi-turn chat
python generate.py --checkpoint checkpoints/sft/sft_final.pt --chat --interactive
```

Generation uses a KV cache that stores only the `n_kv_heads=5` (not `n_heads=20`)
K/V tensors per layer, exploiting GQA for a 4x smaller cache. In `--chat` mode,
`<tool_call>...</tool_call>` blocks are parsed out of the model's raw output into
structured `{"name": ..., "arguments": {...}}` dicts (see `chat_template.parse_tool_calls`)
so a caller can execute the tool and append a `{"role": "tool", "content": ...}`
message before calling again.

## Notes on correctness choices baked into the code

- **RoPE**: "rotate-half" (non-interleaved) convention — the head dim is split in
  half and rotated as a pair; equivalent to the complex-number formulation but
  simpler/faster on GPU.
- **GQA**: K/V projections only produce `n_kv_heads` heads; `repeat_kv` expands them
  to `n_heads` right before the attention call (not before caching), so the KV
  cache itself stays small.
- **DDP grad sync**: `model.require_backward_grad_sync` is only set `True` on the
  final microstep of each gradient-accumulation window, so NCCL all-reduce runs
  once per optimizer step instead of once per microbatch.
- **Checkpoint step semantics**: the saved `step` is the last *completed* step;
  resuming continues at `step + 1` (verified with a bit-identical-continuation test).
- **Loss masking**: pretraining's `(x, y)` are pre-shifted by the data loader, so
  `model.forward` never re-shifts internally. SFT follows the same convention —
  `y[i]` is the label for predicting the token *after* `x[i]`, masked to `-100`
  unless that target token belongs to an assistant turn.
