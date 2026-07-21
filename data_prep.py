"""
data_prep.py — stream HuggingFaceFW/fineweb-edu (sample-100BT), tokenize with
the SmolLM2 tokenizer, and pack tokens end-to-end into flat uint16 .bin
shards of ~100M tokens each.

Usage:
    python data_prep.py --out_dir data/fineweb_edu --target_tokens 55_000_000_000 \
        --num_workers 96

Design notes:
  - We pack tokens *end-to-end* (no per-document padding) with a single EOS
    token appended after each document as a separator. This is the standard
    "packed" pretraining format (nanoGPT / llm.c style) that maximizes token
    utilization per batch.
  - Tokenization is parallelized across a process pool; each worker keeps its
    own tokenizer instance (initialized once via the pool initializer) so we
    don't pay Python-GIL / pickling overhead per document.
  - Shards are plain flat uint16 arrays, so they can later be read back with
    a zero-copy np.memmap (see data.py) — no framework-specific format.
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

_tokenizer = None
_eos_id = None


def _init_worker(tokenizer_name: str):
    global _tokenizer, _eos_id
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # We tokenize whole documents with no truncation and pack them end-to-end
    # ourselves, so the tokenizer's default model_max_length (its notion of a
    # model's context window) is irrelevant here. Raise it so HF stops
    # emitting a "sequence length is longer than..." warning per long doc.
    _tokenizer.model_max_length = int(1e12)
    _eos_id = _tokenizer.eos_token_id
    assert _eos_id is not None, "tokenizer has no eos_token_id"


def _tokenize(example) -> np.ndarray:
    global _tokenizer, _eos_id
    ids = _tokenizer.encode(example["text"], add_special_tokens=False)
    ids.append(_eos_id)
    return np.array(ids, dtype=np.uint16)


def assert_vocab_fits_uint16(vocab_size: int):
    assert vocab_size <= 65535, (
        f"tokenizer vocab size {vocab_size} does not fit in uint16 (max 65535). "
        "NOTE: gated tokenizers like meta-llama/Llama-3.2-1B have ~128k tokens, "
        "which does NOT satisfy this constraint out of the box. Either (a) use "
        "a smaller-vocab tokenizer, or (b) change the .bin dtype to uint32 "
        "everywhere in this file and in data.py (doubles on-disk size, removes "
        "this limit)."
    )


def write_shard(buf: np.ndarray, count: int, out_dir: str, shard_idx: int) -> str:
    path = os.path.join(out_dir, f"fineweb_edu_{shard_idx:06d}.bin")
    buf[:count].tofile(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default="sample-100BT")
    parser.add_argument("--out_dir", type=str, default="data/fineweb_edu")
    parser.add_argument("--shard_size", type=int, default=100_000_000, help="tokens per shard")
    parser.add_argument("--target_tokens", type=int, default=55_000_000_000)
    parser.add_argument("--num_workers", type=int, default=96)
    parser.add_argument("--chunksize", type=int, default=64, help="pool.imap chunksize")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    assert_vocab_fits_uint16(vocab_size)
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None

    print(f"tokenizer={args.tokenizer} vocab_size={vocab_size} eos_id={eos_id}")
    print(f"streaming dataset={args.dataset} subset={args.subset}")

    ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)

    buffer = np.empty(args.shard_size, dtype=np.uint16)
    buf_idx = 0
    shard_idx = 0
    total_written = 0
    docs_seen = 0

    with mp.Pool(args.num_workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for ids in pool.imap(_tokenize, ds, chunksize=args.chunksize):
            docs_seen += 1
            pos = 0
            n = len(ids)
            while pos < n:
                space = args.shard_size - buf_idx
                take = min(space, n - pos)
                buffer[buf_idx:buf_idx + take] = ids[pos:pos + take]
                buf_idx += take
                pos += take
                if buf_idx == args.shard_size:
                    path = write_shard(buffer, buf_idx, args.out_dir, shard_idx)
                    total_written += buf_idx
                    print(f"[shard {shard_idx}] wrote {buf_idx:,} tokens -> {path} "
                          f"(total {total_written:,}/{args.target_tokens:,}, docs {docs_seen:,})")
                    shard_idx += 1
                    buf_idx = 0
            if total_written + buf_idx >= args.target_tokens:
                break

    if buf_idx > 0:
        path = write_shard(buffer, buf_idx, args.out_dir, shard_idx)
        total_written += buf_idx
        print(f"[shard {shard_idx}] wrote final partial shard with {buf_idx:,} tokens -> {path}")
        shard_idx += 1

    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "dtype": "uint16",
        "shard_size": args.shard_size,
        "num_shards": shard_idx,
        "total_tokens": total_written,
        "dataset": args.dataset,
        "subset": args.subset,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done. {shard_idx} shards, {total_written:,} tokens total. meta.json written.")


if __name__ == "__main__":
    main()
