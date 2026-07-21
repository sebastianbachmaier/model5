"""
prepare_pretokenized_shards.py — download Andrej Karpathy's pre-tokenized
FineWeb-Edu shards (GPT-2 tokenizer, llm.c binary format) from
karpathy/fineweb-edu-100B-gpt2-token-shards and repackage them into the flat
uint16 .bin format (no header) that data.py expects, stopping once
--target_tokens tokens have been collected.

Why this exists: tokenizing ~55B tokens ourselves (see data_prep.py) is
CPU-bound and can take a long time / requires a beefy CPU box. This dataset
is the same underlying FineWeb-Edu corpus, already tokenized with GPT-2's
tokenizer, so we can just download and repackage instead of re-tokenizing.

llm.c shard format (see build-nanogpt / llm.c's data_common.h):
    256 x int32 header (1024 bytes) --
        header[0] = magic number 20240520
        header[1] = version
        header[2] = number of uint16 tokens that follow
    followed by that many uint16 tokens.

Usage:
    python prepare_pretokenized_shards.py --out_dir data/fineweb_edu_gpt2 \
        --target_tokens 55_000_000_000
"""
import argparse
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "karpathy/fineweb-edu-100B-gpt2-token-shards"
MAGIC = 20240520
HEADER_INT32S = 256
TOKENIZER = "gpt2"
VOCAB_SIZE = 50257
EOS_ID = 50256


def strip_header(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_INT32S * 4), dtype=np.int32)
        assert header[0] == MAGIC, f"bad magic number in {path}: {header[0]}"
        num_tokens = int(header[2])
        tokens = np.frombuffer(f.read(num_tokens * 2), dtype=np.uint16)
        assert len(tokens) == num_tokens, f"truncated shard {path}"
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/fineweb_edu_gpt2")
    parser.add_argument("--target_tokens", type=int, default=55_000_000_000)
    parser.add_argument("--cache_dir", type=str, default=None,
                         help="huggingface_hub cache dir for the raw downloaded shards")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(f for f in list_repo_files(REPO_ID, repo_type="dataset")
                    if f.endswith(".bin"))
    assert files, f"no .bin files found in {REPO_ID}"

    total_written = 0
    shard_idx = 0
    for fname in files:
        if total_written >= args.target_tokens:
            break

        out_path = os.path.join(args.out_dir, f"fineweb_edu_{shard_idx:06d}.bin")
        if os.path.exists(out_path):
            # Resumable: skip shards we already converted on a prior run.
            num_tokens = os.path.getsize(out_path) // 2
            total_written += num_tokens
            shard_idx += 1
            print(f"[shard {shard_idx - 1}] {out_path} already exists, skipping "
                  f"({num_tokens:,} tokens, total {total_written:,}/{args.target_tokens:,})")
            continue

        local_path = hf_hub_download(REPO_ID, fname, repo_type="dataset",
                                      cache_dir=args.cache_dir)
        tokens = strip_header(local_path)
        tokens.tofile(out_path)
        total_written += len(tokens)
        shard_idx += 1
        print(f"[shard {shard_idx - 1}] {fname} -> {out_path} "
              f"({len(tokens):,} tokens, total {total_written:,}/{args.target_tokens:,})")

    meta = {
        "tokenizer": TOKENIZER,
        "vocab_size": VOCAB_SIZE,
        "eos_id": EOS_ID,
        "dtype": "uint16",
        "num_shards": shard_idx,
        "total_tokens": total_written,
        "dataset": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-100BT",
        "source": REPO_ID,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done. {shard_idx} shards, {total_written:,} tokens total. meta.json written.")


if __name__ == "__main__":
    main()
