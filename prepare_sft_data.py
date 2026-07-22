"""
prepare_sft_data.py — download teknium/OpenHermes-2.5 (general instruction
conversations, no tool-calling) and convert it into the JSONL
{"messages": [...]} format that sft.py expects.

OpenHermes-2.5 is in ShareGPT format: each row has a "conversations" list of
{"from": "system"|"human"|"gpt", "value": "..."} turns. We map
system/human/gpt -> system/user/assistant and drop empty turns (many rows
have an empty "system" turn). No tool_calls are ever set, so every example
is plain conversation, matching sft.py's schema with the tool-call fields
simply omitted.

Usage:
    python prepare_sft_data.py --out_dir data/sft --val_fraction 0.01

    # quick smoke-test subset (e.g. to sanity-check the sft.py pipeline on
    # an in-progress pretraining checkpoint):
    python prepare_sft_data.py --out_dir data/sft_smoke --limit 2000
"""
import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm

REPO_ID = "teknium/OpenHermes-2.5"
ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


def convert(example):
    messages = []
    for turn in example["conversations"]:
        role = ROLE_MAP.get(turn["from"])
        content = (turn.get("value") or "").strip()
        if role is None or not content:
            # Unknown role (e.g. ShareGPT variants) or empty turn (many rows
            # have a blank "system" turn) contribute nothing to the prompt.
            continue
        messages.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" for m in messages):
        return None
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/sft")
    parser.add_argument("--limit", type=int, default=None,
                         help="only keep this many examples (after shuffling) -- useful for a quick smoke test")
    parser.add_argument("--val_fraction", type=float, default=0.01,
                         help="fraction of examples held out into val.jsonl")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true",
                         help="allow overwriting an existing train.jsonl/val.jsonl in --out_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    existing = [p for p in (train_path, val_path) if os.path.exists(p)]
    if existing and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing file(s): {existing}. "
            "Pass --overwrite to replace them, or use a different --out_dir."
        )

    print(f"downloading/loading {REPO_ID} (first run downloads ~2GB, cached after)...", flush=True)
    ds = load_dataset(REPO_ID, split="train")
    print(f"loaded {len(ds):,} raw conversations", flush=True)

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if args.limit is not None:
        indices = indices[:args.limit]

    n_val = int(len(indices) * args.val_fraction)
    val_indices = set(indices[:n_val])

    n_written, n_skipped = 0, 0
    with open(train_path, "w") as train_f, open(val_path, "w") as val_f:
        for i in tqdm(indices, desc="converting", unit="convo"):
            example = convert(ds[i])
            if example is None:
                n_skipped += 1
                continue
            out_f = val_f if i in val_indices else train_f
            out_f.write(json.dumps(example) + "\n")
            n_written += 1

    print(f"wrote {n_written - n_val:,} train / {n_val:,} val examples "
          f"({n_skipped:,} skipped, no assistant turns) -> {train_path}, {val_path}")


if __name__ == "__main__":
    main()
