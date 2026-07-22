"""
sft.py — supervised fine-tuning (instruction + tool-use, no RL) on top of a
pretrained checkpoint.

Data format: JSONL, one conversation per line:
    {"messages": [{"role": "system", "content": "..."},
                  {"role": "user", "content": "..."},
                  {"role": "assistant", "content": "...",
                   "tool_calls": [{"type": "function",
                                    "function": {"name": "...", "arguments": {...}}}]},
                  {"role": "tool", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Suggested data mix (not enforced by this script — just concatenate JSONL
files in whatever ratio you like before pointing --data at them):
    ~60% general instruction-following: OpenHermes-2.5, Tulu-3 SFT mix
    ~40% function/tool-calling:         Glaive-function-calling-v2,
                                         Hermes-Function-Calling,
                                         ToolACE, xLAM function-calling
This mix teaches broad instruction following while still weighting tool-use
heavily enough that the small ~0.5B model reliably learns the <tool_call>
format instead of treating it as a rare edge case.

Launch:
    torchrun --standalone --nproc_per_node=8 sft.py \
        --init_checkpoint checkpoints/pretrain/ckpt_final.pt \
        --data data/sft/train.jsonl --out_dir checkpoints/sft
"""
import argparse
import hashlib
import json
import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from chat_template import add_special_tokens, render_message
from model import ModelConfig, Transformer, count_parameters


def build_example(messages, tokenizer, max_len, add_bos=True):
    """Tokenize a conversation turn-by-turn and track, per input token,
    whether it belongs to an assistant turn (render_message tells us this
    per-segment). We tokenize per-message rather than the whole string at
    once so we know exactly which resulting ids are "assistant" without
    needing an offset-mapping reconciliation step."""
    input_ids = []
    is_assistant = []
    if add_bos and tokenizer.bos_token_id is not None:
        input_ids.append(tokenizer.bos_token_id)
        is_assistant.append(False)
    for msg in messages:
        text, assistant_turn = render_message(msg)
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(ids)
        is_assistant.extend([assistant_turn] * len(ids))
    return input_ids[:max_len], is_assistant[:max_len]


def _cache_path(data_path, tokenizer, max_len):
    # Cache key covers everything that changes tokenization output: the
    # data file's own content (mtime+size is enough, cheap to check),
    # tokenizer vocab (len(tokenizer) captures added special tokens), and
    # max_len (truncation point).
    stat = os.stat(data_path)
    key = f"{data_path}:{stat.st_mtime_ns}:{stat.st_size}:{len(tokenizer)}:{max_len}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return f"{data_path}.tokcache_{digest}.pt"


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, cache_path=None, verbose=True):
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id
        if cache_path and os.path.exists(cache_path):
            if verbose:
                print(f"[sft data] loading cached tokenized dataset from {cache_path}")
            self.examples = torch.load(cache_path)
            return
        self.examples = []
        with open(path) as f:
            lines = f.readlines()
        iterator = tqdm(lines, desc="tokenizing SFT data", unit="convo") if verbose else lines
        for line in iterator:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids, is_assistant = build_example(obj["messages"], tokenizer, max_len)
            if any(is_assistant):  # skip conversations with no trainable tokens
                self.examples.append((ids, is_assistant))
        if cache_path:
            torch.save(self.examples, cache_path)
            if verbose:
                print(f"[sft data] cached tokenized dataset -> {cache_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, is_assistant = self.examples[idx]
        # Standard next-token teacher forcing: x predicts the token that
        # comes right after it, so y[i] = ids[i+1] (masked unless that
        # *target* token is part of an assistant turn) — same convention as
        # the packed pretraining windows in data.py (no extra shift needed
        # inside the model).
        x = ids[:-1]
        y = [ids[i] if is_assistant[i] else -100 for i in range(1, len(ids))]
        pad_len = (self.max_len - 1) - len(x)
        x = x + [self.pad_id] * pad_len
        y = y + [-100] * pad_len
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def get_lr(step, warmup_steps, max_steps, max_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * min(ratio, 1.0)))


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--data", type=str, required=True, help="JSONL file of {'messages': [...]}")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--micro_bsz", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--compile", dest="compile", action="store_true", default=True)
    parser.add_argument("--no_compile", dest="compile", action="store_false")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rank, local_rank, world_size, is_ddp = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    master = rank == 0
    torch.manual_seed(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_special_tokens(tokenizer)  # <|system|> <|user|> <|assistant|> <|tool|> <|eot|>

    ckpt = torch.load(args.init_checkpoint, map_location="cpu")
    config = ModelConfig.from_dict(ckpt["config"])
    model = Transformer(config)
    model.load_state_dict(ckpt["model"])
    if len(tokenizer) != config.vocab_size:
        # SFT added new special tokens on top of the pretrained tokenizer:
        # grow the (tied) embedding/output matrix to match, new rows get
        # freshly initialized (small) weights that fine-tuning will shape.
        if master:
            print(f"[info] resizing embeddings {config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    if master:
        print(f"model config: {model.config}")
        print(f"total parameters: {count_parameters(model):,}")

    raw_model = model
    if args.compile:
        model = torch.compile(model)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Tokenizing the whole file is CPU-bound and identical across ranks, so
    # only rank 0 does it (with a progress bar) and writes a cache keyed off
    # the data file + tokenizer + max_len; other ranks wait at the barrier
    # and then just load that cache instead of redundantly re-tokenizing.
    cache_path = _cache_path(args.data, tokenizer, args.max_len)
    if is_ddp:
        if master:
            SFTDataset(args.data, tokenizer, args.max_len, cache_path=cache_path, verbose=True)
        dist.barrier()
        dataset = SFTDataset(args.data, tokenizer, args.max_len, cache_path=cache_path, verbose=master)
    else:
        dataset = SFTDataset(args.data, tokenizer, args.max_len, cache_path=cache_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if is_ddp else None
    loader = DataLoader(
        dataset, batch_size=args.micro_bsz, sampler=sampler,
        shuffle=(sampler is None), drop_last=True, num_workers=2, pin_memory=True,
    )

    # No weight decay for SFT (per spec) — a single param group is enough.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.0, fused=torch.cuda.is_available(),
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    model.train()
    step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = get_lr(step, args.warmup_steps, total_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.log_interval == 0 and master:
                print(f"epoch {epoch} step {step:6d}/{total_steps} | loss {loss.item():.4f} | "
                      f"lr {lr:.2e} | grad_norm {grad_norm:.2f}")

            if step > 0 and step % args.ckpt_interval == 0 and master:
                path = os.path.join(args.out_dir, f"sft_ckpt_{step:06d}.pt")
                torch.save({"model": raw_model.state_dict(), "step": step, "config": raw_model.config.to_dict()}, path)
                print(f"saved checkpoint -> {path}")

            step += 1

    if master:
        path = os.path.join(args.out_dir, "sft_final.pt")
        torch.save({"model": raw_model.state_dict(), "step": step, "config": raw_model.config.to_dict()}, path)
        print(f"saved final checkpoint -> {path}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
