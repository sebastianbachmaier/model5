"""
train.py — pretraining loop for the ~0.5B Llama-style model, using DDP,
bf16 autocast, and torch.compile.

Launch:
    # single GPU (smoke test)
    python train.py --data_dir data/fineweb_edu --micro_bsz 8 --grad_accum 4

    # 8x H100 node
    torchrun --standalone --nproc_per_node=8 train.py \
        --data_dir data/fineweb_edu --micro_bsz 16 --grad_accum 8
"""
import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist

from data import TokenDataset
from model import ModelConfig, Transformer, count_parameters


def setup_ddp():
    """torchrun sets RANK/LOCAL_RANK/WORLD_SIZE in the environment. Falls
    back to a single-process, single-GPU (or CPU) run otherwise."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    rank, local_rank, world_size = 0, 0, 1
    return rank, local_rank, world_size, False


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Linear warmup, then cosine decay to min_lr, held at min_lr after
    max_steps (in case total_steps estimate is slightly off)."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model, weight_decay, lr, betas, eps):
    """Weight decay only applies to 2D+ parameters (matmul weights); norms
    (1D RMSNorm weight) and embeddings... note embeddings are 2D but Llama-
    style implementations conventionally still decay them since they're a
    matmul weight (tied with lm_head, which we *do* want to regularize).
    Here we follow the common nanoGPT convention: decay everything with
    ndim >= 2, no decay for 1D params (biases/norms — we have no biases)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused AdamW is CUDA-only; fall back to the standard implementation on CPU.
    use_fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=use_fused)


def save_checkpoint(path, raw_model, optimizer, step, config: ModelConfig):
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--n_layers", type=int, default=26)
    parser.add_argument("--n_heads", type=int, default=20)
    parser.add_argument("--n_kv_heads", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    # data
    parser.add_argument("--data_dir", type=str, default="data/fineweb_edu")
    # optimization
    parser.add_argument("--micro_bsz", type=int, default=16, help="per-GPU micro batch size (in sequences)")
    parser.add_argument("--grad_accum", type=int, default=8, help="gradient accumulation steps")
    parser.add_argument("--total_tokens", type=int, default=55_000_000_000)
    parser.add_argument("--max_lr", type=float, default=4e-4)
    parser.add_argument("--min_lr", type=float, default=4e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    # logging / checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--resume", type=str, default=None)
    # misc
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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # If shards were prepped with a tokenizer whose vocab differs from the
    # --vocab_size default, trust the data's meta.json (the embedding table
    # must exactly match the token ids that show up in the .bin shards).
    meta_path = os.path.join(args.data_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta["vocab_size"] != args.vocab_size:
            if master:
                print(f"[warn] overriding --vocab_size {args.vocab_size} -> "
                      f"{meta['vocab_size']} from {meta_path}")
            args.vocab_size = meta["vocab_size"]

    config = ModelConfig(
        dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len, rope_theta=args.rope_theta,
    )

    model = Transformer(config).to(device)
    if master:
        n_params = count_parameters(model)
        print(f"model config: {config}")
        print(f"total parameters: {n_params:,} ({n_params / 1e9:.3f}B)")

    raw_model = model  # keep an un-wrapped handle for state_dict / clip_grad_norm_
    if args.compile:
        model = torch.compile(model)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = configure_optimizer(
        raw_model, args.weight_decay, args.max_lr, (args.beta1, args.beta2), args.eps
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # ckpt["step"] is the last *completed* step (its optimizer update has
        # already been applied), so resume one step past it or we'd redo it.
        start_step = ckpt["step"] + 1
        if master:
            print(f"resumed from {args.resume}, last completed step {ckpt['step']}, continuing at step {start_step}")

    dataset = TokenDataset(args.data_dir, args.max_seq_len, rank=rank, world_size=world_size)
    windows_per_step = args.micro_bsz * args.grad_accum
    dataset.set_position(start_step * windows_per_step)

    tokens_per_step = args.micro_bsz * args.max_seq_len * args.grad_accum * world_size
    total_steps = args.total_tokens // tokens_per_step

    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"world_size={world_size} tokens/step={tokens_per_step:,} total_steps={total_steps:,}")

    model.train()
    t_log = time.time()
    for step in range(start_step, total_steps):
        lr = get_lr(step, args.warmup_steps, total_steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros(1, device=device)
        for micro_step in range(args.grad_accum):
            x, y = dataset.next_batch(args.micro_bsz, device)
            if is_ddp:
                # Only all-reduce gradients on the last microstep of the
                # accumulation window — otherwise DDP would sync (and waste
                # NCCL bandwidth) after every single microbatch.
                model.require_backward_grad_sync = (micro_step == args.grad_accum - 1)
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss_accum += loss.detach()
            loss.backward()

        if is_ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0 and master:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t_log
            toks_per_sec = tokens_per_step * args.log_interval / dt if step > start_step else tokens_per_step / dt
            print(f"step {step:7d}/{total_steps} | loss {loss_accum.item():.4f} | "
                  f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | tok/s {toks_per_sec:,.0f}")
            t_log = time.time()

        if step > 0 and step % args.ckpt_interval == 0 and master:
            path = os.path.join(args.out_dir, f"ckpt_{step:07d}.pt")
            save_checkpoint(path, raw_model, optimizer, step, config)
            print(f"saved checkpoint -> {path}")

    if master:
        path = os.path.join(args.out_dir, "ckpt_final.pt")
        save_checkpoint(path, raw_model, optimizer, total_steps - 1, config)
        print(f"saved final checkpoint -> {path}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
