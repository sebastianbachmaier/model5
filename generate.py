"""
generate.py — autoregressive inference with a KV cache, temperature/top-p
sampling, and a chat mode that understands the same template used by sft.py
(including parsing <tool_call> blocks out of the model's output).

Examples:
    # raw completion
    python generate.py --checkpoint checkpoints/sft/sft_final.pt --prompt "Once upon a time"

    # one-shot chat turn
    python generate.py --checkpoint checkpoints/sft/sft_final.pt --chat \
        --system "You are a helpful assistant." --user "What's 12 * 7?"

    # interactive chat loop
    python generate.py --checkpoint checkpoints/sft/sft_final.pt --chat --interactive
"""
import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from chat_template import add_special_tokens, render_chat, parse_tool_calls, EOT
from model import KVCache, ModelConfig, Transformer


def load_model(checkpoint_path, device, tokenizer=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ModelConfig.from_dict(ckpt["config"])
    model = Transformer(config)
    model.load_state_dict(ckpt["model"])
    if tokenizer is not None and len(tokenizer) != config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    model.eval()
    return model


def apply_repetition_penalty(logits: torch.Tensor, seen_ids: set, penalty: float) -> torch.Tensor:
    """HF-style repetition penalty: shrink the logit of any token already seen
    in the prompt+generation so far towards zero (divide positive logits,
    multiply negative ones), discouraging the model from re-selecting it.
    penalty == 1.0 is a no-op. This directly counters the "answers the same
    question over and over" loop an undertrained model can fall into when it
    doesn't confidently predict the stop token."""
    if penalty == 1.0 or not seen_ids:
        return logits
    idx = torch.tensor(list(seen_ids), device=logits.device, dtype=torch.long)
    seen = logits[:, idx]
    logits[:, idx] = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int = 0) -> torch.Tensor:
    """logits: (1, vocab). Greedy if temperature == 0, else top-k (if top_k >
    0) then temperature + nucleus (top-p) sampling. top_k is applied first
    since it's a hard cap on how many candidates are ever considered --
    useful on a poorly-calibrated (e.g. undertrained) model whose
    distribution is flat enough that top-p alone still leaves a very long,
    low-probability tail available to sample from."""
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    # Keep the smallest set of tokens whose cumulative prob >= top_p; drop
    # everything after that (but always keep at least the top-1 token).
    drop_mask = (cum_probs - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(drop_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    next_in_sorted = torch.multinomial(sorted_probs, 1)
    return torch.gather(sorted_idx, -1, next_in_sorted)


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature, top_p, device, stop_ids, use_kv_cache=True,
             repetition_penalty=1.0, compute_dtype=torch.bfloat16, top_k=0):
    """Autoregressive generation. By default uses a KV cache (the n_kv_heads,
    not n_heads, K/V tensors per layer, exploiting GQA to cut cache memory by
    n_heads/n_kv_heads -- 4x for the default config). If use_kv_cache=False,
    instead recomputes the full sequence from scratch on every step (O(n^2),
    much slower) -- this exactly matches the numerics training uses and
    avoids small bf16 rounding differences between the cached single-token
    SDPA call and a full-sequence SDPA call. Those differences are usually
    negligible, but can get amplified by an early/undertrained model's less
    stable representations, degrading a cached checkpoint's generations far
    more than its no-cache/teacher-forced quality would suggest -- use
    use_kv_cache=False to sanity-check a checkpoint's real quality without
    that artifact, at the cost of much slower generation.

    compute_dtype controls both the autocast precision and the KV cache's
    storage dtype (previously hardcoded to bf16 regardless of what was
    requested here) -- pass torch.float32 to disable autocast entirely for
    the most numerically faithful (but slowest) comparison point, e.g. to
    check whether bf16 rounding is responsible for a quality gap vs. some
    other (e.g. fp16) inference engine on the same checkpoint."""
    config = model.config
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    autocast_enabled = compute_dtype != torch.float32
    dtype_ctx = torch.autocast(device_type=device_type, dtype=compute_dtype, enabled=autocast_enabled)
    seen_ids = set(prompt_ids)

    if not use_kv_cache:
        tokens = list(prompt_ids)
        generated = []
        for _ in range(max_new_tokens):
            if len(tokens) >= config.max_seq_len:
                break
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with dtype_ctx:
                logits, _ = model(input_ids, kv_caches=None, start_pos=0)
            next_logits = logits[:, -1, :].float()
            next_logits = apply_repetition_penalty(next_logits, seen_ids, repetition_penalty)
            next_id = sample_token(next_logits, temperature, top_p, top_k)
            tok = next_id.item()
            if tok in stop_ids:
                break
            generated.append(tok)
            tokens.append(tok)
            seen_ids.add(tok)
        return generated

    head_dim = config.dim // config.n_heads
    kv_caches = [
        KVCache(1, config.max_seq_len, config.n_kv_heads, head_dim, device, compute_dtype)
        for _ in range(config.n_layers)
    ]

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with dtype_ctx:
        logits, _ = model(tokens, kv_caches=kv_caches, start_pos=0)
    start_pos = tokens.size(1)
    next_logits = logits[:, -1, :].float()

    generated = []
    for _ in range(max_new_tokens):
        next_logits = apply_repetition_penalty(next_logits, seen_ids, repetition_penalty)
        next_id = sample_token(next_logits, temperature, top_p, top_k)
        tok = next_id.item()
        if tok in stop_ids:
            break
        generated.append(tok)
        seen_ids.add(tok)
        if start_pos >= config.max_seq_len:
            break  # ran out of cache room
        with dtype_ctx:
            logits, _ = model(next_id, kv_caches=kv_caches, start_pos=start_pos)
        next_logits = logits[:, -1, :].float()
        start_pos += 1

    return generated


def build_stop_ids(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids(EOT)
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        stop_ids.add(eot_id)
    return stop_ids


def chat_turn(model, tokenizer, messages, args, device, stop_ids):
    prompt_text = render_chat(messages, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    gen_ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p, device, stop_ids,
                        use_kv_cache=not args.no_kv_cache, repetition_penalty=args.repetition_penalty,
                        compute_dtype=args.compute_dtype, top_k=args.top_k)
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    content, tool_calls = parse_tool_calls(raw_text)
    return content, tool_calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0,
                         help="0 disables top-k. On a poorly-calibrated (e.g. undertrained) model, "
                              "top-p alone can still leave a very long low-probability tail available "
                              "to sample from -- a hard cap like 40-50 restricts candidates further.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                         help="HF-style repetition penalty (1.0 = disabled). Values like 1.1-1.3 "
                              "discourage the model from re-emitting already-seen tokens -- useful "
                              "for an undertrained checkpoint that loops/repeats instead of stopping.")
    # raw completion mode
    parser.add_argument("--prompt", type=str, default=None)
    # chat mode
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no_kv_cache", action="store_true",
                         help="recompute the full sequence from scratch every step instead of using a KV "
                              "cache (much slower, but matches training's numerics exactly -- useful for "
                              "sanity-checking an early/undertrained checkpoint whose cached generations "
                              "look worse than its real quality due to bf16 rounding differences)")
    parser.add_argument("--device", type=str, default=None,
                         help="e.g. cuda:0, cuda:3, cpu. Defaults to cuda (device 0) if available else cpu. "
                              "Useful to pin inference to a spare/idle GPU (or cpu) so it doesn't contend "
                              "with a training job already using the other GPUs.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                         help="compute precision for both autocast and the KV cache (training used bf16, "
                              "so that's the default, but bf16's low mantissa precision can amplify cached "
                              "-decoding numerical divergence on an undertrained checkpoint -- try fp32 "
                              "(slow but most faithful) or fp16 to check whether precision, not the "
                              "checkpoint itself, explains a quality gap vs. another inference engine)")
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    args.compute_dtype = dtype_map[args.dtype]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.chat:
        add_special_tokens(tokenizer)

    model = load_model(args.checkpoint, device, tokenizer=tokenizer if args.chat else None)
    stop_ids = build_stop_ids(tokenizer)

    if args.chat:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})

        def run_turn(user_text):
            messages.append({"role": "user", "content": user_text})
            content, tool_calls = chat_turn(model, tokenizer, messages, args, device, stop_ids)
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls or None})
            print(f"assistant: {content}")
            if tool_calls:
                print("tool_calls:", json.dumps(tool_calls, indent=2))
                print("(execute the tool(s) above, then feed results back as a "
                      "{'role': 'tool', 'content': ...} message and call again)")
            return content, tool_calls

        if args.interactive:
            print("interactive chat (Ctrl+C to exit)")
            try:
                if args.user:
                    run_turn(args.user)
                while True:
                    user_text = input("user: ")
                    run_turn(user_text)
            except (KeyboardInterrupt, EOFError):
                print()
        else:
            run_turn(args.user or "Hello!")
    else:
        prompt = args.prompt or "Once upon a time"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if tokenizer.bos_token_id is not None:
            prompt_ids = [tokenizer.bos_token_id] + prompt_ids
        gen_ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p, device, stop_ids,
                            use_kv_cache=not args.no_kv_cache, repetition_penalty=args.repetition_penalty,
                            compute_dtype=args.compute_dtype, top_k=args.top_k)
        print(prompt + tokenizer.decode(gen_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
