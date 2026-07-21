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


def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """logits: (1, vocab). Greedy if temperature == 0, else temperature +
    nucleus (top-p) sampling."""
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
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
def generate(model, prompt_ids, max_new_tokens, temperature, top_p, device, stop_ids):
    """Autoregressive generation with a KV cache. The cache only stores the
    n_kv_heads (not n_heads) K/V tensors per layer, exploiting GQA to cut
    cache memory by n_heads/n_kv_heads (4x for the default config)."""
    config = model.config
    head_dim = config.dim // config.n_heads
    kv_caches = [
        KVCache(1, config.max_seq_len, config.n_kv_heads, head_dim, device, torch.bfloat16)
        for _ in range(config.n_layers)
    ]

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    dtype_ctx = torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)

    with dtype_ctx:
        logits, _ = model(tokens, kv_caches=kv_caches, start_pos=0)
    start_pos = tokens.size(1)
    next_logits = logits[:, -1, :].float()

    generated = []
    for _ in range(max_new_tokens):
        next_id = sample_token(next_logits, temperature, top_p)
        tok = next_id.item()
        if tok in stop_ids:
            break
        generated.append(tok)
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
    gen_ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p, device, stop_ids)
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
    # raw completion mode
    parser.add_argument("--prompt", type=str, default=None)
    # chat mode
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        gen_ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p, device, stop_ids)
        print(prompt + tokenizer.decode(gen_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
