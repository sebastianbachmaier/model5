"""
export_hf.py — convert one of our checkpoints (train.py's or sft.py's, same
{"model", "config", "step"} schema) into a standard HuggingFace
LlamaForCausalLM checkpoint directory, as a stepping stone to GGUF.

Why via HF Llama and not a direct GGUF writer: our architecture already
matches HF's LlamaForCausalLM exactly (RMSNorm, rotate-half RoPE, GQA,
SwiGLU, tied embeddings use the same conventions) so no weight permutation
is needed -- we just copy tensors into the equivalent HF module names. Once
in HF format, llama.cpp's own `convert_hf_to_gguf.py` handles the GGUF
tensor layout and tokenizer (vocab/merges/special tokens) export, which is
much more reliable than reimplementing that by hand.

Usage:
    python export_hf.py --checkpoint checkpoints/sft/sft_final.pt \
        --out_dir hf_export/model3-sft --chat

    # then, in a clone of https://github.com/ggml-org/llama.cpp :
    python convert_hf_to_gguf.py /path/to/hf_export/model3-sft \
        --outfile model3.gguf --outtype f16
    # optionally quantize:
    ./llama-quantize model3.gguf model3-q4_k_m.gguf Q4_K_M
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from chat_template import add_special_tokens, EOT
from model import ModelConfig, swiglu_hidden_dim

# Jinja equivalent of chat_template.py's render_chat/render_message, so
# LM Studio / llama.cpp (which apply the GGUF-embedded chat_template to
# format the conversation) wrap turns exactly the way sft.py trained on --
# otherwise they fall back to a generic default template that doesn't match
# what the model actually saw, causing garbled/self-chatting output.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'system' -%}"
    "{{ '<|system|>\n' + message['content'] + '<|eot|>\n' }}"
    "{%- elif message['role'] == 'user' -%}"
    "{{ '<|user|>\n' + message['content'] + '<|eot|>\n' }}"
    "{%- elif message['role'] == 'tool' -%}"
    "{{ '<|tool|>\n' + message['content'] + '<|eot|>\n' }}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{%- set body = message['content'] or '' -%}"
    "{%- if message.get('tool_calls') -%}"
    "{%- for tc in message['tool_calls'] -%}"
    "{%- set body = body + ('\n' if body else '') + '<tool_call>' "
    "+ ({'name': tc['function']['name'], 'arguments': tc['function']['arguments']} | tojson) "
    "+ '</tool_call>' -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{{ '<|assistant|>\n' + body + '<|eot|>\n' }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{ '<|assistant|>\n' }}"
    "{%- endif -%}"
)


def build_hf_config(config: ModelConfig) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.dim,
        intermediate_size=swiglu_hidden_dim(config.dim, config.multiple_of),
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=config.n_kv_heads,
        max_position_embeddings=config.max_seq_len,
        rms_norm_eps=config.norm_eps,
        rope_theta=config.rope_theta,
        tie_word_embeddings=True,
        hidden_act="silu",
    )


def convert_state_dict(sd: dict, n_layers: int) -> dict:
    out = {}
    out["model.embed_tokens.weight"] = sd["tok_embeddings.weight"]
    out["model.norm.weight"] = sd["norm.weight"]
    for i in range(n_layers):
        p = f"layers.{i}."
        q = f"model.layers.{i}."
        out[q + "self_attn.q_proj.weight"] = sd[p + "attention.wq.weight"]
        out[q + "self_attn.k_proj.weight"] = sd[p + "attention.wk.weight"]
        out[q + "self_attn.v_proj.weight"] = sd[p + "attention.wv.weight"]
        out[q + "self_attn.o_proj.weight"] = sd[p + "attention.wo.weight"]
        out[q + "mlp.gate_proj.weight"] = sd[p + "feed_forward.w1.weight"]
        out[q + "mlp.up_proj.weight"] = sd[p + "feed_forward.w3.weight"]
        out[q + "mlp.down_proj.weight"] = sd[p + "feed_forward.w2.weight"]
        out[q + "input_layernorm.weight"] = sd[p + "attention_norm.weight"]
        out[q + "post_attention_layernorm.weight"] = sd[p + "ffn_norm.weight"]
    # lm_head is tied (tie_word_embeddings=True) -- HF re-ties it from
    # embed_tokens automatically, so it doesn't need to be in the state dict.
    return out


def _fix_tokenizer_config(out_dir: str):
    """Work around a currently-unpatched transformers round-trip bug
    (huggingface/transformers#47110): save_pretrained() can write a
    new-style "extra_special_tokens" key as a bare list, but that same
    version's loader only accepts a dict there and crashes with
    `AttributeError: 'list' object has no attribute 'keys'`. The
    "additional_special_tokens" key (legacy mechanism, what actually matters
    for our chat tokens) is written separately and loads fine, so the
    list-shaped "extra_special_tokens" key is redundant -- drop it."""
    path = os.path.join(out_dir, "tokenizer_config.json")
    with open(path) as f:
        cfg = json.load(f)
    if isinstance(cfg.get("extra_special_tokens"), list):
        print("[info] removing buggy list-format 'extra_special_tokens' key from "
              "tokenizer_config.json (see huggingface/transformers#47110)")
        del cfg["extra_special_tokens"]
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--chat", action="store_true",
                         help="pass this for SFT checkpoints so the chat special tokens "
                              "(<|system|> etc.) are registered on the exported tokenizer, "
                              "matching the checkpoint's (possibly resized) vocab_size")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ModelConfig.from_dict(ckpt["config"])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.chat:
        add_special_tokens(tokenizer)
        tokenizer.chat_template = CHAT_TEMPLATE
    assert len(tokenizer) == config.vocab_size, (
        f"tokenizer size {len(tokenizer)} != checkpoint vocab_size {config.vocab_size} "
        f"-- pass --chat if this is an SFT checkpoint (adds the chat special tokens)"
    )

    hf_config = build_hf_config(config)
    model = LlamaForCausalLM(hf_config)
    state_dict = convert_state_dict(ckpt["model"], config.n_layers)
    missing, unexpected = model.model.load_state_dict(
        {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}, strict=True
    )
    assert not missing and not unexpected, (missing, unexpected)

    if args.chat:
        # The model was only ever trained to emit <|eot|> at the end of a
        # turn -- the base tokenizer's <|endoftext|> is never produced mid
        # -conversation. Registering both as EOS in generation_config (the
        # same mechanism Llama-3 uses for <|end_of_text|> + <|eot_id|>) is
        # correct for plain-transformers consumers of this checkpoint.
        #
        # BUT llama.cpp's convert_hf_to_gguf.py never reads
        # generation_config.json at all -- its SpecialVocab loader only
        # looks at config.json / tokenizer_config.json, and only accepts a
        # single int there (a list is silently dropped). So for GGUF/LM
        # Studio to actually stop at <|eot|>, the *single* eos_token has to
        # be <|eot|> itself, in both places that loader checks.
        eot_id = tokenizer.convert_tokens_to_ids(EOT)
        base_eos_id = tokenizer.eos_token_id
        tokenizer.eos_token = EOT  # -> tokenizer_config.json eos_token
        model.config.eos_token_id = eot_id  # -> config.json eos_token_id
        model.generation_config.eos_token_id = [base_eos_id, eot_id]

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    _fix_tokenizer_config(args.out_dir)
    print(f"saved HF checkpoint -> {args.out_dir}")
    print("next: in a clone of https://github.com/ggml-org/llama.cpp, run")
    print(f"  python convert_hf_to_gguf.py {os.path.abspath(args.out_dir)} "
          f"--outfile model3.gguf --outtype f16")


if __name__ == "__main__":
    main()
