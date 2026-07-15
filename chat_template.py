"""
chat_template.py — the ONE chat template shared by sft.py (training) and
generate.py (inference). Keeping this in a single module guarantees the
token sequence the model is trained on is exactly what generate.py builds at
inference time — a common source of silent SFT bugs is a template drift
between training and serving code.

Format per turn:
    <|system|>\n{content}<|eot|>\n
    <|user|>\n{content}<|eot|>\n
    <|assistant|>\n{content}\n<tool_call>{json}</tool_call><|eot|>\n
    <|tool|>\n{content}<|eot|>\n

Tool calls are serialized inline as a single-line JSON object
{"name": ..., "arguments": {...}} wrapped in <tool_call>...</tool_call>, so
next-token prediction learns to emit them like any other text.
"""
import json
import re
from typing import List, Dict, Tuple

SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
TOOL = "<|tool|>"
EOT = "<|eot|>"

# All role-marker tokens that must be registered as single, atomic tokenizer
# tokens (see add_special_tokens) so they can never be split by BPE and are
# stripped by tokenizer.decode(skip_special_tokens=True).
SPECIAL_TOKENS = [SYSTEM, USER, ASSISTANT, TOOL, EOT]


def add_special_tokens(tokenizer):
    """Register the role markers as additional special tokens. Idempotent —
    calling this again on a tokenizer that already has them is a no-op.
    Returns the number of tokens actually added (0 if already present)."""
    return tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})


def format_tool_call(tool_call: Dict) -> str:
    fn = tool_call["function"]
    args = fn["arguments"]
    if isinstance(args, str):
        # Some datasets store arguments as a JSON string rather than a dict.
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    payload = {"name": fn["name"], "arguments": args}
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def render_message(msg: Dict) -> Tuple[str, bool]:
    """Render one message dict to its template text. Returns
    (text, is_assistant) — is_assistant marks spans that should contribute
    to the SFT loss."""
    role = msg["role"]
    if role == "system":
        return f"{SYSTEM}\n{msg['content']}{EOT}\n", False
    if role == "user":
        return f"{USER}\n{msg['content']}{EOT}\n", False
    if role == "tool":
        # A tool's *result* is context the model reads, not something it
        # should learn to generate, so it is masked out like system/user.
        return f"{TOOL}\n{msg['content']}{EOT}\n", False
    if role == "assistant":
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        call_strs = [format_tool_call(c) for c in tool_calls]
        body = content
        if call_strs:
            body = (body + "\n" if body else "") + "\n".join(call_strs)
        return f"{ASSISTANT}\n{body}{EOT}\n", True
    raise ValueError(f"unknown message role: {role!r}")


def render_chat(messages: List[Dict], add_generation_prompt: bool = False) -> str:
    """Render a full conversation to a single string. If
    add_generation_prompt, appends a dangling '<|assistant|>\\n' so the model
    can be prompted to continue generating the next assistant turn."""
    parts = [render_message(m)[0] for m in messages]
    if add_generation_prompt:
        parts.append(f"{ASSISTANT}\n")
    return "".join(parts)


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> Tuple[str, List[Dict]]:
    """Inverse of format_tool_call: pull every <tool_call>{...}</tool_call>
    block out of generated text. Returns (remaining_text, tool_calls) where
    remaining_text has the tool-call spans removed and whitespace collapsed,
    and tool_calls is a list of {"name": ..., "arguments": ...} dicts (blocks
    that fail to parse as JSON are silently skipped, since a model can emit
    malformed JSON — callers should treat an empty list as "no tool call")."""
    tool_calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            tool_calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    remaining = _TOOL_CALL_RE.sub("", text).strip()
    return remaining, tool_calls
