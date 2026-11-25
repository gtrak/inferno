#!/usr/bin/env python3
"""Regenerates the committed test fixtures.

Two things are produced, from two different sources:

1. `model/tokenizer.json` — a deliberately tiny *identity* tokenizer: one
   token per byte, ids equal to byte values, plus `<|im_start|>`/`<|im_end|>`.
   It is generated here from scratch, so the integration tests need no model files
   and every expected id is obvious by inspection (`"Hi"` is `[72, 105]`).

2. `chat_goldens.json` — the chat-template goldens. `rendered` is what Python
   `jinja2` produces from the model's own `chat_template.jinja` (the reference
   implementation), and `ids` is what the **real** `tokenizer.json` of
   `nvidia/Qwen3.8-Flash-Next-NVFP4` produces from that text. Both are copied
   verbatim into the JSON, so the Rust side is compared against the reference
   rather than against itself.

Usage (needs `jinja2` and `tokenizers` in the interpreter):

    python3 host/tests/fixtures/make_fixtures.py \
        --model-dir ~/models/qwen38-flash-next-nvfp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

FIXTURES = pathlib.Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# 1. the identity tokenizer (GPT-2's byte <-> printable-character mapping)
# --------------------------------------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable unicode mapping."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(printable, codepoints)}


def identity_tokenizer() -> dict:
    mapping = bytes_to_unicode()
    vocab = {character: byte for byte, character in mapping.items()}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 256,
                "content": "<|im_end|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 257,
                "content": "<|im_start|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [],
        },
    }


# --------------------------------------------------------------------------
# 2. the chat-template goldens
# --------------------------------------------------------------------------

SYSTEM = {"role": "system", "content": "You are a terse assistant."}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city, e.g. <Paris> & 'Lyon'",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "description": "Forecast length"},
            },
            "required": ["city"],
        },
    },
}

CASES = [
    {
        "name": "system-user-no-thinking",
        "messages": [SYSTEM, {"role": "user", "content": "What is the capital of France?"}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    {
        "name": "user-only-no-thinking",
        "messages": [{"role": "user", "content": "hi"}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    {
        "name": "user-only-thinking-default",
        "messages": [{"role": "user", "content": "hi"}],
        "add_generation_prompt": True,
    },
    {
        "name": "multi-turn-thinking",
        "messages": [
            SYSTEM,
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        "add_generation_prompt": True,
        "enable_thinking": True,
    },
    {
        "name": "content-parts",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one "},
                    {"type": "text", "text": "part two"},
                ],
            }
        ],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    {
        "name": "tools",
        "messages": [SYSTEM, {"role": "user", "content": "weather in Paris?"}],
        "tools": [WEATHER_TOOL],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    {
        "name": "tool-call-round-trip",
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                    }
                ],
            },
            {"role": "tool", "content": "18C"},
        ],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
    {
        "name": "assistant-reasoning",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning_content": "thought"},
        ],
        "add_generation_prompt": True,
        "enable_thinking": True,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=pathlib.Path.home() / "models/qwen38-flash-next-nvfp4",
        help="directory with tokenizer.json and chat_template.jinja",
    )
    arguments = parser.parse_args()

    tokenizer_path = arguments.model_dir / "tokenizer.json"
    template_path = arguments.model_dir / "chat_template.jinja"
    for path in (tokenizer_path, template_path):
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
            return 1

    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template_source = template_path.read_text()
    template = environment.from_string(template_source)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    goldens = {
        "provenance": {
            "reference_renderer": "python jinja2 ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)",
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "template_sha256": hashlib.sha256(template_source.encode()).hexdigest(),
            "encode": "tokenizer.encode(text, add_special_tokens=False)",
        },
        "cases": [],
    }
    for case in CASES:
        kwargs = {
            "messages": case["messages"],
            "add_generation_prompt": case["add_generation_prompt"],
        }
        if "tools" in case:
            kwargs["tools"] = case["tools"]
        if "enable_thinking" in case:
            kwargs["enable_thinking"] = case["enable_thinking"]
        rendered = template.render(**kwargs)
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        entry = dict(case)
        entry["rendered"] = rendered
        entry["ids"] = ids
        goldens["cases"].append(entry)

    model = FIXTURES / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "tokenizer.json").write_text(json.dumps(identity_tokenizer(), indent=2) + "\n")
    (model / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>", "add_bos_token": False}, indent=2) + "\n"
    )
    (model / "chat_template.jinja").write_text(template_source)
    (FIXTURES / "chat_goldens.json").write_text(json.dumps(goldens, indent=2) + "\n")

    print(f"wrote {len(goldens['cases'])} chat goldens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
