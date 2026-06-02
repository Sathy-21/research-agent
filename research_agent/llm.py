"""Thin helpers around the Groq chat-completions API used by every LLM step.

Centralising the API calls here keeps the planning/relevance/synthesis/compose/verify
modules focused on *what* to ask rather than the mechanics of calling the model and
parsing replies. The two public helpers keep the same signatures regardless of provider,
so swapping the LLM (Anthropic -> Gemini -> Groq) touches only this file plus config.

The JSON helper uses Groq's JSON mode (`response_format={"type": "json_object"}`), which
constrains the model to emit a single valid JSON object. We still run `_parse_json` over
the result as a defensive backstop.
"""

from __future__ import annotations

import json
import re
from typing import Any

from groq import Groq


def complete_text(
    client: Groq, *, model: str, system: str, user: str, max_tokens: int
) -> str:
    """Send one user message and return the model's reply as plain text."""
    return _generate(
        client, model=model, system=system, user=user, max_tokens=max_tokens, json_mode=False
    )


def complete_json(
    client: Groq, *, model: str, system: str, user: str, max_tokens: int
) -> Any:
    """Call the model (in JSON mode) and parse its reply as JSON.

    Raises ValueError if no JSON object/array can be found in the response.
    """
    raw = _generate(
        client, model=model, system=system, user=user, max_tokens=max_tokens, json_mode=True
    )
    return _parse_json(raw)


def _generate(
    client: Groq,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    json_mode: bool,
) -> str:
    """Make one Groq chat-completion request and return the reply text ("" if none)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    # JSON mode requires the word "JSON" to appear in the prompt; every JSON caller's
    # prompt already asks for JSON explicitly, so this is satisfied.
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    # content can be None if the model returns nothing; normalise to "" so callers can
    # degrade gracefully instead of hitting None.
    return (response.choices[0].message.content or "").strip()


def _parse_json(raw: str) -> Any:
    """Parse JSON from a model reply, tolerating prose or markdown fences around it."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} or [...] block in the text.
    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError("Model response did not contain valid JSON.")
