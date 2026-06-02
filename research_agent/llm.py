"""Thin helpers around the Anthropic Messages API used by every LLM step.

Centralising the API calls here keeps the planning/synthesis/compose modules focused
on *what* to ask rather than the mechanics of calling the model and parsing replies.
We parse JSON ourselves (rather than relying on a specific SDK structured-output
helper) so the code runs on a wide range of SDK versions and is easy to follow.
"""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic


def complete_text(
    client: anthropic.Anthropic, *, model: str, system: str, user: str, max_tokens: int
) -> str:
    """Send one user message and return the assistant's reply as plain text."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def complete_json(
    client: anthropic.Anthropic, *, model: str, system: str, user: str, max_tokens: int
) -> Any:
    """Call the model and parse its reply as JSON.

    Raises ValueError if no JSON object/array can be found in the response.
    """
    raw = complete_text(client, model=model, system=system, user=user, max_tokens=max_tokens)
    return _parse_json(raw)


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
