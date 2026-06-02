"""Thin helpers around the Google Gemini API used by every LLM step.

Centralising the API calls here keeps the planning/synthesis/compose modules focused
on *what* to ask rather than the mechanics of calling the model and parsing replies.
The two public helpers keep the same signatures regardless of provider, so swapping
the LLM (as we did from Anthropic to Gemini) touches only this file plus config.

Thinking is disabled on every call. Gemini 2.5 Flash is a "thinking" model by
default, and thinking tokens are drawn from `max_output_tokens` — with a small budget
(e.g. the planner's) the model can spend the whole budget thinking and return no text.
Disabling it makes the token budget go entirely to the answer: predictable, faster,
and cheaper. (thinking_budget=0 is supported on Flash; it is not on Gemini 2.5 Pro.)
"""

from __future__ import annotations

import json
import re
from typing import Any

from google import genai
from google.genai import types


def complete_text(
    client: genai.Client, *, model: str, system: str, user: str, max_tokens: int
) -> str:
    """Send one user message and return the model's reply as plain text."""
    return _generate(
        client, model=model, system=system, user=user, max_tokens=max_tokens, json_mode=False
    )


def complete_json(
    client: genai.Client, *, model: str, system: str, user: str, max_tokens: int
) -> Any:
    """Call the model (asking for JSON) and parse its reply as JSON.

    Raises ValueError if no JSON object/array can be found in the response.
    """
    raw = _generate(
        client, model=model, system=system, user=user, max_tokens=max_tokens, json_mode=True
    )
    return _parse_json(raw)


def _generate(
    client: genai.Client,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    json_mode: bool,
) -> str:
    """Make one Gemini request and return the reply text (empty string if none)."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    response = client.models.generate_content(model=model, contents=user, config=config)
    # response.text is None when the model returns no text (e.g. a safety block);
    # normalise to "" so callers can degrade gracefully instead of hitting None.
    return (response.text or "").strip()


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
