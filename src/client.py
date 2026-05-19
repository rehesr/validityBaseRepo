"""Thin OpenRouter client using the OpenAI SDK."""

import os
import time
from typing import Any

from openai import OpenAI

def call_model(
    model_id: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Call a model via OpenRouter and return a full log-friendly record."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.time() - t0) * 1000)
        usage = response.usage
        return {
            "model_id": model_id,
            "latency_ms": latency_ms,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "error": None,
            "response": response.model_dump(mode="json"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "model_id": model_id,
            "latency_ms": int((time.time() - t0) * 1000),
            "content": None,
            "finish_reason": None,
            "input_tokens": None,
            "output_tokens": None,
            "error": str(exc),
            "response": None,
        }
