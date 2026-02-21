"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (Kimi For Coding via Anthropic format).
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "kimi-for-coding"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


# Kimi For Coding pricing (per 1M tokens) - as of 2025-02
# Format: (input_price, cached_price, output_price) in USD
KIMI_PRICING: Dict[str, Tuple[float, float, float]] = {
    "kimi-for-coding": (0.60, 0.06, 2.50),  # $0.60 input, $2.50 output per 1M
}


def fetch_kimi_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Return Kimi For Coding pricing.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    """
    return KIMI_PRICING.copy()


class LLMClient:
    """Kimi For Coding API wrapper (Anthropic format). All LLM calls go through this class."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.kimi.com/coding/",
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    def _calculate_cost(self, usage: Dict[str, Any], model: str) -> float:
        """Calculate cost from token usage using Kimi pricing."""
        pricing = KIMI_PRICING.get(model)
        if not pricing:
            log.debug(f"No pricing found for model {model}")
            return 0.0

        input_price, cached_price, output_price = pricing

        # Anthropic usage format
        prompt_tokens = int(usage.get("input_tokens", 0))
        completion_tokens = int(usage.get("output_tokens", 0))
        cached_tokens = int(usage.get("cache_read_input_tokens", 0))

        # Calculate cost per 1M tokens
        input_cost = (prompt_tokens - cached_tokens) * input_price / 1_000_000
        cached_cost = cached_tokens * cached_price / 1_000_000
        output_cost = completion_tokens * output_price / 1_000_000

        return round(input_cost + cached_cost + output_cost, 6)

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-format messages to Anthropic format."""
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Anthropic uses "user" and "assistant" roles
            if role == "system":
                # System messages handled separately in Anthropic
                continue
            elif role in ("user", "assistant"):
                anthropic_messages.append({"role": role, "content": content})
            else:
                anthropic_messages.append({"role": "user", "content": content})

        return anthropic_messages

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost)."""
        client = self._get_client()

        # Extract system message if present
        system_message = ""
        anthropic_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                anthropic_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        # Build kwargs
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }

        if system_message:
            kwargs["system"] = system_message

        if tools:
            # Convert OpenAI tools to Anthropic tools format
            anthropic_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    anthropic_tools.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    })
            kwargs["tools"] = anthropic_tools

        resp = client.messages.create(**kwargs)

        # Convert Anthropic response to OpenAI-like format for compatibility
        content_text = ""
        tool_calls = []

        for block in resp.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": str(block.input) if block.input else "{}",
                    }
                })

        msg = {
            "content": content_text,
            "role": "assistant",
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Build usage dict (OpenAI-compatible format)
        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_read_input_tokens": getattr(resp.usage, 'cache_read_input_tokens', 0),
        }

        # Calculate cost
        usage["cost"] = self._calculate_cost(usage, model)

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "kimi-for-coding",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        client = self._get_client()

        # Build content blocks for vision
        content_blocks = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": img["url"],
                    }
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": img["base64"],
                    }
                })

        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content_blocks}],
        )

        text = resp.content[0].text if resp.content else ""

        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cost": self._calculate_cost({
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }, model),
        }

        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "kimi-for-coding")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "kimi-for-coding")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
