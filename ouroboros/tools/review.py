"""Multi-model review tool — sends code/text to multiple LLMs for consensus review.

Uses Kimi For Coding API (Anthropic format). Budget is tracked via llm_usage events.
"""

import os
import json
import asyncio
import logging

from ouroboros.utils import utc_now_iso
from ouroboros.tools.registry import ToolEntry, ToolContext
from ouroboros.llm import LLMClient


log = logging.getLogger(__name__)

# Maximum number of models allowed per review
MAX_MODELS = 5
# Concurrency limit for parallel requests
CONCURRENCY_LIMIT = 3


def get_tools():
    """Return list of ToolEntry for registry."""
    return [
        ToolEntry(
            name="multi_model_review",
            schema={
                "name": "multi_model_review",
                "description": (
                    "Send code or text to Kimi For Coding for review. "
                    "Returns structured verdict. Budget is tracked automatically."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The code or text to review",
                        },
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Review instructions — what to check for. "
                                "Fully specified by the LLM at call time."
                            ),
                        },
                    },
                    "required": ["content", "prompt"],
                },
            },
            handler=_handle_multi_model_review,
        )
    ]


def _handle_multi_model_review(ctx: ToolContext, content: str = "", prompt: str = "", **kwargs) -> str:
    """Sync wrapper around async review. Registry calls this."""
    # Ignore models parameter - we only use kimi-for-coding
    try:
        try:
            asyncio.get_running_loop()
            # Already in async context — run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(asyncio.run, _review_async(content, prompt, ctx)).result()
        except RuntimeError:
            # No running loop — safe to use asyncio.run directly
            result = asyncio.run(_review_async(content, prompt, ctx))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        log.error("Review failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Review failed: {e}"}, ensure_ascii=False)


async def _review_async(content: str, prompt: str, ctx: ToolContext):
    """Async orchestration: validate → query → parse → emit → return."""
    # Validation
    if not content:
        return {"error": "content is required"}
    if not prompt:
        return {"error": "prompt is required"}

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set"}

    # Build messages
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]

    # Query Kimi using LLMClient
    client = LLMClient(api_key=api_key)

    try:
        msg, usage = client.chat(
            messages=messages,
            model="kimi-for-coding",
            max_tokens=4096,
        )

        review_result = _parse_response(msg, usage)
        _emit_usage_event(review_result, ctx)

        return {
            "model_count": 1,
            "results": [review_result],
        }
    except Exception as e:
        return {"error": f"Review request failed: {e}"}


def _parse_response(msg: dict, usage: dict) -> dict:
    """Parse response into structured review_result dict."""
    text = msg.get("content", "")

    # Robust verdict parsing: check first 3 lines for PASS/FAIL anywhere (case-insensitive)
    verdict = "UNKNOWN"
    lines = text.split("\n")[:3]
    for line in lines:
        line_upper = line.upper()
        if "PASS" in line_upper:
            verdict = "PASS"
            break
        elif "FAIL" in line_upper:
            verdict = "FAIL"
            break

    return {
        "model": "kimi-for-coding",
        "verdict": verdict,
        "text": text,
        "tokens_in": usage.get("input_tokens", 0),
        "tokens_out": usage.get("output_tokens", 0),
        "cost_estimate": usage.get("cost", 0.0),
    }


def _emit_usage_event(review_result: dict, ctx: ToolContext) -> None:
    """Emit llm_usage event for budget tracking (for ALL cases, including errors)."""
    if ctx is None:
        return

    usage_event = {
        "type": "llm_usage",
        "ts": utc_now_iso(),
        "task_id": ctx.task_id if ctx.task_id else "",
        "usage": {
            "prompt_tokens": review_result["tokens_in"],
            "completion_tokens": review_result["tokens_out"],
            "cost": review_result["cost_estimate"],
        },
        "category": "review",
    }

    if ctx.event_queue is not None:
        try:
            ctx.event_queue.put_nowait(usage_event)
        except Exception:
            # Fallback to pending_events if queue fails
            if hasattr(ctx, "pending_events"):
                ctx.pending_events.append(usage_event)
    elif hasattr(ctx, "pending_events"):
        # No event_queue — use pending_events
        ctx.pending_events.append(usage_event)
