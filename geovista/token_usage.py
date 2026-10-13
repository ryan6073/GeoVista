"""Split chat usage into visual / input-text / output-text tokens.

Fields (aligned with qwen2_5-RL/token_usage.py):
  visual_tokens: image tokens
  input_text_tokens: prompt text tokens (prompt_tokens - visual_tokens)
  output_text_tokens: completion / generated text tokens

Also keeps legacy aliases:
  generated_tokens = input_text_tokens + output_text_tokens
  total_tokens = generated_tokens + visual_tokens

vLLM often omits image_tokens, so visual tokens are recovered from the images
actually sent in the request. Fallback uses Qwen3-VL smart_resize defaults
(patch 16, merge 2) — GeoVista agent_zoom is Qwen3-VL based.
"""

from __future__ import annotations

import base64
import math
from io import BytesIO
from typing import Any

from PIL import Image

# Qwen3-VL preprocessor_config.json defaults.
_PATCH_SIZE = 16
_MERGE_SIZE = 2
_MIN_PIXELS = 65536
_MAX_PIXELS = 16777216


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen3_vl_visual_tokens(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return 0
    factor = _PATCH_SIZE * _MERGE_SIZE
    resized_h, resized_w = _smart_resize(
        height,
        width,
        factor,
        _MIN_PIXELS,
        _MAX_PIXELS,
    )
    grid_h = resized_h // _PATCH_SIZE
    grid_w = resized_w // _PATCH_SIZE
    return (grid_h * grid_w) // (_MERGE_SIZE ** 2)


# Back-compat alias
qwen25_vl_visual_tokens = qwen3_vl_visual_tokens


def _image_size_from_data_url(url: str) -> tuple[int, int] | None:
    if not url:
        return None
    try:
        payload = url.split(",", 1)[-1]
        img = Image.open(BytesIO(base64.b64decode(payload)))
        return img.size
    except Exception:
        return None


def visual_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    visual = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url") or {}
            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
            size = _image_size_from_data_url(url)
            if size is None:
                continue
            width, height = size
            visual += qwen3_vl_visual_tokens(width, height)
    return visual


def visual_tokens_from_usage(usage: dict[str, Any] | None) -> int:
    usage = usage or {}
    visual = _as_int(usage.get("visual_tokens")) or _as_int(usage.get("image_tokens"))
    details = usage.get("prompt_tokens_details") or {}
    if visual <= 0 and isinstance(details, dict):
        visual = _as_int(details.get("image_tokens"))
    return visual


def split_usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int, int]:
    """Return (input_text_tokens, output_text_tokens, visual_tokens)."""
    usage = usage or {}
    prompt = _as_int(usage.get("prompt_tokens"))
    completion = _as_int(usage.get("completion_tokens"))
    visual = visual_tokens_from_usage(usage)
    if prompt and visual:
        visual = min(visual, prompt)
    input_text = _as_int(usage.get("input_text_tokens"))
    output_text = _as_int(usage.get("output_text_tokens"))
    if input_text <= 0:
        input_text = max(prompt - visual, 0)
    if output_text <= 0:
        output_text = completion
    return input_text, output_text, visual


def record_turn_usage(usage: dict[str, Any] | None, messages: list[dict[str, Any]]) -> dict[str, int]:
    usage = dict(usage or {})
    prompt = _as_int(usage.get("prompt_tokens"))
    completion = _as_int(usage.get("completion_tokens"))
    visual = visual_tokens_from_usage(usage)
    if visual <= 0:
        visual = visual_tokens_from_messages(messages)
    if prompt:
        visual = min(visual, prompt)
    input_text = max(prompt - visual, 0)
    generated = input_text + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "visual_tokens": visual,
        "input_text_tokens": input_text,
        "output_text_tokens": completion,
        "generated_tokens": generated,
        "total_tokens": generated + visual,
    }


def usage_dict_from_response(response: Any) -> dict[str, Any]:
    raw_usage = getattr(response, "usage", None)
    if raw_usage is None:
        return {}
    usage: dict[str, Any] = {
        "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(raw_usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(raw_usage, "total_tokens", 0) or 0,
    }
    details = getattr(raw_usage, "prompt_tokens_details", None)
    image = None
    if details is not None:
        image = getattr(details, "image_tokens", None)
        if image is None and isinstance(details, dict):
            image = details.get("image_tokens")
    if image:
        usage["image_tokens"] = int(image)
    return usage


def record_chat_response(messages: list[dict[str, Any]], response: Any) -> dict[str, int]:
    return record_turn_usage(usage_dict_from_response(response), messages)


def summarize_turn_usages(usages: list[dict[str, Any]] | None) -> dict[str, int]:
    input_text_tokens = 0
    output_text_tokens = 0
    visual_tokens = 0
    for usage in usages or []:
        inp, out, vis = split_usage_tokens(usage)
        input_text_tokens += inp
        output_text_tokens += out
        visual_tokens += vis
    generated = input_text_tokens + output_text_tokens
    return {
        "visual_tokens": visual_tokens,
        "input_text_tokens": input_text_tokens,
        "output_text_tokens": output_text_tokens,
        "generated_tokens": generated,
        "total_tokens": generated + visual_tokens,
    }


def attach_usage_fields(row: dict[str, Any], agent: Any) -> None:
    """Write visual/input_text/output_text tokens and n_turns onto a result row."""
    summary = summarize_turn_usages(getattr(agent, "turn_usages", None))
    row["visual_tokens"] = summary["visual_tokens"]
    row["input_text_tokens"] = summary["input_text_tokens"]
    row["output_text_tokens"] = summary["output_text_tokens"]
    row["generated_tokens"] = summary["generated_tokens"]
    row["total_tokens"] = summary["total_tokens"]
    row["n_turns"] = int(getattr(agent, "n_turns", 0) or 0)
