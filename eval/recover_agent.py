"""RecoverAgent and helpers for discard-repeat → force eval.

Shared by main XLRS / XHR / LRS benches (default SOP, no component ablation).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent import futures
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geovista.agent import (  # noqa: E402
    GIVEN_ROI_EVAL_HINT,
    GeoVistaAgent,
    enforce_one_step,
    question_has_given_pixel_roi,
    zoom_in_tool,
)
from geovista.token_usage import record_chat_response  # noqa: E402

MAX_TURN_ERROR = "Error: Max turns reached"
FORCE_TAIL_MCQ = (
    "The observations above are finished. Do not call any tool and do not repeat earlier text. "
    "Output only the final choice letter inside <answer></answer>."
)
FORCE_TAIL_OPEN = (
    "The observations above are finished. Do not call any tool and do not repeat earlier text. "
    "Output only a concise final answer inside <answer></answer>."
)
# Back-compat alias
FORCE_TAIL = FORCE_TAIL_MCQ
FILE_LOCKS: dict[str, threading.Lock] = {}
Image.MAX_IMAGE_PIXELS = None


def row_key(row: dict) -> str:
    return str(row.get("question_id", row.get("index", row.get("id", "Unknown"))))


def progress_path(result_dir: Path, jsonl_path: Path) -> Path:
    folder = result_dir / "_repeat_recover"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (jsonl_path.name + ".jsonl")


def file_lock(path: str) -> threading.Lock:
    if path not in FILE_LOCKS:
        FILE_LOCKS[path] = threading.Lock()
    return FILE_LOCKS[path]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_progress(prog: Path) -> dict:
    done = {}
    if not prog.exists():
        return done
    with prog.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            done[str(item["key"])] = item["row"]
    return done


def backup_once(path: Path) -> None:
    backup = Path(str(path) + ".bak_before_repeat_recover")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"📦 已备份 {backup}", flush=True)


def reconcile(path: Path, prog: Path) -> dict:
    done = load_progress(prog)
    if not done:
        return done
    rows = load_rows(path)
    changed = False
    for i, row in enumerate(rows):
        key = row_key(row)
        if key in done and row.get("model_output") == MAX_TURN_ERROR:
            rows[i] = done[key]
            changed = True
    if changed:
        write_rows(path, rows)
    return done


def save_replacement(path: Path, prog: Path, key: str, new_row: dict) -> None:
    with file_lock(str(path)):
        with prog.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "row": new_row}, ensure_ascii=False) + "\n")
            f.flush()
        rows = load_rows(path)
        replaced = False
        for i, row in enumerate(rows):
            if row_key(row) == key and row.get("model_output") == MAX_TURN_ERROR:
                rows[i] = new_row
                replaced = True
                break
        if not replaced:
            for i, row in enumerate(rows):
                if row_key(row) == key:
                    rows[i] = new_row
                    replaced = True
                    break
        if replaced:
            write_rows(path, rows)


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d+(?:\.\d+)?", "#", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _bboxes(text: str):
    boxes = []
    for match in re.finditer(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        text,
    ):
        boxes.append(tuple(round(float(match.group(i)), 1) for i in range(1, 5)))
    return boxes


def _tool_bboxes(text: str):
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"(\{.*\"bbox\".*\})", text, re.DOTALL)
    boxes = []
    for block in blocks:
        boxes.extend(_bboxes(block))
    return boxes


def _same_box(a, b) -> bool:
    return all(abs(x - y) <= 2 for x, y in zip(a, b))


def intra_cut(text: str):
    yada = re.search(r"yada\.\.\.", text, re.IGNORECASE)
    if yada:
        return text[: yada.start()].rstrip()
    lines = text.splitlines()
    seen = {}
    for i, line in enumerate(lines):
        key = re.sub(r"\s+", " ", line).strip()
        if len(key) >= 24 and key in seen:
            return "\n".join(lines[:i]).rstrip()
        if len(key) >= 24:
            seen[key] = i
    collapsed = re.sub(r"\s+", " ", text)
    window = 48
    seen = {}
    for i in range(0, max(0, len(collapsed) - window), 8):
        chunk = collapsed[i : i + window]
        prev = seen.get(chunk)
        if prev is not None and i - prev >= window:
            needle = chunk[:24]
            first = text.find(needle)
            second = text.find(needle, first + len(needle)) if first >= 0 else -1
            if second > 0:
                return text[:second].rstrip()
            return text[: max(1, int(len(text) * 0.5))].rstrip()
        seen[chunk] = i
    return None


def is_cross_repeat(text: str, previous: list[str]) -> bool:
    current = _norm(text)
    if len(current) >= 80:
        for prev in previous:
            other = _norm(prev)
            if len(other) < 80:
                continue
            if SequenceMatcher(None, current[:900], other[:900]).ratio() >= 0.88:
                return True
    boxes = _tool_bboxes(text)
    old_boxes = []
    for prev in previous:
        old_boxes.extend(_tool_bboxes(prev))
    for box in boxes:
        for old in old_boxes:
            if _same_box(box, old):
                return True
    return False


def extract_answer(raw: str) -> str:
    if not raw:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = re.sub(r"<tool_call>.*?(</tool_call>)?", "", text, flags=re.DOTALL)
    text = text.replace("</answer>", "").strip()
    if re.fullmatch(r"\(?[A-E]\)?", text, re.IGNORECASE):
        return text.replace("(", "").replace(")", "").upper()
    return text


def force_answer(
    client: OpenAI,
    model_name: str,
    first_input: str,
    image: Image.Image,
    thoughts: list[str],
    temperature: float = 0.0,
    *,
    task_family: str = "mcq",
    api_base: str | None = None,
):
    """Force a final answer from clean (non-repeating) prior thoughts only.

    Returns ``(answer, forced, usage_or_None)`` so callers can fold the force
    call into ``agent.turn_usages`` (same idea as qwen2_5-RL/run_xlrs.py).
    """
    # Drop empty / truncated junk; keep only prior good turns.
    later = [t.strip() for t in thoughts if t and t.strip()]
    first_text = re.sub(
        r"\nPlease strictly follow the SOP\. Begin your response with <think>\.\s*$",
        "",
        first_input,
    ).strip()
    tail = FORCE_TAIL_OPEN if task_family == "open" else FORCE_TAIL_MCQ
    pieces = [first_text] + later
    prompt = "\n\n".join(pieces) + "\n\n" + tail
    encoder = GeoVistaAgent(model_name=model_name, api_base=api_base or "http://127.0.0.1:0")
    messages = [
        {
            "role": "system",
            "content": (
                "Answer now from the question and the observations already collected. "
                "Do not call tools. Do not repeat earlier sentences. "
                "Ignore any truncated or repetitive drafts."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoder.encode_image(image)}"}},
            ],
        },
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
            frequency_penalty=0.0,
            stop=["</answer>"],
        )
    except Exception as exc:
        err = str(exc).lower()
        if later and any(tok in err for tok in ("context", "length", "maximum", "too long")):
            short = "\n\n".join([first_text] + ([later[-1]] if later else [])) + "\n\n" + tail
            messages[1]["content"][0]["text"] = short
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2048,
                    frequency_penalty=0.0,
                    stop=["</answer>"],
                )
            except Exception as exc2:
                return f"Error: forced answer failed ({exc2})", True, None
        else:
            return f"Error: forced answer failed ({exc})", True, None

    usage = record_chat_response(messages, response)
    raw = response.choices[0].message.content or ""
    if response.choices[0].finish_reason == "stop" and "<answer>" in raw and "</answer>" not in raw:
        raw += "</answer>"
    ans = extract_answer(raw)
    return (ans if ans else "Error: empty forced answer"), True, usage


class RecoverAgent(GeoVistaAgent):
    """GeoVistaAgent with repeat cutoff; returns (answer, thoughts, forced).

    On repeat: drop the repeating turn entirely, then force-answer from prior turns only.
    """

    def __init__(self, *args, task_family: str = "mcq", **kwargs):
        super().__init__(*args, **kwargs)
        self.task_family = task_family  # 'mcq' or 'open'

    def run_collect(self, user_prompt, image_path, system_prompt, q_id="Unknown", force_at_end=False):
        context_kwargs = {}
        self.reset_memory(system_prompt)
        self.n_turns = 0
        self.n_tool_calls = 0
        self.budget_exhausted = False
        self.turn_usages = []
        original_img = Image.open(image_path).convert("RGB")
        context_kwargs["original_image"] = original_img
        context_kwargs["question"] = user_prompt
        full_thoughts: list[str] = []
        safe_prompt = user_prompt + "\nPlease strictly follow the SOP. Begin your response with <think>."
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": safe_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(original_img)}"}},
                ],
            }
        )

        for _turn in range(1, self.max_turns + 1):
            model_output = self._call_llm()
            if model_output is None:
                return "Error: LLM API Failure or Timeout", full_thoughts, False
            model_output = enforce_one_step(model_output)

            self.n_turns += 1
            # Tentatively append; if this turn is a repeat we pop it and force-answer.
            self.messages.append({"role": "assistant", "content": model_output})
            if "<answer>" in model_output:
                match = re.search(r"<answer>(.*?)</answer>", model_output, re.DOTALL)
                full_thoughts.append(model_output)
                return (match.group(1).strip() if match else "No answer parsed"), full_thoughts, False

            text_without_think = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL).strip()
            if self.task_family == "mcq" and re.fullmatch(r"\(?[A-E]\)?", text_without_think, re.IGNORECASE):
                full_thoughts.append(model_output)
                return text_without_think.replace("(", "").replace(")", "").upper(), full_thoughts, False

            kept = intra_cut(model_output)
            cross = is_cross_repeat(model_output, full_thoughts)
            if kept is not None or cross:
                reason = "段内复读" if kept is not None and not cross else "跨轮复读"
                # Discard the repeating turn completely (do not keep truncated draft).
                self.messages.pop()
                print(
                    f"  ✂️ [Agent {q_id}] {reason}，舍弃该轮 → 强制回答 "
                    f"(保留前 {len(full_thoughts)} 轮)",
                    flush=True,
                )
                ans, forced, usage = force_answer(
                    self.client,
                    self.model_name,
                    safe_prompt,
                    original_img,
                    full_thoughts,
                    self.temperature,
                    task_family=self.task_family,
                )
                if usage:
                    self.turn_usages.append(usage)
                    self.n_turns += 1
                return ans, full_thoughts, forced

            full_thoughts.append(model_output)
            tool_str_raw = self._extract_tool_call(model_output)
            if tool_str_raw:
                if self.max_tool_calls is not None and self.n_tool_calls >= self.max_tool_calls:
                    self.budget_exhausted = True
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"TOOL BUDGET EXHAUSTED: you already used {self.n_tool_calls}/{self.max_tool_calls} "
                                "zoom_in calls. Do NOT call tools again. Produce the final answer now inside "
                                "<answer>...</answer> based on evidence gathered so far."
                            ),
                        }
                    )
                    continue
                try:
                    tool_data = json.loads(tool_str_raw.strip())
                except json.JSONDecodeError:
                    match = re.search(
                        r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+).*?\]",
                        tool_str_raw,
                    )
                    if match:
                        tool_data = {"name": "zoom_in", "arguments": {"bbox": [float(match.group(i)) for i in range(1, 5)]}}
                    else:
                        self.messages.append(
                            {"role": "user", "content": "Malformed JSON. Please output a VALID JSON inside <tool_call>.</tool_call>."}
                        )
                        continue
                tool_name = tool_data.get("name")
                tool_args = tool_data.get("arguments", {})
                if tool_name in self.tools:
                    tool_result = self.tools[tool_name](tool_args, context_kwargs)
                    self.n_tool_calls += 1
                    if "error" in tool_result:
                        self.messages.append({"role": "user", "content": tool_result["error"]})
                    else:
                        observe = tool_result["text"]
                        if self.max_tool_calls is not None:
                            remain = self.max_tool_calls - self.n_tool_calls
                            observe += f"\n[Budget] zoom_in used {self.n_tool_calls}/{self.max_tool_calls} ({remain} remaining)."
                        self.messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": observe},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(tool_result['image'])}"},
                                    },
                                ],
                            }
                        )
                else:
                    self.messages.append({"role": "user", "content": f"Error: Tool [{tool_name}] not found."})
                continue

            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "WARNING: You stopped generating without making a <tool_call> or providing an <answer>. "
                        "If you need to use a tool, output EXACTLY: "
                        '<tool_call>{"name": "zoom_in", "arguments": {"bbox": [x1, y1, x2, y2]}}</tool_call>. '
                        "If you have confidently found the target or finished the reasoning, "
                        "output your final result wrapped in <answer>...</answer>."
                    ),
                }
            )

        if force_at_end:
            print(f"  ✂️ [Agent {q_id}] 达到 max turns，强制回答", flush=True)
            ans, forced, usage = force_answer(
                self.client,
                self.model_name,
                safe_prompt,
                original_img,
                full_thoughts,
                self.temperature,
                task_family=self.task_family,
            )
            if usage:
                self.turn_usages.append(usage)
                self.n_turns += 1
            return ans, full_thoughts, forced
        return MAX_TURN_ERROR, full_thoughts, False


def build_prompt(row: dict) -> str:
    question = str(row.get("input") or row.get("question") or "")
    question = question.replace("please only return answer option.", "").strip()
    if question_has_given_pixel_roi(question):
        question = question + "\n\n" + GIVEN_ROI_EVAL_HINT
    return f"Question: {question}\n\n[System Observation]\nCurrent View: Global View\n"


def image_path_of(row: dict) -> str:
    return row.get("image_absolute_path") or row.get("image") or ""


def attach_stats(new_row: dict, agent: GeoVistaAgent, answer: str, thoughts: list, attempt: int, forced: bool):
    summary = agent.token_summary()
    new_row["model_output"] = answer
    new_row["visual_tokens"] = summary["visual_tokens"]
    new_row["input_text_tokens"] = summary["input_text_tokens"]
    new_row["output_text_tokens"] = summary["output_text_tokens"]
    new_row["generated_tokens"] = summary["generated_tokens"]
    new_row["total_tokens"] = summary["total_tokens"]
    new_row["n_turns"] = summary["n_turns"]
    new_row["n_tool_calls"] = summary.get("n_tool_calls", 0)
    new_row["budget_exhausted"] = summary.get("budget_exhausted", False)
    new_row["retry_attempt"] = attempt
    new_row["forced_on_repeat"] = forced
    new_row["full_thoughts"] = thoughts
    new_row["recovered_max_turns"] = True
