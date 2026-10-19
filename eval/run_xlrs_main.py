"""Main XLRS GeoVista eval (default SOP, no ablation).

- No client-side image downsampling (encode_image max_resolution=None).
- Online repeat cutoff; if Max turns: retry from scratch up to 3 times;
  if all 3 fail, force a final answer (no tools).

Mirrors /srv/user/zhujs/work/etc/qwen2_5-RL/run_repeat_recover.py logic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))

from config import DATASETS  # noqa: E402
from evaluate import eval_xlrs  # noqa: E402
from geovista.agent import zoom_in_tool  # noqa: E402
from recover_agent import (  # noqa: E402
    MAX_TURN_ERROR,
    RecoverAgent,
    attach_stats,
    build_prompt,
    file_lock,
    image_path_of,
    row_key,
)

Image.MAX_IMAGE_PIXELS = None
WRITE_LOCK = threading.Lock()


def progress_path(result_dir: Path, out_name: str) -> Path:
    folder = result_dir / "_xlrs_main_progress"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (out_name + ".jsonl")


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


def append_result(out_path: Path, prog: Path, key: str, row: dict) -> None:
    with file_lock(str(out_path)):
        with prog.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "row": row}, ensure_ascii=False) + "\n")
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_one(row: dict, model_name: str, api_base: str, max_turns: int, sop: str):
    start = time.time()
    new_row = dict(row)
    q_id = row_key(row)
    img = image_path_of(row)
    prompt = build_prompt(row)

    if not img or not os.path.exists(img):
        agent = RecoverAgent(model_name, api_base, max_turns=max_turns, max_tool_calls=None)
        attach_stats(new_row, agent, "Error: Image file not found", [], 0, False)
        new_row["processing_time"] = round(time.time() - start, 2)
        new_row["ablation"] = "default"
        new_row.pop("recovered_max_turns", None)
        return new_row

    try:
        for attempt in range(1, 4):
            agent = RecoverAgent(model_name, api_base, max_turns=max_turns, max_tool_calls=None)
            agent.register_tool("zoom_in", zoom_in_tool)
            answer, thoughts, forced = agent.run_collect(
                prompt, img, sop, q_id=q_id, force_at_end=False
            )
            if answer != MAX_TURN_ERROR:
                attach_stats(new_row, agent, answer, thoughts, attempt, forced)
                new_row["processing_time"] = round(time.time() - start, 2)
                new_row["ablation"] = "default"
                new_row.pop("recovered_max_turns", None)
                return new_row
            print(f"  🔁 [Agent {q_id}] 第 {attempt}/3 次达到 max turns，从头重跑", flush=True)

        agent = RecoverAgent(model_name, api_base, max_turns=max_turns, max_tool_calls=None)
        agent.register_tool("zoom_in", zoom_in_tool)
        print(f"  ✂️ [Agent {q_id}] 3 次都达到 max turns，强制回答", flush=True)
        answer, thoughts, forced = agent.run_collect(
            prompt, img, sop, q_id=q_id, force_at_end=True
        )
        attach_stats(new_row, agent, answer, thoughts, 3, True)
    except Exception as exc:
        agent = RecoverAgent(model_name, api_base, max_turns=max_turns, max_tool_calls=None)
        attach_stats(new_row, agent, f"Error: {exc}", [], 0, False)

    new_row["processing_time"] = round(time.time() - start, 2)
    new_row["ablation"] = "default"
    new_row.pop("recovered_max_turns", None)
    return new_row


def main():
    parser = argparse.ArgumentParser(description="Main XLRS GeoVista (no downsample, repeat×3, force)")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model-name", type=str, default="agent_zoom_step_1120")
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--work-dir", type=str, default="results/xlrs_main_nodownsample")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = Path(__file__).resolve().parents[1] / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{args.model_name}_xlrs.jsonl"
    out_path = work_dir / out_name
    report_path = work_dir / f"{args.model_name}_xlrs_report.txt"
    prog = progress_path(work_dir, out_name)

    cfg = DATASETS["xlrs"]
    sop = cfg["sop"]
    with open(cfg["jsonl_path"], "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    done = load_progress(prog)
    # rebuild output from progress if resuming
    if done and (not out_path.exists() or out_path.stat().st_size == 0):
        with out_path.open("w", encoding="utf-8") as f:
            for row in all_data:
                k = row_key(row)
                if k in done:
                    f.write(json.dumps(done[k], ensure_ascii=False) + "\n")

    pending = [row for row in all_data if row_key(row) not in done]
    max_pix = os.getenv("GEOVISTA_MAX_PIXELS", "1003520")
    max_res = os.getenv("GEOVISTA_MAX_RESOLUTION", "") or "None"
    print(
        f"======== XLRS MAIN START {time.strftime('%c')} ========\n"
        f"model={args.model_name} api={args.api_base}\n"
        f"max_turns={args.max_turns} workers={args.workers}\n"
        f"encode: max_pixels={max_pix} max_resolution={max_res} | repeat×3 then force\n"
        f"pending={len(pending)}/{len(all_data)} → {out_path}",
        flush=True,
    )

    if pending:
        if not out_path.exists():
            out_path.write_text("", encoding="utf-8")
        finished = 0
        total = len(pending)
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futs = {
                executor.submit(run_one, row, args.model_name, args.api_base, args.max_turns, sop): row
                for row in pending
            }
            for fut in futures.as_completed(futs):
                src = futs[fut]
                key = row_key(src)
                try:
                    result = fut.result()
                except Exception as exc:
                    result = dict(src)
                    result["model_output"] = f"Error: {exc}"
                    result["ablation"] = "default"
                append_result(out_path, prog, key, result)
                finished += 1
                ans = str(result.get("model_output", "")).replace("\n", " ")[:40]
                if result.get("forced_on_repeat"):
                    flag = "强制"
                else:
                    flag = f"第{result.get('retry_attempt', 1)}次"
                if finished % 20 == 0 or finished == total:
                    print(f"[xlrs {finished}/{total}] {flag} ID {key} | {ans}", flush=True)

    print("✅ XLRS main inference done", flush=True)

    if not args.skip_eval:
        print(f"======== RE-EVAL START {time.strftime('%c')} ========", flush=True)
        eval_xlrs(str(out_path), str(report_path))
        print(f"======== ALL DONE {time.strftime('%c')} ========", flush=True)


if __name__ == "__main__":
    main()
