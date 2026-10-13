#!/usr/bin/env python3
"""XLRS prompt ablation under corrected 1k settings (same stack as xlrs_georeflex_boost_v2).

- GEOVISTA_MAX_PIXELS=1003520 (set by launcher / env)
- RecoverAgent: repeat×3 on max-turns, then force answer
- Ablation SOPs from ablation_sops.py (Given-ROI aligned)
- max_tool_calls default 8 (fair component comparison)
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

from ablation_sops import ABLATION_ORDER, normalize_ablation, sop_for_dataset  # noqa: E402
from config import DATASETS  # noqa: E402
from evaluate import eval_xlrs  # noqa: E402
from geovista.agent import zoom_in_tool  # noqa: E402
from run_ablation_recover import (  # noqa: E402
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
    folder = result_dir / "_ablation_1k_progress"
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


def run_one(
    row: dict,
    model_name: str,
    api_base: str,
    max_turns: int,
    max_tool_calls: int,
    ablation: str,
    sop: str,
):
    start = time.time()
    new_row = dict(row)
    q_id = row_key(row)
    img = image_path_of(row)
    prompt = build_prompt(row)

    if not img or not os.path.exists(img):
        agent = RecoverAgent(
            model_name, api_base, max_turns=max_turns, max_tool_calls=max_tool_calls
        )
        attach_stats(new_row, agent, "Error: Image file not found", [], 0, False)
        new_row["processing_time"] = round(time.time() - start, 2)
        new_row["ablation"] = ablation
        new_row.pop("recovered_max_turns", None)
        return new_row

    try:
        for attempt in range(1, 4):
            agent = RecoverAgent(
                model_name, api_base, max_turns=max_turns, max_tool_calls=max_tool_calls
            )
            agent.register_tool("zoom_in", zoom_in_tool)
            answer, thoughts, forced = agent.run_collect(
                prompt, img, sop, q_id=q_id, force_at_end=False
            )
            if answer != MAX_TURN_ERROR:
                attach_stats(new_row, agent, answer, thoughts, attempt, forced)
                new_row["processing_time"] = round(time.time() - start, 2)
                new_row["ablation"] = ablation
                new_row.pop("recovered_max_turns", None)
                return new_row
            print(f"  🔁 [{ablation} {q_id}] 第 {attempt}/3 次 max turns，重跑", flush=True)

        agent = RecoverAgent(
            model_name, api_base, max_turns=max_turns, max_tool_calls=max_tool_calls
        )
        agent.register_tool("zoom_in", zoom_in_tool)
        answer, thoughts, forced = agent.run_collect(
            prompt, img, sop, q_id=q_id, force_at_end=True
        )
        attach_stats(new_row, agent, answer, thoughts, 3, forced)
        new_row["processing_time"] = round(time.time() - start, 2)
        new_row["ablation"] = ablation
        new_row.pop("recovered_max_turns", None)
        return new_row
    except Exception as exc:
        agent = RecoverAgent(
            model_name, api_base, max_turns=max_turns, max_tool_calls=max_tool_calls
        )
        attach_stats(new_row, agent, f"Error: {exc}", [], 0, False)
        new_row["processing_time"] = round(time.time() - start, 2)
        new_row["ablation"] = ablation
        return new_row


def run_variant(args, ablation: str, all_data: list) -> Path:
    canon = normalize_ablation(ablation)
    assert canon
    sop = sop_for_dataset(canon, "xlrs", max_tool_calls=args.max_tool_calls)
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = Path(__file__).resolve().parents[1] / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{args.model_name}_xlrs_{canon}.jsonl"
    out_path = work_dir / out_name
    report_path = work_dir / f"{args.model_name}_xlrs_{canon}_report.txt"
    prog = progress_path(work_dir, out_name)

    done = load_progress(prog)
    if done and (not out_path.exists() or out_path.stat().st_size == 0):
        with out_path.open("w", encoding="utf-8") as f:
            for row in all_data:
                k = row_key(row)
                if k in done:
                    f.write(json.dumps(done[k], ensure_ascii=False) + "\n")

    pending = [row for row in all_data if row_key(row) not in done]
    print(
        f"\n======== ABLATION {canon} START {time.strftime('%c')} ========\n"
        f"model={args.model_name} api={args.api_base}\n"
        f"max_turns={args.max_turns} max_tool_calls={args.max_tool_calls} workers={args.workers}\n"
        f"encode max_pixels={os.getenv('GEOVISTA_MAX_PIXELS', '1003520')} "
        f"max_resolution={os.getenv('GEOVISTA_MAX_RESOLUTION', '') or 'None'}\n"
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
                executor.submit(
                    run_one,
                    row,
                    args.model_name,
                    args.api_base,
                    args.max_turns,
                    args.max_tool_calls,
                    canon,
                    sop,
                ): row
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
                    result["ablation"] = canon
                append_result(out_path, prog, key, result)
                finished += 1
                ans = str(result.get("model_output", "")).replace("\n", " ")[:40]
                if finished % 20 == 0 or finished == total:
                    print(f"[{canon} {finished}/{total}] ID {key} | {ans}", flush=True)

    if not args.skip_eval:
        eval_xlrs(str(out_path), str(report_path))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8110/v1")
    parser.add_argument("--model-name", type=str, default="agent_zoom_step_1120")
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--work-dir", type=str, default="results/ablation_prompt_1k")
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(ABLATION_ORDER),
        help="Comma-separated ablation ids",
    )
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    # Ensure train-aligned 1k unless user overrides.
    os.environ.setdefault("GEOVISTA_MAX_PIXELS", "1003520")
    if "GEOVISTA_MAX_RESOLUTION" in os.environ and not os.environ["GEOVISTA_MAX_RESOLUTION"]:
        os.environ.pop("GEOVISTA_MAX_RESOLUTION", None)

    cfg = DATASETS["xlrs"]
    with open(cfg["jsonl_path"], "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    variants = [normalize_ablation(v.strip()) for v in args.variants.split(",") if v.strip()]
    variants = [v for v in variants if v]
    for v in variants:
        run_variant(args, v, all_data)

    print(f"\n✅ Ablation 1k done → {args.work_dir}", flush=True)


if __name__ == "__main__":
    main()
