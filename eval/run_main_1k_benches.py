#!/usr/bin/env python3
"""Corrected-1k main eval for XHR (RSHR) / LRS / XLRS.

Stack matches xlrs_georeflex_boost_v2:
  - GEOVISTA_MAX_PIXELS=1003520
  - default DATASETS SOP (Given-ROI aligned)
  - RecoverAgent: on repeat → discard that turn → force answer
  - max-turns: retry×3 then force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent import futures
from pathlib import Path

from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))

from config import DATASETS  # noqa: E402
from evaluate import eval_lrs, eval_xhr, eval_xlrs  # noqa: E402
from geovista.agent import (  # noqa: E402
    GIVEN_ROI_EVAL_HINT,
    question_has_given_pixel_roi,
    zoom_in_tool,
)
from recover_agent import (  # noqa: E402
    MAX_TURN_ERROR,
    RecoverAgent,
    attach_stats,
    file_lock,
    row_key,
)

Image.MAX_IMAGE_PIXELS = None

EVAL_FN = {"xlrs": eval_xlrs, "xhr": eval_xhr, "lrs": eval_lrs}


def progress_path(result_dir: Path, out_name: str) -> Path:
    folder = result_dir / "_main_1k_progress"
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


def resolve_image(row: dict, dataset: str) -> str:
    cfg = DATASETS[dataset]
    root = cfg.get("image_root") or ""
    raw = (
        row.get("image_absolute_path")
        or row.get("image_path")
        or row.get("image")
        or row.get("Image")
        or ""
    )
    if not raw:
        return ""
    if os.path.isabs(raw) and os.path.exists(raw):
        return raw
    cands = []
    if root:
        cands.append(os.path.join(root, raw))
        cands.append(os.path.join(root, os.path.basename(raw)))
    cands.append(raw)
    for p in cands:
        if p and os.path.exists(p):
            return p
    return cands[0] if cands else ""


def build_prompt_for(row: dict, dataset: str) -> str:
    question = str(
        row.get("input") or row.get("question") or row.get("text") or row.get("Text") or ""
    )
    question = question.replace("please only return answer option.", "").strip()
    if question_has_given_pixel_roi(question):
        question = question + "\n\n" + GIVEN_ROI_EVAL_HINT
    return f"Question: {question}\n\n[System Observation]\nCurrent View: Global View\n"


def run_one(
    row: dict,
    dataset: str,
    model_name: str,
    api_base: str,
    max_turns: int,
    sop: str,
    task_family: str,
):
    start = time.time()
    new_row = dict(row)
    q_id = row_key(row)
    img = resolve_image(row, dataset)
    prompt = build_prompt_for(row, dataset)

    def _agent():
        return RecoverAgent(
            model_name,
            api_base,
            max_turns=max_turns,
            max_tool_calls=None,
            task_family=task_family,
        )

    if not img or not os.path.exists(img):
        agent = _agent()
        attach_stats(new_row, agent, "Error: Image file not found", [], 0, False)
        new_row["processing_time"] = round(time.time() - start, 2)
        new_row["ablation"] = "default_1k"
        new_row.pop("recovered_max_turns", None)
        return new_row

    try:
        for attempt in range(1, 4):
            agent = _agent()
            agent.register_tool("zoom_in", zoom_in_tool)
            answer, thoughts, forced = agent.run_collect(
                prompt, img, sop, q_id=q_id, force_at_end=False
            )
            if answer != MAX_TURN_ERROR:
                attach_stats(new_row, agent, answer, thoughts, attempt, forced)
                new_row["processing_time"] = round(time.time() - start, 2)
                new_row["ablation"] = "default_1k"
                new_row.pop("recovered_max_turns", None)
                return new_row
            print(f"  🔁 [{dataset} {q_id}] 第 {attempt}/3 次 max turns，重跑", flush=True)

        agent = _agent()
        agent.register_tool("zoom_in", zoom_in_tool)
        print(f"  ✂️ [{dataset} {q_id}] 3 次 max turns，强制回答", flush=True)
        answer, thoughts, forced = agent.run_collect(
            prompt, img, sop, q_id=q_id, force_at_end=True
        )
        attach_stats(new_row, agent, answer, thoughts, 3, True)
    except Exception as exc:
        agent = _agent()
        attach_stats(new_row, agent, f"Error: {exc}", [], 0, False)

    new_row["processing_time"] = round(time.time() - start, 2)
    new_row["ablation"] = "default_1k"
    new_row.pop("recovered_max_turns", None)
    return new_row


def run_dataset(args, dataset: str) -> None:
    cfg = DATASETS[dataset]
    sop = cfg["sop"]
    task_family = "open" if dataset == "lrs" else "mcq"
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = Path(__file__).resolve().parents[1] / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{args.model_name}_{dataset}.jsonl"
    out_path = work_dir / out_name
    report_path = work_dir / f"{args.model_name}_{dataset}_report.txt"
    prog = progress_path(work_dir, out_name)

    with open(cfg["jsonl_path"], "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    done = load_progress(prog)
    if done and (not out_path.exists() or out_path.stat().st_size == 0):
        with out_path.open("w", encoding="utf-8") as f:
            for row in all_data:
                k = row_key(row)
                if k in done:
                    f.write(json.dumps(done[k], ensure_ascii=False) + "\n")

    pending = [row for row in all_data if row_key(row) not in done]
    print(
        f"\n======== MAIN-1k {dataset.upper()} START {time.strftime('%c')} ========\n"
        f"model={args.model_name} api={args.api_base}\n"
        f"max_turns={args.max_turns} workers={args.workers} task_family={task_family}\n"
        f"encode max_pixels={os.getenv('GEOVISTA_MAX_PIXELS', '1003520')}\n"
        f"repeat: discard turn → force | pending={len(pending)}/{len(all_data)} → {out_path}",
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
                    dataset,
                    args.model_name,
                    args.api_base,
                    args.max_turns,
                    sop,
                    task_family,
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
                    result["ablation"] = "default_1k"
                append_result(out_path, prog, key, result)
                finished += 1
                ans = str(result.get("model_output", "")).replace("\n", " ")[:40]
                flag = "强制" if result.get("forced_on_repeat") else f"第{result.get('retry_attempt', 1)}次"
                if finished % 20 == 0 or finished == total:
                    print(f"[{dataset} {finished}/{total}] {flag} ID {key} | {ans}", flush=True)

    if not args.skip_eval and dataset in EVAL_FN:
        EVAL_FN[dataset](str(out_path), str(report_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8110/v1")
    parser.add_argument("--model-name", type=str, default="agent_zoom_step_1120")
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--work-dir", type=str, default="results/main_1k_discard_force")
    parser.add_argument(
        "--data",
        nargs="+",
        default=["xhr", "xlrs", "lrs"],
        choices=["xhr", "lrs", "xlrs"],
    )
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("GEOVISTA_MAX_PIXELS", "1003520")
    os.environ.pop("GEOVISTA_MAX_RESOLUTION", None)

    for ds in args.data:
        run_dataset(args, ds)
    print(f"\n✅ main-1k benches done → {args.work_dir}", flush=True)


if __name__ == "__main__":
    main()
