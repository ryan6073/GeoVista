"""Re-run ablation control variants (single_path / multi_roi / evidence) from scratch.

Uses softened SOP + online repeat cutoff + force-at-end (same spirit as
run_repeat_recover). Leaves `full` untouched.

Writes into results/ablation_prompt (backs up prior jsonl as
*.bak_before_control_rerun) and re-evaluates XLRS reports.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent import futures
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))

from run_ablation_recover import (  # noqa: E402
    RecoverAgent,
    attach_stats,
    build_prompt,
    file_lock,
    image_path_of,
    load_rows,
    row_key,
    write_rows,
)
from geovista.agent import zoom_in_tool  # noqa: E402
from ablation_sops import normalize_ablation, sop_for_dataset  # noqa: E402
from evaluate import eval_xlrs  # noqa: E402


CONTROLS = ["single_path", "multi_roi", "evidence"]


def progress_path(result_dir: Path, jsonl_path: Path) -> Path:
    folder = result_dir / "_control_rerun"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (jsonl_path.name + ".jsonl")


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
    backup = Path(str(path) + ".bak_before_control_rerun")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"📦 已备份 {backup}", flush=True)


def save_row(path: Path, prog: Path, key: str, result: dict, index_map: dict) -> None:
    with file_lock(str(path)):
        with prog.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "row": result}, ensure_ascii=False) + "\n")
        rows = load_rows(path)
        replaced = False
        for i, row in enumerate(rows):
            if row_key(row) == key:
                rows[i] = result
                replaced = True
                break
        if not replaced and key in index_map:
            rows[index_map[key]] = result
            replaced = True
        if not replaced:
            rows.append(result)
        write_rows(path, rows)


def rerun_one(row: dict, ablation: str, model_name: str, api_base: str, max_turns: int, max_tool_calls: int):
    start = time.time()
    new_row = dict(row)
    for k in ("forced_on_repeat", "recovered_max_turns", "retry_attempt", "full_thoughts"):
        new_row.pop(k, None)
    q_id = row_key(row)
    img = image_path_of(row)
    prompt = build_prompt(row)
    sop = sop_for_dataset(ablation, "xlrs", max_tool_calls=max_tool_calls)
    if not sop:
        raise ValueError(f"missing sop for {ablation}")

    agent = RecoverAgent(model_name, api_base, max_turns=max_turns, max_tool_calls=max_tool_calls)
    if not img or not os.path.exists(img):
        attach_stats(new_row, agent, "Error: Image file not found", [], 1, False)
        new_row["processing_time"] = round(time.time() - start, 2)
        new_row["ablation"] = ablation
        new_row["control_rerun"] = True
        return new_row

    agent.register_tool("zoom_in", zoom_in_tool)
    try:
        answer, thoughts, forced = agent.run_collect(
            prompt, img, sop, q_id=q_id, force_at_end=True
        )
        attach_stats(new_row, agent, answer, thoughts, 1, forced)
    except Exception as exc:
        attach_stats(new_row, agent, f"Error: {exc}", [], 1, False)
    new_row["processing_time"] = round(time.time() - start, 2)
    new_row["ablation"] = ablation
    new_row["control_rerun"] = True
    new_row.pop("recovered_max_turns", None)
    return new_row


def rerun_variant(
    path: Path,
    ablation: str,
    model_name: str,
    api_base: str,
    max_turns: int,
    max_tool_calls: int,
    workers: int,
):
    backup_once(path)
    prog = progress_path(path.parent, path)
    done = load_progress(prog)
    rows = load_rows(path)
    index_map = {row_key(r): i for i, r in enumerate(rows)}

    if done:
        changed = False
        for i, row in enumerate(rows):
            k = row_key(row)
            if k in done:
                rows[i] = done[k]
                changed = True
        if changed:
            write_rows(path, rows)

    pending = [row for row in rows if row_key(row) not in done]
    print(f"📄 {path.name} [{ablation}] 待重跑 {len(pending)}/{len(rows)}", flush=True)
    if not pending:
        return

    finished = 0
    total = len(pending)
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futs = {
            executor.submit(rerun_one, row, ablation, model_name, api_base, max_turns, max_tool_calls): row
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
                result["control_rerun"] = True
            save_row(path, prog, key, result, index_map)
            finished += 1
            ans = str(result.get("model_output", "")).replace("\n", " ")[:40]
            flag = "强制" if result.get("forced_on_repeat") else "正常"
            if finished % 20 == 0 or finished == total:
                print(f"[{ablation} {finished}/{total}] {flag} ID {key} | {ans}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, default="results/ablation_prompt")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model-name", type=str, default="agent_zoom_step_1120")
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--variants", nargs="+", default=CONTROLS)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = Path(__file__).resolve().parents[1] / result_dir

    print(
        f"======== CONTROL RERUN START {time.strftime('%c')} ========\n"
        f"api={args.api_base} model={args.model_name} "
        f"max_turns={args.max_turns} max_tool_calls={args.max_tool_calls} "
        f"workers={args.workers} variants={args.variants}",
        flush=True,
    )

    for v in args.variants:
        canon = normalize_ablation(v)
        if canon not in CONTROLS:
            print(f"⚠️ skip non-control variant {v}", flush=True)
            continue
        path = result_dir / f"{args.model_name}_xlrs_{canon}.jsonl"
        if not path.exists():
            print(f"⚠️ missing {path}", flush=True)
            continue
        rerun_variant(
            path,
            canon,
            args.model_name,
            args.api_base,
            args.max_turns,
            args.max_tool_calls,
            args.workers,
        )

    print("✅ 三组对照重跑结束", flush=True)

    if not args.skip_eval:
        print(f"======== RE-EVAL START {time.strftime('%c')} ========", flush=True)
        for v in args.variants:
            canon = normalize_ablation(v)
            if canon not in CONTROLS:
                continue
            infer = result_dir / f"{args.model_name}_xlrs_{canon}.jsonl"
            report = result_dir / f"{args.model_name}_xlrs_{canon}_report.txt"
            if infer.exists():
                print(f"# Processing: [XLRS] model=[{args.model_name}] ablation=[{canon}]", flush=True)
                eval_xlrs(str(infer), str(report))
        print(f"======== ALL DONE {time.strftime('%c')} ========", flush=True)


if __name__ == "__main__":
    main()
