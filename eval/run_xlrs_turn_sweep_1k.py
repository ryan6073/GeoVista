#!/usr/bin/env python3
"""XLRS turn-budget sweep under the current 1k discard-force stack.

Same settings as ``run_main_1k_benches.py`` except ``max_turns``:
  - GEOVISTA_MAX_PIXELS=1003520
  - default DATASETS SOP (Given-ROI / ORIGINAL 0–1000 / [OBSERVATION])
  - RecoverAgent: discard repeating turn → force answer
  - max-turns hit: retry×3 then force

Runs XLRS for each turn budget (default 4, 8, 16) into separate work dirs,
writes accuracy report + mean token metrics (visual / input_text / output_text).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))

from evaluate import eval_xlrs  # noqa: E402
from run_main_1k_benches import run_dataset  # noqa: E402

TOKEN_KEYS = (
    "visual_tokens",
    "input_text_tokens",
    "output_text_tokens",
    "generated_tokens",
    "total_tokens",
    "n_turns",
    "n_tool_calls",
)


def summarize_tokens(jsonl_path: Path) -> dict:
    vals = {k: [] for k in TOKEN_KEYS}
    n = 0
    forced = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            if row.get("forced_on_repeat"):
                forced += 1
            for k in TOKEN_KEYS:
                try:
                    vals[k].append(float(row.get(k, 0) or 0))
                except (TypeError, ValueError):
                    vals[k].append(0.0)
    out = {"n": n, "forced_on_repeat": forced}
    for k, arr in vals.items():
        out[f"mean_{k}"] = round(statistics.fmean(arr), 2) if arr else 0.0
        out[f"sum_{k}"] = int(sum(arr)) if arr else 0
    return out


def parse_overall_acc(report_path: Path) -> str:
    if not report_path.exists():
        return "n/a"
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if "OVERALL ACCURACY" in line:
            return line.strip()
    return "n/a"


def write_summary(root: Path, rows: list[dict]) -> Path:
    path = root / "TURN_SWEEP_SUMMARY.md"
    lines = [
        "# XLRS turn-budget sweep (1k discard-force)",
        "",
        "| max_turns | overall | mean visual | mean input_text | mean output_text | mean total | mean n_turns | forced |",
        "|----------:|---------|------------:|----------------:|-----------------:|-----------:|-------------:|-------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['max_turns']} | {r['acc']} | {r['mean_visual_tokens']} | "
            f"{r['mean_input_text_tokens']} | {r['mean_output_text_tokens']} | "
            f"{r['mean_total_tokens']} | {r['mean_n_turns']} | {r['forced_on_repeat']}/{r['n']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = root / "TURN_SWEEP_SUMMARY.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8110/v1")
    parser.add_argument("--model-name", type=str, default="agent_zoom_step_1120")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--turns",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="max_turns budgets to sweep (default: 4 8 16)",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="results/xlrs_turns_1k_discard_force",
        help="parent dir; each budget goes to <root>/turns_<N>/",
    )
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("GEOVISTA_MAX_PIXELS", "1003520")
    os.environ.pop("GEOVISTA_MAX_RESOLUTION", None)

    root = Path(args.root_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    root.mkdir(parents=True, exist_ok=True)

    (root / "EXPERIMENT.md").write_text(
        "\n".join(
            [
                "# XLRS turn-budget sweep (1k discard-force)",
                "",
                f"- model: `{args.model_name}` @ `{args.api_base}`",
                "- `GEOVISTA_MAX_PIXELS=1003520`",
                "- default SOP (Given-ROI / ORIGINAL 0–1000 / [OBSERVATION])",
                "- RecoverAgent: discard repeating turn → force",
                "- max-turns: retry×3 then force",
                f"- turn budgets: {args.turns}",
                "",
                "Token fields per row: `visual_tokens`, `input_text_tokens`, "
                "`output_text_tokens`, `generated_tokens`, `total_tokens`, `n_turns`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary_rows = []
    for turns in args.turns:
        work_dir = root / f"turns_{turns}"
        work_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n######## SWEEP max_turns={turns} → {work_dir} ########\n",
            flush=True,
        )
        ns = argparse.Namespace(
            api_base=args.api_base,
            model_name=args.model_name,
            max_turns=turns,
            workers=args.workers,
            work_dir=str(work_dir),
            data=["xlrs"],
            skip_eval=True,  # evaluate once below so we can attach token summary
        )
        t0 = time.time()
        run_dataset(ns, "xlrs")
        out_path = work_dir / f"{args.model_name}_xlrs.jsonl"
        report_path = work_dir / f"{args.model_name}_xlrs_report.txt"
        if not args.skip_eval:
            eval_xlrs(str(out_path), str(report_path))
        tok = summarize_tokens(out_path)
        row = {
            "max_turns": turns,
            "work_dir": str(work_dir),
            "acc": parse_overall_acc(report_path),
            "elapsed_sec": round(time.time() - t0, 1),
            **tok,
        }
        summary_rows.append(row)
        (work_dir / "token_summary.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[turns={turns}] {row['acc']} | "
            f"mean_visual={row['mean_visual_tokens']} "
            f"mean_in={row['mean_input_text_tokens']} "
            f"mean_out={row['mean_output_text_tokens']} "
            f"mean_total={row['mean_total_tokens']} "
            f"mean_n_turns={row['mean_n_turns']}",
            flush=True,
        )

    summary_path = write_summary(root, summary_rows)
    print(f"\n✅ turn sweep done → {root}\n   summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
