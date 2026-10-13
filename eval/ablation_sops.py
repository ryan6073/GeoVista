"""
Prompt-only component ablation SOPs for GeoVista (no retraining).

Aligned with the corrected 1k main XLRS SOP (Given-ROI, ORIGINAL-image 0–1000,
crop [OBSERVATION], stop-after-tool). Variants only differ in multi-ROI /
evidence / dedup constraints.

Variants (cumulative):
  A / single_path   : depth-first single ROI, no multi-ROI / evidence / dedup
  B / multi_roi     : + multi-ROI candidate plan; sequential branch verification
  C / evidence      : + explicit shared evidence state; no explicit dedup
  D / full          : + region/object de-duplication (full GeoVista interface)

Chain: 单路径 → 多感兴趣区域规划 → 多 ROI + 证据追踪 → 完整 GeoVista

All variants share the same tool schema, coordinate system, and answer format.
Enforce identical tool budgets via --max-tool-calls (recommended: 8).
"""

from __future__ import annotations

ABLATION_ALIASES = {
    "a": "single_path",
    "single_path": "single_path",
    "single-path": "single_path",
    "b": "multi_roi",
    "multi_roi": "multi_roi",
    "multi-roi": "multi_roi",
    "c": "evidence",
    "evidence": "evidence",
    "evidence_tracking": "evidence",
    "d": "full",
    "full": "full",
    "geovista": "full",
}

ABLATION_ORDER = ["single_path", "multi_roi", "evidence", "full"]

ABLATION_META = {
    "single_path": {
        "label": "A. Single-path",
        "multi_roi": False,
        "evidence": False,
        "dedup": False,
    },
    "multi_roi": {
        "label": "B. + Multi-ROI planning",
        "multi_roi": True,
        "evidence": False,
        "dedup": False,
    },
    "evidence": {
        "label": "C. + Evidence tracking",
        "multi_roi": True,
        "evidence": True,
        "dedup": False,
    },
    "full": {
        "label": "D. Full GeoVista",
        "multi_roi": True,
        "evidence": True,
        "dedup": True,
    },
}


def normalize_ablation(name: str | None) -> str | None:
    """Return canonical ablation id, or None to keep dataset-default SOP."""
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in ("", "none", "default", "baseline_sop"):
        return None
    if key not in ABLATION_ALIASES:
        raise ValueError(
            f"Unknown ablation variant '{name}'. "
            f"Choose from: {', '.join(ABLATION_ORDER)} (or A–D)."
        )
    return ABLATION_ALIASES[key]


# ---------------------------------------------------------------------------
# Shared building blocks (aligned with corrected 1k XLRS_SOP)
# ---------------------------------------------------------------------------

_TOOL_RULES = """TOOL CALL FORMAT (identical for all variants; aligned with main 1k SOP):
- Request a high-resolution crop ONLY via:
  <tool_call>{"name": "zoom_in", "arguments": {"bbox": [x1, y1, x2, y2]}}</tool_call>
- Coordinates are discrete relative coords in [0, 1000] w.r.t. the ORIGINAL image
  (top-left = [0,0], bottom-right = [1000,1000]). Never use raw pixel boxes in zoom_in.
- STOP GENERATING immediately after </tool_call>. Do not continue reasoning in the same turn.
- Always reason inside <think>...</think> before a tool call or final answer.
- After a crop returns: write a new <think> with [OBSERVATION] of THIS crop
  (hue/material/shape/texture). Do not answer from the global view alone when a crop was requested.
"""

_GIVEN_ROI = """GIVEN-ROI RULE (OVERRIDES multi-ROI planning when applicable):
- If the question names both "Image resolution: W x H" and a pixel
  "Bounding box: [x1,y1,x2,y2]", that box is the ONLY ROI.
- FIRST convert once to 0-1000: round(x/W*1000), round(y/H*1000).
- zoom_in must use ONLY those four integers — never the raw pixel box.
- Do NOT plan multi-ROI / one ROI per MCQ option for Given-ROI questions.
- After the crop: [OBSERVATION] of THIS crop, then answer from that evidence.
"""

_BUDGET_RULE = """TOOL BUDGET:
- You may call zoom_in at most {max_tool_calls} times for this question.
- When the budget is exhausted or further zooming is unnecessary, produce the final answer.
"""

_EARLY_STOP = """EARLY STOP (IMPORTANT — raises accuracy):
- Prefer answering as soon as evidence is sufficient; do NOT use tools just to fill the budget.
- Keep <think> concise (a few sentences). Do not rewrite long checklists every turn.
- Never repeat the same bbox or the same paragraph. If stuck, answer immediately.
- Valid termination is ALWAYS a letter inside <answer>...</answer> (MCQ) or a short span (open).
"""


def _answer_block(task_family: str) -> str:
    if task_family == "mcq":
        return (
            "FINAL ANSWER:\n"
            "- Output ONLY the letter of the correct choice wrapped in <answer>...</answer>.\n"
            "- Example: <answer>D</answer>."
        )
    return (
        "FINAL ANSWER:\n"
        "- Output a concise answer wrapped in <answer>...</answer>.\n"
        "- Do not include explanations inside <answer>."
    )


def _task_header(task_family: str) -> str:
    if task_family == "mcq":
        return "Task: Visual reasoning over an ultra-high-resolution remote sensing image (multiple choice)."
    return "Task: Visual question answering over an ultra-high-resolution remote sensing image."


# ---------------------------------------------------------------------------
# Variant cores
# ---------------------------------------------------------------------------

def _core_single_path() -> str:
    return """VARIANT CONSTRAINTS — Single-path (NO multi-ROI / evidence / dedup):
1. Maintain ONLY ONE current ROI at a time.
2. From the global view, select at most one region to inspect first (or answer directly if already clear).
3. After a crop returns, you may ONLY refine / continue zooming INSIDE the current branch
   (depth-first). Do NOT jump to an unrelated new region.
4. Do NOT pre-list multiple candidate ROIs, multi-item [PLAN] checklists, or evidence tables.
5. Do NOT perform cross-region de-duplication.
6. Typical trajectory: 0–2 zooms then <answer>. Answer as soon as confident.
"""


def _core_multi_roi() -> str:
    return """VARIANT CONSTRAINTS — Multi-ROI planning (sequential branches, no evidence table / dedup):
1. If Given-ROI applies, follow GIVEN-ROI RULE instead of this section.
2. Otherwise, in the FIRST <think>, write a SHORT [PLAN] with 2–3 candidate ROIs (not more).
   Each item: bbox hint [x1,y1,x2,y2] (0-1000, ORIGINAL image) + one-line verification goal. Use "- [ ]".
3. Execute ROIs SEQUENTIALLY (one zoom_in per turn). After a crop, [OBSERVATION] then briefly note pass/fail.
4. You do NOT need to finish every plan item: if 1–2 inspections already decide the choice, answer now.
5. Do NOT maintain a structured evidence table; do NOT de-duplicate / merge boxes.
6. Keep turns short. Avoid re-printing the full plan every turn — only update the active item.
"""


def _core_evidence() -> str:
    return """VARIANT CONSTRAINTS — Multi-ROI + light evidence tracking (NO de-duplication):
1. If Given-ROI applies, follow GIVEN-ROI RULE; still keep a compact [EVIDENCE] after the crop.
2. Otherwise keep a SHORT Multi-ROI [PLAN] (2–3 ROIs) with sequential verification.
3. Maintain a COMPACT [EVIDENCE] note (update after each tool return; keep under ~6 lines):
   - status: pending/confirmed/rejected per plan item (one line)
   - key_findings: 1–3 short facts that support the choice
4. Aggregate the final answer from [EVIDENCE], not from the last crop alone.
5. De-duplication DISABLED: do not skip overlapping ROIs or merge duplicate detections.
6. Answer as soon as evidence is enough; unfinished plan items are OK.
"""


def _core_full() -> str:
    return """VARIANT CONSTRAINTS — Full GeoVista (Multi-ROI + evidence + de-duplication):
1. If Given-ROI applies, follow GIVEN-ROI RULE (single ROI); still update [EVIDENCE] from that crop.
2. Otherwise keep Multi-ROI [PLAN] with sequential branch verification (bboxes in ORIGINAL 0-1000).
3. Keep and update the shared [EVIDENCE] block after every tool return
   (inspected_rois, verification_status, confirmed/rejected findings, unresolved_subgoals).
4. After each crop: [OBSERVATION] of THIS crop before updating [EVIDENCE].
5. DE-DUPLICATION (REQUIRED when multi-ROI):
   - Before requesting a new ROI, check overlap with already inspected regions; skip near-duplicates
     (high IoU with a visited ROI) unless a distinct verification goal remains.
   - Map all object mentions / boxes back to GLOBAL 0-1000 coordinates.
   - Before answering, merge or match duplicate detections of the same target across branches
     so the same object is not double-counted.
6. Aggregate the final answer from the de-duplicated [EVIDENCE] state.
"""


_CORES = {
    "single_path": _core_single_path,
    "multi_roi": _core_multi_roi,
    "evidence": _core_evidence,
    "full": _core_full,
}


def build_ablation_sop(
    variant: str,
    task_family: str = "mcq",
    max_tool_calls: int = 8,
) -> str:
    """Build a full system SOP for one ablation variant.

    task_family: 'mcq' (XLRS/XHR/MME) or 'open' (LRS-VQA).
    """
    canon = normalize_ablation(variant)
    if canon is None:
        raise ValueError("build_ablation_sop requires a concrete variant, not None/default")
    if task_family not in ("mcq", "open"):
        raise ValueError("task_family must be 'mcq' or 'open'")

    meta = ABLATION_META[canon]
    parts = [
        _task_header(task_family),
        f"Ablation variant: {meta['label']}",
        _TOOL_RULES.strip(),
        _GIVEN_ROI.strip(),
        _BUDGET_RULE.format(max_tool_calls=max_tool_calls).strip(),
        _EARLY_STOP.strip(),
        _CORES[canon]().strip(),
        _answer_block(task_family),
    ]
    return "\n\n".join(parts)


def sop_for_dataset(variant: str | None, dataset_name: str, max_tool_calls: int = 8) -> str | None:
    """Return ablation SOP for a dataset, or None to keep the default DATASETS sop."""
    canon = normalize_ablation(variant)
    if canon is None:
        return None
    task_family = "open" if dataset_name == "lrs" else "mcq"
    return build_ablation_sop(canon, task_family=task_family, max_tool_calls=max_tool_calls)
