import re
import math
import json
import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    log_file_path = 'reward_debug.log'
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - PID:%(process)d - [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)


class AgentZoomRewardManager:
    def __init__(
        self,
        gamma=0.8,
        beta=0.2,
        lambda_g=120.0,
        lambda_c=0.15,
        alpha_plan=1.0,
        beta_plan=0.2,
        gamma_plan=0.8,
    ):
        self.gamma = gamma
        self.beta = beta
        self.lambda_g = lambda_g
        self.lambda_c = lambda_c
        self.alpha_plan = alpha_plan
        self.beta_plan = beta_plan
        self.gamma_plan = gamma_plan

    @staticmethod
    def compute_iou(box1, box2):
        if not box1 or not box2: return 0.0
        
        # 1. 坐标系反转保护 (确保 x1 < x2, y1 < y2)
        b1 = [min(box1[0], box1[2]), min(box1[1], box1[3]), max(box1[0], box1[2]), max(box1[1], box1[3])]
        b2 = [min(box2[0], box2[2]), min(box2[1], box2[3]), max(box2[0], box2[2]), max(box2[1], box2[3])]
        
        x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area <= 0: return 0.0
        
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter_area / float(area1 + area2 - inter_area + 1e-8)

    def get_format_reward(self, current_content):
        has_think = "<think>" in current_content and "</think>" in current_content
        has_tool = "<tool_call>" in current_content and "</tool_call>" in current_content
        has_ans = "<answer>" in current_content and "</answer>" in current_content
        
        if has_think and (has_tool or has_ans):
            if has_tool and has_ans:
                return -self.gamma
            return 1.0
        return -self.gamma

    def get_answer_reward(self, task_type, pred_str, gt_val, gt_bboxes):
        if pred_str is None: return 0.0
        
        try:
            if task_type == 'counting':
                num_match = re.search(r'\d+\.?\d*', pred_str)
                if not num_match: return 0.0
                
                py, gy = float(num_match.group()), float(gt_val)
                denom = math.pow(gy + 1e-6, 2/3)
                r = 1.0 - (abs(py - gy) / denom)
                return max(0.0, float(r))
            else:
                box_match = re.search(r'\[(.*?)\]', pred_str)
                if not box_match: return 0.0
                
                box_nums = [float(x.strip()) for x in box_match.group(1).split(',')]
                if len(box_nums) < 4 or not gt_bboxes: return 0.0
                
                cx_p, cy_p = (box_nums[0]+box_nums[2])/2, (box_nums[1]+box_nums[3])/2
                cx_g, cy_g = (gt_bboxes[0][0]+gt_bboxes[0][2])/2, (gt_bboxes[0][1]+gt_bboxes[0][3])/2
                dist = math.sqrt((cx_p - cx_g)**2 + (cy_p - cy_g)**2)
                return max(0.0, math.tanh(150.0 / (1.5 * dist + 120.0))) 
        except:
            return 0.0

    def get_iou_reward(self, unique_pred_bboxes, gt_bboxes):
        if not gt_bboxes or not unique_pred_bboxes: return 0.0
        
        num_gt = len(gt_bboxes)
        matches = []
        for i, gb in enumerate(gt_bboxes):
            for j, pb in enumerate(unique_pred_bboxes):
                matches.append({'gt_idx': i, 'pred_idx': j, 'iou': self.compute_iou(gb, pb)})
        
        matches.sort(key=lambda x: x['iou'], reverse=True)
        used_gt, used_pred = set(), set()
        total_matched_iou = 0.0
        
        for m in matches:
            if m['gt_idx'] not in used_gt and m['pred_idx'] not in used_pred:
                total_matched_iou += m['iou']
                used_gt.add(m['gt_idx'])
                used_pred.add(m['pred_idx'])
        
        return total_matched_iou / num_gt


_BBOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_CHECK_RE = re.compile(r"^[\-\*\u2022]\s*\[\s*([ xX])\s*\]\s*(.+?)\s*$", re.M)
_MARKER_RE = re.compile(
    r"\[PLAN\]|\[PROGRESS\]|TOOL_EXECUTION_SUCCESS|TOOL_EXECUTION_FAILED|<answer>",
    re.I,
)
_BOUNDARY_RE = re.compile(
    r"</think>|<tool_call>|<answer>|\[PROGRESS\]|\[OBSERVATION\]|\[EVIDENCE\]|\[PLAN\]|TOOL_EXECUTION_",
    re.I,
)


def plan_reward(
    events,
    difficulty,
    alpha_plan=1.0,
    beta_plan=0.2,
    gamma_plan=0.8,
):
    """Score plan execution.

    - Complete, legal execution: +α_plan · D
    - Illegal execution: -γ_plan
    - No plan: -β_plan · D

    Illegal includes empty/duplicate items, multiple plans, tool failure,
    spatial id mismatch (exact rounded bbox id, no IoU soft match), out-of-order
    tracking, progress without a prior successful tool, and answering early.
    """
    plans = [e for e in events if e["type"] == "plan"]
    if not plans:
        return -beta_plan * difficulty

    illegal = -gamma_plan

    if len(plans) != 1:
        return illegal

    items = plans[0].get("items", [])
    if not items or len(items) != len(set(items)):
        return illegal

    next_item = 0
    observed = False
    answered = False

    for event in events:
        kind = event["type"]

        if kind == "tool_result":
            if (
                next_item >= len(items)
                or event.get("plan_id") != items[next_item]
                or not event.get("success", False)
            ):
                return illegal
            observed = True

        elif kind == "progress":
            if (
                next_item >= len(items)
                or event.get("completed") != items[next_item]
                or not observed
            ):
                return illegal
            next_item += 1
            observed = False

        elif kind == "answer":
            answered = True
            break

    if answered and next_item == len(items) and not observed:
        return alpha_plan * difficulty
    return illegal


def _parse_bbox(text):
    match = _BBOX_RE.search(text or "")
    if not match:
        return None
    return tuple(float(match.group(i)) for i in range(1, 5))


def _item_id(text, bbox):
    if bbox is not None:
        return ",".join(str(int(round(v))) for v in bbox)
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _match_completed(body, plan_items):
    """Align a PROGRESS [x] line to a plan item by exact id / text (no IoU)."""
    bbox = _parse_bbox(body)
    ident = _item_id(body, bbox)
    for item in plan_items:
        if item["id"] == ident:
            return item["id"]
    norm = re.sub(r"\s+", " ", body or "").strip().lower()
    for item in plan_items:
        item_text = re.sub(r"\s+", " ", item["text"]).strip().lower()
        if item_text and (item_text in norm or norm in item_text):
            return item["id"]
    return ident


def parse_plan_events(text: str) -> list:
    """Turn an RL trajectory into ordered plan / tool / progress / answer events.

    Plan item ids are the checklist bbox when present, otherwise the item text.
    Tool observations are aligned by the exact rounded bbox id logged in
    TOOL_EXECUTION_* (same id scheme as plan items), not by IoU.
    """
    if not text:
        return []
    events = []
    plan_items = []
    for match in _MARKER_RE.finditer(text):
        kind = match.group(0).upper()
        boundary = _BOUNDARY_RE.search(text, match.end())
        end = boundary.start() if boundary else len(text)
        block = text[match.end():end]
        if kind == "[PLAN]":
            items = []
            for checked in _CHECK_RE.finditer(block):
                if checked.group(1).lower() == "x":
                    continue
                body = checked.group(2).strip()
                bbox = _parse_bbox(body)
                item_id = _item_id(body, bbox)
                items.append(item_id)
                plan_items.append({"id": item_id, "bbox": bbox, "text": body})
            events.append({"type": "plan", "items": items})
        elif kind == "[PROGRESS]":
            for checked in _CHECK_RE.finditer(block):
                if checked.group(1).lower() != "x":
                    continue
                events.append({
                    "type": "progress",
                    "completed": _match_completed(checked.group(2).strip(), plan_items),
                })
        elif "SUCCESS" in kind:
            bbox = _parse_bbox(text[match.end():match.end() + 500])
            events.append({
                "type": "tool_result",
                "plan_id": _item_id("", bbox) if bbox is not None else None,
                "success": True,
            })
        elif "FAILED" in kind:
            bbox = _parse_bbox(text[match.end():match.end() + 300])
            events.append({
                "type": "tool_result",
                "plan_id": _item_id("", bbox) if bbox is not None else None,
                "success": False,
            })
        elif kind == "<ANSWER>":
            events.append({"type": "answer"})
            break
    return events


def compute_score(solution_str: str, ground_truth: str, extra_info: Any = None, data_source: str = None, **kwargs) -> Dict:
    predict_str = solution_str
    manager = AgentZoomRewardManager()
    
    logger.debug("="*60)
    logger.debug("--- Start Computing Reward ---")
    
    # 解析 Parquet 数据
    extra_info = extra_info or {}
    gt_bboxes = extra_info.get('rel_bboxs', [])
    if not gt_bboxes:
        ei_str = extra_info.get('extra_info', '{}')
        ei_data = json.loads(ei_str) if isinstance(ei_str, str) else ei_str
        gt_bboxes = ei_data.get('rel_bboxs', [])

    rm_str = extra_info.get('reward_model', '{}')
    rm_data = json.loads(rm_str) if isinstance(rm_str, str) else rm_str
    task_type = rm_data.get('task_type', extra_info.get('ability', 'counting'))
    gt_val = rm_data.get('ground_truth', ground_truth)

    logger.debug(f"1. Data Parsing: Task Type: {task_type} | GT Val: {gt_val} | GT BBoxes Count: {len(gt_bboxes)}")
    logger.debug(f"   GT BBoxes Type: {type(gt_bboxes)}")

    # --- 1. Format Score ---
    last_think_pos = predict_str.rfind("<think>")
    current_turn_content = predict_str[last_think_pos:] if last_think_pos != -1 else predict_str
    r_format = manager.get_format_reward(current_turn_content)
    
    logger.debug(f"2. Format Score (r_format): {r_format}")

    # --- 2. IOU Score ---
    all_unique_boxes = {} 
    for m in re.finditer(r'Obj:\s*\[(.*?)\]', predict_str):
        try:
            coords = tuple([float(x.strip()) for x in m.group(1).split(',')])
            if len(coords) == 4:
                all_unique_boxes[coords] = list(coords)
        except: continue
    
    unique_pred_bboxes = list(all_unique_boxes.values())
    r_iou = manager.get_iou_reward(unique_pred_bboxes, gt_bboxes)
    
    logger.debug(f"3. IOU positioning status: Number of BBoxes predicted by the model is {len(unique_pred_bboxes)}")
    logger.debug(f"   IOU score (r_iou): {r_iou}")

    # --- 3. Acc Score ---
    ans_match = re.search(r'<answer>(.*?)</answer>', current_turn_content, re.DOTALL)
    final_ans = ans_match.group(1).strip() if ans_match else None
    r_ans = manager.get_answer_reward(task_type, final_ans, gt_val, gt_bboxes)
    
    logger.debug(f"4. Acc Answer Status: Extracted Answer: {final_ans} | Answer Score (r_ans): {r_ans}")

    if task_type == 'counting':
        D = 1.0 - math.exp(-manager.lambda_c * len(gt_bboxes))
    else:
        area = (gt_bboxes[0][2]-gt_bboxes[0][0]) * (gt_bboxes[0][3]-gt_bboxes[0][1]) if gt_bboxes else 1000.0
        D = math.exp(-manager.lambda_g * (area / 1000000.0))

    try:
        plan_events = parse_plan_events(predict_str)
        r_plan = float(
            plan_reward(
                plan_events,
                D,
                alpha_plan=manager.alpha_plan,
                beta_plan=manager.beta_plan,
                gamma_plan=manager.gamma_plan,
            )
        )
    except Exception as exc:
        logger.warning(f"plan reward parse failed ({exc}); assigning penalty")
        plan_events = []
        r_plan = float(-manager.gamma_plan)

    logger.debug(
        f"5. Process classification: Difficulty (D): {D:.4f} | events={len(plan_events)} | Process Score (r_plan): {r_plan:.4f}"
    )

    # 一票否决
    if r_format < 0:
        logger.warning(f"!!! Triggered a veto (r_format < 0)!!!")
        logger.debug("="*60)
        return {"acc": 0.0, "process": 0.0, "format": float(r_format), "iou": 0.0, "score": float(r_format)}

    total_score = float(r_ans + r_plan + r_format + r_iou)
    
    logger.info(f">>> Final Total Score: {total_score:.4f} [Acc:{r_ans:.2f}, Plan:{r_plan:.2f}, Fmt:{r_format:.2f}, IoU:{r_iou:.2f}]")
    logger.debug("="*60)

    return {
        "acc": r_ans,
        "process": r_plan,
        "format": r_format,
        "iou": r_iou,
        "score": total_score
    }