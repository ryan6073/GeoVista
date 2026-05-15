import re
import math
import json
import logging
import os
from typing import List, Dict, Any

# ==========================================
# 0. 日志配置 (Logging Configuration)
# ==========================================
# 创建一个专门针对当前模块的 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 防止重复添加 handler
if not logger.handlers:
    # 设置日志输出文件，你可以修改这个路径，比如 '/workspace/logs/reward_debug.log'
    log_file_path = 'reward_debug.log'
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 设置日志格式：时间 - 进程ID - 级别 - 信息
    formatter = logging.Formatter('%(asctime)s - PID:%(process)d - [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)


class AgentZoomRewardManager:
    def __init__(self, gamma=0.8, beta=0.2, lambda_g=120.0, lambda_c=0.15):
        self.gamma = gamma    
        self.beta = beta      
        self.lambda_g = lambda_g
        self.lambda_c = lambda_c

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
            if has_tool and has_ans: # 同一轮中互斥
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

# ==========================================
# 2. verl 适配层接口
# ==========================================
def compute_score(solution_str: str, ground_truth: str, extra_info: Any = None, data_source: str = None, **kwargs) -> Dict:
    predict_str = solution_str
    manager = AgentZoomRewardManager()
    
    logger.debug("="*60)
    logger.debug("--- 开始计算本次 Reward ---")
    
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

    logger.debug(f"1. 数据解析情况：Task Type: {task_type} | GT Val: {gt_val} | GT BBoxes 数量: {len(gt_bboxes)}")
    logger.debug(f"   GT BBoxes 内容: {gt_bboxes}")

    # --- 1. Format Score ---
    last_think_pos = predict_str.rfind("<think>")
    current_turn_content = predict_str[last_think_pos:] if last_think_pos != -1 else predict_str
    r_format = manager.get_format_reward(current_turn_content)
    
    logger.debug(f"2. 格式得分 (r_format): {r_format}")

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
    
    logger.debug(f"3. IOU 定位情况：模型预测的 BBoxes 数量: {len(unique_pred_bboxes)}")
    logger.debug(f"   定位得分 (r_iou): {r_iou}")

    # --- 3. Acc Score ---
    ans_match = re.search(r'<answer>(.*?)</answer>', current_turn_content, re.DOTALL)
    final_ans = ans_match.group(1).strip() if ans_match else None
    r_ans = manager.get_answer_reward(task_type, final_ans, gt_val, gt_bboxes)
    
    logger.debug(f"4. Acc 答案情况：提取的 Answer: {final_ans} | 答案得分 (r_ans): {r_ans}")

    # --- 4. Process Score (无兜底逻辑) ---
    if task_type == 'counting':
        D = 1.0 - math.exp(-manager.lambda_c * len(gt_bboxes))
    else:
        area = (gt_bboxes[0][2]-gt_bboxes[0][0]) * (gt_bboxes[0][3]-gt_bboxes[0][1]) if gt_bboxes else 1000.0
        D = math.exp(-manager.lambda_g * (area / 1000000.0))

    has_plan = "[PLAN]" in predict_str or "- [ ]" in predict_str or "[PROGRESS]" in predict_str
    r_plan = D if has_plan else -manager.beta * D

    logger.debug(f"5. Process 过程分情况：难度(D): {D:.4f} | 有Plan: {has_plan} | 过程得分(r_plan): {r_plan:.4f}")

    # 一票否决
    if r_format < 0:
        logger.warning(f"！！！触发一票否决 (r_format < 0)！！！")
        logger.debug("="*60)
        return {"acc": 0.0, "process": 0.0, "format": float(r_format), "iou": 0.0, "score": float(r_format)}

    total_score = float(r_ans + r_plan + r_format + r_iou)
    
    logger.info(f">>> 最终总分 (total_score): {total_score:.4f} [Acc:{r_ans:.2f}, Plan:{r_plan:.2f}, Fmt:{r_format:.2f}, IoU:{r_iou:.2f}]")
    logger.debug("="*60)

    return {
        "acc": r_ans,
        "process": r_plan,
        "format": r_format,
        "iou": r_iou,
        "score": total_score
    }