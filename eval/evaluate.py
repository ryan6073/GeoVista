import os
import json
import re
from collections import defaultdict
import nltk
from nltk.corpus import wordnet as wn

# ==============================================================================
# 分类映射表
# ==============================================================================
XLRS_TAXONOMY = {
    "Overall counting": ("Perception", "Counting", "Overall Counting"),
    "Regional counting": ("Perception", "Counting", "Regional Counting"),
    "Overall Land use classification": ("Perception", "Scene Classification", "Overall Land Use Classification"),
    "Regional Land use classification": ("Perception", "Scene Classification", "Regional Land Use Classification"),
    "Object spatial relationship": ("Perception", "Object Spatial Relationship", "Object Spatial Relationship"),
    "Object classification": ("Perception", "Object Properties", "Object Classification"),
    "Object color": ("Perception", "Object Properties", "Object Color"),
    "Object motion state": ("Perception", "Object Properties", "Object Motion State"),
    "Detailed Image Captioning": ("Perception", "Image Captioning", "Detailed Image Captioning"),
    "Fine-grained Visual Grounding": ("Perception", "Visual Grounding", "Fine-grained Visual Grounding"),
    "Environmental condition reasoning": ("Reasoning", "Complex Reasoning", "Environmental Condition Reasoning"),
    "Counting with complex reasoning": ("Reasoning", "Complex Reasoning", "Counting with Complex Reasoning"),
    "Anomaly Detection and Interpretation": ("Reasoning", "Anomaly Reasoning", "Anomaly Detection and Interpretation"),
    "Route planning": ("Reasoning", "Planning", "Route Planning"),
    "Condition-based Visual Grounding": ("Reasoning", "Visual Grounding", "Condition-based Visual Grounding"),
    "Counting with changing detection": ("Reasoning", "Spatiotemporal Reasoning", "Regional Counting with Change Detection")
}

XHR_TAXONOMY = {
    "color": "Perception/Local Attributes/color",
    "shape": "Perception/Local Attributes/shape",
    "detection": "Perception/Local Attributes/detection",
    "object_clasfication": "Perception/Overall Attributes/object_classification",
    "relation": "Perception/Overall Attributes/spatial_relationship",
    "object_grounding": "Perception/Visual Grounding/object_grounding",
    "regional_grouding": "Perception/Visual Grounding/regional_grounding",
    "object_counting": "Perception/Counting/object_counting",
    "regionla_counting": "Perception/Counting/regional_counting",
    "anomaly_detection": "Reasoning/Knowledge-grounded/anomaly_detection",
    "future_prediction_two_image": "Reasoning/Knowledge-grounded/future_prediction_two_image",
    "multi_region_join_contrast_singal": "Reasoning/Logic-grounded/multi_region_joint_contrast_singal",
    "multi_region_join_contrast": "Reasoning/Logic-grounded/multi_region_joint_contrast",
    "object_state_judgement": "Reasoning/Logic-grounded/object_state_judgement",
    "multturn_anomaly_detection": "Multi-turn/Knowledge-grounded/anomaly_detection",
    "multitrun-future_prediction": "Multi-turn/Knowledge-grounded/future_prediction",
    "multiturn_object_state_judgement": "Multi-turn/Logic-grounded/object_state_judgement"
}

# ==============================================================================
# 核心逻辑
# ==============================================================================
def _extract_choice(s: str) -> str:
    if not s or not isinstance(s, str): return ""
    s = s.strip()
    prefixes = ["The best answer is", "The correct answer is", "Answer:", "The answer is", "The best option is", "The correct option is", "Best answer:"]
    for prefix in prefixes:
        s = re.sub(re.escape(prefix), "", s, flags=re.IGNORECASE).strip()

    priority_matches = re.findall(r"(?:\(|^|\s)([A-E])(?:\)|\]|\.|\:|\s|$)", s.upper())
    if priority_matches: return "".join(sorted(list(set(priority_matches))))

    if len(s) < 15:
        short_matches = re.findall(r"[A-E]", s.upper())
        if short_matches: return "".join(sorted(list(set(short_matches))))

    first_char_match = re.search(r"[A-E]", s.upper())
    return first_char_match.group(0) if first_char_match else ""


def _write_hierarchical_report(metrics: dict, output_file: str, name: str, extra_metrics: list = None):
    """新增 extra_metrics 参数，用于插入特定数据集的全局指标（如 MTEM@1）"""
    lines = [f"\n{'='*20} {name} Evaluation Report {'='*20}"]
    t_corr, t_count = 0, 0
    
    for l1, l2_dict in sorted(metrics.items()):
        lines.append(f"\n==================== {l1} (L1 Task) ====================")
        l1_corr, l1_count = 0, 0
        
        for l2, l3_dict in sorted(l2_dict.items()):
            lines.append(f"---------- {l2} (L2 Task)")
            l2_corr, l2_count = 0, 0
            
            for l3, stats in sorted(l3_dict.items()):
                c, f = stats["true"], stats["false"]
                total = c + f
                acc = c / total if total > 0 else 0
                lines.append(f"  [{l3.ljust(40)}]: Acc {acc:.4f} ({c}/{total})")
                l2_corr += c
                l2_count += total
                
            l2_acc = l2_corr / l2_count if l2_count > 0 else 0
            lines.append(f"Sub-Total {l2}: Acc {l2_acc:.4f}")
            l1_corr += l2_corr
            l1_count += l2_count
            
        l1_acc = l1_corr / l1_count if l1_count > 0 else 0
        lines.append(f"Main-Total {l1}: Acc {l1_acc:.4f} ({l1_corr}/{l1_count})")
        t_corr += l1_corr
        t_count += l1_count

    overall_acc = t_corr / t_count if t_count > 0 else 0
    lines.append("\n" + "*" * 50)
    
    # 插入额外指标
    if extra_metrics:
        lines.extend(extra_metrics)
        
    lines.append(f"OVERALL ACCURACY: {overall_acc:.4f} ({t_corr}/{t_count})")
    lines.append("*" * 50)
    
    report = "\n".join(lines)
    print(report)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)


def eval_xlrs(input_file: str, output_file: str):
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"true": 0, "false": 0})))
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            pred = _extract_choice(doc.get("model_output", ""))
            ans = doc.get("answer", "")
            raw_cat = doc.get("category", "")
            cat = raw_cat.split("/")[-1].strip() if "/" in raw_cat else raw_cat.strip()
            l1, l2, l3 = XLRS_TAXONOMY.get(cat, ("Unknown_L1", "Unknown_L2", cat))
            
            is_correct = 1 if set(pred) == set(ans) and pred != "" else 0
            metrics[l1][l2][l3]["true"] += is_correct
            metrics[l1][l2][l3]["false"] += 1 - is_correct
    _write_hierarchical_report(metrics, output_file, "XLRS")


def eval_xhr(input_file: str, output_file: str):
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"true": 0, "false": 0})))
    
    # 追踪多轮交互的指标 (MTEM@1)
    mt_cats = ["multturn_anomaly_detection", "multitrun-future_prediction", "multiturn_object_state_judgement"]
    mt_tracking = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            pred = _extract_choice(doc.get("model_output", ""))
            ans = str(doc.get("answer", "")).split(")")[0].strip().upper()
            
            raw_cat = doc.get("category", "unknown")
            mapped_path = XHR_TAXONOMY.get(raw_cat, f"Other/Other/{raw_cat}")
            parts = mapped_path.split("/")
            l1, l2 = parts[0], parts[1] if len(parts) > 1 else "Unknown"
            l3 = parts[2] if len(parts) > 2 else parts[-1]
            
            is_correct = 1 if set(pred) == set(ans) and pred != "" else 0
            metrics[l1][l2][l3]["true"] += is_correct
            metrics[l1][l2][l3]["false"] += 1 - is_correct
            
            # MTEM@1 计算数据收集
            raw_cat_lower = str(raw_cat).strip().lower()
            if raw_cat_lower in mt_cats:
                img_path = doc.get("image", doc.get("image_path", doc.get("image_id", "unknown")))
                group_key = (raw_cat_lower, img_path)
                mt_tracking[group_key].append(is_correct)
                
    # 计算 MTEM@1 指标列表
    extra_lines = []
    if mt_tracking:
        mtem_correct = sum(1 for hits in mt_tracking.values() if all(h == 1 for h in hits))
        mtem_total = len(mt_tracking)
        mtem_acc = mtem_correct / mtem_total if mtem_total > 0 else 0
        extra_lines.append(f"MTEM@1 ACCURACY: {mtem_acc:.4f} ({mtem_correct}/{mtem_total})")
    else:
        extra_lines.append("MTEM@1 ACCURACY: N/A (No Multi-turn data found)")

    _write_hierarchical_report(metrics, output_file, "XHR", extra_lines)


def _are_synonyms(word1: str, word2: str) -> bool:
    try:
        for s1 in wn.synsets(word1):
            for s2 in wn.synsets(word2):
                if s1.path_similarity(s2) is not None and s1.path_similarity(s2) > 0.8:
                    return True
    except LookupError:
        nltk.download('wordnet')
        return _are_synonyms(word1, word2)
    return False


def eval_lrs(input_file: str, output_file: str):
    all_categories = set()
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    correct, total = 0, 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line)
            gt = str(doc.get('ground_truth', '')).lower().strip()
            ans_raw = doc.get('model_output')
            ans = str(ans_raw).lower().rstrip('.').strip() if ans_raw else ""
            cat = doc.get('category', 'Unknown').lower()
            
            all_categories.add(cat)
            category_total[cat] += 1
            total += 1
            
            if gt == ans or (ans and _are_synonyms(gt, ans)):
                correct += 1
                category_correct[cat] += 1

    lines = ["\n==================== LRS Evaluation Report ===================="]
    lines.append("Category-wise accuracies:")
    for cat in sorted(all_categories):
        c, t = category_correct[cat], category_total[cat]
        lines.append(f"{cat.ljust(20)}: {c}/{t} ({(c/t)*100 if t>0 else 0:.2f}%)")
        
    overall_acc = correct / total if total > 0 else 0
    lines.append("\n" + "*" * 50)
    lines.append(f"OVERALL ACCURACY: {overall_acc*100:.2f}% ({correct}/{total})")
    lines.append("*" * 50)
    
    report = "\n".join(lines)
    print(report)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)


def run_evaluation(dataset_name: str, input_file: str, output_file: str):
    print(f"📊 [Eval] Scoring {dataset_name.upper()}...")
    if dataset_name == 'xlrs': eval_xlrs(input_file, output_file)
    elif dataset_name == 'xhr': eval_xhr(input_file, output_file)
    elif dataset_name == 'lrs': eval_lrs(input_file, output_file)