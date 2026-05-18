import os
from dotenv import load_dotenv

# 自动向上寻找并加载根目录的 .env 文件
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

# ==========================================
# 环境变量读取 (带有 Fallback 默认值)
# ==========================================
BENCHMARK_ROOT = os.getenv("BENCHMARK_ROOT", "/srv/user/zhujs/benchmark")
WORK_DIR = os.getenv("WORK_DIR", "/srv/user/zhujs/work/qwen2_5-RL/result")
API_BASE = os.getenv("API_BASE", "http://localhost:8100/v1")
API_KEY = os.getenv("API_KEY", "empty")

# ==========================================
# 算法 SOP 定义
# ==========================================
XLRS_SOP = """Task: Complex Visual Reasoning and Interpretation.
Algorithm SOP (STRICT CHECKLIST PROCESS):
1. PLAN PHASE: In your first <think> block, carefully analyze the question, the options, and the global visual context. If fine-grained details or specific regional features are needed to deduce the correct option, plan to inspect that area.
2. EXECUTION RULE: If the target region or contextual clues are unclear in the global view, request a high-resolution crop via:
   <tool_call>{"name": "zoom_in", "arguments": {"bbox": [x1, y1, x2, y2]}}</tool_call>
   (Coordinates are 0-1000 relative to the image size).
   🛑 STOP GENERATING AFTER THE TOOL CALL.
3. DISCOVERY FORMAT: Upon receiving the cropped image, strictly continue your reasoning inside a new <think> block, evaluating how the new visual evidence aligns with the multiple-choice options.
4. FINAL ANSWER: Once you have confidently deduced the best interpretation, output ONLY the letter of the correct choice wrapped in <answer>...</answer>. For example: <answer>D</answer>."""

XHR_SOP = """Task: High-Resolution Remote Sensing Analysis & Anomaly Detection.
Algorithm SOP (STRICT CHECKLIST PROCESS):
1. PLAN PHASE: In your first <think> block, analyze the question and choices. If the question asks about a specific anomaly, texture, or fine detail in a particular region, explicitly plan to inspect that area.
2. EXECUTION RULE: If the target region or objects are too small or unclear in the global view to make a confident decision, request a high-resolution crop via:
   <tool_call>{"name": "zoom_in", "arguments": {"bbox": [x1, y1, x2, y2]}}</tool_call>
   (Coordinates are 0-1000 relative to the image size).
   🛑 STOP GENERATING AFTER THE TOOL CALL.
3. DISCOVERY FORMAT: Upon receiving the cropped image, strictly continue your reasoning inside a new <think> block, paying close attention to color variations, textures, or anomalies mentioned in the options.
4. FINAL ANSWER: Once you have confidently determined the correct option, output ONLY the letter of the correct choice wrapped in <answer>...</answer>. For example: <answer>B</answer>."""

LRS_SOP = """Task: Visual Reasoning and Question Answering.
Algorithm SOP (STRICT CHECKLIST PROCESS):
1. PLAN PHASE: In your first <think> block, analyze the question and write an explicit [PLAN] to locate the requested objects/regions.
2. EXECUTION RULE: If the objects are too small or unclear in the global view, request a high-resolution crop via:
   <tool_call>{"name": "zoom_in", "arguments": {"bbox": [x1, y1, x2, y2]}}</tool_call>
   (Coordinates are 0-1000 relative to the image size).
   🛑 STOP GENERATING AFTER THE TOOL CALL.
3. DISCOVERY FORMAT: Upon receiving the cropped image, strictly continue your reasoning inside a new <think> block.
4. FINAL ANSWER: Once you have confidently answered the question, output your final concise answer wrapped in <answer>...</answer>."""

# ==========================================
# 数据集元数据路由表
# ==========================================
DATASETS = {
    "xlrs": {
        "jsonl_path": os.path.join(BENCHMARK_ROOT, "XLRSBench/xlrs_final.jsonl"),
        "image_root": "", # XLRS 使用的是绝对路径
        "sop": XLRS_SOP
    },
    "xhr": {
        "jsonl_path": os.path.join(BENCHMARK_ROOT, "XHRBench/xhrbench.jsonl"),
        "image_root": os.path.join(BENCHMARK_ROOT, "XHRBench"),
        "sop": XHR_SOP
    },
    "lrs": {
        "jsonl_path": os.path.join(BENCHMARK_ROOT, "LRS-VQA/LRS_VQA_merged.jsonl"),
        "image_root": os.path.join(BENCHMARK_ROOT, "LRS-VQA/image"),
        "sop": LRS_SOP
    }
}