import os
import re
import json
import base64
import math
import time
from io import BytesIO
from typing import Optional, Sequence
from PIL import Image
from openai import OpenAI

from geovista.token_usage import record_chat_response, summarize_turn_usages

# 允许加载大图。Qwen-VL 对短边 < 28 的裁剪会直接报错，这里把过小的 crop 补成至少 28px。
Image.MAX_IMAGE_PIXELS = None
_MIN_IMAGE_SIDE = 28
# Eval-only: enlarge tiny crops so fine attributes remain readable (GeoReflex-style).
_EVAL_UPSAMPLE_MIN_SIDE = 224
# Tiny given-ROI boxes on 10k images (3–8 units) → expand before crop.
_EVAL_MIN_CROP_SIDE_1000 = 20
# Qwen2/3-VL smart_resize rejects extreme aspect ratios.
_QWEN_VL_MAX_ASPECT = 199.0
# Train-aligned area budget (SFT image_max_pixels). Override via env.
_DEFAULT_MAX_PIXELS = int(os.getenv("GEOVISTA_MAX_PIXELS", "1003520"))
# Optional longest-side cap (e.g. 2048). Empty / 0 = disabled.
_env_max_res = os.getenv("GEOVISTA_MAX_RESOLUTION", "").strip()
_DEFAULT_MAX_RESOLUTION = int(_env_max_res) if _env_max_res.isdigit() and int(_env_max_res) > 0 else None
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.I)
_ANSWER_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.I)
_RES_RE = re.compile(
    r"(?:Image resolution|resolution)\s*:\s*(\d+)\s*[x×X]\s*(\d+)",
    re.I,
)
_GIVEN_PIXEL_BBOX_RE = re.compile(
    r"Bounding box:\s*\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*"
    r"-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]",
    re.I,
)

GIVEN_ROI_EVAL_HINT = (
    "[Given-ROI] This question already names Image resolution W x H and a pixel Bounding box.\n"
    "In <think> FIRST compute 0-1000: round(x/W*1000), round(y/H*1000). "
    "zoom_in must use ONLY those four integers — never the raw pixel box.\n"
    "After the crop returns: write a complete [OBSERVATION] of the object in THIS crop "
    "(hue, material, shape). Do not answer from the global view."
)


def parse_image_resolution(text: Optional[str]) -> Optional[tuple[int, int]]:
    """Parse ``Image resolution: W x H`` from XLRS/XHR prompts."""
    if not text:
        return None
    match = _RES_RE.search(text)
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return width, height


def maybe_pixel_bbox_to_1000(
    bbox: Sequence[float],
    image_size: tuple[int, int],
    question: Optional[str] = None,
) -> list[float]:
    """Map a pixel-space bbox onto 0–1000; leave already-normalized boxes alone.

    Policies often copy the question's absolute pixel box into ``zoom_in``. Treating
    those as 0–1000 collapses the crop to an edge strip. Convert when any
    coordinate is outside 0–1000.
    """
    box = [float(v) for v in bbox[:4]]
    if max(abs(v) for v in box) <= 1000.0:
        return box
    ref_w, ref_h = image_size
    stated = parse_image_resolution(question)
    if stated is not None:
        stated_w, stated_h = stated
        if max(box[0], box[2]) <= stated_w * 1.02 and max(box[1], box[3]) <= stated_h * 1.02:
            ref_w, ref_h = stated_w, stated_h
    ref_w = max(int(ref_w), 1)
    ref_h = max(int(ref_h), 1)
    return [
        box[0] / ref_w * 1000.0,
        box[1] / ref_h * 1000.0,
        box[2] / ref_w * 1000.0,
        box[3] / ref_h * 1000.0,
    ]


def upsample_small_image(image: Image.Image, min_side: int = _EVAL_UPSAMPLE_MIN_SIDE) -> Image.Image:
    """Bicubic-upscale so the short side is at least ``min_side``."""
    width, height = image.size
    short = min(width, height)
    if short <= 0 or short >= min_side:
        return image
    scale = min_side / short
    return image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.BICUBIC,
    )


def question_has_given_pixel_roi(question: Optional[str]) -> bool:
    if not question:
        return False
    return parse_image_resolution(question) is not None and bool(
        _GIVEN_PIXEL_BBOX_RE.search(question)
    )


def expand_bbox_min_side_1000(
    bbox: Sequence[float], min_side: int = _EVAL_MIN_CROP_SIDE_1000
) -> list[float]:
    """Grow a 0–1000 box around its center so both sides are ≥ ``min_side``."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if min_side <= 0:
        return [x1, y1, x2, y2]
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if width < min_side:
        x1, x2 = cx - min_side / 2.0, cx + min_side / 2.0
    if height < min_side:
        y1, y2 = cy - min_side / 2.0, cy + min_side / 2.0

    def _shift(lo: float, hi: float) -> tuple[float, float]:
        if hi - lo > 1000.0:
            return 0.0, 1000.0
        if lo < 0.0:
            hi -= lo
            lo = 0.0
        if hi > 1000.0:
            lo -= hi - 1000.0
            hi = 1000.0
        return max(0.0, lo), min(1000.0, hi)

    x1, x2 = _shift(x1, x2)
    y1, y2 = _shift(y1, y2)
    return [x1, y1, x2, y2]


def pad_for_qwen_vl(image: Image.Image) -> Image.Image:
    """Pad so Qwen-VL image processor accepts the crop (min side + aspect)."""
    w, h = image.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (_MIN_IMAGE_SIDE, _MIN_IMAGE_SIDE), (0, 0, 0))
    need_w, need_h = w, h
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio >= _QWEN_VL_MAX_ASPECT:
        if w >= h:
            need_h = max(h, math.ceil(w / _QWEN_VL_MAX_ASPECT), _MIN_IMAGE_SIDE)
        else:
            need_w = max(w, math.ceil(h / _QWEN_VL_MAX_ASPECT), _MIN_IMAGE_SIDE)
    need_w = max(need_w, _MIN_IMAGE_SIDE)
    need_h = max(need_h, _MIN_IMAGE_SIDE)
    if need_w == w and need_h == h:
        return image
    canvas = Image.new("RGB", (need_w, need_h), (0, 0, 0))
    canvas.paste(image, (0, 0))
    return canvas


def enforce_one_step(out: str) -> str:
    """At most one tool call per turn; a tool call beats a same-turn answer."""
    text = out or ""
    if re.search(r"</think>", text, re.I) and not re.search(r"<think\b", text, re.I):
        text = "<think>\n" + text.lstrip()
    tools = list(_TOOL_CALL_RE.finditer(text))
    if tools:
        first = tools[0]
        before = _ANSWER_RE.sub("", text[: first.start()])
        tool_block = first.group(0)
        if not tool_block.endswith("</tool_call>"):
            tool_block = tool_block.rstrip() + "</tool_call>"
        return before.rstrip() + "\n\n" + tool_block + "\n"
    am = _ANSWER_RE.search(text)
    if am:
        return text[: am.end()].rstrip() + "\n"
    return text


def zoom_in_tool(arguments: dict, context: dict) -> dict:
    bbox = arguments.get("bbox")

    if not isinstance(bbox, list) or len(bbox) < 4:
        return {"error": f"Invalid bbox argument. Expected exactly 4 numbers [x1, y1, x2, y2]."}

    original_image = context.get("original_image")
    width, height = original_image.size
    question = context.get("question") or ""

    try:
        bbox = maybe_pixel_bbox_to_1000(bbox[:4], (width, height), question)
        bbox = expand_bbox_min_side_1000(bbox, _EVAL_MIN_CROP_SIDE_1000)
        x1 = max(0, int((bbox[0] / 1000.0) * width))
        y1 = max(0, int((bbox[1] / 1000.0) * height))
        x2 = min(width, int((bbox[2] / 1000.0) * width))
        y2 = min(height, int((bbox[3] / 1000.0) * height))

        if x2 <= x1:
            x2 = min(width, x1 + 10)
        if y2 <= y1:
            y2 = min(height, y1 + 10)

        cropped_img = pad_for_qwen_vl(original_image.crop((x1, y1, x2, y2)))
        cropped_img = upsample_small_image(cropped_img)
        return {
            "text": (
                f"System Return: Cropped image for bbox {[round(v, 1) for v in bbox]} "
                "(0-1000, relative to the original image). "
                "Write a new <think> with [OBSERVATION] of THIS crop "
                "(hue/material/shape/texture); do not answer from the global view alone."
            ),
            "image": cropped_img,
        }
    except Exception as e:
        return {"error": f"Internal logic error during cropping: {e}."}

class GeoVistaAgent:
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str = "empty",
        max_turns: int = 30,
        max_tool_calls: Optional[int] = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        # 传入动态配置的 api_key
        self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=300.0)
        self.max_turns = max_turns
        # None = unlimited (legacy). Ablation should pass e.g. 8.
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        
        self.tools = {}
        self.messages = []
        self.n_turns = 0
        self.n_tool_calls = 0
        self.budget_exhausted = False
        self.turn_usages: list[dict] = []
        
    def register_tool(self, tool_name: str, func: callable):
        self.tools[tool_name] = func
        # 保持原有的注册提示
        # print(f"🔧 tool registered: [{tool_name}]")

    @staticmethod
    def encode_image(
        image: Image.Image,
        max_resolution: Optional[int] = _DEFAULT_MAX_RESOLUTION,
        max_pixels: Optional[int] = _DEFAULT_MAX_PIXELS,
    ) -> str:
        """将 PIL 图像转换为 Base64。

        默认保留面积降采样（≈1M，对齐 SFT）。可用环境变量覆盖：
          GEOVISTA_MAX_PIXELS / GEOVISTA_MAX_RESOLUTION（如 2048）。
        max_resolution: None/0 = 不按最长边 thumbnail；>0 则先 thumbnail。
        max_pixels: 0/None = 不按面积收缩。
        """
        img = upsample_small_image(image.copy())
        if max_resolution and (img.width > max_resolution or img.height > max_resolution):
            img.thumbnail((max_resolution, max_resolution), Image.Resampling.LANCZOS)
        if max_pixels and img.width * img.height > max_pixels:
            scale = (max_pixels / (img.width * img.height)) ** 0.5
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    def reset_memory(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def _call_llm(self):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=2048,
                frequency_penalty=0.0,
                stop=["</tool_call>", "</answer>"] 
            )
        except Exception as e:
            print(f"❌ LLM API Exception ({e}). Pausing for 10s to prevent avalanche...")
            time.sleep(10)
            return None
        
        raw_output = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        self._record_usage(response)

        if finish_reason == "stop":
            if "<tool_call>" in raw_output and "</tool_call>" not in raw_output:
                raw_output += "</tool_call>"
            elif "<answer>" in raw_output and "</answer>" not in raw_output:
                raw_output += "</answer>"
        elif finish_reason == "length":
            print("⚠️ Warning: exceed max tokens.")
                
        return raw_output

    def _record_usage(self, response) -> None:
        self.turn_usages.append(record_chat_response(self.messages, response))

    def token_summary(self) -> dict:
        summary = summarize_turn_usages(self.turn_usages)
        summary["n_turns"] = self.n_turns
        summary["n_tool_calls"] = self.n_tool_calls
        summary["budget_exhausted"] = self.budget_exhausted
        return summary

    def _extract_tool_call(self, text: str) -> str:
        if not text: return None
        match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if match: return match.group(1)
        
        match = re.search(r'(\{.*"name"\s*:.*\})', text, re.DOTALL)
        if match: 
            tool_str = match.group(1)
            if tool_str.count('{') > tool_str.count('}'):
                tool_str += '}' * (tool_str.count('{') - tool_str.count('}'))
            return tool_str
        return None

    def run(self, user_prompt: str, image_path: str, system_prompt: str, context_kwargs: dict = None, q_id: str = "Unknown"):
        if context_kwargs is None:
            context_kwargs = {}
            
        # 多线程环境下建议关闭逐个文件的起始提示，避免终端混乱
        # print(f"\n🚀 Starting GeoVista Agent | Target Image: {os.path.basename(image_path)}")
        
        self.reset_memory(system_prompt)
        self.n_turns = 0
        self.n_tool_calls = 0
        self.budget_exhausted = False
        self.turn_usages = []
        original_img = Image.open(image_path).convert("RGB")
        context_kwargs['original_image'] = original_img
        # Keep the raw question text so zoom_in can recover pixel→0-1000 bboxes.
        context_kwargs.setdefault("question", user_prompt)
        
        safe_prompt = user_prompt + "\nPlease strictly follow the SOP. Begin your response with <think>."
        
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": safe_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(original_img)}"}}
            ]
        })

        for turn in range(1, self.max_turns + 1):
            model_output = self._call_llm()

            if model_output is None:
                return "Error: LLM API Failure or Timeout"
            model_output = enforce_one_step(model_output)

            self.n_turns += 1
            self.messages.append({"role": "assistant", "content": model_output})            
            # 解析答案
            if "<answer>" in model_output:
                answer_match = re.search(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
                final_answer = answer_match.group(1).strip() if answer_match else "No answer parsed"
                return final_answer
            
            # 无视 <think>，强行提取可能的单选答案 (针对选择题的保底逻辑)
            text_without_think = re.sub(r'<think>.*?</think>', '', model_output, flags=re.DOTALL).strip()
            if re.fullmatch(r'\(?[A-E]\)?', text_without_think, re.IGNORECASE):
                return text_without_think.replace("(", "").replace(")", "").upper()
                
            tool_str_raw = self._extract_tool_call(model_output)
            
            if tool_str_raw:
                if self.max_tool_calls is not None and self.n_tool_calls >= self.max_tool_calls:
                    self.budget_exhausted = True
                    self.messages.append({
                        "role": "user",
                        "content": (
                            f"TOOL BUDGET EXHAUSTED: you already used {self.n_tool_calls}/{self.max_tool_calls} "
                            "zoom_in calls. Do NOT call tools again. Produce the final answer now inside "
                            "<answer>...</answer> based on evidence gathered so far."
                        ),
                    })
                    continue

                try:
                    tool_str = tool_str_raw.strip()
                    tool_data = json.loads(tool_str)
                except json.JSONDecodeError:
                    # 暴力正则抢救机制
                    match = re.search(r'\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+).*?\]', tool_str_raw)
                    if match:
                        tool_data = {"name": "zoom_in", "arguments": {"bbox": [float(m) for m in match.groups()]}}
                    else:
                        self.messages.append({"role": "user", "content": "Malformed JSON. Please output a VALID JSON inside <tool_call>.</tool_call>."})
                        continue
                        
                tool_name = tool_data.get("name")
                tool_args = tool_data.get("arguments", {})
                
                if tool_name in self.tools:
                    # print(f"🔧 Scheduler successfully intercepted tool call: [{tool_name}] Arguments: {tool_args}")
                    tool_func = self.tools[tool_name]
                    tool_result = tool_func(tool_args, context_kwargs)
                    self.n_tool_calls += 1
                    
                    if "error" in tool_result:
                        self.messages.append({"role": "user", "content": tool_result["error"]})
                    else:
                        observe_text = tool_result["text"]
                        if self.max_tool_calls is not None:
                            remain = self.max_tool_calls - self.n_tool_calls
                            observe_text += (
                                f"\n[Budget] zoom_in used {self.n_tool_calls}/{self.max_tool_calls} "
                                f"({remain} remaining)."
                            )
                        self.messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": observe_text},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(tool_result['image'])}"}}
                            ]
                        })
                else:
                    print(f"⚠️ Tool [{tool_name}] is not registered!")
                    self.messages.append({"role": "user", "content": f"Error: Tool [{tool_name}] not found."})
            else:
                warning_msg = (
                    "WARNING: You stopped generating without making a <tool_call> or providing an <answer>. "
                    "If you need to use a tool, output EXACTLY: <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>. "
                    "If you have confidently found the target or finished the reasoning, output your final result wrapped in <answer>...</answer>."
                )
                self.messages.append({"role": "user", "content": warning_msg})
                continue

        if self.budget_exhausted:
            return "Error: Tool budget exhausted without answer"
        # print("\n⚠️ Reached maximum turns, forcing termination.")
        return "Error: Max turns reached"