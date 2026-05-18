import os
import re
import json
import base64
import time
from io import BytesIO
from PIL import Image
from openai import OpenAI

# 允许加载大图
Image.MAX_IMAGE_PIXELS = None

def zoom_in_tool(arguments: dict, context: dict) -> dict:
    bbox = arguments.get("bbox")
    
    if not isinstance(bbox, list) or len(bbox) < 4:
        return {"error": f"Invalid bbox argument. Expected exactly 4 numbers [x1, y1, x2, y2]."}
        
    original_image = context.get("original_image")
    width, height = original_image.size
    
    try:
        bbox = bbox[:4]
        # 确保使用的是 0-1000 的离散相对坐标系
        x1 = max(0, int((bbox[0] / 1000.0) * width))
        y1 = max(0, int((bbox[1] / 1000.0) * height))
        x2 = min(width, int((bbox[2] / 1000.0) * width))
        y2 = min(height, int((bbox[3] / 1000.0) * height))
        
        # 防止裁剪出面积为0的非法图像
        if x2 <= x1: x2 = min(width, x1 + 10)
        if y2 <= y1: y2 = min(height, y1 + 10)
        
        cropped_img = original_image.crop((x1, y1, x2, y2))
        return {
            "text": f"System Return: Cropped image for bbox {bbox}. Please continue your analysis INSIDE a new <think> block.",
            "image": cropped_img
        }
    except Exception as e:
        return {"error": f"Internal logic error during cropping: {e}."}


class GeoVistaAgent:
    def __init__(self, model_name: str, api_base: str, api_key: str = "empty", max_turns: int = 30, temperature: float = 0.0):
        self.model_name = model_name
        # 传入动态配置的 api_key
        self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=300.0)
        self.max_turns = max_turns
        self.temperature = temperature
        
        self.tools = {}
        self.messages = []
        
    def register_tool(self, tool_name: str, func: callable):
        self.tools[tool_name] = func
        # 保持原有的注册提示
        # print(f"🔧 tool registered: [{tool_name}]")

    @staticmethod
    def encode_image(image: Image.Image, max_resolution: int = 1024) -> str:
        """将 PIL 图像转换为 Base64，附带防爆显存保护"""
        img = image.copy()
        if img.width > max_resolution or img.height > max_resolution:
            img.thumbnail((max_resolution, max_resolution), Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def reset_memory(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def _call_llm(self):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=1024,
                frequency_penalty=1.0,
                stop=["</tool_call>", "</answer>"] 
            )
        except Exception as e:
            print(f"❌ LLM API Exception ({e}). Pausing for 10s to prevent avalanche...")
            time.sleep(10)
            return None
        
        raw_output = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        if finish_reason == "stop":
            if "<tool_call>" in raw_output and "</tool_call>" not in raw_output:
                raw_output += "</tool_call>"
            elif "<answer>" in raw_output and "</answer>" not in raw_output:
                raw_output += "</answer>"
        elif finish_reason == "length":
            print("⚠️ Warning: exceed max tokens.")
                
        return raw_output

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
        original_img = Image.open(image_path).convert("RGB")
        context_kwargs['original_image'] = original_img
        
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
                    
                    if "error" in tool_result:
                        self.messages.append({"role": "user", "content": tool_result["error"]})
                    else:
                        self.messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": tool_result["text"]},
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

        # print("\n⚠️ Reached maximum turns, forcing termination.")
        return "Error: Max turns reached"