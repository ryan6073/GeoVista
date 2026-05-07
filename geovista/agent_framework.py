import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

class GeoVistaAgent:
    def __init__(self, model_name: str, api_base: str, max_turns: int = 15, temperature: float = 0.0):
        self.model_name = model_name
        self.client = OpenAI(api_key="empty", base_url=api_base)
        self.max_turns = max_turns
        self.temperature = temperature
        
        self.tools = {}
        self.messages = []
        
    def register_tool(self, tool_name: str, func: callable):
        self.tools[tool_name] = func
        print(f"🔧 tool registered: [{tool_name}]")

    @staticmethod
    def encode_image(image: Image.Image) -> str:
        """将 PIL 图像转换为 Base64"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def reset_memory(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def _call_llm(self):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=1024,
            frequency_penalty=1.0,
            stop=["</tool_call>", "</answer>"] 
        )
        
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
        match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if match: return match.group(1)
        
        match = re.search(r'(\{.*"name"\s*:.*\})', text, re.DOTALL)
        if match: 
            tool_str = match.group(1)
            if tool_str.count('{') > tool_str.count('}'):
                tool_str += '}' * (tool_str.count('{') - tool_str.count('}'))
            return tool_str
        return None

    def run(self, user_prompt: str, image_path: str, system_prompt: str, context_kwargs: dict = None):
        if context_kwargs is None:
            context_kwargs = {}
            
        print(f"\n🚀 Starting GeoVista Agent | Target Image: {os.path.basename(image_path)}")
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
            print(f"\n" + "="*50)

            print(f"🔄 Turn {turn} (thinking...)")
            print("="*50)
            
            model_output = self._call_llm()
            print(f"🤖 Model Output:\n\n{model_output}\n")
            self.messages.append({"role": "assistant", "content": model_output})
            
            if "<answer>" in model_output:
                answer_match = re.search(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
                final_answer = answer_match.group(1).strip() if answer_match else "No answer parsed"
                print(f"\n🎉 [Task Completed] Agent returned final answer: {final_answer}")
                return final_answer
                
            tool_str_raw = self._extract_tool_call(model_output)
            
            if tool_str_raw:
                try:
                    tool_str = tool_str_raw.strip()
                    tool_data = json.loads(tool_str)
                    
                    tool_name = tool_data.get("name")
                    tool_args = tool_data.get("arguments", {})
                    
                    if tool_name in self.tools:
                        print(f"🔧 Scheduler successfully intercepted tool call: [{tool_name}] Arguments: {tool_args}")
                        
                        tool_func = self.tools[tool_name]
                        tool_result = tool_func(tool_args, context_kwargs)
                        
                        self.messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": tool_result["text"]},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(tool_result['image'])}"}}
                            ]
                        })
                    else:
                        print(f"⚠️ Tool [{tool_name}] is not registered!")
                        break
                        
                except Exception as e:
                    print(f"❌ Failed to parse or execute tool: {e}\nOriginal: {tool_str}")
                    break
            else:
                print("⚠️ Agent unexpectedly stopped (no tool called and no answer provided). Triggering general error correction mechanism...")
                warning_msg = (
                    "WARNING: You stopped generating without making a <tool_call> or providing an <answer>. "
                    "If you need to use a tool, output EXACTLY: <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>. "
                    "If you have confidently found the target or finished the reasoning, output your final result wrapped in <answer>...</answer>."
                )
                self.messages.append({"role": "user", "content": warning_msg})
                continue

        print("\n⚠️ Reached maximum turns, forcing termination.")
        return None


def zoom_in_tool(arguments: dict, context: dict) -> dict:
    bbox = arguments.get("bbox")
    original_image = context.get("original_image")
    width, height = original_image.size
    
    x1 = max(0, int((bbox[0] / 1000.0) * width))
    y1 = max(0, int((bbox[1] / 1000.0) * height))
    x2 = min(width, int((bbox[2] / 1000.0) * width))
    y2 = min(height, int((bbox[3] / 1000.0) * height))
    
    cropped_img = original_image.crop((x1, y1, x2, y2))
    return {
        "text": f"System Return: Cropped image for bbox {bbox}. Please continue your analysis INSIDE a new <think> block.",
        "image": cropped_img
    }

if __name__ == "__main__":
    VLLM_API = "http://localhost:8011/v1"
    MODEL_NAME = "qwen3-8b"
    
    agent = GeoVistaAgent(model_name=MODEL_NAME, api_base=VLLM_API)
    agent.register_tool("zoom_in", zoom_in_tool)
    
    test_image = "/srv/user/zhujs/data/tmp/counting/teacher/img/1ef2c9ce.png"
    
    SYSTEM_PROMPT = """Task: Precise Object Counting.
    Algorithm SOP (STRICT CHECKLIST PROCESS):
    1. PLAN PHASE: In your first <think> block, write an explicit [PLAN] containing the regions to inspect. Then copy it exactly into a [PROGRESS] checklist.
    2. EXECUTION RULE: If there is an unchecked [ ] region in your [PROGRESS], request a crop via:
    <tool_call>{"name": "zoom_in", "arguments": {"bbox": [x1, y1, x2, y2]}}</tool_call>
    STOP GENERATING AFTER THE TOOL CALL.
    3. DISCOVERY FORMAT: Upon receiving the cropped image, strictly follow this format in your <think> block:
    * Obj: [xmin, ymin, xmax, ymax]
    * Obj: [xmin, ymin, xmax, ymax]
    [PROGRESS]
    - [x] [Region you just inspected]
    - [ ] [Next region to inspect]
    4. FINAL AGGREGATION: When all regions are [x], output the final integer in <answer>.
    """
    counting_prompt = "Question: How many planes are there in the picture?\n\n[System Observation]\nCurrent View: 1ef2c9ce (Global View)\n"
    
    print("\n\n>>> Starting Test Scenario 3: Counting Cascaded Counting")
    agent.run(user_prompt=counting_prompt, image_path=test_image, system_prompt=SYSTEM_PROMPT)