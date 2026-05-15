import io
import re
import json
import uuid
import logging
from PIL import Image
from typing import Optional, List, Dict, Any
from verl.workers.agent.tool_envs import ToolBase

logger = logging.getLogger(__name__)

class VisualToolBox(ToolBase):
    """
    AgentZoom 物理执行工具箱 (魔改版)
    1. 适配 env_name: visual_toolbox
    2. 支持 [0, 1000] 归一化坐标。
    3. 观测输出完全对齐 SFT 数据格式 (TOOL_EXECUTION_SUCCESS)。
    """
    name = "visual_toolbox"

    def __init__(self, _name=None, _desc=None, _params=None, **kwargs):
        # 显式使用类定义的 name 确保与数据集 env_name 匹配
        super().__init__(name=self.name)
        self.multi_modal_data = None
        self.width = 1000 # 默认值，在 reset 中更新
        self.height = 1000

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        """同步 verl 的多模态环境状态"""
        self.multi_modal_data = origin_multi_modal_data
        if 'image' in self.multi_modal_data and self.multi_modal_data['image']:
            img = self.multi_modal_data['image'][0]
            # 兼容 PIL Image 或字节流
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img)).convert('RGB')
                self.multi_modal_data['image'][0] = img # 更新回缓存
            
            self.width, self.height = img.size
        return raw_prompt

    def extract_action(self, action_string: str) -> Optional[str]:
        """提取模型输出的 <tool_call> 块"""
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        核心执行逻辑：
        映射坐标 -> 裁剪图像 -> 构造 SFT 风格的观测反馈
        """
        # 1. 拦截答案
        if "<answer>" in action_string:
            return "", 0.0, True, {"status": "completed"}

        # 2. 提取 Action
        action_raw = self.extract_action(action_string)
        if not action_raw:
            return "Error: Invalid tool call format.", 0.0, False, {"status": "failed"}

        try:
            call_data = json.loads(action_raw.strip())
            tool_name = call_data.get("name")
            args = call_data.get("arguments", {})

            if tool_name == "zoom_in":
                # 获取参数
                bbox_1000 = args.get("bbox")
                source_id = args.get("source_image_id", "unknown")
                
                if not bbox_1000 or len(bbox_1000) != 4:
                    raise ValueError("bbox must be [x1, y1, x2, y2] in 0-1000 scale.")

                # 3. 坐标映射: [0, 1000] -> 实际像素
                left = int((bbox_1000[0] / 1000.0) * self.width)
                top = int((bbox_1000[1] / 1000.0) * self.height)
                right = int((bbox_1000[2] / 1000.0) * self.width)
                bottom = int((bbox_1000[3] / 1000.0) * self.height)

                # 边界剪裁与修正 (保持裁剪区域有效)
                x1, x2 = min(left, right), max(left, right)
                y1, y2 = min(top, bottom), max(top, bottom)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.width, x2), min(self.height, y2)

                # 4. 执行物理裁剪
                img = self.multi_modal_data['image'][0]
                cropped_img = img.crop((x1, y1, x2, y2))
                
                # 生成新的 View ID (模拟 SFT 数据中的随机后缀)
                new_view_id = f"{source_id}_{uuid.uuid4().hex[:8]}"

                # 5. 构造与 SFT 完全一致的 [System Observation]
                obs_text = (
                    f"TOOL_EXECUTION_SUCCESS\n\n"
                    f"[System Observation]\n"
                    f"Current View: {new_view_id}\n"
                    f"Parent Image: {source_id}\n"
                    f"Zoom Path: {source_id} -> {new_view_id} (bbox: {bbox_1000})\n"
                    f"<image>"
                )

                # 构建符合 verl 渲染要求的 Observation 字典
                obs = {
                    "prompt": f"\n<|im_start|>user\n{obs_text}<|im_end|>\n<|im_start|>assistant\n",
                    "multi_modal_data": {"image": [cropped_img]}
                }
                
                return obs, 0.0, False, {"status": "success", "new_view": new_view_id}

            else:
                raise ValueError(f"Unknown tool name: {tool_name}")

        except Exception as e:
            error_msg = f"TOOL_EXECUTION_FAILED: {str(e)}"
            logger.error(error_msg)
            obs = f"\n<|im_start|>user\n{error_msg}<|im_end|>\n<|im_start|>assistant\n"
            return obs, 0.0, False, {"status": "error"}