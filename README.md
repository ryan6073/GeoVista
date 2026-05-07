# GeoVista

GeoVista 是一款专为超高分辨率（UHR）遥感图像设计的 Agentic VLM 调度与训练框架。该框架基于主动感知策略，支持“零调用秒答”、“工具调用”以及“裸体 JSON 容错”。

## 🌟 核心特性

* **尺度不变的主动感知**：系统采用离散的 0-1000 相对坐标系表示边界框（Bounding Boxes），以保证遥感目标在不同视场下的尺度不变性。在数据处理与工具调用时，系统会优先从高分辨率的原始图像上进行物理裁剪，从而有效防止目标丢失。
* **APEX-GRO 训练集**：项目内置支持 APEX-GRO 等训练集，这套数据集专门用于监督微调（SFT）和智能体能力初始化。
* **通用型 Agent 调度框架**：内置调度器，提供防暴走截断与通用型正则解析。框架支持为 Agent 注册物理级裁剪工具（`zoom_in`），支持模型的多轮思考（`<think>`）与目标计数级联推理。
* **基于 GRPO 的强化学习对齐**：支持利用组相对策略优化（GRPO）算法进行强化学习微调。系统自定义了多维度的奖励函数（Format, IOU, Accuracy, Process/Plan），引导模型输出严格遵守 SOP 的推理轨迹。

## 📁 项目结构

* **`geovista/agent_framework.py`**: 核心推理与调度框架，负责调用底层 vLLM 引擎，管理多轮对话状态，并拦截调度 `zoom_in` 等工具操作。
* **`training/SFT/`**: 监督微调配置模块。支持 Qwen2.5-VL 模型的 LoRA 微调，包含了 `APEX-GRO`、`HighRS` 和 `LRS_GRO` 等多套视觉大模型微调方案与配置参数。
* **`training/RL/`**: 强化学习训练模块，基于 verl 框架构建。
    * `geovista_grpo.yaml`: GRPO 算法的训练超参数与并行配置。
    * `parallel_env.py`: 适配 vLLM 的多模态并行 rollout 环境，支持图像长宽比校验与多模态数据输入预处理。
    * `visual_toolbox_geovista.py`: GeoVista 的物理执行工具箱。负责将 0-1000 的相对坐标映射到实际像素，执行物理裁剪，并构造符合 SFT 风格的观测反馈（Observation）。
    * `vl_agent_geovista.py`: GeoVista 的定制化奖励函数管理器（Reward Manager），从回答格式（Format）、边界框 IoU、最终答案准确率（Acc）以及规划过程（Process）等四个维度计算奖励得分。

## 🚀 快速开始

### 1. 监督微调 (SFT)
项目提供了基于 Qwen2.5-VL-7B-Instruct 模型的 LoRA 训练配置。使用 LLaMA-Factory 运行以下命令即可启动训练：
```bash
llamafactory-cli train training/SFT/qwen2_5_vl_agent_sft.yaml
```

### 2. 强化学习对齐 (RL - GRPO)
基于 verl 框架进行强化学习微调，使用提供的 YAML 配置文件：
```bash
python -m verl.trainer.main_ppo --config-path training/RL/geovista_grpo.yaml
```

### 3. Agent 推理部署
调用 `GeoVistaAgent` 进行推理，系统会自动注册 `zoom_in` 工具并启动级联计数等任务：
```python
from geovista.agent_framework import GeoVistaAgent, zoom_in_tool

agent = GeoVistaAgent(model_name="qwen3-8b", api_base="http://localhost:8011/v1")
agent.register_tool("zoom_in", zoom_in_tool)
agent.run(user_prompt="...", image_path="...", system_prompt="...")
```
