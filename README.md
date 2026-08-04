<div align="center">
  <h1 style="font-size: 32px; font-weight: bold; margin: 0; display: inline-block;">
    <img src="assets/logo_geovista.png" alt="logo" height="50" style="vertical-align: middle; margin-right: 8px; display: inline-block;">
    GeoVista: Visually Grounded Active Perception for Vision-Language Understanding of Ultra-High-Resolution Remote Sensing Images
  </h1>
  
  <br clear="all">
  <br>

  <!-- 四个徽章水平放置 -->
  <p align="center">
    <a href="https://arxiv.org/abs/2605.14475">
      <img src="https://img.shields.io/badge/ArXiv-GeoVista-brown?logo=arxiv" alt="Paper">
    </a>
    <a href="https://huggingface.co/collections/ryan6073/apex-gro">
      <img src="https://img.shields.io/badge/🤗 huggingface-Dataset-blue" alt="dataset">
    </a>
    <a href="https://huggingface.co/collections/ryan6073/geovista">
      <img src="https://img.shields.io/badge/🤗 huggingface-Model-purple" alt="checkpoint">
    </a>
    <a href="https://ryan6073.github.io/GeoVista/">
      <img src="https://img.shields.io/badge/-HomePage-black?logo=github" alt="homepage">
    </a>
  </p>
</div>

---

<!-- OverView 图片居中 -->
<p align="center">
  <img src="assets/comparison_of_methods.jpg" alt="OverView">
</p>

## Contents:

1. [Getting Started](#getting-started)
2. [Demo](#demo)
3. [Benchmark](#benchmark)
4. [Evaluation](#evaluation)
5. [Training](#training)
6. [License](#license)
7. [Citation](#citation)
8. [Acknowledgement](#acknowledgement)

## Getting Started 

### Installation

我们提供两种环境配置方式。您可以选择传统的 Conda 方式，或者使用极速的 `uv` 工具栈进行部署。

**方式一：纯 Conda 部署 (传统且稳定)**
```bash
conda create -n geovista python=3.10 -y
conda activate geovista

# 安装核心依赖 (基于 vLLM, verl 与 LLaMA-Factory)
pip install -r requirements.txt
# 单独安装 flash-attn 以加速注意力计算
pip install flash-attn --no-build-isolation

```

**方式二：纯 uv 部署 (极速构建，强烈推荐)**

```bash
# 如果尚未安装 uv，请先执行: pip install uv
uv venv --python 3.10
source .venv/bin/activate  # Windows 用户请使用 .venv\Scripts\activate

# 使用 uv pip 进行极速依赖解析和安装
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation

```

### Pre-trained Models

项目支持加载经过 SFT 或 RL 对齐的模型权重：

* **GeoVista-7B-Instruct**: [https://huggingface.co/ryan6073/GeoVista-7B-Instruct]
* **GeoVista-7B-Preview**: [https://huggingface.co/ryan6073/GeoVista-7B-Preview]

## Demo

您可以使用内置的 `GeoVistaAgent` 启动单张图像的交互式主动感知任务。框架会自动注册物理执行工具，并处理模型在全局与局部视图间的多轮推理与容错逻辑。

```python
from geovista.agent import GeoVistaAgent, zoom_in_tool

# 1. 初始化 Agent (支持 vLLM 本地服务或 OpenAI 兼容中转站 API)
agent = GeoVistaAgent(
    model_name="geovista-7b-preview", 
    api_base="http://localhost:8100/v1"
)

# 2. 注册高分辨率裁剪工具
agent.register_tool("zoom_in", zoom_in_tool)

# 3. 定义标准操作程序 (SOP)
SYSTEM_PROMPT = """Task: Precise Object Counting.
Algorithm SOP (STRICT CHECKLIST PROCESS):
1. PLAN PHASE: In your first <think> block, write an explicit [PLAN] containing the regions to inspect.
2. EXECUTION RULE: If the target region is unclear, request a crop via:
   <tool_call>{"name": "zoom_in", "arguments": {"bbox": [x1, y1, x2, y2]}}</tool_call>
   STOP GENERATING AFTER THE TOOL CALL.
"""

# 4. 启动主动感知推理
final_answer = agent.run(
    user_prompt="图中共有多少架小型民航客机？请通过局部放大确认。",
    image_path="path/to/uhr_image.tif",
    system_prompt=SYSTEM_PROMPT
)
print(f"Final Answer: {final_answer}")

```

## Benchmark

我们提供了一个多维度的评估基准工具箱，支持对模型在高分辨率场景下的视觉推理、主动感知与工具调用能力进行自动化打分。目前完整支持以下三大基准测试：

* **XLRSBench**: 复杂的视觉推理与综合判读 (包含 16 项细分感知与推理能力)。
* **XHRBench**: 高分辨率遥感异常检测与细粒度感知 (支持 **MTEM@1** 多轮主动感知评测)。
* **LRS-VQA**: 基础视觉问答与空间关系推理。

> **⚠️ 坐标系规范 (Coordinate System Spec):** > 在工具调用环节，模型输出的裁剪包围盒（BBox）必须严格遵循 `[x1, y1, x2, y2]` 格式，且统一使用相对原始图像尺寸的 **0-1000 离散相对坐标系**（左上角为 `[0, 0]`，右下角为 `[1000, 1000]`）。框架底层的 Agent 会自动将其映射并裁剪为真实的物理像素。

---

## Evaluation

基于我们提供的模块化评测套件，您可以通过统一的命令行接口 (CLI) 轻松发起多并发的推理和自动化打分。

在运行评估前，请确保您已在项目根目录创建并配置了 `.env` 文件（参考 `.env.example`，配置您的 `API_BASE` 与数据集路径 `BENCHMARK_ROOT`）。

```bash
# 1. 完整运行推理与评估 (同时评测 XLRS, XHR, LRS 三个数据集)
python eval/main.py --data xlrs xhr lrs --model geovista-7b-preview --mode all --api-nproc 5

# 2. 仅进行并发推理
python eval/main.py --data xhr --model geovista-7b-preview --mode infer --api-nproc 10

# 3. 仅生成结构化评测报告
python eval/main.py --data xlrs xhr lrs --model geovista-7b-preview --mode eval

# 4. 自动对比多组模型权重
python eval/main.py --data xlrs --model agent_step_1000 agent_step_2000 --mode all

```

## Training 

### 1. 监督微调 (SFT)

利用 APEX-GRO 数据集对模型进行指令微调：

```bash
llamafactory-cli train training/SFT/qwen2_5_vl_agent_sft.yaml

```

### 2. 强化学习对齐 (RL - GRPO)

通过 GRPO 算法进行强化学习微调，系统将根据回答格式、IOU、准确率及推理过程提供多维度奖励：

```bash
python -m verl.trainer.main_ppo --config-path training/RL/geovista_grpo.yaml

```

## License 

本项目采用 **MIT License**。

## Citation 

如果您觉得本项目对您的研究有所帮助，请考虑引用：

```bibtex
@article{zhu2026geovista,
  title={GeoVista: Visually Grounded Active Perception for Vision-Language Understanding of Ultra-High-Resolution Remote Sensing Images},
  author={Zhu, Jiashun and Fu, Ronghao and Hu, Jiasen and Huang, Jing and Xing, Nachuan and Yang, Bo},
  journal={arXiv preprint arXiv:2605.14475},
  year={2026}
}

```

## Acknowledgement 

* 本项目基于 [vLLM](https://github.com/vllm-project/vllm), [verl](https://github.com/verl-project/verl) 以及 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) 构建。
* 感谢 [Qwen](https://github.com/QwenLM/Qwen) 团队提供的强大视觉语言基础模型。

## Star History

<a href="https://www.star-history.com/?repos=ryan6073%2FGeoVista&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ryan6073/GeoVista&type=date&theme=dark&legend=top-left&sealed_token=91yLJLmyVFkwi770rivIdZ6U2WcIb9023c-O4oKWUQW5V2DB4RVNGhhNsAIVxJNwOSlaumGyvdUQZkzZ3ZMNRmZ6Xb7-VyQqpg5YXYQ0oWJFMZBqK7LsFA" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ryan6073/GeoVista&type=date&legend=top-left&sealed_token=91yLJLmyVFkwi770rivIdZ6U2WcIb9023c-O4oKWUQW5V2DB4RVNGhhNsAIVxJNwOSlaumGyvdUQZkzZ3ZMNRmZ6Xb7-VyQqpg5YXYQ0oWJFMZBqK7LsFA" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ryan6073/GeoVista&type=date&legend=top-left&sealed_token=91yLJLmyVFkwi770rivIdZ6U2WcIb9023c-O4oKWUQW5V2DB4RVNGhhNsAIVxJNwOSlaumGyvdUQZkzZ3ZMNRmZ6Xb7-VyQqpg5YXYQ0oWJFMZBqK7LsFA" />
 </picture>
</a>