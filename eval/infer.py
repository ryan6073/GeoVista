import os
import sys
import json
import threading
import concurrent.futures
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geovista.agent import (
    GeoVistaAgent,
    GIVEN_ROI_EVAL_HINT,
    question_has_given_pixel_roi,
    zoom_in_tool,
)
from config import DATASETS
from ablation_sops import sop_for_dataset, normalize_ablation, ABLATION_META

write_lock = threading.Lock()


def process_single_task(
    data_row: dict,
    dataset_name: str,
    model_name: str,
    api_base: str,
    api_key: str,
    ablation: str = None,
    max_tool_calls: int = None,
    max_turns: int = 30,
) -> dict:
    config = DATASETS[dataset_name]

    agent = GeoVistaAgent(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
    )
    agent.register_tool("zoom_in", zoom_in_tool)

    q_id = str(data_row.get("question_id", data_row.get("index", "Unknown")))

    question_text = data_row.get("input", data_row.get("text", ""))
    question_text = question_text.replace("please only return answer option.", "").strip()

    img_key = data_row.get("image_absolute_path") or data_row.get("image_path") or data_row.get("image") or data_row.get("Image", "")
    test_image = os.path.join(config["image_root"], img_key) if config["image_root"] else img_key

    if not os.path.exists(test_image):
        data_row["model_output"] = "Error: Image file not found"
        _attach_run_stats(data_row, agent, ablation)
        return data_row

    if question_has_given_pixel_roi(question_text):
        question_text = question_text + "\n\n" + GIVEN_ROI_EVAL_HINT
    prompt = f"Question: {question_text}\n\n[System Observation]\nCurrent View: Global View\n"

    # Prompt-only ablation: swap SOP; keep same tool / budget / model.
    ablation_sop = sop_for_dataset(ablation, dataset_name, max_tool_calls=max_tool_calls or 8)
    system_prompt = ablation_sop if ablation_sop is not None else config["sop"]

    try:
        final_ans = agent.run(
            user_prompt=prompt,
            image_path=test_image,
            system_prompt=system_prompt,
            q_id=q_id,
        )
        data_row["model_output"] = final_ans
    except Exception as e:
        data_row["model_output"] = f"Error: {str(e)}"

    _attach_run_stats(data_row, agent, ablation)
    return data_row


def _attach_run_stats(data_row: dict, agent, ablation: str = None) -> None:
    """Write visual/input_text/output_text tokens and turn/tool counts onto the result row."""
    summary = agent.token_summary()
    data_row["visual_tokens"] = summary["visual_tokens"]
    data_row["input_text_tokens"] = summary.get("input_text_tokens", 0)
    data_row["output_text_tokens"] = summary.get("output_text_tokens", 0)
    data_row["generated_tokens"] = summary["generated_tokens"]
    data_row["total_tokens"] = summary["total_tokens"]
    data_row["n_turns"] = summary["n_turns"]
    data_row["n_tool_calls"] = summary.get("n_tool_calls", 0)
    data_row["budget_exhausted"] = summary.get("budget_exhausted", False)
    canon = normalize_ablation(ablation)
    data_row["ablation"] = canon or "default"
    if canon:
        data_row["ablation_label"] = ABLATION_META[canon]["label"]


def run_inference(
    dataset_name: str,
    model_name: str,
    api_base: str,
    api_key: str,
    workers: int,
    output_path: str,
    ablation: str = None,
    max_tool_calls: int = None,
    max_turns: int = 30,
):
    config = DATASETS.get(dataset_name)
    if not config:
        print(f"❌ Error: Dataset '{dataset_name}' not found in config.")
        return

    print(f"📦 [Infer] Loading {dataset_name.upper()} dataset...")
    with open(config["jsonl_path"], 'r', encoding='utf-8') as f:
        all_data = [json.loads(line.strip()) for line in f if line.strip()]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    open(output_path, 'w', encoding='utf-8').close()

    canon = normalize_ablation(ablation)
    abl_tag = canon or "default"
    print(
        f"🚀 [Infer] Starting | Model: {model_name} | Ablation: {abl_tag} | "
        f"max_tool_calls={max_tool_calls} | Threads: {workers} | Total: {len(all_data)}"
    )

    with tqdm(total=len(all_data), desc=f"Infer {dataset_name.upper()}/{abl_tag}", unit="it") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_data = {
                executor.submit(
                    process_single_task,
                    row,
                    dataset_name,
                    model_name,
                    api_base,
                    api_key,
                    ablation,
                    max_tool_calls,
                    max_turns,
                ): row
                for row in all_data
            }

            for future in concurrent.futures.as_completed(future_to_data):
                result_row = future.result()
                with write_lock:
                    with open(output_path, 'a', encoding='utf-8') as out_f:
                        out_f.write(json.dumps(result_row, ensure_ascii=False) + '\n')
                pbar.update(1)
