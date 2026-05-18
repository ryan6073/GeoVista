import os
import sys
import json
import threading
import concurrent.futures
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geovista.agent import GeoVistaAgent, zoom_in_tool
from config import DATASETS

write_lock = threading.Lock()

# 增加 api_key 参数
def process_single_task(data_row: dict, dataset_name: str, model_name: str, api_base: str, api_key: str) -> dict:
    config = DATASETS[dataset_name]
    
    # 初始化时传入 api_key
    agent = GeoVistaAgent(model_name=model_name, api_base=api_base, api_key=api_key)
    agent.register_tool("zoom_in", zoom_in_tool)
    
    q_id = str(data_row.get("question_id", data_row.get("index", "Unknown")))
    
    question_text = data_row.get("input", data_row.get("text", ""))
    question_text = question_text.replace("please only return answer option.", "").strip()
    
    img_key = data_row.get("image_absolute_path") or data_row.get("image_path") or data_row.get("image") or data_row.get("Image", "")
    test_image = os.path.join(config["image_root"], img_key) if config["image_root"] else img_key
    
    if not os.path.exists(test_image):
        data_row["model_output"] = "Error: Image file not found"
        return data_row
        
    prompt = f"Question: {question_text}\n\n[System Observation]\nCurrent View: Global View\n"
    
    try:
        final_ans = agent.run(
            user_prompt=prompt, 
            image_path=test_image, 
            system_prompt=config["sop"], 
            q_id=q_id
        )
        data_row["model_output"] = final_ans
    except Exception as e:
        data_row["model_output"] = f"Error: {str(e)}"
        
    return data_row


# 增加 api_key 参数
def run_inference(dataset_name: str, model_name: str, api_base: str, api_key: str, workers: int, output_path: str):
    config = DATASETS.get(dataset_name)
    if not config:
        print(f"❌ Error: Dataset '{dataset_name}' not found in config.")
        return
        
    print(f"📦 [Infer] Loading {dataset_name.upper()} dataset...")
    with open(config["jsonl_path"], 'r', encoding='utf-8') as f:
        all_data = [json.loads(line.strip()) for line in f if line.strip()]
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    open(output_path, 'w', encoding='utf-8').close() 

    print(f"🚀 [Infer] Starting | Model: {model_name} | Threads: {workers} | Total: {len(all_data)}")
    
    with tqdm(total=len(all_data), desc=f"Infer {dataset_name.upper()}", unit="it") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_data = {
                # 传入 api_key
                executor.submit(process_single_task, row, dataset_name, model_name, api_base, api_key): row 
                for row in all_data
            }
            
            for future in concurrent.futures.as_completed(future_to_data):
                result_row = future.result()
                with write_lock:
                    with open(output_path, 'a', encoding='utf-8') as out_f:
                        out_f.write(json.dumps(result_row, ensure_ascii=False) + '\n')
                pbar.update(1)