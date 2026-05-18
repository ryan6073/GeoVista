import argparse
import os

# 从配置中导入环境变量读取的默认值
from config import WORK_DIR, API_BASE, API_KEY
from infer import run_inference
from evaluate import run_evaluation

def parse_args():
    parser = argparse.ArgumentParser(description="GeoVista / AgentZoom Multi-Benchmark Toolkit")
    
    parser.add_argument('--data', nargs='+', type=str, required=True, 
                        choices=['xlrs', 'xhr', 'lrs'], 
                        help='List of datasets to run (e.g., --data xlrs xhr)')
                        
    parser.add_argument('--model', nargs='+', type=str, required=True, 
                        help='List of models to evaluate (e.g., --model agent_zoom_step_1400)')
                        
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'infer', 'eval'],
                        help='Execution mode: all (infer+eval), infer (inference only), eval (evaluation only)')
    
    # 优先使用命令行的值，如果没有提供，则 fallback 到 config.py 中从 .env 读到的值
    parser.add_argument('--api-base', type=str, default=API_BASE,
                        help='vLLM API Base URL or OpenAI proxy endpoint')
                        
    parser.add_argument('--api-key', type=str, default=API_KEY,
                        help='API Key for the model service')
                        
    parser.add_argument('--api-nproc', type=int, default=5,
                        help='Number of concurrent threads for calling the API')
                        
    parser.add_argument('--work-dir', type=str, default=WORK_DIR,
                        help='Directory to save the inference results and evaluation reports')
                        
    return parser.parse_args()


def main():
    args = parse_args()
    
    for model_name in args.model:
        for dataset in args.data:
            print(f"\n{'#'*60}")
            print(f"# Processing: [{dataset.upper()}] with model: [{model_name}]")
            print(f"#{'#'*59}")
            
            infer_output_file = os.path.join(args.work_dir, f"{model_name}_{dataset}.jsonl")
            eval_output_file = os.path.join(args.work_dir, f"{model_name}_{dataset}_report.txt")
            
            if args.mode in ['all', 'infer']:
                run_inference(
                    dataset_name=dataset,
                    model_name=model_name,
                    api_base=args.api_base,
                    api_key=args.api_key,       # 新增
                    workers=args.api_nproc,
                    output_path=infer_output_file
                )
                
            if args.mode in ['all', 'eval']:
                if not os.path.exists(infer_output_file):
                    print(f"❌ Error: Could not find inference output at {infer_output_file}")
                    print("Please run with '--mode infer' or '--mode all' first.")
                    continue
                    
                run_evaluation(
                    dataset_name=dataset,
                    input_file=infer_output_file,
                    output_file=eval_output_file
                )

if __name__ == "__main__":
    main()