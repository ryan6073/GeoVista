import argparse
import os

# 从配置中导入环境变量读取的默认值
from config import WORK_DIR, API_BASE, API_KEY
from infer import run_inference
from evaluate import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="GeoVista Multi-Benchmark Toolkit"
    )

    parser.add_argument('--data', nargs='+', type=str, required=True,
                        choices=['mme', 'xlrs', 'xhr', 'lrs'],
                        help='List of datasets to run (e.g., --data mme xlrs xhr)')

    parser.add_argument('--model', nargs='+', type=str, required=True,
                        help='List of models to evaluate (e.g., --model GeoEyes)')

    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'infer', 'eval'],
                        help='Execution mode: all (infer+eval), infer, eval')

    parser.add_argument('--api-base', type=str, default=API_BASE,
                        help='vLLM API Base URL or OpenAI proxy endpoint')

    parser.add_argument('--api-key', type=str, default=API_KEY,
                        help='API Key for the model service')

    parser.add_argument('--api-nproc', type=int, default=5,
                        help='Number of concurrent threads for calling the API')

    parser.add_argument('--work-dir', type=str, default=WORK_DIR,
                        help='Directory to save inference results and evaluation reports')

    parser.add_argument(
        '--max-tool-calls', type=int, default=None,
        help='Hard cap on zoom_in calls per question. Omit = unlimited.',
    )

    parser.add_argument(
        '--max-turns', type=int, default=30,
        help='Max LLM turns per question (default 30).',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    for model_name in args.model:
        for dataset in args.data:
            stem = f"{model_name}_{dataset}"
            print(f"\n{'#'*60}")
            print(f"# Processing: [{dataset.upper()}] model=[{model_name}]")
            print(f"#   max_tool_calls={args.max_tool_calls} max_turns={args.max_turns}")
            print(f"#{'#'*59}")

            infer_output_file = os.path.join(args.work_dir, f"{stem}.jsonl")
            eval_output_file = os.path.join(args.work_dir, f"{stem}_report.txt")

            if args.mode in ['all', 'infer']:
                run_inference(
                    dataset_name=dataset,
                    model_name=model_name,
                    api_base=args.api_base,
                    api_key=args.api_key,
                    workers=args.api_nproc,
                    output_path=infer_output_file,
                    max_tool_calls=args.max_tool_calls,
                    max_turns=args.max_turns,
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
