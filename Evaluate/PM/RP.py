import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate RP task (state_understand)")
    parser.add_argument("--result_path", type=str, required=True, help="Path to result JSON file")
    parser.add_argument("--benchmark_path", type=str, required=True, help="Path to benchmark JSON file")
    args = parser.parse_args()

    try:
        with open(args.result_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
    except Exception as e:
        print(f"Error reading result file: {e}")
        sys.exit(1)

    try:
        with open(args.benchmark_path, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except Exception as e:
        print(f"Error reading benchmark file: {e}")
        sys.exit(1)

    for item in result_data:
        if item.get("result_time") == -2:
            print("Error: Found sample with result_time=-2. Exiting.")
            sys.exit(1)
            
    if len(result_data) != len(benchmark_data):
        print(f"Error: Sample counts mismatch. Result: {len(result_data)}, Benchmark: {len(benchmark_data)}")
        sys.exit(1)
        
    total_samples = len(result_data)
    if total_samples == 0:
        print("No samples to evaluate.")
        sys.exit(0)
        
    benchmark_dict = {item.get('id'): item for item in benchmark_data if 'id' in item}
    
    time_correct_count = 0
    exact_correct_count = 0
    
    for i, res_item in enumerate(result_data):
        res_id = res_item.get('id')
        if res_id is not None and res_id in benchmark_dict:
            bench_item = benchmark_dict[res_id]
        else:
            bench_item = benchmark_data[i]
            
        result_time = res_item.get("result_time")
        result_text = res_item.get("result_text")
        
        answers = bench_item.get("answer", [])
        if not answers:
            continue
            
        trigger_time = answers[0].get("trigger_time")
        bench_text = answers[0].get("text")
        
        if result_time is None or trigger_time is None:
            continue
            
        if (trigger_time - 1) <= result_time <= (trigger_time + 1):
            time_correct_count += 1
            if str(result_text).strip().lower() == str(bench_text).strip().lower():
                exact_correct_count += 1
                
    time_accuracy = time_correct_count / total_samples
    exact_accuracy = exact_correct_count / total_samples
    
    print(f"Total samples: {total_samples}")
    print(f"Time Correct: {time_correct_count} ({time_accuracy:.2%})")
    print(f"Full Correct: {exact_correct_count} ({exact_accuracy:.2%})")

if __name__ == "__main__":
    main()
