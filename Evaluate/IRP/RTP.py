import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate RTP task (event detection)")
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

    bench_dict = {item['id']: item for item in benchmark_data if 'id' in item}
    
    if len(result_data) != len(benchmark_data):
        print(f"Error: Sample counts mismatch. Result: {len(result_data)}, Benchmark: {len(benchmark_data)}")
        sys.exit(1)

    correct_samples = 0
    total_samples = len(result_data)

    for i, item in enumerate(result_data):
        result_time = item.get('result_time')
        
        if result_time == -2:
            print(f"Error: Result time is -2 for sample index {i}. Exiting.")
            sys.exit(1)
            
        item_id = item.get('id')
        if item_id is not None and item_id in bench_dict:
            bench_item = bench_dict[item_id]
        else:
            bench_item = benchmark_data[i]
            
        answers = item.get('answer', [])
        if not answers and 'answer' in bench_item:
            answers = bench_item['answer']
        
        if answers and isinstance(answers, list) and len(answers) > 0:
            trigger_time = answers[0].get('trigger_time')
            if trigger_time is not None:
                if trigger_time - 1 <= result_time <= trigger_time + 1:
                    correct_samples += 1

    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    print(f"Total samples: {total_samples}")
    print(f"Correct samples: {correct_samples}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
