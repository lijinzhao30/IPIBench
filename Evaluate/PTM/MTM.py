import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate MTM task (multi_task)")
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
            bench_data = json.load(f)
    except Exception as e:
        print(f"Error reading benchmark file: {e}")
        sys.exit(1)

    for item in result_data:
        result_time = item.get('result_time', [])
        if isinstance(result_time, list) and -2 in result_time:
            print(f"Error: Found sample with -2 in result_time (id: {item.get('id', 'unknown')}). Exiting.")
            sys.exit(1)
        elif result_time == -2:
            print(f"Error: Found sample with -2 in result_time (id: {item.get('id', 'unknown')}). Exiting.")
            sys.exit(1)

    if len(result_data) != len(bench_data):
        print(f"Error: Sample counts mismatch. Result: {len(result_data)}, Benchmark: {len(bench_data)}")
        sys.exit(1)

    bench_dict = {item['id']: item for item in bench_data if 'id' in item}

    total_samples = len(result_data)
    correct_count = 0

    for res_item in result_data:
        res_id = res_item.get('id')
        if res_id not in bench_dict:
            print(f"Warning: Result id {res_id} not found in benchmark.")
            continue
            
        bench_item = bench_dict[res_id]
        result_time = res_item.get('result_time', [])
        
        try:
            answer = bench_item['answer']
            trigger_time_0 = answer[0]['trigger_time']
            trigger_time_1 = answer[1]['trigger_time']
        except (KeyError, IndexError):
            continue

        if not isinstance(result_time, list) or len(result_time) < 2:
            continue

        if (trigger_time_0 - 1 <= result_time[0] <= trigger_time_0 + 1) and \
           (trigger_time_1 - 1 <= result_time[1] <= trigger_time_1 + 1):
            correct_count += 1

    accuracy = correct_count / total_samples if total_samples > 0 else 0
    print(f"Total samples: {total_samples}")
    print(f"Correct: {correct_count} ({accuracy:.2%})")

if __name__ == "__main__":
    main()
