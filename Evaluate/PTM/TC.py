import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate TC task (cancel_complete / cancel_incomplete)")
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

    if len(result_data) != len(benchmark_data):
        print(f"Error: Sample counts mismatch. Result: {len(result_data)}, Benchmark: {len(benchmark_data)}")
        sys.exit(1)

    for i, item in enumerate(result_data):
        result_time = item.get('result_time')
        if result_time == -2:
            print(f"Error: Found unprocessed sample (result_time = -2) at index {i}. Exiting.")
            sys.exit(1)

    correct_time_only = 0
    correct_strict = 0
    total = len(result_data)
    
    bench_dict = {item['id']: item for item in benchmark_data if 'id' in item}
    
    for item in result_data:
        try:
            item_id = item.get('id')
            if item_id in bench_dict:
                bench_item = bench_dict[item_id]
            else:
                bench_item = item # Fallback if IDs mismatch or zip
                
            result_time = float(item.get('result_time', -999))
            trigger_time = float(bench_item['answer'][0]['trigger_time'])
            
            time_match = abs(result_time - trigger_time) <= 1.0
            
            if time_match:
                correct_time_only += 1
                
                cancel_list = item.get('cancel', [])
                cancel_match = bool(cancel_list) and all(
                    'silence' in str(c).lower() and 'attention' not in str(c).lower() 
                    for c in cancel_list
                )
                
                if cancel_match:
                    correct_strict += 1
        except Exception as e:
            pass

    print(f"Total samples: {total}")
    accuracy_time = correct_time_only / total if total > 0 else 0
    accuracy_strict = correct_strict / total if total > 0 else 0
    print(f"Time Correct: {correct_time_only} ({accuracy_time:.2%})")
    print(f"Strict Correct: {correct_strict} ({accuracy_strict:.2%})")

if __name__ == "__main__":
    main()
