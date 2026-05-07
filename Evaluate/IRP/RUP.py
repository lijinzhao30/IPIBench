import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate RUP task (repeat detection)")
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
        result_time_list = item.get('result_time', [])
        
        if not isinstance(result_time_list, list):
            print(f"Error: result_time is not a list at index {i}.")
            sys.exit(1)
            
        if -2 in result_time_list:
            print(f"Error: result_time list contains -2 at index {i}. Exiting.")
            sys.exit(1)
            
        item_id = item.get('id')
        if item_id is not None and item_id in bench_dict:
            bench_item = bench_dict[item_id]
        else:
            bench_item = benchmark_data[i]
            
        answers = item.get('answer', [])
        if not answers and 'answer' in bench_item:
            answers = bench_item['answer']
        
        if len(result_time_list) != len(answers):
            continue
            
        is_sample_correct = True
        for j in range(len(result_time_list)):
            r_time = result_time_list[j]
            trigger_time = answers[j].get('trigger_time')
            
            if trigger_time is None:
                is_sample_correct = False
                break
                
            if not (trigger_time - 1 <= r_time <= trigger_time + 1):
                is_sample_correct = False
                break
                
        if is_sample_correct:
            correct_samples += 1

    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    print(f"Total samples: {total_samples}")
    print(f"Correct samples: {correct_samples}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
