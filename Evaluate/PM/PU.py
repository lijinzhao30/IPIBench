import json
import sys
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate PU task (attribute_understand)")
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

    bench_dict = {}
    for i, b_item in enumerate(benchmark_data):
        item_id = b_item.get('id', i)
        bench_dict[item_id] = b_item
        
    time_correct_count = 0
    fully_correct_count = 0
    total_samples = len(result_data)

    for i, r_item in enumerate(result_data):
        result_time = r_item.get('result_time')
        
        if result_time == -2:
            print(f"Error: result_time is -2 for sample {r_item.get('id', i)}. Exiting.")
            sys.exit(1)
            
        item_id = r_item.get('id', i)
        if item_id not in bench_dict:
            print(f"Error: Result ID {item_id} not found in Benchmark.")
            sys.exit(1)
            
        b_item = bench_dict[item_id]
        answers = r_item.get('answer', [])
        if not answers and 'answer' in b_item:
            answers = b_item['answer']
            
        is_time_correct = False
        if answers and isinstance(answers, list) and len(answers) > 0:
            trigger_time = answers[0].get('trigger_time')
            if trigger_time is not None:
                if trigger_time - 1 <= result_time <= trigger_time + 1:
                    is_time_correct = True
                    time_correct_count += 1
                    
        if is_time_correct:
            result_text = str(r_item.get('result_text', '')).strip().lower()
            is_fully_correct = False
            answer_list = []
            
            if 'number' in b_item and b_item['number']:
                answer_list = b_item.get('answer_list', [])
                if not answer_list:
                    answer_list = [str(b_item['number'])]
            elif 'color' in b_item and b_item['color']:
                answer_list = b_item.get('answer_list', [])
                if not answer_list:
                    answer_list = [str(b_item['color'])]
            elif 'material' in b_item and b_item['material']:
                answer_list = b_item.get('answer_list', [])
                if not answer_list:
                    answer_list = [str(b_item['material'])]
            
            if answer_list:
                answer_list = [str(a).strip().lower() for a in answer_list if str(a).strip()]
                for ans in answer_list:
                    if result_text in ans or ans in result_text:
                        is_fully_correct = True
                        break
                
            if is_fully_correct:
                fully_correct_count += 1

    print(f"Total samples: {total_samples}")
    print(f"Time Correct: {time_correct_count} ({(time_correct_count/total_samples*100) if total_samples else 0:.2f}%)")
    print(f"Full Correct: {fully_correct_count} ({(fully_correct_count/total_samples*100) if total_samples else 0:.2f}%)")

if __name__ == "__main__":
    main()
