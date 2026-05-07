import json
import sys
import re
import argparse

def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def normalize_word_endings(word):
    word = normalize_text(word)
    if word.endswith("'s"):
        word = word[:-2]
    word = re.sub(r"('s|es|s)\b", "", word)
    return word

def check_match(extracted_text, answer_list):
    if not extracted_text or not answer_list:
        return False
    ext_norm = normalize_word_endings(extracted_text)
    for ans in answer_list:
        ans_norm = normalize_word_endings(ans)
        if ext_norm == ans_norm or ext_norm in ans_norm or ans_norm in ext_norm:
            return True
    return False

def extract_target_text(item_result, item_bench):
    result_text = normalize_text(item_result.get('result_text', ''))
    article_match = re.search(r'\b(the|a|an)\b\s+(.*)', result_text)
    if article_match:
        return article_match.group(2).strip()
        
    preposition = normalize_text(item_bench.get('preposition', ''))
    if preposition:
        prep_pattern = r'\b' + re.escape(preposition) + r'\b\s+(.*)'
        prep_match = re.search(prep_pattern, result_text)
        if prep_match:
            return prep_match.group(1).strip()
            
    return result_text

def main():
    parser = argparse.ArgumentParser(description="Evaluate PT task (spatial_understand)")
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
        if item.get('result_time') == -2:
            print(f"Error: Found sample with result_time == -2 (id: {item.get('id', 'unknown')}). Exiting.")
            sys.exit(1)

    if len(result_data) != len(bench_data):
        print(f"Error: Sample counts do not match! Result: {len(result_data)}, Benchmark: {len(bench_data)}")
        sys.exit(1)

    bench_dict = {item['id']: item for item in bench_data if 'id' in item}
    
    total_samples = len(result_data)
    time_correct_count = 0
    full_correct_count = 0

    for res_item in result_data:
        res_id = res_item.get('id')
        if res_id not in bench_dict:
            print(f"Warning: Result id {res_id} not found in benchmark.")
            continue
            
        bench_item = bench_dict[res_id]
        res_time = res_item.get('result_time', -999)
        
        try:
            trigger_time = bench_item['answer'][0]['trigger_time']
        except (KeyError, IndexError):
            continue
            
        time_is_correct = False
        if trigger_time - 1 <= res_time <= trigger_time + 1:
            time_is_correct = True
            time_correct_count += 1
            
        if not time_is_correct:
            continue

        text_is_correct = False
        if "ego_spatial" in bench_item:
            try:
                target_text = normalize_text(bench_item['answer'][0]['text'])
                result_text = normalize_text(res_item.get('result_text', ''))
                if result_text == target_text:
                    text_is_correct = True
            except (KeyError, IndexError):
                pass
        elif "exo_spatial" in bench_item:
            answer_list = bench_item.get('answer_list', [])
            if answer_list:
                extracted_text = extract_target_text(res_item, bench_item)
                if check_match(extracted_text, answer_list):
                    text_is_correct = True

        if text_is_correct:
            full_correct_count += 1

    print(f"Total Samples: {total_samples}")
    print(f"Time Correct: {time_correct_count} ({(time_correct_count/total_samples*100) if total_samples else 0:.2f}%)")
    print(f"Full Correct: {full_correct_count} ({(full_correct_count/total_samples*100) if total_samples else 0:.2f}%)")

if __name__ == "__main__":
    main()
