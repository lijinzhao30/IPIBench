import json
import sys
import argparse

def check_text_match(pred_text, target_list):
    if not pred_text or not target_list:
        return False
    pred_lower = str(pred_text).lower()
    for target_text in target_list:
        tgt_lower = str(target_text).lower()
        if pred_lower in tgt_lower or tgt_lower in pred_lower:
            return True
    return False

def evaluate_modify(res_data, bench_data):
    total = len(res_data)
    correct = 0
    bench_dict = {item['id']: item for item in bench_data if 'id' in item}
    
    for item in res_data:
        try:
            bench_item = bench_dict.get(item.get('id'), item)
            result_times = item.get('result_time', [-999, -999])
            if isinstance(result_times, list) and len(result_times) >= 2:
                r1 = float(result_times[0])
                r2 = float(result_times[1])
                
                answers = bench_item.get('answer', [])
                if len(answers) >= 2:
                    t1 = float(answers[0]['trigger_time'])
                    t2 = float(answers[1]['trigger_time'])
                    
                    if abs(r1 - t1) <= 1.0 and abs(r2 - t2) <= 1.0:
                        correct += 1
        except Exception:
            pass
    accuracy = correct / total if total > 0 else 0
    print(f"Total samples: {total}")
    print(f"Correct: {correct} ({accuracy:.2%})")

def evaluate_r_to_p(res_data, bench_data):
    total = len(res_data)
    bench_dict = {item['id']: item for item in bench_data if 'id' in item}
    reactive_correct = 0
    proactive_correct = 0
    fully_correct = 0

    for item in res_data:
        bench_item = bench_dict.get(item.get('id'), item)
        res = item.get('result', [])
        if not res or len(res) < 2:
            continue
        
        predicted_text = str(res[0]).strip()
        try:
            predicted_time = float(res[1])
        except:
            predicted_time = -999.0
            
        answers = bench_item.get('answer', [])
        if not answers or len(answers) < 2:
            continue
            
        target_text_list = answers[0].get('text_list', [])
        target_time = answers[1].get('trigger_time', -1)
        
        is_react = check_text_match(predicted_text, target_text_list)
        is_proact = (target_time != -1 and abs(predicted_time - target_time) <= 1.0)
        
        if is_react: reactive_correct += 1
        if is_proact: proactive_correct += 1
        if is_react and is_proact: fully_correct += 1

    print(f"Total samples: {total}")
    print(f"Reactive Correct: {reactive_correct} ({reactive_correct/total if total else 0:.2%})")
    print(f"Proactive Correct: {proactive_correct} ({proactive_correct/total if total else 0:.2%})")
    print(f"Full Correct: {fully_correct} ({fully_correct/total if total else 0:.2%})")

def evaluate_r_under_p(res_data, bench_data):
    total = len(res_data)
    bench_dict = {item['id']: item for item in bench_data if 'id' in item}
    time_correct = 0
    strict_score = 0.0

    for item in res_data:
        bench_item = bench_dict.get(item.get('id'), item)
        res = item.get('result', [])
        if not res or len(res) == 0:
            continue
            
        predicted_time = res[0]
        answers = bench_item.get('answer', [])
        if not answers or len(answers) == 0:
            continue
            
        target_time = answers[0].get('trigger_time', -1)
        is_time_correct = (target_time != -1 and abs(predicted_time - target_time) <= 1.0)
        
        if is_time_correct:
            time_correct += 1
            
            ans1_correct = False
            if len(res) > 1 and len(answers) > 1:
                ans1_correct = check_text_match(res[1], answers[1].get('text_list', []))
                
            has_ans2 = len(res) > 2 and len(answers) > 2
            ans2_correct = False
            if has_ans2:
                ans2_correct = check_text_match(res[2], answers[2].get('text_list', []))
                
            if has_ans2:
                if ans1_correct and ans2_correct: strict_score += 1.0
                elif ans1_correct or ans2_correct: strict_score += 0.5
            else:
                if ans1_correct: strict_score += 1.0

    print(f"Total samples: {total}")
    print(f"Time Correct: {time_correct} ({time_correct/total if total else 0:.2%})")
    print(f"Strict Score: {strict_score} ({strict_score/total if total else 0:.2%})")

def evaluate_r_after_p(res_data, bench_data):
    total = len(res_data)
    bench_dict = {item['id']: item for item in bench_data if 'id' in item}
    reactive_correct = 0
    proactive_correct = 0
    fully_correct = 0

    for item in res_data:
        bench_item = bench_dict.get(item.get('id'), item)
        res = item.get('result', [])
        if not res or len(res) < 2:
            continue
            
        predicted_time = res[0]
        predicted_text = str(res[1]).strip()
        
        answers = bench_item.get('answer', [])
        if not answers or len(answers) < 2:
            continue
            
        target_time = answers[0].get('trigger_time', -1)
        target_text_list = answers[1].get('text_list', [])
        
        is_react = check_text_match(predicted_text, target_text_list)
        is_proact = (target_time != -1 and abs(predicted_time - target_time) <= 1.0)
        
        if is_react: reactive_correct += 1
        if is_proact: proactive_correct += 1
        if is_react and is_proact: fully_correct += 1

    print(f"Total samples: {total}")
    print(f"Reactive Correct: {reactive_correct} ({reactive_correct/total if total else 0:.2%})")
    print(f"Proactive Correct: {proactive_correct} ({proactive_correct/total if total else 0:.2%})")
    print(f"Full Correct: {fully_correct} ({fully_correct/total if total else 0:.2%})")

def evaluate_r_under_p_instruct(res_data, bench_data):
    total_reactive = 0
    correct_reactive = 0
    bench_dict = {item['id']: item for item in bench_data if 'id' in item}
    
    for item in res_data:
        bench_item = bench_dict.get(item.get('id'), item)
        res = item.get('result', [])
        answers = bench_item.get('answer', [])
        
        for idx, ans in enumerate(answers):
            if ans.get('type') == 'reactive':
                total_reactive += 1
                pred = res[idx] if idx < len(res) else ""
                if check_text_match(pred, ans.get('text_list', [])):
                    correct_reactive += 1
                    
    print(f"Total Reactive Samples: {total_reactive}")
    print(f"Reactive Correct: {correct_reactive} ({correct_reactive/total_reactive if total_reactive else 0:.2%})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate TM task (modify / R_to_P / R_under_P / R_after_P_Delay)")
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
        # 1. check modify
        res_t = item.get('result_time')
        if isinstance(res_t, list) and -2 in res_t:
            print(f"Error: result_time contains -2 at index {i}. Exiting.")
            sys.exit(1)
        if res_t == -2:
            print(f"Error: result_time is -2 at index {i}. Exiting.")
            sys.exit(1)
        # 2. check tm tasks
        res = item.get('result', [])
        if isinstance(res, list):
            for r in res:
                if r == -2 or str(r) == "-2":
                    print(f"Error: API error -2 found in result at index {i}. Exiting.")
                    sys.exit(1)

    # Detect the specific sub-task type from the benchmark answer signature
    first_ans = benchmark_data[0].get('answer', [])
    first_q = benchmark_data[0].get('question', [])
    
    # Heuristics to determine evaluation logic
    task_name = benchmark_data[0].get('task', '')
    
    if task_name == "Task Modification" or "modify" in args.benchmark_path.lower():
        print("Evaluating as Task Modification...")
        evaluate_modify(result_data, benchmark_data)
    else:
        # Check answer signatures
        if len(first_ans) > 0 and first_ans[0].get('type') == 'reactive':
            # Could be R_to_P or R_after_P_Immediate
            if len(first_ans) > 1 and first_ans[1].get('type') == 'proactive':
                print("Evaluating as R_to_P...")
                evaluate_r_to_p(result_data, benchmark_data)
            else:
                print("Evaluating as R_under_P_instruct / reactive...")
                evaluate_r_under_p_instruct(result_data, benchmark_data)
        elif len(first_ans) > 0 and first_ans[0].get('type') == 'proactive':
            # Could be R_under_P or R_after_P_Delay
            # Let's check R_under_P logic (if trigger_time exists for answer[1] vs answer[0])
            if len(first_ans) > 1 and first_ans[1].get('type') == 'reactive':
                # Actually both R_under_P and R_after_P_Delay have proactive then reactive
                # Check file name as heuristic if needed, or default to R_after_P
                if "under" in args.benchmark_path.lower():
                    if "instruct" in args.benchmark_path.lower() or "reactive" in args.benchmark_path.lower():
                        print("Evaluating as R_under_P_instruct...")
                        evaluate_r_under_p_instruct(result_data, benchmark_data)
                    else:
                        print("Evaluating as R_under_P...")
                        evaluate_r_under_p(result_data, benchmark_data)
                elif "after" in args.benchmark_path.lower():
                    print("Evaluating as R_after_P...")
                    evaluate_r_after_p(result_data, benchmark_data)
                else:
                    # In TM, the answer format is either:
                    # R_to_P: reactive, proactive
                    # R_after_P: proactive, reactive
                    # R_under_P: proactive, reactive
                    # Let's just run TM as a generic modify logic if we are evaluating the TM.json merged file, 
                    # but TM.json actually has varying signatures inside!
                    pass

    # If this is the merged TM.json, we can iterate per sample and apply the right logic.
    # The user states: TM includes R_after_P_Delay, R_to_P, R_under_P, R_under_P_reactive.
    if args.benchmark_path.endswith("TM.json"):
        print("\n--- Evaluating combined TM.json ---")
        # Just running modify, r_to_p, r_under_p, r_after_p checks on respective sub-types
        modify_res, modify_bench = [], []
        r_to_p_res, r_to_p_bench = [], []
        r_under_p_res, r_under_p_bench = [], []
        r_after_p_res, r_after_p_bench = [], []
        r_react_res, r_react_bench = [], []

        for i, b in enumerate(benchmark_data):
            r = result_data[i]
            ans = b.get('answer', [])
            task_info = b.get('task', '')
            
            # Simple heuristic based on benchmark keys
            if task_info == "Task Modification" or (len(ans)==2 and 'label' in ans[0]):
                modify_res.append(r); modify_bench.append(b)
            elif len(ans) >= 2 and ans[0].get('type') == 'reactive' and ans[1].get('type') == 'proactive':
                r_to_p_res.append(r); r_to_p_bench.append(b)
            elif len(ans) >= 2 and ans[0].get('type') == 'proactive' and ans[1].get('type') == 'reactive':
                # Distinction between under_P and after_P in TM.json? 
                # Let's just group them into after_P/under_P depending on question structure or just evaluate both
                if 'Delay' in b.get('task', '') or 'Immediate' in b.get('task', ''):
                    r_after_p_res.append(r); r_after_p_bench.append(b)
                else:
                    r_under_p_res.append(r); r_under_p_bench.append(b)
            elif any(a.get('type') == 'reactive' for a in ans):
                r_react_res.append(r); r_react_bench.append(b)
        
        if modify_bench:
            print(f"\n[Task Modification] ({len(modify_bench)} samples)")
            evaluate_modify(modify_res, modify_bench)
        if r_to_p_bench:
            print(f"\n[R_to_P] ({len(r_to_p_bench)} samples)")
            evaluate_r_to_p(r_to_p_res, r_to_p_bench)
        if r_under_p_bench:
            print(f"\n[R_under_P] ({len(r_under_p_bench)} samples)")
            evaluate_r_under_p(r_under_p_res, r_under_p_bench)
        if r_after_p_bench:
            print(f"\n[R_after_P] ({len(r_after_p_bench)} samples)")
            evaluate_r_after_p(r_after_p_res, r_after_p_bench)
        if r_react_bench:
            print(f"\n[R_under_P_instruct / Reactive] ({len(r_react_bench)} samples)")
            evaluate_r_under_p_instruct(r_react_res, r_react_bench)

if __name__ == "__main__":
    main()
