import argparse
import json
import os
import glob
import time
import torch
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_image

# ==========================================
# 超参数配置
# ==========================================
INPUT_JSON = ""
OUTPUT_JSON = ""
FRAME_BASE_DIR = ""
MODEL_PATH = ""

BATCH_SIZE = 1
MAX_TOKENS = 500
MAX_FRAMES = 16

def extract_result(text: str) -> str:
    """从模型输出中提取 <attention> 或 <silence>"""
    if text is None:
        return "Error: Empty response"
        
    text_lower = text.lower()
    if "<attention>" in text_lower:
        return "<attention>"
    elif "<silence>" in text_lower:
        return "<silence>"
    return text

class SampleState:
    def __init__(self, sample):
        self.sample = sample
        self.sample_id = str(sample['id'])
        self.video_st = int(sample.get('video_st', 0))
        self.video_ed = int(sample.get('video_ed', 0))
        
        self.answers = sample.get('answer', [])
        self.full_user_prompt = sample['question'][0]['text']
        
        self.num_answers = len(self.answers)
        self.result_time_list = [-1] * self.num_answers
        
        self.current_ans_idx = 0
        self.target_times = []
        self.current_time_idx = 0
        self.done = False
        
        if self.num_answers == 0:
            self.done = True
        else:
            self._init_current_answer_target_times()

    def _init_current_answer_target_times(self):
        trigger_time = int(self.answers[self.current_ans_idx].get('trigger_time', 0))
        start_time = max(self.video_st, trigger_time - 4)
        end_time = min(self.video_ed, trigger_time + 4)
        if start_time <= end_time:
            self.target_times = list(range(start_time, end_time + 1))
        else:
            self.target_times = []
        self.current_time_idx = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--frame_dir', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=False, default="")

    parser.add_argument('--api_endpoint', type=str, required=False, default="")

    parser.add_argument('--api_version', type=str, required=False, default="")

    parser.add_argument('--model_name', type=str, required=False, default="")

    parser.add_argument('--model_path', type=str, required=False, default="")

    args = parser.parse_args()
    
    global INPUT_JSON, OUTPUT_JSON, FRAME_BASE_DIR, API_KEY, API_ENDPOINT, API_VERSION, MODEL_NAME, MODEL_PATH

    
    INPUT_JSON = args.input

    
    OUTPUT_JSON = args.output

    
    FRAME_BASE_DIR = args.frame_dir

    
    if hasattr(args, 'api_key') and args.api_key: API_KEY = args.api_key

    
    if hasattr(args, 'api_endpoint') and args.api_endpoint: API_ENDPOINT = args.api_endpoint

    
    if hasattr(args, 'api_version') and args.api_version: API_VERSION = args.api_version

    
    if hasattr(args, 'model_name') and args.model_name: MODEL_NAME = args.model_name

    
    if hasattr(args, 'model_path') and args.model_path: MODEL_PATH = args.model_path

    if not os.path.exists(INPUT_JSON):
        print(f"Input file not found: {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    # 增量读取和断点续跑
    processed_data = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    processed_data[str(item['id'])] = item
        except Exception as e:
            print(f"Failed to load existing output json: {e}")

    # Load Model
    print("Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    print("Model loaded.")

    unprocessed_samples = [SampleState(s) for s in data if str(s['id']) not in processed_data]
    
    # 清理掉一开始就无效的
    for state in list(unprocessed_samples):
        if state.done:
            unprocessed_samples.remove(state)
            sample_copy = dict(state.sample)
            sample_copy['result_time'] = []
            processed_data[state.sample_id] = sample_copy
            
    active_states = []
    results = processed_data.copy()

    print(f"Starting batch processing. Batch size: {BATCH_SIZE}, Samples to process: {len(unprocessed_samples)}")

    while unprocessed_samples or active_states:
        # Fill active states up to BATCH_SIZE
        while len(active_states) < BATCH_SIZE and unprocessed_samples:
            state = unprocessed_samples.pop(0)
            
            # 如果某个区间的 target_times 为空，自动跳到下一个 answer
            while not state.done and not state.target_times:
                state.current_ans_idx += 1
                if state.current_ans_idx >= state.num_answers:
                    state.done = True
                else:
                    state._init_current_answer_target_times()
            
            if state.done:
                sample_copy = dict(state.sample)
                sample_copy['result_time'] = state.result_time_list
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                continue
                
            print(f"Starting sample {state.sample_id} Answer {state.current_ans_idx}...")
            active_states.append(state)
            
        if not active_states:
            break
            
                # Prepare batch (Serial Processing)
        output_texts = []
        for state in active_states:
            x = state.target_times[state.current_time_idx]
            
            frame_start = max(state.video_st, x - MAX_FRAMES + 1)
            selected_frames = list(range(frame_start, x + 1))
            if len(selected_frames) > MAX_FRAMES:
                selected_frames = selected_frames[-MAX_FRAMES:]
                
            video_frames = []
            for frame_idx in selected_frames:
                frame_path_jpg = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.jpg")
                frame_path_png = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.png")
                frame_path = frame_path_jpg if os.path.exists(frame_path_jpg) else frame_path_png
                
                if os.path.exists(frame_path):
                    video_frames.append(frame_path)
            
            question = state.full_user_prompt
            if video_frames:
                question += '\n' + ''.join(['<image>\n' for _ in video_frames])
            else:
                print(f"Warning: No frames collected for sample {state.sample_id}")
            
            pixel_values_list = []
            num_patches_list = []
            for frame_path in video_frames:
                pv = load_image(frame_path, max_num=1).to(torch.bfloat16).cuda()
                pixel_values_list.append(pv)
                num_patches_list.append(pv.size(0))
                
            if pixel_values_list:
                pixel_values = torch.cat(pixel_values_list, dim=0)
            else:
                pixel_values = None
                
            generation_config = dict(max_new_tokens=MAX_TOKENS, do_sample=False)
            
            response = model.chat(tokenizer, pixel_values, question, generation_config,
                                     num_patches_list=num_patches_list,
                                     history=None, return_history=False)
            output_texts.append(response)
        
        # Process results
        next_active_states = []
        for state, out_text in zip(active_states, output_texts):
            x = state.target_times[state.current_time_idx]
            parsed_result = extract_result(out_text)
            print(f"Sample {state.sample_id} Answer {state.current_ans_idx} at X={x}: {parsed_result}")
            
            if parsed_result == "<attention>":
                state.result_time_list[state.current_ans_idx] = x
                state.current_ans_idx += 1
                if state.current_ans_idx >= state.num_answers:
                    state.done = True
                else:
                    state._init_current_answer_target_times()
            else:
                state.current_time_idx += 1
                if state.current_time_idx >= len(state.target_times):
                    # 没命中，当前 answer 留为 -1（初始化的值）
                    state.current_ans_idx += 1
                    if state.current_ans_idx >= state.num_answers:
                        state.done = True
                    else:
                        state._init_current_answer_target_times()
                        
            # 预处理下一段为空区间的跳过逻辑
            while not state.done and not state.target_times:
                state.current_ans_idx += 1
                if state.current_ans_idx >= state.num_answers:
                    state.done = True
                else:
                    state._init_current_answer_target_times()
                    
            if state.done:
                # Save to results
                sample_copy = dict(state.sample)
                sample_copy['result_time'] = state.result_time_list
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                print(f"Completed sample {state.sample_id}, Result Time List: {state.result_time_list}")
            else:
                next_active_states.append(state)
                
        active_states = next_active_states

    print("All done!")

if __name__ == "__main__":
    main()