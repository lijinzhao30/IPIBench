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

def extract_attention(text: str) -> bool:
    """从模型输出中提取 <attention>"""
    if text is None:
        return False
    text_lower = text.lower()
    return "<attention>" in text_lower

class SampleState:
    def __init__(self, sample):
        self.sample = sample
        self.sample_id = str(sample['id'])
        self.video_st = int(sample.get('video_st', 0))
        self.video_ed = int(sample.get('video_ed', 0))
        
        self.questions = sample.get('question', [])
        self.answers = sample.get('answer', [])
        
        self.valid = len(self.questions) >= 2 and len(self.answers) >= 1
        
        self.first_attention_time = -1
        self.ans2 = ""
        self.last_test_time = -1
        
        self.current_interval = 0  # 0: first interval, 1: second interval
        self.current_idx = 0
        self.done = False
        
        self.target_times_1 = []
        self.target_times_2 = []
        
        if self.valid:
            self.text1 = self.questions[0]['text']
            self.text2 = self.questions[1]['text']
            self.t2 = int(self.questions[1]['t'])
            
            trigger_time_1 = int(self.answers[0]['trigger_time'])
            
            # Interval 1
            start_test = max(self.video_st, trigger_time_1 - 4)
            end_test = min(self.t2 - 1, trigger_time_1 + 4)
            if start_test <= end_test:
                self.target_times_1 = list(range(start_test, end_test + 1))
            
            # Interval 2
            self.end_frame2_time = self.t2 - 1
            self.target_times_2 = [self.end_frame2_time]
        else:
            self.first_attention_time = -2
            self.done = True

def get_current_target_times(state: SampleState) -> List[int]:
    if state.current_interval == 0:
        return state.target_times_1
    else:
        return state.target_times_2

def prepare_message_for_state(state: SampleState, x: int) -> list:
    content = []
    history = []
    
    if state.current_interval == 0:
        content.append({
            "type": "text",
            "text": state.text1
        })
        
        frame_start = max(state.video_st, x - MAX_FRAMES + 1)
        selected_frames = list(range(frame_start, x + 1))
        
        for frame_idx in selected_frames:
            frame_path_jpg = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.jpg")
            frame_path_png = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.png")
            frame_path = frame_path_jpg if os.path.exists(frame_path_jpg) else frame_path_png
            
            if os.path.exists(frame_path):
                content.append({
                    "type": "image",
                    "image": f"file://{frame_path}",
                    "max_pixels": 360 * 420,
                })
    elif state.current_interval == 1:
        # Interval 2
        if state.first_attention_time != -1:
            attention_val = "<attention>"
            split_time = state.first_attention_time
        else:
            attention_val = "<silence>"
            split_time = state.last_test_time

        content.append({
            "type": "text",
            "text": state.text1
        })
        
        frame_start_2 = max(state.video_st, state.end_frame2_time - MAX_FRAMES + 1)
        selected_frames_2 = list(range(frame_start_2, state.end_frame2_time + 1))
        
        frames_part1 = [f for f in selected_frames_2 if f <= split_time]
        frames_part2 = [f for f in selected_frames_2 if f > split_time]
        
        for frame_idx in frames_part1:
            frame_path_jpg = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.jpg")
            frame_path_png = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.png")
            frame_path = frame_path_jpg if os.path.exists(frame_path_jpg) else frame_path_png
            
            if os.path.exists(frame_path):
                content.append({
                    "type": "image",
                    "image": f"file://{frame_path}",
                    "max_pixels": 360 * 420,
                })
                
        history.append({
            "role": "user",
            "content": content
        })
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": attention_val}]
        })
        content = []
        
        for frame_idx in frames_part2:
            frame_path_jpg = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.jpg")
            frame_path_png = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.png")
            frame_path = frame_path_jpg if os.path.exists(frame_path_jpg) else frame_path_png
            
            if os.path.exists(frame_path):
                content.append({
                    "type": "image",
                    "image": f"file://{frame_path}",
                    "max_pixels": 360 * 420,
                })
                
        content.append({
            "type": "text",
            "text": state.text2
        })
        
    history.append({
        "role": "user",
        "content": content
    })
    return history

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
                    if 'result' in item and isinstance(item['result'], list) and len(item['result']) >= 2:
                        if item['result'][0] != -2:
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
    
    active_states = []
    results = processed_data.copy()

    print(f"Starting batch processing. Batch size: {BATCH_SIZE}, Samples to process: {len(unprocessed_samples)}")

    while unprocessed_samples or active_states:
        # Fill active states up to BATCH_SIZE
        while len(active_states) < BATCH_SIZE and unprocessed_samples:
            state = unprocessed_samples.pop(0)
            
            # 预检查跳过空的区间
            while not state.done:
                curr_times = get_current_target_times(state)
                if not curr_times:
                    # 空区间，跳到下一个
                    state.current_interval += 1
                    state.current_idx = 0
                    if state.current_interval > 1:
                        state.done = True
                else:
                    break
                    
            if state.done:
                sample_copy = dict(state.sample)
                sample_copy['result'] = [state.first_attention_time, state.ans2]
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                continue
                
            print(f"Starting sample {state.sample_id} Interval {state.current_interval + 1}...")
            active_states.append(state)
            
        if not active_states:
            break
            
        # Prepare batch
        batch_messages = []
        for state in active_states:
            curr_times = get_current_target_times(state)
            x = curr_times[state.current_idx]
            
            history = prepare_message_for_state(state, x)
            batch_messages.append(history)
            if state.current_interval == 0:
                state.last_test_time = x
            
        # Perform inference serially
        output_texts = []
        for state, history_or_msg in zip(active_states, batch_messages):
            try:
                x = get_current_target_times(state)[state.current_idx]
            except NameError:
                x = state.target_times[state.current_time_idx] if hasattr(state, 'current_time_idx') else state.target_times[state.current_idx]
            
            # Since Qwen used apply_chat_template with a history format, 
            # we need to translate history_or_msg back to internVL's inputs.
            # `batch_messages` was a list of lists of dicts (messages) or a single list.
            if isinstance(history_or_msg, list) and len(history_or_msg) > 0 and isinstance(history_or_msg[0], list):
                # Unpack if nested list
                history_or_msg = history_or_msg[0]
                
            internvl_history = []
            pixel_values_list = []
            num_patches_list = []
            
            # Parse history_or_msg (which is a list of {"role": ..., "content": [...]})
            for i, turn in enumerate(history_or_msg):
                if turn['role'] == 'user':
                    turn_text = ""
                    turn_image_paths = []
                    for item in turn['content']:
                        if item['type'] == 'text':
                            turn_text = item['text']
                        elif item['type'] == 'image':
                            path = item['image'].replace('file://', '')
                            turn_image_paths.append(path)
                    
                    q_text = turn_text
                    if turn_image_paths:
                        # Append <image> placeholders for this turn
                        q_text += '\n' + ''.join(['<image>\n' for _ in turn_image_paths])
                    
                    if i == len(history_or_msg) - 1:
                        # Current question
                        question = q_text
                        for path in turn_image_paths:
                            pv = load_image(path, max_num=1).to(torch.bfloat16).cuda()
                            pixel_values_list.append(pv)
                            num_patches_list.append(pv.size(0))
                    else:
                        # historical user question
                        # (We don't support historical images easily with InternVL multi-turn without packing them all into the current call, 
                        # so we just maintain text if there are no images, or if there are images we might need to adjust.
                        # Wait, InternVL multi-turn allows passing history as [(q, a), (q, a)]. We just append.)
                        # We will append to history after seeing assistant's reply.
                        pending_q = q_text
                elif turn['role'] == 'assistant':
                    ans_text = ""
                    for item in turn['content']:
                        if item['type'] == 'text':
                            ans_text = item['text']
                    internvl_history.append((pending_q, ans_text))
                    
            if pixel_values_list:
                pixel_values = torch.cat(pixel_values_list, dim=0)
            else:
                pixel_values = None
                
            generation_config = dict(max_new_tokens=MAX_TOKENS, do_sample=False)
            
            response = model.chat(tokenizer, pixel_values, question, generation_config,
                                     num_patches_list=num_patches_list,
                                     history=internvl_history if internvl_history else None, 
                                     return_history=False)
            output_texts.append(response)
        
        # Process results
        next_active_states = []
        for state, out_text in zip(active_states, output_texts):
            curr_times = get_current_target_times(state)
            x = curr_times[state.current_idx]
            
            print(f"Sample {state.sample_id} Interval {state.current_interval + 1} at X={x}: {out_text}")
            
            if state.current_interval == 0:
                if extract_attention(out_text):
                    state.first_attention_time = x
                    # 命中后，直接跳到下一个区间
                    state.current_interval += 1
                    state.current_idx = 0
                else:
                    state.current_idx += 1
                    if state.current_idx >= len(curr_times):
                        # 测完了本区间，没命中，跳到下一个区间
                        state.current_interval += 1
                        state.current_idx = 0
            else:
                # Interval 2: 记录所有的 ans2
                state.ans2 = out_text
                state.current_idx += 1
                if state.current_idx >= len(curr_times):
                    # 测完了第二个区间，直接结束
                    state.current_interval += 1
                    state.current_idx = 0
                    
            # 预处理空区间
            while not state.done and state.current_interval <= 1:
                next_times = get_current_target_times(state)
                if not next_times:
                    state.current_interval += 1
                else:
                    break
                    
            if state.current_interval > 1:
                state.done = True
                    
            if state.done:
                # Save to results
                sample_copy = dict(state.sample)
                sample_copy['result'] = [state.first_attention_time, state.ans2]
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                print(f"Completed sample {state.sample_id}, Result: {[state.first_attention_time, state.ans2]}")
            else:
                next_active_states.append(state)
                
        active_states = next_active_states

    print("All done!")

if __name__ == "__main__":
    main()
