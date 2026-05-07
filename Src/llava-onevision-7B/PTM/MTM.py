import argparse
import json
import os
import glob
import time
import torch
import re
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# ==========================================
# 超参数配置
# ==========================================
INPUT_JSON = ""
OUTPUT_JSON = ""
FRAME_BASE_DIR = ""
MODEL_PATH = ""

BATCH_SIZE = 16
MAX_TOKENS = 500
MAX_FRAMES = 16

def extract_result(text: str) -> bool:
    """如果模型输出中包含 <attention>，则返回 True，否则返回 False"""
    if not text:
        return False
        
    if "<attention>" in text.lower():
        return True
    return False

class SampleState:
    def __init__(self, sample):
        self.sample = sample
        self.sample_id = str(sample['id'])
        self.video_st = int(sample['video_st'])
        self.video_ed = int(sample['video_ed'])
        
        trigger_time0 = int(sample['answer'][0]['trigger_time']) 
        trigger_time1 = int(sample['answer'][1]['trigger_time']) 
        end_time0 = int(sample['answer'][0]['end_time'])
        self.full_user_prompt = sample['question'][0]['text']
        
        # 两个测试区间
        start_time1 = max(self.video_st, trigger_time0 - 4)
        end_time1 = min(trigger_time0 + 4, end_time0 + 1)
        self.target_times1 = list(range(start_time1, end_time1 + 1)) if start_time1 <= end_time1 else []
        
        start_time2 = max(end_time0 + 1, trigger_time1 - 4)
        end_time2 = min(self.video_ed, trigger_time1 + 4)
        self.target_times2 = list(range(start_time2, end_time2 + 1)) if start_time2 <= end_time2 else []
        
        self.result_time = [-1, -1]
        
        self.current_interval = 0  # 0 or 1
        self.current_idx = 0
        self.done = False
        
        if not self.target_times1 and not self.target_times2:
            self.done = True

def get_current_target_times(state: SampleState) -> List[int]:
    return state.target_times1 if state.current_interval == 0 else state.target_times2

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
                    item.pop('result_text', None)
                    processed_data[str(item['id'])] = item
        except Exception as e:
            print(f"Failed to load existing output json: {e}")

    # Load Model
    print("Loading model...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = 'left'
    print("Model loaded.")

    unprocessed_samples = [SampleState(s) for s in data if str(s['id']) not in processed_data]
    
    # 清理一上来就结束的任务
    for state in list(unprocessed_samples):
        if state.done:
            unprocessed_samples.remove(state)
            sample_copy = dict(state.sample)
            sample_copy.pop('result_text', None)
            sample_copy['result_time'] = state.result_time
            processed_data[state.sample_id] = sample_copy
            
    active_states = []
    results = processed_data.copy()

    print(f"Starting batch processing. Batch size: {BATCH_SIZE}, Samples to process: {len(unprocessed_samples)}")

    while unprocessed_samples or active_states:
        # Fill active states up to BATCH_SIZE
        while len(active_states) < BATCH_SIZE and unprocessed_samples:
            state = unprocessed_samples.pop(0)
            
            # 预处理空区间
            while not state.done:
                curr_times = get_current_target_times(state)
                if not curr_times:
                    state.current_interval += 1
                    state.current_idx = 0
                    if state.current_interval > 1:
                        state.done = True
                else:
                    break
                    
            if state.done:
                sample_copy = dict(state.sample)
                sample_copy['result_time'] = state.result_time
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                continue
                
            print(f"Starting sample {state.sample_id} Interval {state.current_interval}...")
            active_states.append(state)
            
        if not active_states:
            break
            
        # Prepare batch
        batch_messages = []
        for state in active_states:
            curr_times = get_current_target_times(state)
            x = curr_times[state.current_idx]
            current_end = x
            
            content = []
            content.append({
                "type": "text",
                "text": state.full_user_prompt
            })
            
            # 收集从 video_st 到 current_end 的图片
            all_frames = list(range(state.video_st, current_end + 1))
            if len(all_frames) > MAX_FRAMES:
                selected_frames = all_frames[-MAX_FRAMES:]
            else:
                selected_frames = all_frames
                
            video_frames = []
            for frame_idx in selected_frames:
                frame_path_jpg = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.jpg")
                frame_path_png = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.png")
                frame_path = frame_path_jpg if os.path.exists(frame_path_jpg) else frame_path_png
                
                if os.path.exists(frame_path):
                    video_frames.append(f"file://{frame_path}")
            
            if video_frames:
                for frame_path in video_frames:
                    content.append({
                        "type": "image",
                        "image": frame_path,
                        "max_pixels": 360 * 420,
                    })
            else:
                print(f"Warning: No frames collected for sample {state.sample_id} from {state.video_st} to {current_end}")
                
            message = {
                "role": "user",
                "content": content
            }
            batch_messages.append([message])
            
        # Preparation for llava-onevision inference
        # 1. 转换 batch_messages 和抽取 PIL 图像
        llava_batch_messages = []
        batch_images = []
        
        for msg_list in batch_messages:
            msg = msg_list[0]
            content_list = msg["content"]
            
            new_content_list = []
            sample_images = []
            
            for item in content_list:
                if item["type"] == "text":
                    new_content_list.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    img_path = item["image"].replace("file://", "")
                    try:
                        img = Image.open(img_path).convert("RGB")
                        sample_images.append(img)
                        # llava-ov prompt 里只需要 {"type": "image"}
                        new_content_list.append({"type": "image"})
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        
            llava_batch_messages.append([{"role": "user", "content": new_content_list}])
            batch_images.append(sample_images)
            
        # 2. apply_chat_template 构建文本 prompt
        prompts = [
            processor.apply_chat_template(conv, add_generation_prompt=True)
            for conv in llava_batch_messages
        ]
        
        # 3. flatten batch_images，因为 processor 接受一维的图像列表或者二维嵌套
        # 官方对于多图的 batch 处理，目前 AutoProcessor 支持 list of lists
        # processor(images=batch_images, text=prompts, ...)
        
        try:
            inputs = processor(
                images=batch_images if any(batch_images) else None,
                text=prompts,
                padding=True,
                return_tensors="pt"
            ).to(model.device, torch.float16)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except Exception as e:
            print(f"Inference error: {e}")
            output_texts = ["<error>"] * len(active_states)
        
        # Process results
        next_active_states = []
        for state, out_text in zip(active_states, output_texts):
            curr_times = get_current_target_times(state)
            x = curr_times[state.current_idx]
            
            has_attention = extract_result(out_text)
            print(f"Sample {state.sample_id} Interval {state.current_interval} at X={x}: Raw={repr(out_text)}, HasAttention={has_attention}")
            
            if out_text == "<error>":
                state.result_time[state.current_interval] = -2
                state.current_interval += 1
                state.current_idx = 0
                continue

            if has_attention:
                state.result_time[state.current_interval] = x
                # 命中后，直接跳到下一个区间
                state.current_interval += 1
                state.current_idx = 0
            else:
                state.current_idx += 1
                if state.current_idx >= len(curr_times):
                    # 测完了本区间，没命中，跳到下一个区间
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
                sample_copy['result_time'] = state.result_time
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                print(f"Completed sample {state.sample_id}, Result Time: {state.result_time}")
            else:
                next_active_states.append(state)
                
        active_states = next_active_states

    print("All done!")

if __name__ == "__main__":
    main()