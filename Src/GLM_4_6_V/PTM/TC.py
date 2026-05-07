import argparse
import json
import os
import glob
import time
import base64
import threading
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
from openai import AzureOpenAI

# ==========================================
# 超参数配置
# ==========================================
INPUT_JSON = ""
OUTPUT_JSON = ""
FRAME_BASE_DIR = ""

API_ENDPOINT = ""
API_KEY = ""
API_VERSION = ""
MODEL_NAME = "glm-4.6v"

MAX_TOKENS = 500
NUM_THREADS = 16
MAX_FRAMES = 16

# 用于保护文件写入的锁
write_lock = threading.Lock()

def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def init_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

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

def test_sample(client: AzureOpenAI, sample: Dict[str, Any]) -> Tuple[int, List[str]]:
    """测试单个样本，返回 result_time 和 cancel_list"""
    sample_id = str(sample['id'])
    video_st = int(sample.get('video_st', 0))
    video_ed = int(sample.get('video_ed', 0))
    
    questions = sample.get('question', [])
    answers = sample.get('answer', [])
    
    valid = len(questions) >= 2 and len(answers) >= 2
    if not valid:
        return -1, []
        
    q0_text = questions[0]['text']
    q1_text = questions[1]['text']
    q1_t = int(questions[1]['t'])
    
    ans0_t = int(answers[0]['trigger_time'])
    ans1_t = int(answers[1]['trigger_time'])
    
    # Interval 1
    start_time_1 = max(video_st, ans0_t - 4)
    end_time_1 = min(video_ed, q1_t - 1, ans0_t + 4)
    target_times_1 = list(range(start_time_1, end_time_1 + 1)) if start_time_1 <= end_time_1 else []
        
    # Interval 2
    start_time_2 = max(q1_t, ans1_t - 2)
    end_time_2 = min(video_ed, ans1_t + 4)
    target_times_2 = list(range(start_time_2, end_time_2 + 1)) if start_time_2 <= end_time_2 else []
    
    result_time = -1
    cancel_list = []

    # 测试 Interval 1
    for x in target_times_1:
        content_payload = [{"type": "text", "text": q0_text}]
        
        frame_start = max(video_st, x - MAX_FRAMES + 1)
        selected_frames = list(range(frame_start, x + 1))
            
        for frame_idx in selected_frames:
            frame_path = os.path.join(FRAME_BASE_DIR, sample_id, f"{frame_idx}.jpg")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(FRAME_BASE_DIR, sample_id, f"{frame_idx}.png")
                
            if os.path.exists(frame_path):
                base64_img = encode_image_to_base64(frame_path)
                mime_type = "image/png" if frame_path.endswith('.png') else "image/jpeg"
                content_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_img}"
                    }
                })
                
        content_payload.append({
            "type": "text",
            "text": "Now, considering all the instructions above, determine:\nAt the current moment, should you output <attention> or <silence>? You must output exactly: <attention> or <silence>. Do not output anything else."
        })
        
        messages = [{"role": "user", "content": content_payload}]
        
        try:
            log_id = f'glm-46v-eval-run-{sample_id}-int1-{x}'
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False,
                max_tokens=MAX_TOKENS,
                extra_headers={"X-TT-LOGID": log_id}
            )
            raw_output = response.choices[0].message.content
            parsed_result = extract_result(raw_output)
            print(f"Sample {sample_id} Interval 1 at X={x}: {parsed_result}")
            
            if parsed_result == "<attention>":
                result_time = x
                break
        except Exception as e:
            print(f"Error calling API for sample {sample_id} at Int1 X={x}: {e}")
            return -2, cancel_list

    # 测试 Interval 2
    for x in target_times_2:
        content_payload = [{"type": "text", "text": q0_text}]
        
        frame_start = max(video_st, x - MAX_FRAMES + 1)
        selected_frames = list(range(frame_start, x + 1))
            
        frames_1 = [f for f in selected_frames if f < q1_t]
        frames_2 = [f for f in selected_frames if f >= q1_t]
        
        for frame_idx in frames_1:
            frame_path = os.path.join(FRAME_BASE_DIR, sample_id, f"{frame_idx}.jpg")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(FRAME_BASE_DIR, sample_id, f"{frame_idx}.png")
            if os.path.exists(frame_path):
                base64_img = encode_image_to_base64(frame_path)
                mime_type = "image/png" if frame_path.endswith('.png') else "image/jpeg"
                content_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_img}"
                    }
                })
                
        content_payload.append({"type": "text", "text": q1_text})
        
        for frame_idx in frames_2:
            frame_path = os.path.join(FRAME_BASE_DIR, sample_id, f"{frame_idx}.jpg")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(FRAME_BASE_DIR, sample_id, f"{frame_idx}.png")
            if os.path.exists(frame_path):
                base64_img = encode_image_to_base64(frame_path)
                mime_type = "image/png" if frame_path.endswith('.png') else "image/jpeg"
                content_payload.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_img}"
                    }
                })
                
        content_payload.append({
            "type": "text",
            "text": "Now, considering all the instructions above, determine:\nAt the current moment, should you output <attention> or <silence>? You must output exactly: <attention> or <silence>. Do not output anything else."
        })
        
        messages = [{"role": "user", "content": content_payload}]
        
        try:
            log_id = f'glm-46v-eval-run-{sample_id}-int2-{x}'
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False,
                max_tokens=MAX_TOKENS,
                extra_headers={"X-TT-LOGID": log_id}
            )
            raw_output = response.choices[0].message.content
            print(f"Sample {sample_id} Interval 2 at X={x}: {raw_output}")
            cancel_list.append(raw_output)
            
        except Exception as e:
            print(f"Error calling API for sample {sample_id} at Int2 X={x}: {e}")
            return -2, cancel_list

    return result_time, cancel_list

def process_sample(client: AzureOpenAI, sample: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = str(sample['id'])
    print(f"Starting sample {sample_id}...")
    
    result_time, cancel_list = test_sample(client, sample)
    
    sample_copy = dict(sample)
    sample_copy['result_time'] = result_time
    sample_copy['cancel'] = cancel_list
    
    return sample_copy

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
    
    processed_data = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    processed_data[str(item['id'])] = item
        except Exception as e:
            print(f"Failed to load existing output json: {e}")

    unprocessed_samples = [s for s in data if str(s['id']) not in processed_data]
    
    # 预处理无效样本
    for sample in list(unprocessed_samples):
        valid = len(sample.get('question', [])) >= 2 and len(sample.get('answer', [])) >= 2
        if not valid:
            unprocessed_samples.remove(sample)
            sample_copy = dict(sample)
            sample_copy['result_time'] = -1
            sample_copy['cancel'] = []
            processed_data[str(sample['id'])] = sample_copy
            
            with write_lock:
                sorted_data = [processed_data[k] for k in sorted(processed_data.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)

    print(f"Total samples to process: {len(unprocessed_samples)}")

    if not unprocessed_samples:
        print("All samples already processed.")
        return

    client = init_client()

    def worker(sample):
        try:
            result = process_sample(client, sample)
            with write_lock:
                processed_data[str(result['id'])] = result
                sorted_data = [processed_data[k] for k in sorted(processed_data.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Thread failed on sample {sample['id']}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(worker, sample): sample for sample in unprocessed_samples}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                success = future.result()
                if success:
                    print(f"Sample {sample['id']} processed and saved.")
            except Exception as exc:
                print(f"Sample {sample['id']} generated an exception: {exc}")

    print("All done!")

if __name__ == "__main__":
    main()
