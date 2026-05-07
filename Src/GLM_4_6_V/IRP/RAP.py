import argparse
import json
import os
import glob
import time
import base64
import threading
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import List, Dict, Any
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
    return text  # 如果都没找到，返回原输出

def test_sample(client: AzureOpenAI, sample: Dict[str, Any]) -> int:
    """测试单个样本，在 +-4 个时间点内进行评测，遇到 yes 即返回该时间点 x，否则返回 -1"""
    sample_id = str(sample['id'])
    video_st = int(sample['video_st'])
    video_ed = int(sample['video_ed'])
    
    # 获取真正的 trigger_time
    trigger_time = int(sample['answer'][0]['trigger_time']) 
    
    # 直接使用刚刚改写好的 question[0]['text'] 作为 prompt，不再需要额外加字
    full_user_prompt = sample['question'][0]['text']
    
    # 待测试的时间点集合: trigger_time - 4 到 trigger_time + 4
    # 如果超出 video_st 或 video_ed，则有多少测多少
    start_time = max(video_st, trigger_time - 4)
    end_time = min(video_ed, trigger_time + 4)
    target_times = list(range(start_time, end_time + 1))
    
    for x in target_times:
        current_end = x

        content_payload = [
            {
                "type": "text",
                "text": full_user_prompt
            }
        ]

        # 收集从 video_st 到 current_end 的图片
        all_frames = list(range(video_st, current_end + 1))
        
        # 如果帧数超过上限，只保留最近的 MAX_FRAMES 帧
        if len(all_frames) > MAX_FRAMES:
            selected_frames = all_frames[-MAX_FRAMES:]
        else:
            selected_frames = all_frames
            
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
            else:
                pass

        if len(content_payload) == 1:
            print(f"Warning: No frames collected for sample {sample_id} from {video_st} to {current_end}")
            continue

        # 组装请求
        messages = [
            {
                "role": "user",
                "content": content_payload
            }
        ]

        try:
            # 动态生成 LOGID
            log_id = f'glm-46v-eval-run-{sample_id}-{x}'
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False,
                max_tokens=MAX_TOKENS,
                extra_headers={"X-TT-LOGID": log_id}
            )
            raw_output = response.choices[0].message.content
            parsed_result = extract_result(raw_output)
            print(f"Sample {sample_id} at X={x}: {parsed_result}")
            
            # 重点：一旦回答 yes，直接记录时间点并结束测试
            if parsed_result == "<attention>":
                return x
                
        except Exception as e:
            print(f"Error calling API for sample {sample_id} at X={x}: {e}")
            return -2
            
    # 如果全都没有出现 yes，则记录为 -1
    return -1

def process_sample(client: AzureOpenAI, sample: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个样本并返回结果对象"""
    sample_id = str(sample['id'])
    
    print(f"Starting sample {sample_id}...")
    
    result_time = test_sample(client, sample)
    
    sample_copy = dict(sample)
    sample_copy['result_time'] = result_time
            
    print(f"Completed sample {sample_id}, Result Time: {result_time}")
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

    client = init_client()

    print(f"Starting multi-threaded processing. Threads: {NUM_THREADS}, Samples to process: {len(data)}")
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for sample in data:
            sample_id = str(sample['id'])
            if sample_id in processed_data:
                print(f"Skipping sample {sample_id}, already processed.")
                continue
            futures.append(executor.submit(process_sample, client, sample))
        
        for future in as_completed(futures):
            try:
                res = future.result()
                sample_id = str(res['id'])
                with write_lock:
                    processed_data[sample_id] = res
                    # 每次完成一个样本就写入文件
                    sorted_data = [processed_data[k] for k in sorted(processed_data.keys(), key=lambda x: int(x))]
                    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                        json.dump(sorted_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"A thread raised an exception: {e}")

    print("All done!")

if __name__ == "__main__":
    main()
