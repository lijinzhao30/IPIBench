import argparse
import json
import os
import time
import base64
import threading
import sys
import uuid
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
MODEL_NAME = "gemini-3-pro-preview-new"

MAX_TOKENS = 4096
NUM_THREADS = 16
MAX_FRAMES = 16

# 用于保护文件写入的锁
write_lock = threading.Lock()

def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def init_client(log_id: str) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
        default_headers={"X-TT-LOGID": log_id}
    )

def extract_attention(text: str) -> bool:
    """从模型输出中提取 <attention>"""
    if text is None:
        return False
    text_lower = text.lower()
    return "<attention>" in text_lower

def append_frames_to_payload(payload: list, sample_id: str, frame_list: list):
    """将指定列表中的帧加载为 base64 并 append 到 payload 中"""
    for frame_idx in frame_list:
        frame_path = os.path.join(FRAME_BASE_DIR, sample_id, f"{frame_idx}.png")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(FRAME_BASE_DIR, sample_id, f"{frame_idx}.jpg")
            
        if os.path.exists(frame_path):
            base64_img = encode_image_to_base64(frame_path)
            mime_type = "image/png" if frame_path.endswith('.png') else "image/jpeg"
            payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_img}"
                }
            })

def run_model_inference(payload: list, max_retries: int = 3):
    """封装 API 调用和重试逻辑"""
    for attempt in range(max_retries):
        log_id = str(uuid.uuid4())
        client = init_client(log_id)
        try:
            if len(payload) > 0 and isinstance(payload[0], dict) and "role" in payload[0]:
                messages = payload
            else:
                messages = [
                    {
                        "role": "user",
                        "content": payload
                    }
                ]
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"API Error after {max_retries} attempts: {e}")
                return None
            time.sleep(2)
    return None

def test_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """测试单个样本的逻辑"""
    sample_id = str(sample['id'])
    questions = sample.get('question', [])
    answers = sample.get('answer', [])
    video_st = int(sample.get('video_st'))
    video_ed = int(sample.get('video_ed'))
    
    # 初始化 result 为 [-1, ""]
    # result[0]: 第一次检测到 <attention> 的时刻（或者没有找到则为-1，报错为-2）
    # result[1]: 第二次回答的输出
    result = [-1, ""]
    
    try:
        if len(questions) < 2 or len(answers) < 1:
            result[0] = -2
            sample['result'] = result
            return sample
            
        text1 = questions[0]['text']
        text2 = questions[1]['text']
        t2 = int(questions[1]['t'])
        
        trigger_time_1 = int(answers[0]['trigger_time'])
        
        # ==========================================
        # 1. 第一次问答测试: 寻找 <attention>
        # ==========================================
        start_test = max(video_st, trigger_time_1 - 4)
        end_test = min(t2 - 1, trigger_time_1 + 4)
        
        first_attention_time = -1
        last_test_time = -1
        
        for x in range(start_test, end_test + 1):
            last_test_time = x
            frame_start_x = max(video_st, x - MAX_FRAMES + 1)
            selected_frames_x = list(range(frame_start_x, x + 1))
            
            payload = [{"type": "text", "text": text1}]
            append_frames_to_payload(payload, sample_id, selected_frames_x)
            
            resp = run_model_inference(payload)
            if resp is None:
                result[0] = -2
                sample['result'] = result
                return sample
                
            if extract_attention(resp):
                first_attention_time = x
                result[0] = first_attention_time
                break
                
        # 如果一直没有 <attention>，则 result[0] 记录为 -1
        if first_attention_time == -1:
            result[0] = -1

        # ==========================================
        # 2. 第二次问答测试
        # ==========================================
        if first_attention_time != -1:
            attention_val = "<attention>"
            split_time = first_attention_time
        else:
            attention_val = "<silence>"
            split_time = last_test_time

        # 第二次请求受限于 MAX_FRAMES 滑窗
        # 取 [video_st, t2 - 1] 这个范围里的最后最多 MAX_FRAMES 帧
        # frame1 是这部分帧里面小于等于 split_time 的
        # frame2 是这部分帧里面大于 split_time 的
        end_frame2_time = t2 - 1
        frame_start_2 = max(video_st, end_frame2_time - MAX_FRAMES + 1)
        selected_frames_2 = list(range(frame_start_2, end_frame2_time + 1))
        
        # 根据 split_time 进行切分
        frames_part1 = [f for f in selected_frames_2 if f <= split_time]
        frames_part2 = [f for f in selected_frames_2 if f > split_time]
        
        content1 = [{"type": "text", "text": text1}]
        append_frames_to_payload(content1, sample_id, frames_part1)
        
        messages2 = [
            {"role": "user", "content": content1},
            {"role": "assistant", "content": [{"type": "text", "text": attention_val}]}
        ]
        
        content2 = []
        append_frames_to_payload(content2, sample_id, frames_part2)
        content2.append({"type": "text", "text": text2})
        
        messages2.append({"role": "user", "content": content2})
        
        ans2 = run_model_inference(messages2)
        if ans2 is None:
            result[0] = -2
            sample['result'] = result
            return sample
            
        result[1] = ans2

    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        result[0] = -2
        
    sample['result'] = result
    return sample

def process_sample_with_resume(sample: Dict[str, Any]) -> Dict[str, Any]:
    """带有断点续传检查的执行逻辑"""
    
    # 断点续传检查: 如果 result 已经存在且 result[0] 不是 -2(即之前的执行未抛错)
    # 则认定为该样本已经评测过了，跳过
    if 'result' in sample and isinstance(sample['result'], list) and len(sample['result']) >= 2:
        if sample['result'][0] != -2:
            return sample

    # 重新跑模型
    updated_sample = test_sample(sample)
    
    # 写回文件
    with write_lock:
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
            
            for idx, item in enumerate(current_data):
                if str(item['id']) == str(updated_sample['id']):
                    current_data[idx] = updated_sample
                    break
                    
            tmp_file = OUTPUT_JSON + ".tmp"
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(current_data, f, ensure_ascii=False, indent=4)
            os.replace(tmp_file, OUTPUT_JSON)
        except Exception as e:
            print(f"Failed to write result to disk: {e}")
            
    return updated_sample

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
        print(f"Input file {INPUT_JSON} not found!")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    # 第一次运行如果输出文件不存在，则基于 INPUT 复制一份作为起点
    if not os.path.exists(OUTPUT_JSON):
        with open(INPUT_JSON, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=4)
    else:
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)

    print(f"Starting evaluation on {len(data)} samples using {NUM_THREADS} threads...")
    
    processed_count = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process_sample_with_resume, sample): sample for sample in data}
        
        for future in as_completed(futures):
            processed_count += 1
            sample = futures[future]
            try:
                res = future.result()
                status = res.get('result', [])
                if len(status) > 1:
                    print(f"[{processed_count}/{len(data)}] Processed ID: {sample['id']} -> Attention Time: {status[0]}")
                else:
                    print(f"[{processed_count}/{len(data)}] Processed ID: {sample['id']} -> Result: Empty")
            except Exception as e:
                print(f"[{processed_count}/{len(data)}] Error on ID {sample['id']}: {e}")

    print("Evaluation finished!")

if __name__ == "__main__":
    main()
