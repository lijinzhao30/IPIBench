#!/bin/bash

BASE_DIR="/mnt/bn/2d-videos/ljz/Streaming_AI2"
EVAL_DIR="$BASE_DIR/Evaluate"
BENCHMARK_DIR="$BASE_DIR/Benchmark"
RESULT_DIR="$BASE_DIR/Result"

MODELS=(
  "GLM_4_6_V"
  "GPT_4o"
  "GPT_5_4"
  "GPT_5_4_4frames"
  "Gemini_2_5"
  "Gemini_3"
  "InternVL3-8B"
  "Qwen3_5_VL_Plus"
  "Qwen3_5_VL_Plus_4frames"
  "Qwen3_VL"
  "Qwen3_VL_235B"
  "Qwen3_VL_32B"
  "Qwen3_VL_Plus"
  "Qwen3_VL_abalte"
  "Qwen3_VL_greedy"
  "llava-onevision-7B"
)

TASKS=(
  "IRP/RAP"
  "IRP/RTP"
  "IRP/RUP"
  "PM/PT"
  "PM/PU"
  "PM/RP"
  "PTM/MTM"
  "PTM/TC"
  "PTM/TM"
)

for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Evaluating model: $model"
    echo "========================================"
    
    for task in "${TASKS[@]}"; do
        bench_file="$BENCHMARK_DIR/${task}.json"
        
        # Result files might be stored under their new structure (IRP, PM, PTM) 
        # or we might need to find them if they're still in level1-level4 format.
        # But if the pipeline outputs to the same structure:
        res_file="$RESULT_DIR/$model/${task}.json"
        
        # If the result file doesn't exist under the new structure, we can try to warn
        # Assuming the new pipeline will output to Result/$model/IRP/RAP.json, etc.
        if [ ! -f "$res_file" ]; then
            echo "Skipping $task for $model (Result file not found: $res_file)"
            continue
        fi
        
        if [ ! -f "$bench_file" ]; then
            echo "Skipping $task for $model (Benchmark file not found: $bench_file)"
            continue
        fi
        
        echo ">>> Running eval for $task"
        python "$EVAL_DIR/${task}.py" --result_path "$res_file" --benchmark_path "$bench_file"
        echo ""
    done
done
