#!/bin/bash

set -e

MODEL="GPT_5_4"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$BASE_DIR")")"


API_KEY="X4ULI3VkxrtRWIVPvA1280rWkwdueKm1_GPT_AK"
API_ENDPOINT="https://aidp.bytedance.net/api/modelhub/online/responses"
API_VERSION="2024-03-01-preview"
MODEL_NAME="gpt-5.4-2026-03-05"
MODEL_PATH=""

echo "Starting evaluation for all levels in $MODEL..."

echo "Running IRP/RAP.py..."
python3 "${BASE_DIR}/IRP/RAP.py" \
    --input "${PROJECT_ROOT}/Benchmark/IRP/RAP.json" \
    --output "${PROJECT_ROOT}/Result/${MODEL}/IRP/RAP.json" \
    --frame_dir "${PROJECT_ROOT}/Image/1fps/IRP/RAP" \
    --api_key "${API_KEY}" \
    --api_endpoint "${API_ENDPOINT}" \
    --api_version "${API_VERSION}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}"
echo "Finished IRP/RAP.py"
echo "--------------------------------------------------------"

echo "Running IRP/RTP.py..."
python3 "${BASE_DIR}/IRP/RTP.py" \
    --input "${PROJECT_ROOT}/Benchmark/IRP/RTP.json" \
    --output "${PROJECT_ROOT}/Result/${MODEL}/IRP/RTP.json" \
    --frame_dir "${PROJECT_ROOT}/Image/1fps/IRP/RTP" \
    --api_key "${API_KEY}" \
    --api_endpoint "${API_ENDPOINT}" \
    --api_version "${API_VERSION}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}"
echo "Finished IRP/RTP.py"
echo "--------------------------------------------------------"

echo "Running IRP/RUP.py..."
python3 "${BASE_DIR}/IRP/RUP.py" \
    --input "${PROJECT_ROOT}/Benchmark/IRP/RUP.json" \
    --output "${PROJECT_ROOT}/Result/${MODEL}/IRP/RUP.json" \
    --frame_dir "${PROJECT_ROOT}/Image/1fps/IRP/RUP" \
    --api_key "${API_KEY}" \
    --api_endpoint "${API_ENDPOINT}" \
    --api_version "${API_VERSION}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}"
echo "Finished IRP/RUP.py"
echo "--------------------------------------------------------"

echo "Running PM/PT.py..."
python3 "${BASE_DIR}/PM/PT.py" \
    --input "${PROJECT_ROOT}/Benchmark/PM/PT.json" \
    --output "${PROJECT_ROOT}/Result/${MODEL}/PM/PT.json" \
    --frame_dir "${PROJECT_ROOT}/Image/1fps/PM/PT" \
    --api_key "${API_KEY}" \
    --api_endpoint "${API_ENDPOINT}" \
    --api_version "${API_VERSION}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}"
echo "Finished PM/PT.py"
echo "--------------------------------------------------------"

echo "Running PM/PU.py..."
python3 "${BASE_DIR}/PM/PU.py" \
    --input "${PROJECT_ROOT}/Benchmark/PM/PU.json" \
    --output "${PROJECT_ROOT}/Result/${MODEL}/PM/PU.json" \
    --frame_dir "${PROJECT_ROOT}/Image/1fps/PM/PU" \
    --api_key "${API_KEY}" \
    --api_endpoint "${API_ENDPOINT}" \
    --api_version "${API_VERSION}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}"
echo "Finished PM/PU.py"
echo "--------------------------------------------------------"

echo "Running PM/RP.py..."
python3 "${BASE_DIR}/PM/RP.py" \
    --input "${PROJECT_ROOT}/Benchmark/PM/RP.json" \
    --output "${PROJECT_ROOT}/Result/${MODEL}/PM/RP.json" \
    --frame_dir "${PROJECT_ROOT}/Image/1fps/PM/RP" \
    --api_key "${API_KEY}" \
    --api_endpoint "${API_ENDPOINT}" \
    --api_version "${API_VERSION}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}"
echo "Finished PM/RP.py"
echo "--------------------------------------------------------"

echo "Running PTM/MTM.py..."
python3 "${BASE_DIR}/PTM/MTM.py" \
    --input "${PROJECT_ROOT}/Benchmark/PTM/MTM.json" \
    --output "${PROJECT_ROOT}/Result/${MODEL}/PTM/MTM.json" \
    --frame_dir "${PROJECT_ROOT}/Image/1fps/PTM/MTM" \
    --api_key "${API_KEY}" \
    --api_endpoint "${API_ENDPOINT}" \
    --api_version "${API_VERSION}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}"
echo "Finished PTM/MTM.py"
echo "--------------------------------------------------------"

echo "Running PTM/TC.py..."
python3 "${BASE_DIR}/PTM/TC.py" \
    --input "${PROJECT_ROOT}/Benchmark/PTM/TC.json" \
    --output "${PROJECT_ROOT}/Result/${MODEL}/PTM/TC.json" \
    --frame_dir "${PROJECT_ROOT}/Image/1fps/PTM/TC" \
    --api_key "${API_KEY}" \
    --api_endpoint "${API_ENDPOINT}" \
    --api_version "${API_VERSION}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}"
echo "Finished PTM/TC.py"
echo "--------------------------------------------------------"

echo "Running PTM/TM.py..."
python3 "${BASE_DIR}/PTM/TM.py" \
    --input "${PROJECT_ROOT}/Benchmark/PTM/TM.json" \
    --output "${PROJECT_ROOT}/Result/${MODEL}/PTM/TM.json" \
    --frame_dir "${PROJECT_ROOT}/Image/1fps/PTM/TM" \
    --api_key "${API_KEY}" \
    --api_endpoint "${API_ENDPOINT}" \
    --api_version "${API_VERSION}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}"
echo "Finished PTM/TM.py"
echo "--------------------------------------------------------"

echo "All evaluations completed for $MODEL!"
