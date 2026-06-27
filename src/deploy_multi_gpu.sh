#!/bin/bash
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# your base model path
MODEL="/data/gyc/EVOLLM/gemma-2-2b-it"

# max number of LoRA weights to load
MAX_LORAS=10

# log directory name
ROOT="multi_gpu_lora_logs"

# max LoRA rank
MAX_LORA_RANK=16

# define the mapping of GPU and port
# e.g., GPU0 runs port 9112, 9113, GPU1 runs port 9114, 9115
declare -A GPU_PORTS
GPU_PORTS[0]="9112"
GPU_PORTS[1]="9113"
GPU_PORTS[2]="9114"
GPU_PORTS[3]="9115"
GPU_PORTS[4]="9116"
GPU_PORTS[5]="9117"

mkdir -p vllm_logs/$ROOT

COMMON_ARGS="--model $MODEL \
    --trust-remote-code \
    --enable-lora \
    --seed 42 \
    --max-lora-rank $MAX_LORA_RANK \
    --max-loras $MAX_LORAS \
    --max-cpu-loras $MAX_LORAS \
    --max-model-len 8092 \
    --gpu-memory-utilization 0.9 "

for gpu in "${!GPU_PORTS[@]}"; do
    for port in ${GPU_PORTS[$gpu]}; do
        echo "Starting API server on GPU $gpu, port $port..."
        CUDA_VISIBLE_DEVICES=$gpu nohup python -m vllm.entrypoints.openai.api_server \
            $COMMON_ARGS \
            --port $port > vllm_logs/$ROOT/gpu${gpu}_port${port}.log 2>&1 &
    done
done
