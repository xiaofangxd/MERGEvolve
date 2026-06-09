import os
import random
import socket
import subprocess
import sys
import time
from contextlib import closing
from typing import List
import argparse
import requests
from loguru import logger

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, required=True)
    parser.add_argument("--gpu_ids", type=int, nargs="+", required=True)
    
    return parser.parse_args()

def find_free_port(start_port: int=9000, end_port: int=65535) -> int:
    while True:
        port = random.randint(start_port, end_port)
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.bind(("", port))
                print("Found free port:", port)
                return port
        except socket.error:
            continue

def check_port_in_use(port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind(("", port))
            return False
        except socket.error:
            return True

def online_load_lora(base_url: str, lora_name: str, lora_path: str):
    counter = 1
    while True:
        try:
            response = requests.post(
                f"{base_url}"+"load_lora_adapter",
                json = {
                    "lora_name": lora_name,
                    "lora_path": lora_path
                }
            )
            time.sleep(3)
            assert response.status_code == 200, f"Failed to load LORA: {response.text}"
            break
        except Exception as e:
            logger.warning(f"Load LORA Error: {e}, retry at {min(counter, 10)} seconds ...")
            time.sleep(min(counter, 10))
            counter += 1
            continue

def online_unload_lora(base_url: str, lora_name: str):
    while True:
        try:
            response = requests.post(
                f"{base_url}"+"unload_lora_adapter",
                json = {
                    "lora_name": lora_name
                }
            )
            assert response.status_code == 200, f"Failed to unload LORA: {response.text}"
            break
        except Exception as e:
            logger.warning(f"Unload LORA Error: {e}, retry ...")
            time.sleep(1)
            continue

def run_vllm_server(model_name_or_path: str, seed: int | None, gpu_id: int):
    try:
        port = find_free_port()
    except RuntimeError as e:
        raise RuntimeError(f"{e}, Failed to start serve on GPU {gpu_id}.")
    
    time.sleep(random.randint(1,5))
    env = {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "CUDA_VISIBLE_DEVICES": str(gpu_id)
    }
    log_file = f"ms_workspace/log/vllm_server_gpu{gpu_id}_port{port}.log"
    cmd = f"""
    nohup {sys.executable} -m vllm.entrypoints.openai.api_server \
    --model {model_name_or_path} \
    --trust-remote-code \
    --enable-lora \
    --seed {seed} \
    --max-loras 20 \
    --max-cpu-loras 20 \
    --gpu-memory-utilization 0.90 \
    --port {port} > {log_file} 2>&1 &
    """
    process = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        executable="/bin/bash"
    )
    return dict(process=process, port=port, gpu_id=gpu_id)
    
def main(model_name_or_path: str, max_workers: int, gpu_ids: List[int], start_port: int = 9000):
    processes = []
    assert len(gpu_ids) >= max_workers, "Not enough GPUs for workers."
    for i, gpu_id in enumerate(gpu_ids):
        process = run_vllm_server(
            model_name_or_path=model_name_or_path,
            seed=42,
            gpu_id=gpu_id,
        )
        processes.append(process)
    
    return processes
            

if __name__ == "__main__":
    config = get_args()
    processes = main(
        model_name_or_path=config.model_name_or_path,
        max_workers=config.max_workers,
        gpu_ids=config.gpu_ids
    )
    print("Processes started.")
    
    ports = [p["port"] for p in processes]
    
    print("Ports:", ports)