import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
from loguru import logger
from peft import LoraConfig
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from pathlib import Path


def load_lora_weight(lora_path: str):
    return load_file(os.path.join(lora_path, "adapter_model.safetensors"))

def save_lora_weight(lora_weight, lora_path: str, tokenizer: AutoTokenizer | str, config: LoraConfig | str):
    assert tokenizer is not None, "Tokenizer must be provided (for vllm evaluate)."
    assert config is not None, "LoraConfig must be provided."
    
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    if isinstance(config, str):
        config = LoraConfig.from_pretrained(config)
        
    tokenizer.save_pretrained(lora_path)
    config.save_pretrained(lora_path)

    save_file(lora_weight, filename=os.path.join(lora_path, "adapter_model.safetensors"))
    # wait for save completed.
    time.sleep(1)


def plot_optimization_curves(
    state: Dict, 
    save_dir: str = "figures",
    config_info: Dict = None,
    title: str = None
):
    # create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # extract data
    steps = sorted([int(k.split('_')[1]) for k in state.keys() if k.startswith('step_')])
    global_max = [state[f'step_{step}']['global_max_f'] for step in steps]
    average = [state[f'step_{step}']['ave_f'] for step in steps]

    # create figure
    fig = plt.figure(figsize=(12, 8))
    
    # main plot occupies 80% of the height
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    ax1.plot(steps, global_max, 'r-', label='Global Maximum', marker='o')
    ax1.plot(steps, average, 'g-', label='Population Average', marker='^')
    
    ax1.set_xlabel('Optimization Step')
    ax1.set_ylabel('Fitness Value')
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('PSO Optimization Progress')
    ax1.grid(True)
    ax1.legend()
    
    if not isinstance(config_info, dict):
        config_info = asdict(config_info)
        
    if config_info:
        info_text = []
        if 'N' in config_info:
            info_text.append(f"Population Size: {config_info['N']}")
        if 'max_iter' in config_info:
            info_text.append(f"Max Iterations: {config_info['max_iter']}")
        if 'model_name_or_path' in config_info:
            info_text.append(f"Model: {os.path.basename(config_info['model_name_or_path'])}")
        if 'seed' in config_info:
            info_text.append(f"Seed: {config_info['seed']}")
            
        # add other important parameters
        important_params = ['lambda_step', 'phi_lambda', 'phi_inertia', 'phi_cognitive', 'phi_social', 'phi_repel']
        for param in important_params:
            if param in config_info:
                info_text.append(f"{param}: {config_info[param]}")
                
        # add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # info_text.append(f"Generated: {timestamp}")
        
        # create text box at the bottom
        ax2 = plt.subplot2grid((5, 1), (4, 0))
        ax2.axis('off')
        ax2.text(0.5, 0.5, ' | '.join(info_text),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transAxes,
                wrap=True)

    # adjust layout
    plt.tight_layout()

    # generate file name (include timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'pso_optimization_curves_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    
    # save image
    plt.savefig(filepath, dpi=400, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Optimization curves saved to: {filepath}")
