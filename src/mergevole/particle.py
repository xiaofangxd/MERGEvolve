import json
import os
from typing import Dict, List

import torch
from loguru import logger
from src.base.base_individual import BaseIndividual

class Particle(BaseIndividual):
    def __init__(
        self, id: str, x: Dict, parent: List[str], 
        weight_path: str, model_name_or_path: str, lora_config_path: str, 
        seed: int = 42, reset_iter: int = 3
    ):
        super().__init__(
            id=id, x=x, weight_path=weight_path,
            model_name_or_path=model_name_or_path,
            lora_config_path=lora_config_path, seed=seed
        )
        # record parent
        self.parent = parent
        self.p = x
        self.v = None
        
        self.best_weight_path = weight_path
        self.reset_iter = reset_iter
        self.patience_counter = 0
        
    def reset_velocity(self, ):
        self.v = dict()
        for key in self.x.keys():
            self.v[key] = torch.zeros_like(self.x[key])
    
    def save_particle(self, save_path: str):
        self.save_individual(save_path=save_path)
    
    def init_velocity(self, lora: dict):
        assert set(lora.keys()) == set(self.x.keys()), "The architecture of the two LORAs must be the same."
        self.v = dict()
        
        for key in lora.keys():
            self.v[key] = lora[key] - self.x[key]
        
    def update_velocity(self, global_max_weight: Dict, global_min_weight: Dict, r: List[float] | None, phi: List[float], C: float):
        # v = r*phi*v + r*phi*(p-x) + r*phi*(max-x) - r*phi*(min-x)
        for key in self.x.keys():
            try:
                self.v[key] = r[0] * phi[0] * self.v[key] \
                    + r[1] * phi[1] * (self.p[key] - self.x[key]) \
                        + r[2] * phi[2] * (global_max_weight[key] - self.x[key]) \
                            - r[3] * phi[3]* (global_min_weight[key] - self.x[key])
                self.v[key] = 1/C * self.v[key]
            except Exception as e:
                logger.error(f"Error processing key {key}")
                logger.error(f"Shapes: v={self.v[key].shape}, p={self.p[key].shape}, x={self.x[key].shape}")
                logger.error(f"max={global_max_weight[key].shape}, min={global_min_weight[key].shape}")
                raise e
            
    def update_weight(self, _lambda: float):
        for key in self.x.keys():
            self.x[key] = self.x[key] + _lambda * self.v[key]
    
    def update_position(self, f: float, path:str):
        if self.current_f > self.best_f:
            self.best_f = self.current_f
            # update position
            self.p = self.x
            self.best_weight_path = path
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.reset_iter:
                logger.info(f"Resetting particle {self.id} ...")
                self.patience_counter = 0
                self.x = self.p
                self.save_particle(save_path=path)
                self.reset_velocity()
                self.current_f = self.best_f
        
        os.makedirs(self.weight_path, exist_ok=True)
        # save current state
        with open(os.path.join(self.weight_path, "state.json"), "w") as f:
            f.write(
                json.dumps({
                    "id": self.id,
                    "parent": self.parent,
                    "patience_counter": self.patience_counter,
                    "best_fitness_score": self.best_fitness_score,
                    "fitness_score": self.fitness_score
                }, indent=4)
            )
