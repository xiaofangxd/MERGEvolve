from typing import Dict, List
from transformers import AutoTokenizer
from loguru import logger
from src.utils import save_lora_weight
from abc import ABC, abstractmethod
from openai import OpenAI
from src.evaluate.factory import EvaluatorFactory
from src.evaluate.eval import Method
import random
import os, hashlib

class BaseIndividual(ABC):
    def __init__(
        self, id: str, x: Dict, weight_path: str, model_name_or_path: str, lora_config_path: str, seed: int = 42, 
        tokenizer=None,
    ):
        self.id = id
        self.x = x
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_name_or_path)
        
        self.weight_path = weight_path
        self.config_path = lora_config_path
        self.seed = seed
        
        self.best_fitness_score = -100
        self.fitness_score = -100
        self.task_scores = {}
        self.best_task_scores = {}

        self.evaluated = {}
        self.predictions = {}
    
    def save_individual(self, save_path):
        save_lora_weight(
            lora_weight=self.x,
            lora_path=save_path,
            tokenizer=self.tokenizer,
            config=self.config_path
        )
        self.weight_path = save_path
        safetensors = os.path.join(save_path, "adapter_model.safetensors")
        try:
            if os.path.isfile(safetensors):
                size = os.path.getsize(safetensors)
                with open(safetensors, "rb") as f:
                    head = f.read(1024 * 1024) 
                md5_head = hashlib.md5(head).hexdigest()
                logger.info(f"[DEBUG] Saved LoRA -> {safetensors} | size={size} bytes | md5(head,1MB)={md5_head}")
            else:
                logger.warning(f"[DEBUG] LoRA file not found right after save: {safetensors}")
        except Exception as e:
            logger.warning(f"[DEBUG] Failed to stat/hash LoRA file {safetensors}: {e}")

    def fitness(self, task: str, llm: OpenAI, lora_path: str, split: str, calculate_ppl: bool=False, return_predictions: bool=False, **kwargs) -> Dict:
        """Calculate the fitness of the individual.
        Returns:
            Dict: The fitness score, id and weight path of the individual.
        """
        logger.info(f"[DEBUG] fitness() start | id={self.id} | task={task} | split={split} | lora_path={lora_path}")

        if task not in self.evaluated:
            self.evaluated[task] = False

        if self.evaluated[task] and not return_predictions:
            logger.info(f"[DEBUG] fitness() using CACHED result | id={self.id} | task={task} | lora_path={lora_path}")
            return {
                "id": self.id,
                "path": lora_path,
                "score": self.task_scores[task]
            }
        else:
            logger.info(f"[DEBUG] fitness() will RUN evaluation | id={self.id} | task={task} | lora_path={lora_path}")
        
        evaluator = EvaluatorFactory().get_evaluator(task=task)
        result = evaluator.evaluate(
            method=Method.API, 
            llm=llm, 
            lora_name=f"individual-{self.id}-{random.randint(0, 10000)}", 
            lora_path=lora_path, 
            split=split, 
            calculate_ppl=calculate_ppl,
            return_predictions=return_predictions, 
            **kwargs
        )
        self.task_scores[task] = result['score']
        self.evaluated[task] = True
        
        if return_predictions:
            try:
                task_predictions = dict(sorted(result['predictions'].items(), key=lambda x: int(x[0])))
                self.predictions[task] = task_predictions
                return {
                    "id": self.id,
                    "path": lora_path,
                    "score": result['score'],
                    "predictions": task_predictions
                }
            except:
                logger.warning(f"No predictions found for individual {self.id}")
                self.predictions = {}
                return {
                    "id": self.id,
                    "path": lora_path,
                    "score": result['score'],
                    "predictions": {}
                }
        if calculate_ppl:
            logger.info(f"Perplexity for individual {self.id} is {result['perplexity']}")
            return {
                "id": self.id,
                "path": lora_path,
                "score": result['score'],
                "perplexity": result['perplexity']
            }
        return {
            "id": self.id,
            "path": lora_path,
            "score": result['score']
        }
    
    def update_fitness(self, tasks: List[str], task_weights: List[float]) -> None:
        """Update the fitness of the individual.
        Args:
            task_weights: Dictionary mapping task names to their weights.
        """
        if not self.task_scores:
            logger.warning(f"No task scores found for individual {self.id}")
            return
        
        weighted_score = 0
        for task, weight in zip(tasks, task_weights):
            if task not in self.task_scores:
                logger.warning(f"Task {task} not found in task scores for individual {self.id}")
                continue
            weighted_score += weight * self.task_scores[task]
        
        self.fitness_score = weighted_score
        if weighted_score > self.best_fitness_score:
            self.best_fitness_score = weighted_score
            self.best_task_scores = self.task_scores.copy()