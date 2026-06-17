from abc import ABC, abstractmethod
from typing import Dict, List, Any
import json
import os
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.evaluate.eval import CombineMethod
from src.utils import load_lora_weight
from openai import OpenAI
import uuid
import matplotlib.pyplot as plt


class BaseMethod(ABC):
    """Base class for all methods."""
    def __init__(self, config):
        """Initialize base method with common attributes."""
        self.config = config
        self.config.validate()
        
        # Common configuration
        self.tasks = config.tasks
        self.test_tasks = config.test_tasks
        self.task_weights = config.task_weights
        self.seed = config.seed
        self.combine_method = config.combine_method
        self.model_name_or_path = config.model_name_or_path
        self.llm_base_url = config.llm_base_url
        self.pools = config.pools
        self.max_workers = len(self.llm_base_url)
        self.plot_enabled = config.plot_enabled
        
        # Global state tracking
        self.global_max_fitness_score = -100
        self.global_min_fitness_score = 100
        self.global_max_fitness_path = ""
        self.global_min_fitness_path = ""
        self.global_max_task_scores = {}
        self.global_min_task_scores = {}
        self.global_max_fitness_weight = dict()
        self.global_min_fitness_weight = dict()
        
        # Early stopping state
        self.patience_flag = True
        self.global_patience_counter = 0
        self.early_stop = config.early_stop
        self.early_stop_iter = config.early_stop_iter
        
        # Initialize workspace and state
        self.state = {}
        self.init_workspace()
        self.init_models()
        
        # Save initial config
        self.config.save(self.workspace)
    
    def init_workspace(self) -> None:
        """Initialize the workspace with a standardized directory structure."""
        model = self.model_name_or_path.split("/")[-1]
        self.id = uuid.uuid4().hex
        task_name = "_".join(self.tasks)
        method_name = self.__class__.__name__.lower()
        
        self.workspace = os.path.join(
            f"{method_name}_workspace",
            task_name,
            f"N{self.config.N}_{self.combine_method.value}",
            model,
            f"{method_name.upper()}-{self.id}"
        )
        logger.info(f"Workspace: {self.workspace}")
        os.makedirs(self.workspace, exist_ok=True)
    
    def init_models(self) -> None:
        """Initialize OpenAI API clients for each base URL."""
        self.llms = [
            OpenAI(
                base_url=base_url,
                api_key=f"{self.__class__.__name__.lower()}_api_key"
            ) for base_url in self.llm_base_url
        ]
    
    def update_global(self, id: str, fitness_score: float, path: str, task_scores: Dict[str, float]) -> None:
        """Update global state with new fitness information."""
        logger.info(f"Individual {id} fitness: {fitness_score:.4f}")
        if fitness_score > self.global_max_fitness_score:
            self.global_max_fitness_score = fitness_score
            self.global_max_fitness_path = path
            self.global_max_task_scores = task_scores.copy()
            self.global_max_fitness_weight = load_lora_weight(path)
            logger.info(f"Global max updated: {self.global_max_fitness_score:.4f}")
            if task_scores:
                logger.info("Best individual task scores:")
                for task, score in task_scores.items():
                    logger.info(f"  - Task {task}: {score:.4f}")
            self.patience_flag = False
            
        if fitness_score < self.global_min_fitness_score:
            self.global_min_fitness_score = fitness_score
            self.global_min_fitness_path = path
            self.global_min_task_scores = task_scores.copy()
            self.global_min_fitness_weight = load_lora_weight(path)
            logger.info(f"Global min updated: {self.global_min_fitness_score:.4f}")
    
    def report_state(self, step: int) -> None:
        state = self.state[f"step_{step}"]
        global_max_fitness_score = state["global_max_fitness_score"]
        global_min_fitness_score = state["global_min_fitness_score"]
        average_fitness_score = state["average_fitness_score"]
        logger.info(
            f"Step: {step}, Global max: {global_max_fitness_score:.4f}, Global min: {global_min_fitness_score:.4f}, Average fitness score: {average_fitness_score:.4f}"
        )
        
    def save_final_state(self, individuals: List, time: float) -> None:
        """Save final state and generate plots."""
        # Calculate weighted test scores
        weighted_scores = {
            individual.id: sum(
                self.state['test'][task][individual.id] * self.task_weights[idx]
                for idx, task in enumerate(self.test_tasks)
            )
            for individual in individuals
        }
        
        test_id = max(weighted_scores, key=weighted_scores.get)
        test_score = weighted_scores[test_id]

        self.state['final'] = {
            "test_id": test_id,
            "test_score": test_score,
            "total_time": time,
        }
        
        # Save state
        self.save_optim_state(self.state)
        logger.info(
            f"Best individual id: {test_id}, "
            f"Test performance: {test_score:.4f}"
        )
        
    def evaluate_single_task(self, individuals: List, task: str, split: str = "valid", return_predictions: bool = False) -> Dict[str, Any]:
        """Evaluate individuals on a single task."""
        task_scores = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for idx, individual in enumerate(individuals):
                futures.append(
                    executor.submit(
                        individual.fitness,
                        task=task,
                        llm=self.llms[idx % len(self.llms)],
                        lora_path=individual.weight_path,
                        split=split,
                        return_predictions=return_predictions
                    )
                )
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    entry = {
                        "score": result["score"],
                        "path": result["path"]
                    }
                    if "predictions" in result:
                        entry["predictions"] = result["predictions"]
                    
                    task_scores[result["id"]] = entry
                    
                except Exception as e:
                    logger.error(f"Error processing future result: {str(e)}")
                
        return task_scores
    
    def compute_weighted_score(self, task_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
        """Compute the weighted score for the individuals in all tasks.
        Only includes individuals that have been successfully evaluated on ALL tasks.
        Skips individuals with missing results in any task (e.g., due to evaluation failures).
        """
        weighted_scores = dict()
        assert hasattr(self, "task_weights"), logger.error("Task weights are not set.")
        assert hasattr(self, "tasks"), logger.error("Tasks are not set.")

        # Intersect individual ids shared across all tasks to avoid KeyError (some tasks may fail to evaluate)
        valid_individual_ids = set(task_scores[self.tasks[0]].keys())
        for task in self.tasks[1:]:
            valid_individual_ids &= set(task_scores[task].keys())

        skipped = set(task_scores[self.tasks[0]].keys()) - valid_individual_ids
        if skipped:
            logger.warning(f"Skipping {len(skipped)} individual(s) without scores in all tasks: {skipped}")

        for individual_id in valid_individual_ids:
            scores = {}
            total_score = 0
            for task, weight in zip(self.tasks, self.task_weights):
                task_score = task_scores[task][individual_id]['score']
                scores[task] = task_score
                total_score += weight * task_score

            weighted_scores[individual_id] = {
                "weighted_score": total_score,
                "task_scores": scores,
                "path": task_scores[self.tasks[0]][individual_id]['path']
            }

        return weighted_scores

    def evaluate(self, individuals: List, split: str = "valid", return_predictions: bool = False) -> Dict[str, Any]:
        """Evaluate the models."""
        logger.info(f"Start evaluating {len(individuals)} individuals on {len(self.tasks)} tasks...")
        
        if split != "valid":
            logger.warning(f"Evaluate split is not valid, got {split}.")
        
        # 1. Evaluate on each task separately
        all_task_scores = dict()
        for task in self.tasks:
            logger.info(f"Evaluating on task: {task}")
            all_task_scores[task] = self.evaluate_single_task(
                individuals=individuals, 
                task=task, 
                split=split, 
                return_predictions=return_predictions 
            )
        
        # 2. Compute the weighted score
        weighted_scores = self.compute_weighted_score(all_task_scores)
        
        # Merge per-task predictions back into the final result when requested
        if return_predictions:
            for pid, result in weighted_scores.items():
                result['predictions'] = {}
                for task in self.tasks:
                    if pid in all_task_scores[task] and 'predictions' in all_task_scores[task][pid]:
                        result['predictions'][task] = all_task_scores[task][pid]['predictions']
        
        # 3. Update individual fitness scores
        for individual in individuals:
            if individual.id in weighted_scores:
                individual.update_fitness(tasks=self.tasks, task_weights=self.task_weights)
                        
        # 4. Log results
        for individual_id, result in weighted_scores.items():
            logger.info(f"Individual {individual_id}:")
            logger.info(f"  Weighted score: {result['weighted_score']:.4f}")
            for task, score in result['task_scores'].items():
                logger.info(f"  - Task {task}: {score:.4f}")
                
        return weighted_scores
    
    def save_optim_state(self, state: Dict):
        """Save optimization state to JSON file."""
        with open(os.path.join(self.workspace, "state.json"), "w") as f:
            json.dump(state, indent=4, ensure_ascii=False, fp=f)
    
    @abstractmethod
    def search(self) -> None:
        """Execute the optimization search process."""
        pass
    
    def update_optim_state(self, step: int, time: float, weighted_scores: Dict[str, Dict]=None)-> None:
        if weighted_scores:
            task_stats = {task: {"max": -float("inf"), "min": float("inf"), "sum": 0} for task in self.tasks}
            weighted_stats = {"max": -float("inf"), "min": float("inf"), "sum": 0}

            # Collect statistics
            for individual_data in weighted_scores.values():
                # Update weighted-score statistics
                weighted_score = individual_data["weighted_score"]
                weighted_stats["max"] = max(weighted_stats["max"], weighted_score)
                weighted_stats["min"] = min(weighted_stats["min"], weighted_score)
                weighted_stats["sum"] += weighted_score
                
                # Update per-task statistics
                for task, score in individual_data["task_scores"].items():
                    task_stats[task]["max"] = max(task_stats[task]["max"], score)
                    task_stats[task]["min"] = min(task_stats[task]["min"], score)
                    task_stats[task]["sum"] += score
        
            n_individuals = len(weighted_scores)
            
            self.state[f"step_{step}"] = {
                "global_max_fitness_path": self.global_max_fitness_path,
                "global_max_fitness_score": self.global_max_fitness_score,
                "global_min_fitness_path": self.global_min_fitness_path,
                "global_min_fitness_score": self.global_min_fitness_score,
                "average_fitness_score": sum([i.fitness_score for i in self.individuals])/len(self.individuals),
                "consume_time": time,
                "weighted_scores": {
                    "max": weighted_stats["max"],
                    "min": weighted_stats["min"],
                    "avg": weighted_stats["sum"] / n_individuals,
                },
                "task_scores": {
                    task: {"max": task_stats[task]["max"], "min": task_stats[task]["min"], "avg": task_stats[task]["sum"] / n_individuals} for task in self.tasks
                }
            }
        else:
            self.state[f"step_{step}"] = {
                "global_max_fitness_path": self.global_max_fitness_path,
                "global_max_fitness_score": self.global_max_fitness_score,
                "global_min_fitness_path": self.global_min_fitness_path,
                "global_min_fitness_score": self.global_min_fitness_score,
                "all_fitness_score": [i.fitness_score for i in self.individuals],
                "average_fitness_score": sum([i.fitness_score for i in self.individuals])/len(self.individuals),
                "consume_time": time,
            }
    
    def plot_optimization_curves(self):
        """Plot optimization curves including:
        - Global max fitness score
        - Average fitness score
        - Per-task scores
        """
        steps = sorted([int(k.split('_')[1]) for k in self.state.keys() if k.startswith('step_')])
        
        # Prepare data
        global_max_scores = []
        average_scores = []
        task_scores = {task: [] for task in self.tasks}
        
        for step in steps:
            step_data = self.state[f'step_{step}']
            global_max_scores.append(step_data['global_max_fitness_score'])
            average_scores.append(step_data['average_fitness_score'])
            
            if 'task_scores' in step_data:
                for task in self.tasks:
                    task_scores[task].append(step_data['task_scores'][task]['avg'])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot global max and average scores
        plt.plot(steps, global_max_scores, 'b-', label='Global Max', marker='o')
        plt.plot(steps, average_scores, 'r--', label='Average', marker='s')
        
        # Plot task scores if available
        colors = ['g', 'm', 'c', 'y']
        for i, task in enumerate(self.tasks):
            if task_scores[task]:
                plt.plot(steps, task_scores[task], f'{colors[i%len(colors)]}--', 
                        label=f'Task: {task}', marker='.')
        
        plt.xlabel('Step')
        plt.ylabel('Score')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_dir = os.path.join(self.workspace, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'optimization_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Optimization curves saved to {plot_dir}/optimization_curves.png")

    def generate_plots(self):
        """Generate all plots."""
        logger.info("Generating plots...")
        
        try:
            self.plot_optimization_curves()
        except Exception as e:
            logger.error(f"Error generating optimization curves: {e}")
        
        logger.info("All plots generated successfully.")
