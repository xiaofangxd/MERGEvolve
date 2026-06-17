from dataclasses import dataclass, field
from typing import List, Optional
from src.base.base_config import BaseConfig


@dataclass
class PSOConfig(BaseConfig):
    # Candidate expert LoRA directories to merge from.
    pools: List[str] = field(default_factory=list)

    # Optional cache of pre-computed expert scores (directory or json file).
    # Leave as None to always re-evaluate experts.
    initial_scores_path: Optional[str] = None
    task_name_for_scores: Optional[str] = None
    strict_cache_only: bool = False

    N: int = 10
    max_iter: int = 50
    init_step_scale: float = 1.0
    tau: float = 1
    lambda_step: float = 0.5
    phi_lambda: float = 0.95
    phi_inertia: float = 0.2
    phi_cognitive: float = 0.2
    phi_social: float = 0.2
    phi_repel: float = 0.1
    update_mode: str = "es"
    num_selected_experts: int = 10
    alpha: float = 0.1
    sigma: float = 0.1
    n_samples: int = 10
    unable_random: bool = False
    max_workers: int = 1

    def validate(self):
        super().validate()
        if not self.pools:
            raise ValueError("LoRA pools must be specified (use --lora_dir).")
        if self.N < 1:
            raise ValueError("N must be greater than 0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be greater than 0")
        if not (0 <= self.lambda_step <= 1):
            raise ValueError("lambda_step must be in range [0, 1]")
        if not (0 <= self.phi_lambda <= 1):
            raise ValueError("phi_lambda must be in range [0, 1]")
        if not all(0 <= phi <= 1 for phi in [self.phi_cognitive, self.phi_inertia, self.phi_repel, self.phi_social]):
            raise ValueError("phi values must be in range [0, 1]")
        if self.max_workers < 1:
            raise ValueError("max_workers must be greater than 0")
        if len(self.llm_base_url) < self.max_workers:
            raise ValueError("Not enough llm_base_url for workers")
