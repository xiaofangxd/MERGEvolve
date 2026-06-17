import json
import os
import random
import time
import uuid
from typing import Dict, List, Any

import numpy as np
import torch
from loguru import logger
from pathlib import Path

from src.mergevole.config import PSOConfig
from src.mergevole.particle import Particle
from src.utils import load_lora_weight
from src.base.base_method import BaseMethod
from concurrent.futures import ThreadPoolExecutor, as_completed 


def _cmaes_utilities(lam: int) -> np.ndarray:
    """CMA-ES style recombination weights, recentered to sum to zero (rank-based)."""
    assert lam >= 1
    cutoff = np.log(lam / 2.0 + 1.0)
    raw = np.array([max(0.0, cutoff - np.log(k)) for k in range(1, lam + 1)], dtype=np.float64)
    s = raw.sum()
    w = raw / s if s > 0 else np.zeros(lam, dtype=np.float64)
    u = w - 1.0 / lam  
    return u

def _rank_based_utilities(performance: List[float], higher_is_better: bool = True) -> np.ndarray:
    """Map raw performances to rank-based utilities (invariant to score scale)."""
    lam = len(performance)
    u_rank = _cmaes_utilities(lam)
    idx = np.argsort(performance)
    if higher_is_better:
        idx = idx[::-1]
    u = np.zeros(lam, dtype=np.float64)
    for j, i in enumerate(idx):
        u[i] = u_rank[j]
    return u

def _signed_merge_by_utilities(
    lora_state_dicts: List[Dict[str, torch.Tensor]],
    utilities: np.ndarray,
    ref_idx: int | None = None,
    center: str = "none",           
    step_scale: float = 1,        
) -> tuple[Dict[str, torch.Tensor], dict]:
    """Fuse a set of LoRA state dicts into a single initial point theta0.

    Each expert is weighted by its utility and (optionally) centered around the
    population mean, yielding theta0 = mu + step_scale * sum_i(u_i * lora_i).
    """
    lam = len(lora_state_dicts)
    assert lam == len(utilities)

    if ref_idx is None:
        ref_idx = int(np.argmax(np.abs(utilities)))

    all_keys = set().union(*[sd.keys() for sd in lora_state_dicts])

    if center == "mean":
        mu: Dict[str, torch.Tensor] = {}
        for kname in all_keys:
            ref_t = None
            for sd in lora_state_dicts:
                if kname in sd:
                    ref_t = sd[kname]
                    break
            acc = torch.zeros_like(ref_t)
            cnt = 0
            for sd in lora_state_dicts:
                if kname in sd:
                    acc = acc + sd[kname]
                    cnt += 1
            mu[kname] = acc / max(cnt, 1)
    else:
        mu = {
            kname: torch.zeros_like(next(sd[kname] for sd in lora_state_dicts if kname in sd))
            for kname in all_keys
        }

    S: Dict[str, torch.Tensor] = {}
    for kname in all_keys:
        ref_t = None
        for sd in lora_state_dicts:
            if kname in sd:
                ref_t = sd[kname]
                break
        acc = torch.zeros_like(ref_t)
        for u_i, sd in zip(utilities, lora_state_dicts):
            t = sd.get(kname, torch.zeros_like(ref_t))
            acc = acc + float(u_i) * t
        S[kname] = acc

    renorm_scale = 1
    theta0 = {kname: (mu[kname] + step_scale * renorm_scale * S[kname]) for kname in all_keys}
    meta = {
        "ref_idx": ref_idx,
        "renorm_scale": float(renorm_scale),
        "step_scale": float(step_scale),
    }
    return theta0, meta


def _zscore_softmax_utilities(
    performance: List[float],
    tau: float = 0.2,
    higher_is_better: bool = True,
    eps: float = 1e-8
) -> np.ndarray:
    """Convert performances into normalized weights via z-score + temperature softmax."""
    s = np.asarray(performance, dtype=np.float64)
    if not higher_is_better:
        s = -s
    mu = s.mean()
    sd = s.std(ddof=0)
    z = (s - mu) / (sd + eps)
    logits = z / max(tau, eps)
    logits = logits - logits.max() 
    e = np.exp(logits)
    w = e / (e.sum() + eps)
    return w



class MERGEVOLE(BaseMethod):
    """Swarm/evolutionary optimizer that searches the LoRA weight space.

    Starting from a utility-weighted fusion of candidate expert LoRAs, MERGEVOLE
    iteratively refines a single merged adapter (theta) using either a Particle
    Swarm Optimization (``pso``) or an Evolution Strategy (``es``) update rule,
    with fitness measured by downstream task accuracy served through vLLM.
    """

    def __init__(self, config: PSOConfig) -> None:
        super().__init__(config)
        self.config = config
        self.config.validate()
        self.config.save(path=self.workspace)

        self.global_patience_counter = 0
        self.patience_flag = True


        self.lora_config = self.config.pools[0]

        self.theta: Dict[str, torch.Tensor] | None = None
        self.sigma = getattr(self.config, "sigma", 0.1)
        self.update_mode = getattr(self.config, "update_mode", "es") 

        self.initial_scores_path = getattr(self.config, "initial_scores_path", None)
        self.task_name_for_scores = getattr(self.config, "task_name_for_scores", None)
        self.strict_cache_only = getattr(self.config, "strict_cache_only", False) 
        self.allow_cache_update = getattr(self.config, "allow_cache_update", True)

        self._cached_scores_map: dict[str, float] = {}


    # cache
    def _resolve_task_name_for_scores(self) -> str:
        if self.task_name_for_scores:
            return str(self.task_name_for_scores)
        if isinstance(self.tasks, (list, tuple)) and len(self.tasks) > 0:
            return str(self.tasks[0])
        return "default"

    def _resolve_scores_file(self, task_name: str) -> str | None:
        root = self.initial_scores_path
        if not root:
            return None
        p = Path(root)
        if p.is_dir():
            return str(p / f"{task_name}.json")
        return str(p)

    def _load_scores_json(self, fp: str | None) -> dict[str, float]:
        if not fp:
            return {}
        p = Path(fp)
        if not p.exists():
            return {}
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "experts" in data and isinstance(data["experts"], dict):
                return {str(k): float(v) for k, v in data["experts"].items()}
            else:
                return {str(k): float(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"[CACHE] Failed to read scores json {fp}: {e}")
            return {}

    def _save_scores_json_incremental(self, fp: str | None, update_map: dict[str, float], task_name: str | None = None):
        if not fp or not self.allow_cache_update:
            return
        p = Path(fp)
        p.parent.mkdir(parents=True, exist_ok=True)
        old = self._load_scores_json(fp)
        old.update({str(k): float(v) for k, v in update_map.items()})

        payload = {"experts": old}
        if task_name is not None:
            payload["task"] = task_name
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"[CACHE] Saved/updated {len(update_map)} scores → {p}")
        except Exception as e:
            logger.warning(f"[CACHE] Failed to write scores json {fp}: {e}")

    def random_r(self) -> None:
        if self.unable_random:
            self.r = [1, 1, 1, 1]
        else:
            r = random.Random(self.seed)
            self.r = [r.random() for _ in range(4)]

    def compute_C(self) -> None:
        self.C = sum([self.r[i] * self.phi[i] for i in range(4)])

    def update_lambda(self) -> None:
        self._lambda *= self.config.phi_lambda

    def print_config(self) -> None:
        logger.info(
            f"Config: N={self.config.N}, max_iter={self.config.max_iter}, "
            f"tasks={self.tasks}, task_weights={self.task_weights}, "
            f"max_workers={self.config.max_workers}\n"
            f"Combine method: {self.combine_method}\n"
            f"Weights (φ): inertia={self.config.phi_inertia:.3f}, "
            f"cognitive={self.config.phi_cognitive:.3f}, "
            f"social={self.config.phi_social:.3f}, "
            f"repel={self.config.phi_repel:.3f}, "
            f"lambda={self._lambda:.3f}, "
            f"phi_lambda={self.config.phi_lambda:.3f}\n"
            f"Random (r): unable={self.unable_random}, "
            f"inertia={self.r[0]:.3f}, "
            f"cognitive={self.r[1]:.3f}, "
            f"social={self.r[2]:.3f}, "
            f"repel={self.r[3]:.3f}, "
            f"sigma={self.sigma}, "
            f"alpha={self.config.alpha}, "
            f"init_step_scale={getattr(self.config, 'init_step_scale', 1.0)}, "
            f"tau={getattr(self.config, 'tau', 1)}"
        )

    def initialize(self) -> None:
        """Sample candidate experts, score them, and fuse into the initial theta."""
        logger.info("Initializing individuals ...")
        start_time = time.time()
        self.individuals = []

        num_to_select = getattr(self.config, "num_selected_experts", len(self.pools))
        num_to_select = min(num_to_select, len(self.pools))
        if num_to_select <= 0:
            raise ValueError("No experts available or selected for initialization.")

        selected_experts = random.sample(self.pools, num_to_select)
        logger.info(f"Selected {len(selected_experts)} experts for initialization:")
        for i, expert_path in enumerate(selected_experts):
            expert_name = os.path.basename(expert_path.rstrip('/'))
            logger.info(f"  Expert {i+1}: {expert_name} (path: {expert_path})")

        for i, expert in enumerate(selected_experts):
            expert_weight = load_lora_weight(expert)
            particle_id = uuid.uuid4().hex
            p = Particle(
                id=i,
                x=expert_weight,
                parent=expert,
                weight_path=os.path.join(self.workspace, f"particle_{particle_id}"),
                model_name_or_path=self.model_name_or_path,
                lora_config_path=self.lora_config,
            )
            p.save_particle(save_path=p.weight_path)
            p.evaluated = {task: False for task in self.tasks}
            self.individuals.append(p)

        task_name = self._resolve_task_name_for_scores()
        cache_file = self._resolve_scores_file(task_name)
        cached = self._load_scores_json(cache_file)
        self._cached_scores_map = cached 

        performance_map: dict[str, float] = {}
        need_eval: list[Particle] = []

        for p in self.individuals:
            exp_path = p.parent if isinstance(p.parent, str) else p.parent[0]
            if exp_path in cached:
                performance_map[exp_path] = float(cached[exp_path])
            else:
                need_eval.append(p)

        if need_eval:
            if self.strict_cache_only:
                miss = [p.parent if isinstance(p.parent, str) else p.parent[0] for p in need_eval]
                raise RuntimeError(
                    f"[CACHE] strict_cache_only=True, but {len(miss)} experts are missing in {cache_file}:\n" +
                    "\n".join(miss)
                )
            logger.info(f"[CACHE] {len(need_eval)} experts missing cached score → evaluating them once...")
            eval_res = self.evaluate(individuals=need_eval, split="valid")

            new_insert = {}
            for p in need_eval:
                matched = None
                for _, v in eval_res.items():
                    if v.get("path") == p.weight_path:
                        matched = v
                        break
                if matched is None:
                    logger.warning(f"[CACHE] Cannot match eval result for {p.weight_path}, fallback to first entry.")
                    matched = list(eval_res.values())[0]

                score = float(matched["weighted_score"])
                exp_path = p.parent if isinstance(p.parent, str) else p.parent[0]
                performance_map[exp_path] = score
                new_insert[exp_path] = score

            self._save_scores_json_incremental(cache_file, new_insert, task_name=task_name)
            self._cached_scores_map = self._load_scores_json(cache_file)

        initial_cached_scores = {}
        for p in self.individuals:
            exp_path = p.parent if isinstance(p.parent, str) else p.parent[0]
            score = performance_map[exp_path]
            initial_cached_scores[str(p.id)] = {
                'weighted_score': score,
                'path': p.weight_path
            }


        performance = []
        for p in self.individuals:
            exp_path = p.parent if isinstance(p.parent, str) else p.parent[0]
            performance.append(performance_map[exp_path])
        logger.info(f"[CACHE] Loaded initial scores for {len(performance)} experts (avg={np.mean(performance):.4f})")

        tau = float(getattr(self.config, "tau", 1))
        utilities = _zscore_softmax_utilities(performance, tau=tau, higher_is_better=True)

        expert_loras = [p.x for p in self.individuals]
        init_step_scale = float(getattr(self.config, "init_step_scale", 1.0))
        theta0, meta = _signed_merge_by_utilities(
            expert_loras, utilities, ref_idx=None, center="none", step_scale=init_step_scale
        )

        self.theta = {k: v.clone() for k, v in theta0.items()}
        logger.info(f"[INIT] utilities(softmax,sum=1)={np.round(utilities,4)} "
                    f"(zscore+softmax, step_scale={meta['step_scale']:.4f}, renorm_scale={meta['renorm_scale']:.4f}, tau={tau})")

        logger.info("Evaluating the fused model (theta) after initialization...")
        theta_particle = self._create_particle_from_theta()
        theta_scores = self.evaluate(individuals=[theta_particle], split="valid")
        
        if theta_scores:
            theta_result = list(theta_scores.values())[0]
            theta_score = float(theta_result['weighted_score'])
            logger.info(f"Fused model (theta) performance: Weighted score = {theta_score:.4f}")
            self.performance_parent = theta_score
            self.global_max_fitness_score = theta_score
            self.global_max_fitness_path = theta_result['path']
            self.global_max_task_scores = theta_result['task_scores']
        else:
            logger.warning("Failed to evaluate fused theta; fallback to parent_score = max(expert).")
            self.performance_parent = float(np.max(performance))

        end_time = time.time()
        logger.info(f"Initialization finished in {end_time-start_time:.2f} seconds.")


    def velocity_update(self) -> None:
        for p in self.individuals:
            assert hasattr(self, "global_max_fitness_weight") and hasattr(self, "global_min_fitness_weight"), \
                "Global max and min fitness weight must be set."
            p.update_velocity(
                global_max_weight=self.global_max_fitness_weight,
                global_min_weight=self.global_min_fitness_weight,
                r=self.r,
                phi=self.phi,
                C=self.C
            )

    def weight_update(self, step: int) -> None:
        for p in self.individuals:
            self.update_lambda()
            p.update_weight(_lambda=self._lambda)
            save_path = os.path.join(self.workspace, f"particle_{p.id}")
            p.save_particle(save_path=save_path)
            p.evaluated = {task: False for task in self.tasks}

    def individual_generate(self, step: int, sigma: float) -> List[Dict[str, torch.Tensor]]:
        """ES exploration: sample n Gaussian perturbations around theta.

        Returns the noise vectors (epsilons) used later to estimate the gradient.
        """
        n = getattr(self.config, "n_samples", 10)
        epsilons: List[Dict[str, torch.Tensor]] = []
        new_individuals: List[Particle] = []

        for _ in range(n):
            eps: Dict[str, torch.Tensor] = {}
            new_x: Dict[str, torch.Tensor] = {}
            for key, tensor in self.theta.items():
                noise = torch.normal(mean=0.0, std=1, size=tensor.shape, device=tensor.device)
                eps[key] = noise
                new_x[key] = self.theta[key] + sigma * noise

            epsilons.append(eps)

            particle_id = uuid.uuid4().hex
            p = Particle(
                id=particle_id,
                x=new_x,
                parent="theta+noise",
                weight_path=os.path.join(self.workspace, f"particle_{particle_id}"),
                model_name_or_path=self.model_name_or_path,
                lora_config_path=self.lora_config,
            )
            p.save_particle(save_path=p.weight_path)
            p.evaluated = {task: False for task in self.tasks}
            new_individuals.append(p)

        self.individuals = new_individuals
        return epsilons

    def update_theta(self, step: int, performance: List[float], epsilons: List[Dict[str, torch.Tensor]],
                     alpha: float, sigma: float):
        """Apply the ES natural-gradient step: theta <- theta + (alpha / (sigma*n)) * sum_i(u_i * eps_i)."""
        if not hasattr(self, 'performance_parent'):
            raise ValueError("self.performance_parent (parent theta score) must be set before calling update_theta.")

        n = len(performance)
        assert n == len(epsilons), "performance and epsilons must have the same length"

        u = _rank_based_utilities(performance, higher_is_better=True) 
        logger.info(f"[STEP {step}] z-score utilities (sum≈0): {np.round(u, 4)}")
        logger.info(f"[STEP {step}] Performance per sample: {np.round(performance, 4)}")

        update: Dict[str, torch.Tensor] = {key: torch.zeros_like(val) for key, val in self.theta.items()}
        for u_i, eps in zip(u, epsilons):
            coef = float(u_i)
            if coef == 0.0:
                continue
            for key in eps.keys():
                update[key] = update[key] + coef * eps[key] 

        step_scale = alpha / (sigma * n + 1e-12)
        for key in self.theta.keys():
            self.theta[key] = self.theta[key] + step_scale * update[key]

        self.performance_parent = max(performance)





    def single_search(self, step: int) -> None:
        """Run one optimization iteration."""
        start_time = time.time()
        logger.info(f"Start searching for {step} steps ...")
        logger.info("Update velocity & weight ...")

        epsilons = None

        if self.update_mode == "pso":
            self.velocity_update()
            self.weight_update(step=step)
        elif self.update_mode == "es":
            theta_parent_particle = self._create_particle_from_theta()
            parent_scores = self.evaluate(individuals=[theta_parent_particle], split="valid")
            
            if parent_scores:
                self.performance_parent = list(parent_scores.values())[0]['weighted_score']
                logger.info(f"[ES] parent theta score = {self.performance_parent:.4f}")
            else:
                logger.warning("[ES] Failed to evaluate parent theta; keep previous performance_parent.")

            epsilons = self.individual_generate(step=step, sigma=self.sigma)
        else:
            raise ValueError(f"Unknown update_mode: {self.update_mode}")

        logger.info("Evaluating update position ...")
        logger.info("[DEBUG] Evaluating the following lora paths:")
        for ind in self.individuals:
            logger.info(f"    id={ind.id}  path={ind.weight_path}")

        weighted_scores = self.evaluate(individuals=self.individuals, split="valid")
        

        performance: List[float] = []
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result['weighted_score'],
                path=result['path'],
                task_scores=result['task_scores']
            )
            performance.append(result['weighted_score'])

        if self.update_mode == "es":
            assert epsilons is not None, "epsilons must not be None in ES mode"
            self.update_theta(
                step=step,
                performance=performance,
                epsilons=epsilons,
                alpha=getattr(self.config, "alpha", 0.1),
                sigma=self.sigma
            )

        end_time = time.time()
        logger.info(f"Step {step} finished in {end_time - start_time:.2f} seconds.")
        logger.info(f"Step {step} completed. Updated sigma: {self.sigma}")

        self.update_optim_state(step=step, time=end_time - start_time, weighted_scores=weighted_scores)
        self.save_optim_state(state=self.state)
        self.report_state(step=step)

    def search(self):
        """Full optimization loop: init, iterate with early stopping, then test the best adapter."""
        start_time = time.time()
        self.print_config()
        self.initialize()
        logger.info("Collecting Pre-Swarm Correctness Data...")
        pre_swarm_results = self.evaluate(
            individuals=self.individuals, 
            split="test",  
            return_predictions=True
        )
        analysis_dir = Path(self.workspace) / "analysis_data"
        analysis_dir.mkdir(parents=True, exist_ok=True) 
        pre_path = analysis_dir / "pre_swarm_analysis.pt"
        logger.info(f"Saving Pre-Swarm results to: {pre_path.absolute()}")
        torch.save(pre_swarm_results, os.path.join(self.workspace, "pre_swarm_analysis.pt"))

        for step in range(1, self.config.max_iter + 1):
            self.single_search(step=step)

            if self.patience_flag:
                self.global_patience_counter += 1
                if self.global_patience_counter > self.early_stop_iter and self.early_stop:
                    logger.info("Early stop!")
                    break
            else:
                self.global_patience_counter = 0
                
        if hasattr(self, "global_max_fitness_weight") and self.global_max_fitness_weight is not None:
            logger.info("Re-evaluating the best individual on test split ...")
            best_particle = Particle(
                id="best_final",
                x=self.global_max_fitness_weight,
                parent="best_global",
                weight_path=self.global_max_fitness_path,
                model_name_or_path=self.model_name_or_path,
                lora_config_path=self.lora_config,
            )
            best_result = self.evaluate(individuals=[best_particle], split="test")
            logger.info(f"[TEST] Final best individual test result: {best_result}")
        else:
            logger.warning("No global_max_fitness_weight found. Skip best test evaluation.")

        end_time = time.time()
        try:
            self.save_final_state(individuals=self.individuals, time=end_time - start_time)
        except Exception as e:
            self.save_optim_state(self.state)
            logger.error(f"Error saving final state: {e}")

        if self.plot_enabled:
            try:
                self.generate_plots()
            except Exception as e:
                logger.error(f"Error generating plots: {e}")
        logger.info("Collecting Post-Swarm Correctness Data...")
        post_swarm_results = self.evaluate(
            individuals=self.individuals, 
            split="test", 
            return_predictions=True
        )
        post_path = analysis_dir / "post_swarm_analysis.pt"
        logger.info(f"Saving Post-Swarm results to: {post_path.absolute()}")
        torch.save(post_swarm_results, os.path.join(self.workspace, "post_swarm_analysis.pt"))


    def _create_particle_from_theta(self) -> Particle:
        """Materialize the current theta as a saved Particle for evaluation."""
        assert self.theta is not None, "theta is not initialized"
        particle_id = uuid.uuid4().hex
        particle_path = os.path.join(self.workspace, f"theta_{particle_id}")
        p = Particle(
            id=particle_id,
            x=self.theta,
            parent="theta",
            weight_path=particle_path,
            model_name_or_path=self.model_name_or_path,
            lora_config_path=self.lora_config,
        )
        p.save_particle(save_path=particle_path)
        return p