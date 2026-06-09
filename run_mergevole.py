import argparse
import os

from src.mergevole.mergevole import MERGEVOLE
from src.mergevole.config import PSOConfig
from src.utils import get_base_url


def parse_args():
    parser = argparse.ArgumentParser(description="MERGEVOLE: Swarm Optimization for LoRA Weight Combination (PSO / ES)")

    # === Core Model and Data Paths ===
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the base model")
    parser.add_argument("--lora_dir", type=str, required=True,
                        help="Directory containing the candidate expert LoRA checkpoints (one sub-directory per expert)")

    # Task Configuration
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="Validation tasks (e.g., mmlu hellaswag)")
    parser.add_argument("--test_tasks", type=str, nargs="+", required=True,
                        help="Final test tasks")
    parser.add_argument("--task_weights", type=float, nargs="+",
                        help="Weights for each task (will be normalized to sum to 1)")

    # Optimization Method
    parser.add_argument("--update_mode", type=str, default="es", choices=["pso", "es"],
                        help="Update strategy: 'pso' (Particle Swarm) or 'es' (Evolution Strategy)")

    # === ES params ===
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Learning rate for ES update")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Noise std for ES perturbations")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of perturbations per step in ES")
    parser.add_argument("--tau", type=float, default=1,
                        help="Temperature for Softmax utilities (lower value = sharper selection). Recommended 0.1 for 7B models.")

    # vLLM / service config
    parser.add_argument("--ports", type=int, nargs="+",
                        default=[9113, 9114, 9115, 9116],
                        help="List of ports for vLLM services")

    # Population / search config
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of iterations")
    parser.add_argument("--num_selected_experts", type=int, default=10,
                        help="Number of experts sampled from --lora_dir to initialize the search")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Early Stopping
    parser.add_argument("--early_stop", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--early_stop_iter", type=int, default=5,
                        help="Patience for early stopping")

    # === PSO params (only used when --update_mode pso) ===
    parser.add_argument("--phi_inertia", type=float, default=0.2)
    parser.add_argument("--phi_cognitive", type=float, default=0.3)
    parser.add_argument("--phi_social", type=float, default=0.4)
    parser.add_argument("--phi_repel", type=float, default=0.05)
    parser.add_argument("--phi_lambda", type=float, default=0.95)
    parser.add_argument("--lambda_step", type=float, default=1.0)
    parser.add_argument("--unable_random", action="store_true")

    # Combine Method
    parser.add_argument("--combine_method", type=str, default="ties", choices=["linear", "ties", "magnitude"],
                        help="Method to combine weights")

    args = parser.parse_args()

    # normalize weights
    if args.task_weights:
        total = sum(args.task_weights)
        if total > 0:
            args.task_weights = [w / total for w in args.task_weights]
        else:
            raise ValueError("Sum of task weights must be positive.")

    return args


def collect_lora_pools(lora_dir: str):
    """List candidate expert LoRA sub-directories under ``lora_dir``."""
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(f"lora_dir not found: {lora_dir}")
    pools = sorted(
        os.path.join(lora_dir, d)
        for d in os.listdir(lora_dir)
        if os.path.isdir(os.path.join(lora_dir, d))
    )
    if not pools:
        raise ValueError(f"No expert LoRA sub-directories found under {lora_dir}")
    return pools


def main():
    args = parse_args()

    config = PSOConfig(
        tasks=args.tasks,
        test_tasks=args.test_tasks,
        task_weights=args.task_weights,

        max_iter=args.iters,
        llm_base_url=get_base_url(args.ports),
        combine_method=args.combine_method,

        plot_enabled=False,
        model_name_or_path=args.model_path,

        # Candidate experts
        pools=collect_lora_pools(args.lora_dir),
        num_selected_experts=args.num_selected_experts,

        # PSO params
        phi_inertia=args.phi_inertia,
        phi_cognitive=args.phi_cognitive,
        phi_social=args.phi_social,
        phi_repel=args.phi_repel,
        phi_lambda=args.phi_lambda,
        lambda_step=args.lambda_step,
        unable_random=args.unable_random,

        # ES params
        update_mode=args.update_mode,
        alpha=args.alpha,
        sigma=args.sigma,
        n_samples=args.n_samples,
        tau=args.tau,

        # early stop
        early_stop=args.early_stop,
        early_stop_iter=args.early_stop_iter,
        seed=args.seed
    )

    print("Running MERGEVOLE with config:")
    print(config)

    mergevole = MERGEVOLE(config=config)
    mergevole.search()


if __name__ == "__main__":
    main()
